/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*-----------------------------------------------------------------------------------------------100
Dump per-atom data to user-specified file(s) in the extended XYZ format
--------------------------------------------------------------------------------------------------*/

#include "dump_xyz.cuh"
#include "force/force.cuh"
#include "force/neighbor.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>
#include <vector>

static __global__ void gpu_sum(const int N, const double* g_data, double* g_data_sum)
{
  int number_of_rounds = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  s_data[threadIdx.x] = 0.0;
  for (int round = 0; round < number_of_rounds; ++round) {
    int n = threadIdx.x + round * 1024;
    if (n < N) {
      s_data[threadIdx.x] += g_data[n + blockIdx.x * N];
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      s_data[threadIdx.x] += s_data[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    g_data_sum[blockIdx.x] = s_data[0];
  }
}

static void generate_fibonacci_directions(const int M, std::vector<double>& directions)
{
  directions.resize(3 * M);
  const double golden = PI * (3.0 - sqrt(5.0));
  for (int m = 0; m < M; ++m) {
    const double z = 1.0 - 2.0 * (m + 0.5) / double(M);
    const double xy = sqrt(fmax(0.0, 1.0 - z * z));
    const double theta = m * golden;
    directions[m] = xy * cos(theta);
    directions[m + M] = xy * sin(theta);
    directions[m + 2 * M] = z;
  }
}

// One warp (32 threads) per atom. Each thread holds Q directions, so M = 32 * Q.
// blockDim.x must be a multiple of 32.
template <int Q>
static __global__ void gpu_voronoi_volume(
  const int N,
  const Box box,
  const double R,
  const double* __restrict__ directions,
  const int* __restrict__ NN,
  const int* __restrict__ NL,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  double* __restrict__ volume)
{
  constexpr unsigned FULL = 0xffffffffu;
  constexpr int M = 32 * Q;

  const int lane = threadIdx.x & 31;
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (i >= N)
    return;

  const double inv_R = 1.0 / R;
  const double cutoff2 = 4.0 * R * R;
  const double* dir_x = directions;
  const double* dir_y = directions + M;
  const double* dir_z = directions + 2 * M;

  double q[Q];
  double nx[Q];
  double ny[Q];
  double nz[Q];

#pragma unroll
  for (int a = 0; a < Q; ++a) {
    const int m = lane + 32 * a;
    nx[a] = dir_x[m];
    ny[a] = dir_y[m];
    nz[a] = dir_z[m];
    q[a] = inv_R;
  }

  int count = 0;
  if (lane == 0)
    count = NN[i];
  count = __shfl_sync(FULL, count, 0);

  const double x1 = x[i];
  const double y1 = y[i];
  const double z1 = z[i];

  for (int k = 0; k < count; ++k) {
    double bx = 0.0;
    double by = 0.0;
    double bz = 0.0;
    int active = 0;

    if (lane == 0) {
      const int j = NL[i + k * N];
      double dx = x[j] - x1;
      double dy = y[j] - y1;
      double dz = z[j] - z1;
      apply_mic(box, dx, dy, dz);
      const double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > 0.0 && d2 < cutoff2) {
        const double scale = 2.0 / d2;
        bx = scale * dx;
        by = scale * dy;
        bz = scale * dz;
        active = 1;
      }
    }

    active = __shfl_sync(FULL, active, 0);
    if (active == 0)
      continue;

    bx = __shfl_sync(FULL, bx, 0);
    by = __shfl_sync(FULL, by, 0);
    bz = __shfl_sync(FULL, bz, 0);

#pragma unroll
    for (int a = 0; a < Q; ++a) {
      const double projection = bx * nx[a] + by * ny[a] + bz * nz[a];
      q[a] = fmax(q[a], projection);
    }
  }

  double sum = 0.0;
#pragma unroll
  for (int a = 0; a < Q; ++a) {
    const double radius = 1.0 / q[a];
    sum += radius * radius * radius;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(FULL, sum, offset);
  }

  if (lane == 0) {
    volume[i] = (4.0 * PI / (3.0 * M)) * sum;
  }
}

Dump_XYZ::Dump_XYZ(const char** param, int num_param, const std::vector<Group>& groups, Atom& atom) 
{
  is_nep_charge = check_is_nep_charge();

  parse(param, num_param, groups);
  if (atom.unwrapped_position.size() < atom.number_of_atoms * 3) {
    atom.unwrapped_position.resize(atom.number_of_atoms * 3);
    atom.unwrapped_position.copy_from_device(atom.position_per_atom.data());
  }
  if (atom.position_temp.size() < atom.number_of_atoms * 3) {
    atom.position_temp.resize(atom.number_of_atoms * 3);
  }
  property_name = "dump_xyz";
}

void Dump_XYZ::parse(const char** param, int num_param, const std::vector<Group>& groups)
{
  printf("Dump extended XYZ.\n");

  if (num_param < 5) {
    PRINT_INPUT_ERROR("dump_xyz should have at least 4 parameters.\n");
  }

  // grouping_method
  if (!is_valid_int(param[1], &grouping_method_)) {
    PRINT_INPUT_ERROR("grouping method of dump_xyz should be integer.");
  }
  if (grouping_method_ < 0) {
    printf("    for the whole system.\n");
  } else {
    if (grouping_method_ >= int(groups.size())) {
      PRINT_INPUT_ERROR("grouping method exceeds the bound.");
    }
    printf("    for grouping method %d.\n", grouping_method_);
  }

  // group_id
  if (!is_valid_int(param[2], &group_id_)) {
    PRINT_INPUT_ERROR("group id of dump_xyz should be integer.");
  }
  if (grouping_method_ >= 0) {
    if (group_id_ >= groups[grouping_method_].number) {
      PRINT_INPUT_ERROR("group id exceeds the bound.");
    }
    if (group_id_ < 0) {
      PRINT_INPUT_ERROR("group id is negative.");
    }
    printf("    for group id %d.\n", group_id_);
  }

  if (!is_valid_int(param[3], &dump_interval_)) {
    PRINT_INPUT_ERROR("dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("dump interval should > 0.");
  } else {
    printf("    every %d steps.\n", dump_interval_);
  }

  // filename
  std::string filename_temp = param[4];
  printf("    into file %s.\n", filename_temp.c_str());
  if (filename_temp.back() == '*') {
    separated_ = 1;
    filename_ = filename_temp.substr(0, filename_temp.size() - 1);
  } else {
    separated_ = 0;
    filename_ = filename_temp;
  }

  for (int m = 5; m < num_param; ++m) {
    if (strcmp(param[m], "velocity") == 0) {
      quantities.has_velocity_ = true;
      printf("    has velocity.\n");
    }
    if (strcmp(param[m], "force") == 0) {
      quantities.has_force_ = true;
      printf("    has force.\n");
    }
    if (strcmp(param[m], "potential") == 0) {
      quantities.has_potential_ = true;
      printf("    has potential.\n");
    }
    if (strcmp(param[m], "unwrapped_position") == 0) {
      quantities.has_unwrapped_position_ = true;
      printf("    has unwrapped position.\n");
    }
    if (strcmp(param[m], "mass") == 0) {
      quantities.has_mass_ = true;
      printf("    has mass.\n");
    }
    if (strcmp(param[m], "charge") == 0) {
      quantities.has_charge_ = true;
      if (is_nep_charge){
        printf("    has charge predicted by NEP-charge.\n");
      } else {
        printf("    has charge specified in model.xyz.\n");
      }
    }
    if (strcmp(param[m], "bec") == 0) {
      quantities.has_bec_ = true;
      if (is_nep_charge){
        printf("    has BEC predicted by NEP-charge.\n");
      } else {
        PRINT_INPUT_ERROR("Cannot output BEC for a non-NEP-charge model.\n");
      }
    }
    if (strcmp(param[m], "virial") == 0) {
      quantities.has_virial_ = true;
      printf("    has virial.\n");
    }
    if (strcmp(param[m], "group") == 0) {
      quantities.has_group_ = true;
      printf("    has group.\n");
    }
    if (strcmp(param[m], "volume") == 0) {
      if (m + 2 >= num_param) {
        PRINT_INPUT_ERROR("dump_xyz volume should be followed by R and the number of directions.\n");
      }
      if (!is_valid_real(param[m + 1], &voronoi_radius_)) {
        PRINT_INPUT_ERROR("Voronoi radius R should be a number.\n");
      }
      if (!(voronoi_radius_ > 0.0)) {
        PRINT_INPUT_ERROR("Voronoi radius R should > 0.\n");
      }
      if (!is_valid_int(param[m + 2], &voronoi_directions_)) {
        PRINT_INPUT_ERROR("number of Voronoi directions should be an integer.\n");
      }
      if (voronoi_directions_ != 128 && voronoi_directions_ != 256) {
        PRINT_INPUT_ERROR("number of Voronoi directions should be 128 or 256.\n");
      }
      quantities.has_volume_ = true;
      printf("    has ball-restricted Voronoi volume.\n");
      printf("        R = %g Angstrom.\n", voronoi_radius_);
      printf("        directions = %d.\n", voronoi_directions_);
      m += 2;
    }
  }
}

void Dump_XYZ::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (separated_ == 0) {
    fid_ = my_fopen(filename_.c_str(), "a");
  }

  gpu_total_virial_.resize(6);
  cpu_total_virial_.resize(6);
  if (quantities.has_force_) {
    cpu_force_per_atom_.resize(atom.number_of_atoms * 3);
  }
  if (quantities.has_potential_) {
    cpu_potential_per_atom_.resize(atom.number_of_atoms);
  }
  if (quantities.has_unwrapped_position_) {
    cpu_unwrapped_position_.resize(atom.number_of_atoms * 3);
  }
  if (quantities.has_virial_) {
    cpu_virial_per_atom_.resize(atom.number_of_atoms * 9);
  }
  if (quantities.has_bec_) {
    cpu_bec_.resize(atom.number_of_atoms * 9);
  }
  if (quantities.has_volume_) {
    const int N = atom.number_of_atoms;
    const double rc = 2.0 * voronoi_radius_;
    const double sphere = (4.0 / 3.0) * PI * rc * rc * rc;
    int mn = int(0.2 * sphere) + 64;
    if (mn < 256)
      mn = 256;
    if (mn > 1024)
      mn = 1024;
    voronoi_mn_ = mn;

    gpu_volume_per_atom_.resize(N);
    cpu_volume_per_atom_.resize(N);
    voronoi_cell_count_.resize(N);
    voronoi_cell_count_sum_.resize(N);
    voronoi_cell_contents_.resize(N);
    voronoi_NN_.resize(N);
    voronoi_NL_.resize(N * voronoi_mn_);

    std::vector<double> cpu_directions;
    generate_fibonacci_directions(voronoi_directions_, cpu_directions);
    gpu_voronoi_directions_.resize(cpu_directions.size());
    gpu_voronoi_directions_.copy_from_host(cpu_directions.data());
  }
}

void Dump_XYZ::output_line2(
  const double time,
  const Box& box,
  std::vector<Group>& groups,
  const std::vector<std::string>& cpu_atom_symbol,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& gpu_thermo)
{
  // time
  fprintf(fid_, "Time=%.8f", time * TIME_UNIT_CONVERSION); // output time is in units of fs

  // PBC
  fprintf(
    fid_, " pbc=\"%c %c %c\"", box.pbc_x ? 'T' : 'F', box.pbc_y ? 'T' : 'F', box.pbc_z ? 'T' : 'F');

  // box
  fprintf(
    fid_,
    " Lattice=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\"",
    box.cpu_h[0],
    box.cpu_h[3],
    box.cpu_h[6],
    box.cpu_h[1],
    box.cpu_h[4],
    box.cpu_h[7],
    box.cpu_h[2],
    box.cpu_h[5],
    box.cpu_h[8]);

  // energy and virial (symmetric tensor) in eV, and stress (symmetric tensor) in eV/A^3
  double cpu_thermo[8];
  gpu_thermo.copy_to_host(cpu_thermo, 8);
  const int N = virial_per_atom.size() / 9;
  gpu_sum<<<6, 1024>>>(N, virial_per_atom.data(), gpu_total_virial_.data());
  gpu_total_virial_.copy_to_host(cpu_total_virial_.data());

  fprintf(fid_, " energy=%.8f", cpu_thermo[1]);
  fprintf(
    fid_,
    " virial=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\"",
    cpu_total_virial_[0],
    cpu_total_virial_[3],
    cpu_total_virial_[4],
    cpu_total_virial_[3],
    cpu_total_virial_[1],
    cpu_total_virial_[5],
    cpu_total_virial_[4],
    cpu_total_virial_[5],
    cpu_total_virial_[2]);
  fprintf(
    fid_,
    " stress=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\"",
    cpu_thermo[2],
    cpu_thermo[5],
    cpu_thermo[6],
    cpu_thermo[5],
    cpu_thermo[3],
    cpu_thermo[7],
    cpu_thermo[6],
    cpu_thermo[7],
    cpu_thermo[4]);

  if (quantities.has_volume_) {
    fprintf(
      fid_,
      " voronoi_method=\"ball_restricted\""
      " voronoi_radius=%.8f voronoi_directions=%d",
      voronoi_radius_,
      voronoi_directions_);
  }

  // Properties
  fprintf(fid_, " Properties=species:S:1:pos:R:3");

  if (quantities.has_mass_) {
    fprintf(fid_, ":mass:R:1");
  }
  if (quantities.has_charge_) {
    fprintf(fid_, ":charge:R:1");
  }
  if (quantities.has_bec_) {
    fprintf(fid_, ":bec:R:9");
  }
  if (quantities.has_velocity_) {
    fprintf(fid_, ":vel:R:3");
  }
  if (quantities.has_force_) {
    fprintf(fid_, ":forces:R:3");
  }
  if (quantities.has_potential_) {
    fprintf(fid_, ":energy_atom:R:1");
  }
  if (quantities.has_unwrapped_position_) {
    fprintf(fid_, ":unwrapped_position:R:3");
  }
  if (quantities.has_virial_) {
    fprintf(fid_, ":virial:R:9");
  }
  if (quantities.has_group_) {
    const int num_grouping_methods = groups.size();
    fprintf(fid_, ":group:I:%d", num_grouping_methods);
  }
  if (quantities.has_volume_) {
    fprintf(fid_, ":volume_atom:R:1");
  }

  // Over
  fprintf(fid_, "\n");
}

void Dump_XYZ::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& groups,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  if ((step + 1) % dump_interval_ != 0)
    return;

  if (quantities.has_volume_) {
    const int N = atom.number_of_atoms;
    if (int(voronoi_NN_.size()) != N) {
      gpu_volume_per_atom_.resize(N);
      cpu_volume_per_atom_.resize(N);
      voronoi_cell_count_.resize(N);
      voronoi_cell_count_sum_.resize(N);
      voronoi_cell_contents_.resize(N);
      voronoi_NN_.resize(N);
      voronoi_NL_.resize(N * voronoi_mn_);
    }

    find_neighbor(
      0,
      N,
      2.0 * voronoi_radius_,
      box,
      atom.type,
      atom.position_per_atom,
      voronoi_cell_count_,
      voronoi_cell_count_sum_,
      voronoi_cell_contents_,
      voronoi_NN_,
      voronoi_NL_);

    const double min_thickness = 4.0 * voronoi_radius_;
    if ((box.pbc_x && box.thickness_x < min_thickness) ||
        (box.pbc_y && box.thickness_y < min_thickness) ||
        (box.pbc_z && box.thickness_z < min_thickness)) {
      PRINT_INPUT_ERROR(
        "Periodic box thickness is smaller than 4R. Increase the box or decrease Voronoi R so that the 2R neighborhood is complete under the minimum-image convention.\n");
    }

    if (N > 0) {
      std::vector<int> cpu_NN(N);
      voronoi_NN_.copy_to_host(cpu_NN.data());
      int max_nn = 0;
      for (int n = 0; n < N; ++n) {
        if (cpu_NN[n] > max_nn)
          max_nn = cpu_NN[n];
      }
      if (max_nn >= voronoi_mn_) {
        PRINT_INPUT_ERROR(
          "Voronoi neighbor list overflow. Decrease R or use a less dense structure.\n");
      }

      constexpr int BLOCK = 128;
      constexpr int ATOMS_PER_BLOCK = BLOCK / 32;
      const int grid = (N + ATOMS_PER_BLOCK - 1) / ATOMS_PER_BLOCK;
      const int offset = atom.position_per_atom.size() / 3;
      if (voronoi_directions_ == 128) {
        gpu_voronoi_volume<4><<<grid, BLOCK>>>(
          N,
          box,
          voronoi_radius_,
          gpu_voronoi_directions_.data(),
          voronoi_NN_.data(),
          voronoi_NL_.data(),
          atom.position_per_atom.data(),
          atom.position_per_atom.data() + offset,
          atom.position_per_atom.data() + 2 * offset,
          gpu_volume_per_atom_.data());
      } else {
        gpu_voronoi_volume<8><<<grid, BLOCK>>>(
          N,
          box,
          voronoi_radius_,
          gpu_voronoi_directions_.data(),
          voronoi_NN_.data(),
          voronoi_NL_.data(),
          atom.position_per_atom.data(),
          atom.position_per_atom.data() + offset,
          atom.position_per_atom.data() + 2 * offset,
          gpu_volume_per_atom_.data());
      }
      GPU_CHECK_KERNEL
      gpu_volume_per_atom_.copy_to_host(cpu_volume_per_atom_.data());
    }
  }

  int number_of_atoms_to_dump = atom.number_of_atoms;
  if (grouping_method_ >= 0) {
    number_of_atoms_to_dump = groups[grouping_method_].cpu_size[group_id_];
  }

  atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
  if (quantities.has_mass_) {
    atom.mass.copy_to_host(atom.cpu_mass.data());
  }
  if (quantities.has_charge_) {
    if (is_nep_charge) {
      GPU_Vector<float>& nep_charge = force.potentials[0]->get_charge_reference();
      nep_charge.copy_to_host(atom.cpu_charge.data());
    } else {
      atom.charge.copy_to_host(atom.cpu_charge.data());
    }
  }
  if (quantities.has_bec_) {
    GPU_Vector<float>& gpu_bec = force.potentials[0]->get_bec_reference();
    gpu_bec.copy_to_host(cpu_bec_.data());
  }
  if (quantities.has_velocity_) {
    atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());
  }
  if (quantities.has_force_) {
    atom.force_per_atom.copy_to_host(cpu_force_per_atom_.data());
  }
  if (quantities.has_potential_) {
    atom.potential_per_atom.copy_to_host(cpu_potential_per_atom_.data());
  }
  if (quantities.has_unwrapped_position_) {
    atom.unwrapped_position.copy_to_host(cpu_unwrapped_position_.data());
  }
  if (quantities.has_virial_) {
    atom.virial_per_atom.copy_to_host(cpu_virial_per_atom_.data());
  }

  if (separated_) {
    std::string filename = filename_ + std::to_string(step + 1);
    fid_ = my_fopen(filename.data(), "w");
  }

  // line 1
  fprintf(fid_, "%d\n", number_of_atoms_to_dump);

  // line 2
  output_line2(global_time, box, groups, atom.cpu_atom_symbol, atom.virial_per_atom, thermo);

  // other lines
  for (int n = 0; n < number_of_atoms_to_dump; n++) {

    int m = n;
    if (grouping_method_ >= 0) {
      int group_size_sum = groups[grouping_method_].cpu_size_sum[group_id_];
      m = groups[grouping_method_].cpu_contents[group_size_sum + n];
    }

    fprintf(fid_, "%s", atom.cpu_atom_symbol[m].c_str());
    for (int d = 0; d < 3; ++d) {
      fprintf(fid_, " %.8f", atom.cpu_position_per_atom[m + atom.number_of_atoms * d]);
    }
    if (quantities.has_mass_) {
      fprintf(fid_, " %.8f", atom.cpu_mass[m]);
    }
    if (quantities.has_charge_) {
      fprintf(fid_, " %.8f", atom.cpu_charge[m]);
    }
    if (quantities.has_bec_) {
      for (int d = 0; d < 9; ++d) {
        fprintf(fid_, " %.8f", cpu_bec_[m + atom.number_of_atoms * d]);
      }
    }
    if (quantities.has_velocity_) {
      const double natural_to_A_per_fs = 1.0 / TIME_UNIT_CONVERSION;
      for (int d = 0; d < 3; ++d) {
        fprintf(
          fid_, " %.8f", atom.cpu_velocity_per_atom[m + atom.number_of_atoms * d] * natural_to_A_per_fs);
      }
    }
    if (quantities.has_force_) {
      for (int d = 0; d < 3; ++d) {
        fprintf(fid_, " %.8f", cpu_force_per_atom_[m + atom.number_of_atoms * d]);
      }
    }
    if (quantities.has_potential_) {
      fprintf(fid_, " %.8f", cpu_potential_per_atom_[m]);
    }
    if (quantities.has_unwrapped_position_) {
      for (int d = 0; d < 3; ++d) {
        fprintf(fid_, " %.8f", cpu_unwrapped_position_[m + atom.number_of_atoms * d]);
      }
    }
    if (quantities.has_virial_) {
      const int index[9] = {0, 3, 4, 6, 1, 5, 7, 8, 2};
      for (int d = 0; d < 9; ++d) {
        fprintf(fid_, " %.8f", cpu_virial_per_atom_[m + atom.number_of_atoms * index[d]]);
      }
    }
    if (quantities.has_group_) {
      for (int d = 0; d < groups.size(); ++d) {
        fprintf(fid_, " %d", groups[d].cpu_label[m]);
      }
    }
    if (quantities.has_volume_) {
      fprintf(fid_, " %.10g", cpu_volume_per_atom_[m]);
    }
    fprintf(fid_, "\n");
  }
  if (separated_ == 0) {
    fflush(fid_);
  } else {
    fclose(fid_);
  }
}

void Dump_XYZ::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (separated_ == 0) {
    fclose(fid_);
  }
}
