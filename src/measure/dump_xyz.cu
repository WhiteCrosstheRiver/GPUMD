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
#include <chrono>
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

static void set_voronoi_params(
  const char* radius_token,
  const char* direction_token,
  double& radius,
  int& directions)
{
  double new_radius = 0.0;
  int new_directions = 0;
  if (!is_valid_real(radius_token, &new_radius)) {
    PRINT_INPUT_ERROR("Voronoi radius R should be a number.\n");
  }
  if (!(new_radius > 0.0)) {
    PRINT_INPUT_ERROR("Voronoi radius R should > 0.\n");
  }
  if (!is_valid_int(direction_token, &new_directions)) {
    PRINT_INPUT_ERROR("number of Voronoi directions should be an integer.\n");
  }
  if (new_directions != 128 && new_directions != 256) {
    PRINT_INPUT_ERROR("number of Voronoi directions should be 128 or 256.\n");
  }
  if (radius > 0.0) {
    if (new_radius != radius || new_directions != directions) {
      PRINT_INPUT_ERROR(
        "dump_xyz volume, stress, and pressure must use the same R and number of directions.\n");
    }
    return;
  }
  radius = new_radius;
  directions = new_directions;
}

// If the next token is a number, R and M must both be present and valid; otherwise RM is omitted.
static int consume_optional_voronoi(
  const char** param, const int num_param, const int m, double& radius, int& directions)
{
  if (m + 1 >= num_param) {
    return 0;
  }
  double dummy = 0.0;
  if (!is_valid_real(param[m + 1], &dummy)) {
    return 0;
  }
  if (m + 2 >= num_param) {
    PRINT_INPUT_ERROR(
      "Voronoi radius R should be followed by the number of directions (128 or 256).\n");
  }
  set_voronoi_params(param[m + 1], param[m + 2], radius, directions);
  return 2;
}

static __global__ void gpu_find_neighbor_capped(
  const Box box,
  const int N,
  const int MN,
  const int* __restrict__ cell_counts,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  int* NN,
  int* NL,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const double cutoff_square)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }
  int count = 0;
  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];
  int cell_id;
  int cell_id_x;
  int cell_id_y;
  int cell_id_z;
  find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

  const int z_lim = box.pbc_z ? 2 : 0;
  const int y_lim = box.pbc_y ? 2 : 0;
  const int x_lim = box.pbc_x ? 2 : 0;

  for (int k = -z_lim; k <= z_lim; ++k) {
    for (int j = -y_lim; j <= y_lim; ++j) {
      for (int i = -x_lim; i <= x_lim; ++i) {
        int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
        if (cell_id_x + i < 0)
          neighbor_cell += nx;
        if (cell_id_x + i >= nx)
          neighbor_cell -= nx;
        if (cell_id_y + j < 0)
          neighbor_cell += ny * nx;
        if (cell_id_y + j >= ny)
          neighbor_cell -= ny * nx;
        if (cell_id_z + k < 0)
          neighbor_cell += nz * ny * nx;
        if (cell_id_z + k >= nz)
          neighbor_cell -= nz * ny * nx;

        const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
        const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];
        for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
          const int n2 = cell_contents[num_atoms_previous_cells + m];
          if (n1 == n2) {
            continue;
          }
          double x12 = x[n2] - x1;
          double y12 = y[n2] - y1;
          double z12 = z[n2] - z1;
          apply_mic(box, x12, y12, z12);
          const double d2 = x12 * x12 + y12 * y12 + z12 * z12;
          if (d2 < cutoff_square) {
            if (count < MN) {
              NL[count * N + n1] = n2;
            }
            ++count;
          }
        }
      }
    }
  }
  NN[n1] = count;
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

std::unique_ptr<Property> create_dump_xyz(
  const char** param, int num_param, const std::vector<Group>& groups, Atom& atom)
{
  return std::unique_ptr<Property>(new Dump_XYZ(param, num_param, groups, atom));
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
    } else if (strcmp(param[m], "speed") == 0) {
      quantities.has_speed_ = true;
      printf("    has speed.\n");
    } else if (strcmp(param[m], "force") == 0) {
      quantities.has_force_ = true;
      printf("    has force.\n");
    } else if (strcmp(param[m], "force_norm") == 0) {
      quantities.has_force_norm_ = true;
      printf("    has force_norm.\n");
    } else if (strcmp(param[m], "potential") == 0) {
      quantities.has_potential_ = true;
      printf("    has potential.\n");
    } else if (strcmp(param[m], "unwrapped_position") == 0) {
      quantities.has_unwrapped_position_ = true;
      printf("    has unwrapped position.\n");
    } else if (strcmp(param[m], "mass") == 0) {
      quantities.has_mass_ = true;
      printf("    has mass.\n");
    } else if (strcmp(param[m], "charge") == 0) {
      quantities.has_charge_ = true;
      if (is_nep_charge){
        printf("    has charge predicted by NEP-charge.\n");
      } else {
        printf("    has charge specified in model.xyz.\n");
      }
    } else if (strcmp(param[m], "bec") == 0) {
      quantities.has_bec_ = true;
      if (is_nep_charge){
        printf("    has BEC predicted by NEP-charge.\n");
      } else {
        PRINT_INPUT_ERROR("Cannot output BEC for a non-NEP-charge model.\n");
      }
    } else if (strcmp(param[m], "virial") == 0) {
      quantities.has_virial_ = true;
      printf("    has virial.\n");
    } else if (strcmp(param[m], "group") == 0) {
      quantities.has_group_ = true;
      printf("    has group.\n");
    } else if (strcmp(param[m], "stress") == 0) {
      quantities.has_stress_ = true;
      printf("    has per-atom stress.\n");
      const int consumed = consume_optional_voronoi(param, num_param, m, voronoi_radius_, voronoi_directions_);
      if (consumed > 0) {
        printf("        R = %g Angstrom.\n", voronoi_radius_);
        printf("        directions = %d.\n", voronoi_directions_);
        m += consumed;
      }
    } else if (strcmp(param[m], "stress_norm") == 0) {
      quantities.has_stress_norm_ = true;
      printf("    has stress_norm.\n");
      const int consumed = consume_optional_voronoi(param, num_param, m, voronoi_radius_, voronoi_directions_);
      if (consumed > 0) {
        printf("        R = %g Angstrom.\n", voronoi_radius_);
        printf("        directions = %d.\n", voronoi_directions_);
        m += consumed;
      }
    } else if (strcmp(param[m], "volume") == 0) {
      if (m + 2 >= num_param) {
        PRINT_INPUT_ERROR("dump_xyz volume should be followed by R and the number of directions.\n");
      }
      set_voronoi_params(param[m + 1], param[m + 2], voronoi_radius_, voronoi_directions_);
      quantities.has_volume_ = true;
      printf("    has ball-restricted Voronoi volume.\n");
      printf("        R = %g Angstrom.\n", voronoi_radius_);
      printf("        directions = %d.\n", voronoi_directions_);
      m += 2;
    } else if (strcmp(param[m], "pressure") == 0) {
      quantities.has_pressure_ = true;
      printf("    has per-atom and box pressure.\n");
      const int consumed = consume_optional_voronoi(param, num_param, m, voronoi_radius_, voronoi_directions_);
      if (consumed > 0) {
        printf("        R = %g Angstrom.\n", voronoi_radius_);
        printf("        directions = %d.\n", voronoi_directions_);
        m += consumed;
      }
    } else {
      std::string msg = "Unknown identifier in dump_xyz command: ";
      msg += param[m];
      msg += ".\n";
      PRINT_INPUT_ERROR(msg.c_str());
    }
  }
  if ((quantities.has_stress_ || quantities.has_stress_norm_ || quantities.has_pressure_) &&
      !(voronoi_radius_ > 0.0)) {
    PRINT_INPUT_ERROR(
      "dump_xyz stress, stress_norm, and pressure need R and the number of directions, or a volume keyword with those values.\n");
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
  if (quantities.has_force_ || quantities.has_force_norm_) {
    cpu_force_per_atom_.resize(atom.number_of_atoms * 3);
  }
  if (quantities.has_potential_) {
    cpu_potential_per_atom_.resize(atom.number_of_atoms);
  }
  if (quantities.has_unwrapped_position_) {
    cpu_unwrapped_position_.resize(atom.number_of_atoms * 3);
  }
  if (quantities.has_virial_ || quantities.has_stress_ || quantities.has_stress_norm_ ||
      quantities.has_pressure_) {
    cpu_virial_per_atom_.resize(atom.number_of_atoms * 9);
  }
  if (quantities.has_bec_) {
    cpu_bec_.resize(atom.number_of_atoms * 9);
  }
  if (quantities.has_volume_ || quantities.has_stress_ || quantities.has_stress_norm_ ||
      quantities.has_pressure_) {
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
  if (quantities.has_pressure_) {
    fprintf(fid_, " pressure=%.8f", cpu_box_pressure_);
  }

  if (quantities.has_volume_ || quantities.has_stress_ || quantities.has_stress_norm_ ||
      quantities.has_pressure_) {
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
  if (quantities.has_speed_) {
    fprintf(fid_, ":speed:R:1");
  }
  if (quantities.has_force_) {
    fprintf(fid_, ":forces:R:3");
  }
  if (quantities.has_force_norm_) {
    fprintf(fid_, ":force_norm:R:1");
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
  if (quantities.has_stress_) {
    fprintf(fid_, ":stress:R:9");
  }
  if (quantities.has_stress_norm_) {
    fprintf(fid_, ":stress_norm:R:1");
  }
  if (quantities.has_pressure_) {
    fprintf(fid_, ":pressure:R:1");
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

  if (quantities.has_volume_ || quantities.has_stress_ || quantities.has_stress_norm_ ||
      quantities.has_pressure_) {
    const int N = atom.number_of_atoms;
    const double min_thickness = 4.0 * voronoi_radius_;
    if ((box.pbc_x && box.thickness_x < min_thickness) ||
        (box.pbc_y && box.thickness_y < min_thickness) ||
        (box.pbc_z && box.thickness_z < min_thickness)) {
      PRINT_INPUT_ERROR(
        "Periodic box thickness is smaller than 4R. Increase the box or decrease Voronoi R so that the 2R neighborhood is complete under the minimum-image convention.\n");
    }

    if (int(voronoi_NN_.size()) != N) {
      gpu_volume_per_atom_.resize(N);
      cpu_volume_per_atom_.resize(N);
      voronoi_cell_count_.resize(N);
      voronoi_cell_count_sum_.resize(N);
      voronoi_cell_contents_.resize(N);
      voronoi_NN_.resize(N);
      voronoi_NL_.resize(N * voronoi_mn_);
    }

    const auto t_vol0 = std::chrono::steady_clock::now();
    const double rc = 2.0 * voronoi_radius_;
    const double rc_cell_list = 0.5 * rc;
    int num_bins[3];
    box.get_num_bins(rc_cell_list, num_bins);
    const int offset = atom.position_per_atom.size() / 3;
    const double* x = atom.position_per_atom.data();
    const double* y = x + offset;
    const double* z = x + 2 * offset;
    int max_nn = 0;
    bool overflow = true;
    for (int attempt = 0; attempt < 8 && overflow; ++attempt) {
      if (int(voronoi_NL_.size()) < N * voronoi_mn_) {
        voronoi_NL_.resize(N * voronoi_mn_);
      }
      find_cell_list(
        rc_cell_list,
        num_bins,
        box,
        atom.position_per_atom,
        voronoi_cell_count_,
        voronoi_cell_count_sum_,
        voronoi_cell_contents_);
      const int block_size = 256;
      const int grid_size = (N + block_size - 1) / block_size;
      gpu_find_neighbor_capped<<<grid_size, block_size>>>(
        box,
        N,
        voronoi_mn_,
        voronoi_cell_count_.data(),
        voronoi_cell_count_sum_.data(),
        voronoi_cell_contents_.data(),
        voronoi_NN_.data(),
        voronoi_NL_.data(),
        x,
        y,
        z,
        num_bins[0],
        num_bins[1],
        num_bins[2],
        2.0 / rc,
        rc * rc);
      GPU_CHECK_KERNEL
      CHECK(gpuDeviceSynchronize());
      max_nn = 0;
      if (N > 0) {
        std::vector<int> cpu_NN(N);
        voronoi_NN_.copy_to_host(cpu_NN.data());
        for (int n = 0; n < N; ++n) {
          if (cpu_NN[n] > max_nn)
            max_nn = cpu_NN[n];
        }
      }
      if (max_nn <= voronoi_mn_) {
        overflow = false;
      } else if (max_nn > 1024) {
        PRINT_INPUT_ERROR(
          "Voronoi neighbor list overflow (more than 1024 neighbors). Decrease R or use a less dense structure.\n");
      } else {
        voronoi_mn_ = max_nn;
      }
    }
    if (overflow) {
      PRINT_INPUT_ERROR("Voronoi neighbor list overflow. Decrease R or use a less dense structure.\n");
    }
    const auto t_vol1 = std::chrono::steady_clock::now();

    if (N > 0) {
      constexpr int BLOCK = 128;
      constexpr int ATOMS_PER_BLOCK = BLOCK / 32;
      const int grid = (N + ATOMS_PER_BLOCK - 1) / ATOMS_PER_BLOCK;
      if (voronoi_directions_ == 128) {
        gpu_voronoi_volume<4><<<grid, BLOCK>>>(
          N,
          box,
          voronoi_radius_,
          gpu_voronoi_directions_.data(),
          voronoi_NN_.data(),
          voronoi_NL_.data(),
          x,
          y,
          z,
          gpu_volume_per_atom_.data());
      } else {
        gpu_voronoi_volume<8><<<grid, BLOCK>>>(
          N,
          box,
          voronoi_radius_,
          gpu_voronoi_directions_.data(),
          voronoi_NN_.data(),
          voronoi_NL_.data(),
          x,
          y,
          z,
          gpu_volume_per_atom_.data());
      }
      GPU_CHECK_KERNEL
      CHECK(gpuDeviceSynchronize());
      gpu_volume_per_atom_.copy_to_host(cpu_volume_per_atom_.data());
    }
    const auto t_vol2 = std::chrono::steady_clock::now();
    const double ms_neighbor =
      1.0e-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(t_vol1 - t_vol0).count();
    const double ms_kernel =
      1.0e-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(t_vol2 - t_vol1).count();
    printf(
      "    Voronoi volume: neighbor %.3f ms, kernel+copy %.3f ms "
      "(N=%d, M=%d, maxNN=%d)\n",
      ms_neighbor,
      ms_kernel,
      N,
      voronoi_directions_,
      max_nn);
  }

  int number_of_atoms_to_dump = atom.number_of_atoms;
  if (grouping_method_ >= 0) {
    number_of_atoms_to_dump = groups[grouping_method_].cpu_size[group_id_];
  }

  atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
  if (quantities.has_mass_ || quantities.has_stress_ || quantities.has_stress_norm_ ||
      quantities.has_pressure_) {
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
  if (quantities.has_velocity_ || quantities.has_speed_ || quantities.has_stress_ ||
      quantities.has_stress_norm_ || quantities.has_pressure_) {
    atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());
  }
  if (quantities.has_force_ || quantities.has_force_norm_) {
    atom.force_per_atom.copy_to_host(cpu_force_per_atom_.data());
  }
  if (quantities.has_potential_) {
    atom.potential_per_atom.copy_to_host(cpu_potential_per_atom_.data());
  }
  if (quantities.has_unwrapped_position_) {
    atom.unwrapped_position.copy_to_host(cpu_unwrapped_position_.data());
  }
  if (quantities.has_virial_ || quantities.has_stress_ || quantities.has_stress_norm_ ||
      quantities.has_pressure_) {
    atom.virial_per_atom.copy_to_host(cpu_virial_per_atom_.data());
  }

  if (quantities.has_pressure_) {
    const int N = atom.number_of_atoms;
    double sum_A = 0.0;
    for (int i = 0; i < N; ++i) {
      const double vx = atom.cpu_velocity_per_atom[i];
      const double vy = atom.cpu_velocity_per_atom[i + N];
      const double vz = atom.cpu_velocity_per_atom[i + N * 2];
      const double wxx = cpu_virial_per_atom_[i];
      const double wyy = cpu_virial_per_atom_[i + N];
      const double wzz = cpu_virial_per_atom_[i + N * 2];
      sum_A += wxx + wyy + wzz + atom.cpu_mass[i] * (vx * vx + vy * vy + vz * vz);
    }
    cpu_box_pressure_ = PRESSURE_UNIT_CONVERSION * sum_A / (3.0 * box.get_volume());
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
    if (quantities.has_speed_) {
      const double vx = atom.cpu_velocity_per_atom[m];
      const double vy = atom.cpu_velocity_per_atom[m + atom.number_of_atoms];
      const double vz = atom.cpu_velocity_per_atom[m + atom.number_of_atoms * 2];
      fprintf(fid_, " %.8f", sqrt(vx * vx + vy * vy + vz * vz) / TIME_UNIT_CONVERSION);
    }
    if (quantities.has_force_) {
      for (int d = 0; d < 3; ++d) {
        fprintf(fid_, " %.8f", cpu_force_per_atom_[m + atom.number_of_atoms * d]);
      }
    }
    if (quantities.has_force_norm_) {
      const double fx = cpu_force_per_atom_[m];
      const double fy = cpu_force_per_atom_[m + atom.number_of_atoms];
      const double fz = cpu_force_per_atom_[m + atom.number_of_atoms * 2];
      fprintf(fid_, " %.8f", sqrt(fx * fx + fy * fy + fz * fz));
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
    if (quantities.has_stress_ || quantities.has_stress_norm_) {
      const int index[9] = {0, 3, 4, 6, 1, 5, 7, 8, 2};
      const int va[9] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
      const int vb[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
      const double vol = cpu_volume_per_atom_[m];
      const double mass = atom.cpu_mass[m];
      const double v[3] = {
        atom.cpu_velocity_per_atom[m],
        atom.cpu_velocity_per_atom[m + atom.number_of_atoms],
        atom.cpu_velocity_per_atom[m + atom.number_of_atoms * 2]};
      double sigma[9];
      double frobenius2 = 0.0;
      for (int d = 0; d < 9; ++d) {
        const double w = cpu_virial_per_atom_[m + atom.number_of_atoms * index[d]];
        sigma[d] = (w + mass * v[va[d]] * v[vb[d]]) / vol;
        frobenius2 += sigma[d] * sigma[d];
      }
      if (quantities.has_stress_) {
        for (int d = 0; d < 9; ++d) {
          fprintf(fid_, " %.8f", sigma[d]);
        }
      }
      if (quantities.has_stress_norm_) {
        fprintf(fid_, " %.8f", sqrt(frobenius2));
      }
    }
    if (quantities.has_pressure_) {
      const int N = atom.number_of_atoms;
      const double vx = atom.cpu_velocity_per_atom[m];
      const double vy = atom.cpu_velocity_per_atom[m + N];
      const double vz = atom.cpu_velocity_per_atom[m + N * 2];
      const double A = cpu_virial_per_atom_[m] + cpu_virial_per_atom_[m + N] +
                       cpu_virial_per_atom_[m + N * 2] +
                       atom.cpu_mass[m] * (vx * vx + vy * vy + vz * vz);
      fprintf(fid_, " %.8f", PRESSURE_UNIT_CONVERSION * A / (3.0 * cpu_volume_per_atom_[m]));
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
