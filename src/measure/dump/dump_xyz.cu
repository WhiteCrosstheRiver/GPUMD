/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*-----------------------------------------------------------------------------------------------100
Dump per-atom data to user-specified file(s) in the extended XYZ format
--------------------------------------------------------------------------------------------------*/

#include "dump_xyz.cuh"
#include "force/force.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "measure/parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>

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

Dump_XYZ::Dump_XYZ(
  const std::vector<std::string>& tokens,
  const std::vector<Group>& groups,
  Atom& atom,
  bool is_nep_charge_input)
{
  is_nep_charge = is_nep_charge_input;

  parse(tokens, groups);

  if (quantities.has_unwrapped_position_) {
    atom.enable_unwrapped_position();
  }

  action_name = "dump_xyz";
}

void Dump_XYZ::parse(
  const std::vector<std::string>& tokens, const std::vector<Group>& groups)
{
  const int num_param = tokens.size();
  printf("Dump extended XYZ.\n");

  if (num_param < 3) {
    PRINT_INPUT_ERROR("dump_xyz should have at least 2 parameters.\n");
  }

  // The old syntax started with <grouping_method> <group_id> <interval>, so tokens[2] and
  // tokens[3] were both integers. In the current syntax tokens[2] is the file name and tokens[3]
  // is an option or a quantity keyword, neither of which is an integer.
  int scratch;
  if (
    num_param >= 4 && is_valid_int(tokens[2], &scratch) &&
    is_valid_int(tokens[3], &scratch)) {
    PRINT_INPUT_ERROR(
      "dump_xyz no longer takes <grouping_method> <group_id> as its first two "
      "parameters. Use dump_xyz <interval> <filename> [group <grouping_method> "
      "<group_id>] instead.");
  }

  if (!is_valid_int(tokens[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("dump interval should > 0.");
  } else {
    printf("    every %d steps.\n", dump_interval_);
  }

  // filename
  std::string filename_temp = tokens[2];
  printf("    into file %s.\n", filename_temp.c_str());
  if (filename_temp.back() == '*') {
    separated_ = 1;
    filename_ = filename_temp.substr(0, filename_temp.size() - 1);
  } else {
    separated_ = 0;
    filename_ = filename_temp;
  }

  bool group_seen = false;
  bool precision_seen = false;
  bool split_seen = false;

  for (int m = 3; m < num_param; ++m) {
    if (tokens[m] == "spin") {
      has_spin_ = true;
      continue;
    }
    if (tokens[m] == "mforce") {
      has_mforce_ = true;
      continue;
    }

    if (tokens[m] == "spin_velocity") {
      has_spin_velocity_ = true;
      continue;
    }

    if (tokens[m] == "speed") {
      has_speed_ = true;
      continue;
    }
    if (tokens[m] == "force_norm") {
      has_force_norm_ = true;
      continue;
    }
    if (tokens[m] == "volume") {
      if (m + 2 >= num_param) {
        PRINT_INPUT_ERROR("dump_xyz volume should be followed by R and the number of directions.\n");
      }
      voronoi_.set(tokens[m + 1], tokens[m + 2]);
      has_volume_ = true;
      m += 2;
      continue;
    }
    if (tokens[m] == "stress" || tokens[m] == "stress_norm" || tokens[m] == "pressure") {
      (tokens[m] == "stress" ? has_stress_ : tokens[m] == "pressure" ? has_pressure_ : has_stress_norm_) = true;
      // R and the number of directions are optional here, but come as a pair.
      double probe = 0.0;
      if (m + 1 < num_param && is_valid_real(tokens[m + 1], &probe)) {
        if (m + 2 >= num_param) {
          PRINT_INPUT_ERROR("Voronoi radius R should be followed by the number of directions.\n");
        }
        voronoi_.set(tokens[m + 1], tokens[m + 2]);
        m += 2;
      }
      continue;
    }

    if (tokens[m] == "group") {
      if (group_seen) {
        PRINT_INPUT_ERROR("Option 'group' is specified more than once in dump_xyz.\n");
      }
      // A bare 'group' used to be the quantity that writes the group labels as a column, so say
      // what it is called now rather than only complaining about the missing arguments.
      int probe;
      if (
        m + 2 >= num_param || !is_valid_int(tokens[m + 1], &probe) ||
        !is_valid_int(tokens[m + 2], &probe)) {
        PRINT_INPUT_ERROR(
          "Option 'group' should be followed by a grouping method and a group ID. The quantity "
          "that writes group labels as a column is now called 'group_labels'.");
      }
      parse_group(tokens, false, groups, m, grouping_method_, group_id_);
      group_seen = true;
      continue;
    }
    if (tokens[m] == "precision") {
      if (precision_seen) {
        PRINT_INPUT_ERROR("Option 'precision' is specified more than once in dump_xyz.\n");
      }
      parse_precision(tokens, m, precision_);
      precision_seen = true;
      continue;
    }
    if (tokens[m] == "split") {
      if (split_seen) {
        PRINT_INPUT_ERROR("Option 'split' is specified more than once in dump_xyz.\n");
      }
      if (m + 1 >= num_param || !is_valid_int(tokens[m + 1], &split_frames_) ||
          split_frames_ <= 0) {
        PRINT_INPUT_ERROR("Option 'split' requires a positive number of frames.\n");
      }
      ++m;
      split_seen = true;
      printf("    split every %d frames.\n", split_frames_);
      continue;
    }
    if (!parse_dump_quantity(tokens[m], quantities, is_nep_charge, groups, "dump_xyz")) {
      PRINT_INPUT_ERROR("Unrecognized argument in dump_xyz.\n");
    }
  }

  if (needs_stress() && !voronoi_.enabled()) {
    PRINT_INPUT_ERROR(
      "dump_xyz stress, stress_norm and pressure need R and the number of directions, given after "
      "them or after volume.\n");
  }
  if (needs_volume()) {
    printf(
      "    has Voronoi volume with R = %g A and %d directions.\n",
      voronoi_.radius(),
      voronoi_.directions());
  }

  if (split_frames_ > 0 && separated_) {
    PRINT_INPUT_ERROR("Option 'split' cannot be combined with a filename ending in '*'.\n");
  }

  if (grouping_method_ < 0) {
    printf("    for the whole system.\n");
  }
}

void Dump_XYZ::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (split_frames_ > 0) {
    splitter_.reset(filename_, split_frames_);
    fid_ = nullptr;
  } else if (separated_ == 0) {
    fid_ = my_fopen(filename_.c_str(), "a");
  }

  // %g rather than %f so that quantities of any magnitude keep all their significant digits;
  // %.9g round-trips a float and %.17g round-trips a double.
  fmt_ = (precision_ == 1) ? " %.9g" : " %.17g";

  gpu_total_virial_.resize(6);
  cpu_total_virial_.resize(6);
  if (quantities.has_force_ || has_force_norm_) {
    cpu_force_per_atom_.resize(atom.number_of_atoms * 3);
  }
  if ((has_spin_ || has_mforce_ || has_spin_velocity_) && !atom.has_spin) {
    PRINT_INPUT_ERROR(
      "dump_xyz spin/mforce/spin_velocity require a NEP-spin model and spin:R:3 in model.xyz.");
  }
  if (has_spin_velocity_ && !atom.spin_velocity_initialized) {
    PRINT_INPUT_ERROR("dump_xyz spin_velocity requires TSPIN or spin_vel:R:3 in model.xyz.");
  }

  if (has_spin_) {
    cpu_spin_per_atom_.resize(atom.number_of_atoms * 3);
  }
  if (has_mforce_) {
    cpu_mforce_per_atom_.resize(atom.number_of_atoms * 3);
  }
  if (has_spin_velocity_) {
    cpu_spin_velocity_per_atom_.resize(atom.number_of_atoms * 3);
  }

  if (quantities.has_potential_) {
    cpu_potential_per_atom_.resize(atom.number_of_atoms);
  }
  if (quantities.has_unwrapped_position_) {
    cpu_unwrapped_position_.resize(atom.number_of_atoms * 3);
  }
  if (quantities.has_virial_ || needs_stress()) {
    cpu_virial_per_atom_.resize(atom.number_of_atoms * 9);
  }
  if (quantities.has_bec_) {
    cpu_bec_.resize(atom.number_of_atoms * 9);
  }
}

void Dump_XYZ::print_tensor(const char* name, const double* tensor)
{
  // fmt_ carries a leading space, which is wanted between values but not right after the quote
  const char* separated_fmt = fmt_.c_str();
  const char* first_fmt = separated_fmt + 1;
  fprintf(fid_, " %s=\"", name);
  for (int d = 0; d < 9; ++d) {
    fprintf(fid_, (d == 0) ? first_fmt : separated_fmt, tensor[d]);
  }
  fprintf(fid_, "\"");
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
  const double lattice[9] = {
    box.cpu_h[0],
    box.cpu_h[3],
    box.cpu_h[6],
    box.cpu_h[1],
    box.cpu_h[4],
    box.cpu_h[7],
    box.cpu_h[2],
    box.cpu_h[5],
    box.cpu_h[8]};
  print_tensor("Lattice", lattice);

  // energy and virial (symmetric tensor) in eV, and stress (symmetric tensor) in eV/A^3
  double cpu_thermo[8];
  gpu_thermo.copy_to_host(cpu_thermo, 8);
  const int N = virial_per_atom.size() / 9;
  gpu_sum<<<6, 1024>>>(N, virial_per_atom.data(), gpu_total_virial_.data());
  gpu_total_virial_.copy_to_host(cpu_total_virial_.data());

  fprintf(fid_, " energy=");
  fprintf(fid_, fmt_.c_str() + 1, cpu_thermo[1]);

  const double virial[9] = {
    cpu_total_virial_[0],
    cpu_total_virial_[3],
    cpu_total_virial_[4],
    cpu_total_virial_[3],
    cpu_total_virial_[1],
    cpu_total_virial_[5],
    cpu_total_virial_[4],
    cpu_total_virial_[5],
    cpu_total_virial_[2]};
  print_tensor("virial", virial);

  const double stress[9] = {
    cpu_thermo[2],
    cpu_thermo[5],
    cpu_thermo[6],
    cpu_thermo[5],
    cpu_thermo[3],
    cpu_thermo[7],
    cpu_thermo[6],
    cpu_thermo[7],
    cpu_thermo[4]};
  print_tensor("stress", stress);
  if (has_pressure_) {
    fprintf(fid_, " pressure=");
    fprintf(fid_, fmt_.c_str() + 1, box_pressure_);
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
  if (has_spin_) {
    fprintf(fid_, ":spin:R:3");
  }
  if (has_mforce_) {
    fprintf(fid_, ":mforce:R:3");
  }
  if (has_spin_velocity_) {
    fprintf(fid_, ":spin_vel:R:3");
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
  if (has_speed_) {
    fprintf(fid_, ":speed:R:1");
  }
  if (has_force_norm_) {
    fprintf(fid_, ":force_norm:R:1");
  }
  if (has_volume_) {
    fprintf(fid_, ":volume_atom:R:1");
  }
  if (has_stress_) {
    fprintf(fid_, ":stress:R:9");
  }
  if (has_stress_norm_) {
    fprintf(fid_, ":stress_norm:R:1");
  }
  if (has_pressure_) {
    fprintf(fid_, ":pressure:R:1");
  }

  // Over
  fprintf(fid_, "\n");
}

void Dump_XYZ::end_of_step(
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

  int number_of_atoms_to_dump = atom.number_of_atoms;
  if (grouping_method_ >= 0) {
    number_of_atoms_to_dump = groups[grouping_method_].cpu_size[group_id_];
  }

  atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
  if (quantities.has_mass_ || needs_stress()) {
    atom.mass.copy_to_host(atom.cpu_mass.data());
  }
  if (quantities.has_charge_) {
    if (is_nep_charge) {
      GPU_Vector<float>& nep_charge = force.get_potential(0).get_charge_reference();
      nep_charge.copy_to_host(atom.cpu_charge.data());
    } else {
      atom.charge.copy_to_host(atom.cpu_charge.data());
    }
  }
  if (quantities.has_bec_) {
    GPU_Vector<float>& gpu_bec = force.get_potential(0).get_bec_reference();
    gpu_bec.copy_to_host(cpu_bec_.data());
  }
  if (quantities.has_velocity_ || has_speed_ || needs_stress()) {
    atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());
  }
  if (quantities.has_force_ || has_force_norm_) {
    atom.force_per_atom.copy_to_host(cpu_force_per_atom_.data());
  }
  if (has_spin_) {
    atom.spin_per_atom.copy_to_host(cpu_spin_per_atom_.data());
  }
  if (has_mforce_) {
    atom.mforce_per_atom.copy_to_host(cpu_mforce_per_atom_.data());
  }
  if (has_spin_velocity_) {
    atom.spin_velocity_per_atom.copy_to_host(cpu_spin_velocity_per_atom_.data());
  }

  if (quantities.has_potential_) {
    atom.potential_per_atom.copy_to_host(cpu_potential_per_atom_.data());
  }
  if (quantities.has_unwrapped_position_) {
    atom.unwrapped_position.copy_to_host(cpu_unwrapped_position_.data());
  }
  if (quantities.has_virial_ || needs_stress()) {
    atom.virial_per_atom.copy_to_host(cpu_virial_per_atom_.data());
  }
  if (needs_volume()) {
    voronoi_.compute(box, atom, cpu_volume_per_atom_);
  }
  if (has_pressure_) {
    // Every atom and the box volume, even when only a group is dumped.
    const int N = atom.number_of_atoms;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
      sum += cpu_virial_per_atom_[i] + cpu_virial_per_atom_[i + N] +
             cpu_virial_per_atom_[i + 2 * N] + atom.cpu_mass[i] * kinetic_trace(atom, i);
    }
    box_pressure_ = PRESSURE_UNIT_CONVERSION * sum / (3.0 * box.get_volume());
  }

  if (separated_) {
    std::string filename = filename_ + std::to_string(step + 1);
    fid_ = my_fopen(filename.data(), "w");
  } else if (split_frames_ > 0 && splitter_.needs_new_file()) {
    if (fid_ != nullptr) {
      fclose(fid_);
    }
    const std::string filename = splitter_.next_filename();
    fid_ = my_fopen(filename.c_str(), "w");
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
      fprintf(fid_, fmt_.c_str(), atom.cpu_position_per_atom[m + atom.number_of_atoms * d]);
    }
    if (quantities.has_mass_) {
      fprintf(fid_, fmt_.c_str(), atom.cpu_mass[m]);
    }
    if (quantities.has_charge_) {
      fprintf(fid_, fmt_.c_str(), atom.cpu_charge[m]);
    }
    if (quantities.has_bec_) {
      for (int d = 0; d < 9; ++d) {
        fprintf(fid_, fmt_.c_str(), cpu_bec_[m + atom.number_of_atoms * d]);
      }
    }
    if (quantities.has_velocity_) {
      const double natural_to_A_per_fs = 1.0 / TIME_UNIT_CONVERSION;
      for (int d = 0; d < 3; ++d) {
        fprintf(
          fid_,
          fmt_.c_str(),
          atom.cpu_velocity_per_atom[m + atom.number_of_atoms * d] * natural_to_A_per_fs);
      }
    }
    if (quantities.has_force_) {
      for (int d = 0; d < 3; ++d) {
        fprintf(fid_, fmt_.c_str(), cpu_force_per_atom_[m + atom.number_of_atoms * d]);
      }
    }
    if (has_spin_) {
      for (int d = 0; d < 3; ++d) {
        fprintf(fid_, fmt_.c_str(), cpu_spin_per_atom_[m + atom.number_of_atoms * d]);
      }
    }
    if (has_mforce_) {
      for (int d = 0; d < 3; ++d) {
        fprintf(fid_, fmt_.c_str(), cpu_mforce_per_atom_[m + atom.number_of_atoms * d]);
      }
    }
    if (has_spin_velocity_) {
      const double natural_to_per_fs = 1.0 / TIME_UNIT_CONVERSION;
      for (int d = 0; d < 3; ++d) {
        fprintf(
          fid_,
          fmt_.c_str(),
          cpu_spin_velocity_per_atom_[m + atom.number_of_atoms * d] * natural_to_per_fs);
      }
    }

    if (quantities.has_potential_) {
      fprintf(fid_, fmt_.c_str(), cpu_potential_per_atom_[m]);
    }
    if (quantities.has_unwrapped_position_) {
      for (int d = 0; d < 3; ++d) {
        fprintf(fid_, fmt_.c_str(), cpu_unwrapped_position_[m + atom.number_of_atoms * d]);
      }
    }
    if (quantities.has_virial_) {
      const int index[9] = {0, 3, 4, 6, 1, 5, 7, 8, 2};
      for (int d = 0; d < 9; ++d) {
        fprintf(fid_, fmt_.c_str(), cpu_virial_per_atom_[m + atom.number_of_atoms * index[d]]);
      }
    }
    if (quantities.has_group_) {
      for (int d = 0; d < groups.size(); ++d) {
        fprintf(fid_, " %d", groups[d].cpu_label[m]);
      }
    }
    write_extras(atom, m);
    fprintf(fid_, "\n");
  }
  if (separated_) {
    fclose(fid_);
  } else {
    fflush(fid_);
    if (split_frames_ > 0) {
      splitter_.frame_written();
    }
  }
}

// |v|^2 in natural units.
double Dump_XYZ::kinetic_trace(const Atom& atom, const int i) const
{
  const int N = atom.number_of_atoms;
  double v2 = 0.0;
  for (int d = 0; d < 3; ++d) {
    const double v = atom.cpu_velocity_per_atom[i + N * d];
    v2 += v * v;
  }
  return v2;
}

void Dump_XYZ::write_extras(const Atom& atom, const int m)
{
  const int N = atom.number_of_atoms;
  if (has_speed_) {
    fprintf(fid_, fmt_.c_str(), sqrt(kinetic_trace(atom, m)) / TIME_UNIT_CONVERSION);
  }
  if (has_force_norm_) {
    double f2 = 0.0;
    for (int d = 0; d < 3; ++d) {
      f2 += cpu_force_per_atom_[m + N * d] * cpu_force_per_atom_[m + N * d];
    }
    fprintf(fid_, fmt_.c_str(), sqrt(f2));
  }
  if (has_volume_) {
    fprintf(fid_, fmt_.c_str(), cpu_volume_per_atom_[m]);
  }
  if (has_stress_ || has_stress_norm_) {
    // sigma = (W + m v v^T) / Omega, in eV/A^3, in the component order of virial.
    const int index[9] = {0, 3, 4, 6, 1, 5, 7, 8, 2};
    const int va[9] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    const int vb[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    double sigma[9];
    double norm2 = 0.0;
    for (int d = 0; d < 9; ++d) {
      const double w = cpu_virial_per_atom_[m + N * index[d]];
      const double vv = atom.cpu_velocity_per_atom[m + N * va[d]] * atom.cpu_velocity_per_atom[m + N * vb[d]];
      sigma[d] = (w + atom.cpu_mass[m] * vv) / cpu_volume_per_atom_[m];
      norm2 += sigma[d] * sigma[d];
    }
    if (has_stress_) {
      for (int d = 0; d < 9; ++d) {
        fprintf(fid_, fmt_.c_str(), sigma[d]);
      }
    }
    if (has_stress_norm_) {
      fprintf(fid_, fmt_.c_str(), sqrt(norm2));
    }
  }
  if (has_pressure_) {
    const double trace = cpu_virial_per_atom_[m] + cpu_virial_per_atom_[m + N] +
                         cpu_virial_per_atom_[m + 2 * N] + atom.cpu_mass[m] * kinetic_trace(atom, m);
    fprintf(fid_, fmt_.c_str(), PRESSURE_UNIT_CONVERSION * trace / (3.0 * cpu_volume_per_atom_[m]));
  }
}

void Dump_XYZ::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (separated_ == 0 && fid_ != nullptr) {
    fclose(fid_);
    fid_ = nullptr;
  }
}
