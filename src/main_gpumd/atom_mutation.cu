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

/*----------------------------------------------------------------------------80
Sole entry that may change atom count N. CPU arrays are the source of truth;
GPU buffers are rebuilt (GPU_Vector::resize frees then mallocs).
------------------------------------------------------------------------------*/

#include "atom_mutation.cuh"
#include "force/force.cuh"
#include "model/read_xyz.cuh"
#include "utilities/error.cuh"
#include <chrono>

static void copy_soa3(
  const std::vector<double>& src, int n_src, int i_src, std::vector<double>& dst, int n_dst, int i_dst)
{
  for (int d = 0; d < 3; ++d) {
    dst[i_dst + n_dst * d] = src[i_src + n_src * d];
  }
}

static void rebuild_groups(std::vector<Group>& groups, int N)
{
  for (size_t m = 0; m < groups.size(); ++m) {
    groups[m].find_size(N, static_cast<int>(m));
    groups[m].find_contents(N);
  }
}

static void rebuild_type_size(Atom& atom)
{
  int max_type = -1;
  for (int t : atom.cpu_type) {
    if (t > max_type) {
      max_type = t;
    }
  }
  const int n_types = max_type + 1;
  if (n_types > (int)atom.cpu_type_size.size()) {
    atom.cpu_type_size.resize(n_types, 0);
  }
  for (int& count : atom.cpu_type_size) {
    count = 0;
  }
  for (int t : atom.cpu_type) {
    if (t >= 0 && t < (int)atom.cpu_type_size.size()) {
      atom.cpu_type_size[t]++;
    }
  }
}

static void resize_optional_atom_buffers(Atom& atom, int N)
{
  if (atom.unwrapped_position.size() > 0) {
    atom.unwrapped_position.resize(N * 3);
  }
  if (atom.position_temp.size() > 0) {
    atom.position_temp.resize(N * 3);
  }
  if (atom.heat_per_atom.size() > 0) {
    atom.heat_per_atom.resize(N * 5);
  }
}

void AtomMutation::sync_cpu_from_gpu(Atom& atom)
{
  const auto time_begin = std::chrono::high_resolution_clock::now();
  const int N = atom.number_of_atoms;
  if (atom.position_per_atom.size() == static_cast<size_t>(N) * 3 &&
      atom.cpu_position_per_atom.size() == static_cast<size_t>(N) * 3) {
    atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
  }
  if (atom.velocity_per_atom.size() == static_cast<size_t>(N) * 3 &&
      atom.cpu_velocity_per_atom.size() == static_cast<size_t>(N) * 3) {
    atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());
  }
  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_used = time_finish - time_begin;
  printf("Time used for CPU sync = %g second.\n", time_used.count());
}

void AtomMutation::rebuild_after_mutation(
  Atom& atom, std::vector<Group>& groups, GPU_Vector<double>& thermo, Force& force)
{
  if (atom.number_of_beads > 0) {
    PRINT_INPUT_ERROR("Cannot change the number of atoms when PIMD beads are allocated.");
  }

  const int N = static_cast<int>(atom.cpu_type.size());
  if (N < 1) {
    PRINT_INPUT_ERROR("Cannot mutate to zero atoms.");
  }

  const auto time_begin = std::chrono::high_resolution_clock::now();
  atom.number_of_atoms = N;
  rebuild_groups(groups, N);
  rebuild_type_size(atom);
  allocate_memory_gpu(groups, atom, thermo);
  resize_optional_atom_buffers(atom, N);
  force.update_number_of_atoms(N);
  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_used = time_finish - time_begin;
  printf("Time used for atom-count rebuild = %g second.\n", time_used.count());
}

void AtomMutation::append_atoms(
  Atom& atom,
  std::vector<Group>& groups,
  GPU_Vector<double>& thermo,
  Force& force,
  const NewAtoms& added)
{
  if (added.n <= 0) {
    return;
  }
  if ((int)added.type.size() != added.n || (int)added.mass.size() != added.n ||
      (int)added.charge.size() != added.n || (int)added.symbol.size() != added.n ||
      (int)added.position.size() != added.n * 3 || (int)added.velocity.size() != added.n * 3) {
    PRINT_INPUT_ERROR("NewAtoms arrays do not match n.");
  }

  const int N_old = static_cast<int>(atom.cpu_type.size());
  const int N_new = N_old + added.n;
  const int n_group = static_cast<int>(groups.size());
  if (!added.group_label.empty() && (int)added.group_label.size() != added.n * n_group) {
    PRINT_INPUT_ERROR("NewAtoms group_label size does not match n * number of grouping methods.");
  }

  Atom new_atoms;
  new_atoms.cpu_type.resize(N_new);
  new_atoms.cpu_mass.resize(N_new);
  new_atoms.cpu_charge.resize(N_new);
  new_atoms.cpu_atom_symbol.resize(N_new);
  new_atoms.cpu_position_per_atom.resize(N_new * 3);
  new_atoms.cpu_velocity_per_atom.resize(N_new * 3);

  for (int n = 0; n < N_old; ++n) {
    new_atoms.cpu_type[n] = atom.cpu_type[n];
    new_atoms.cpu_mass[n] = atom.cpu_mass[n];
    new_atoms.cpu_charge[n] = (n < (int)atom.cpu_charge.size()) ? atom.cpu_charge[n] : 0.0f;
    new_atoms.cpu_atom_symbol[n] = atom.cpu_atom_symbol[n];
    copy_soa3(atom.cpu_position_per_atom, N_old, n, new_atoms.cpu_position_per_atom, N_new, n);
    copy_soa3(atom.cpu_velocity_per_atom, N_old, n, new_atoms.cpu_velocity_per_atom, N_new, n);
  }
  for (int k = 0; k < added.n; ++k) {
    const int dst = N_old + k;
    new_atoms.cpu_type[dst] = added.type[k];
    new_atoms.cpu_mass[dst] = added.mass[k];
    new_atoms.cpu_charge[dst] = added.charge[k];
    new_atoms.cpu_atom_symbol[dst] = added.symbol[k];
    copy_soa3(added.position, added.n, k, new_atoms.cpu_position_per_atom, N_new, dst);
    copy_soa3(added.velocity, added.n, k, new_atoms.cpu_velocity_per_atom, N_new, dst);
  }

  std::vector<std::vector<int>> new_labels(n_group);
  for (int m = 0; m < n_group; ++m) {
    new_labels[m].resize(N_new);
    for (int n = 0; n < N_old; ++n) {
      new_labels[m][n] = groups[m].cpu_label[n];
    }
    for (int k = 0; k < added.n; ++k) {
      new_labels[m][N_old + k] =
        added.group_label.empty() ? 0 : added.group_label[k * n_group + m];
    }
  }

  atom.cpu_type.swap(new_atoms.cpu_type);
  atom.cpu_mass.swap(new_atoms.cpu_mass);
  atom.cpu_charge.swap(new_atoms.cpu_charge);
  atom.cpu_atom_symbol.swap(new_atoms.cpu_atom_symbol);
  atom.cpu_position_per_atom.swap(new_atoms.cpu_position_per_atom);
  atom.cpu_velocity_per_atom.swap(new_atoms.cpu_velocity_per_atom);
  for (int m = 0; m < n_group; ++m) {
    groups[m].cpu_label.swap(new_labels[m]);
  }

  rebuild_after_mutation(atom, groups, thermo, force);
}

void AtomMutation::remove_atoms(
  Atom& atom,
  std::vector<Group>& groups,
  GPU_Vector<double>& thermo,
  Force& force,
  const std::vector<char>& delete_mask)
{
  const int N_old = static_cast<int>(atom.cpu_type.size());
  if ((int)delete_mask.size() != N_old) {
    PRINT_INPUT_ERROR("delete_mask size does not match the number of atoms.");
  }

  int n_keep = 0;
  for (int n = 0; n < N_old; ++n) {
    if (!delete_mask[n]) {
      ++n_keep;
    }
  }
  if (n_keep == N_old) {
    return;
  }
  if (n_keep < 1) {
    PRINT_INPUT_ERROR("Cannot delete all atoms. At least one atom must remain.");
  }

  const int n_group = static_cast<int>(groups.size());
  Atom kept;
  kept.cpu_type.resize(n_keep);
  kept.cpu_mass.resize(n_keep);
  kept.cpu_charge.resize(n_keep);
  kept.cpu_atom_symbol.resize(n_keep);
  kept.cpu_position_per_atom.resize(n_keep * 3);
  kept.cpu_velocity_per_atom.resize(n_keep * 3);
  std::vector<std::vector<int>> new_labels(n_group);
  for (int m = 0; m < n_group; ++m) {
    new_labels[m].resize(n_keep);
  }

  int cur = 0;
  for (int n = 0; n < N_old; ++n) {
    if (delete_mask[n]) {
      continue;
    }
    kept.cpu_type[cur] = atom.cpu_type[n];
    kept.cpu_mass[cur] = atom.cpu_mass[n];
    kept.cpu_charge[cur] = atom.cpu_charge[n];
    kept.cpu_atom_symbol[cur] = atom.cpu_atom_symbol[n];
    copy_soa3(atom.cpu_position_per_atom, N_old, n, kept.cpu_position_per_atom, n_keep, cur);
    copy_soa3(atom.cpu_velocity_per_atom, N_old, n, kept.cpu_velocity_per_atom, n_keep, cur);
    for (int m = 0; m < n_group; ++m) {
      new_labels[m][cur] = groups[m].cpu_label[n];
    }
    ++cur;
  }

  atom.cpu_type.swap(kept.cpu_type);
  atom.cpu_mass.swap(kept.cpu_mass);
  atom.cpu_charge.swap(kept.cpu_charge);
  atom.cpu_atom_symbol.swap(kept.cpu_atom_symbol);
  atom.cpu_position_per_atom.swap(kept.cpu_position_per_atom);
  atom.cpu_velocity_per_atom.swap(kept.cpu_velocity_per_atom);
  for (int m = 0; m < n_group; ++m) {
    groups[m].cpu_label.swap(new_labels[m]);
  }

  rebuild_after_mutation(atom, groups, thermo, force);
}
