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
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>

template <typename T>
static void copy_soa3(
  const std::vector<T>& src, int n_src, int i_src, std::vector<T>& dst, int n_dst, int i_dst)
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

// position_image only exists once unwrapped positions are enabled.
static std::vector<int> read_position_image(Atom& atom)
{
  std::vector<int> image;
  if (atom.unwrapped_position.size() > 0) {
    image.resize(atom.position_image.size());
    atom.position_image.copy_to_host(image.data());
  }
  return image;
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

void AtomMutation::rebuild(
  Atom& atom,
  const Box& box,
  std::vector<Group>& groups,
  GPU_Vector<double>& thermo,
  Force& force,
  const std::vector<int>& image)
{
  const int N = static_cast<int>(atom.cpu_type.size());
  atom.number_of_atoms = N;
  rebuild_groups(groups, N);
  rebuild_type_size(atom);
  allocate_memory_gpu(groups, atom, thermo);
  if (atom.heat_per_atom.size() > 0) {
    atom.heat_per_atom.resize(N * 5);
  }
  if (atom.unwrapped_position.size() > 0) {
    atom.position_image.resize(N * 3);
    atom.position_image.copy_from_host(image.data());
    atom.unwrapped_position.resize(N * 3);
    atom.update_unwrapped_position(box);
  }
  force.update_number_of_atoms(N);
}

static void check_mutable(const Atom& atom)
{
  if (atom.number_of_beads > 0) {
    PRINT_INPUT_ERROR("Cannot change the number of atoms when PIMD beads are allocated.");
  }
  if (atom.has_spin) {
    PRINT_INPUT_ERROR("Cannot change the number of atoms in a spin system.");
  }
}

void AtomMutation::append_atoms(
  Atom& atom,
  const Box& box,
  std::vector<Group>& groups,
  GPU_Vector<double>& thermo,
  Force& force,
  const NewAtoms& added)
{
  if (added.n <= 0) {
    return;
  }
  check_mutable(atom);
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

  const std::vector<int> old_image = read_position_image(atom);
  std::vector<int> image(old_image.empty() ? 0 : N_new * 3, 0);
  if (!old_image.empty()) {
    for (int n = 0; n < N_old; ++n) {
      copy_soa3(old_image, N_old, n, image, N_new, n);
    }
  }

  Atom new_atoms;
  new_atoms.cpu_type.resize(N_new);
  new_atoms.cpu_mass.resize(N_new);
  new_atoms.cpu_charge.resize(N_new);
  new_atoms.cpu_atom_symbol.resize(N_new);
  new_atoms.cpu_fixed.assign(N_new, 0);
  new_atoms.cpu_shell_frozen.assign(N_new, 0); // appended atoms start mobile
  new_atoms.cpu_position_per_atom.resize(N_new * 3);
  new_atoms.cpu_velocity_per_atom.resize(N_new * 3);

  for (int n = 0; n < N_old; ++n) {
    new_atoms.cpu_type[n] = atom.cpu_type[n];
    new_atoms.cpu_mass[n] = atom.cpu_mass[n];
    new_atoms.cpu_charge[n] = (n < (int)atom.cpu_charge.size()) ? atom.cpu_charge[n] : 0.0f;
    new_atoms.cpu_atom_symbol[n] = atom.cpu_atom_symbol[n];
    new_atoms.cpu_fixed[n] = (n < (int)atom.cpu_fixed.size()) ? atom.cpu_fixed[n] : 0;
    new_atoms.cpu_shell_frozen[n] =
      (n < (int)atom.cpu_shell_frozen.size()) ? atom.cpu_shell_frozen[n] : 0;
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
    new_labels[m] = groups[m].cpu_label;
    new_labels[m].resize(N_new, 0);
    // New atoms start unfixed. Their model label waits for the end of the run.
    std::vector<int>& requested =
      groups[m].cpu_label_user.empty() ? new_labels[m] : groups[m].cpu_label_user;
    requested.resize(N_new, 0);
    for (int k = 0; k < added.n; ++k) {
      requested[N_old + k] = added.group_label.empty() ? 0 : added.group_label[k * n_group + m];
    }
  }

  atom.cpu_type.swap(new_atoms.cpu_type);
  atom.cpu_mass.swap(new_atoms.cpu_mass);
  atom.cpu_charge.swap(new_atoms.cpu_charge);
  atom.cpu_atom_symbol.swap(new_atoms.cpu_atom_symbol);
  atom.cpu_fixed.swap(new_atoms.cpu_fixed);
  atom.cpu_shell_frozen.swap(new_atoms.cpu_shell_frozen);
  atom.cpu_position_per_atom.swap(new_atoms.cpu_position_per_atom);
  atom.cpu_velocity_per_atom.swap(new_atoms.cpu_velocity_per_atom);
  for (int m = 0; m < n_group; ++m) {
    groups[m].cpu_label.swap(new_labels[m]);
  }

  rebuild(atom, box, groups, thermo, force, image);
}

void AtomMutation::remove_atoms(
  Atom& atom,
  const Box& box,
  std::vector<Group>& groups,
  GPU_Vector<double>& thermo,
  Force& force,
  const std::vector<char>& delete_mask)
{
  check_mutable(atom);
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
  const std::vector<int> old_image = read_position_image(atom);
  std::vector<int> image(old_image.empty() ? 0 : n_keep * 3, 0);

  Atom kept;
  kept.cpu_type.resize(n_keep);
  kept.cpu_mass.resize(n_keep);
  kept.cpu_charge.resize(n_keep);
  kept.cpu_atom_symbol.resize(n_keep);
  kept.cpu_fixed.assign(n_keep, 0);
  kept.cpu_shell_frozen.assign(n_keep, 0);
  kept.cpu_position_per_atom.resize(n_keep * 3);
  kept.cpu_velocity_per_atom.resize(n_keep * 3);
  std::vector<std::vector<int>> new_labels(n_group);
  std::vector<std::vector<int>> new_user_labels(n_group);
  for (int m = 0; m < n_group; ++m) {
    new_labels[m].resize(n_keep);
    new_user_labels[m].resize(groups[m].cpu_label_user.empty() ? 0 : n_keep);
  }

  // prefix-sum of kept atoms so the compaction loop can be parallel
  std::vector<int> keep_index(N_old);
  int cur = 0;
  for (int n = 0; n < N_old; ++n) {
    if (delete_mask[n]) {
      keep_index[n] = -1;
    } else {
      keep_index[n] = cur++;
    }
  }
  {
    const int n_thread = std::min(8, std::max(1, N_old / 65536));
    const int chunk = (N_old + n_thread - 1) / n_thread;
    std::vector<std::thread> workers;
    workers.reserve(n_thread);
    for (int t = 0; t < n_thread; ++t) {
      const int lo = t * chunk;
      const int hi = std::min(N_old, lo + chunk);
      if (lo >= hi) {
        break;
      }
      workers.emplace_back([&, lo, hi]() {
        for (int n = lo; n < hi; ++n) {
          const int dst = keep_index[n];
          if (dst < 0) {
            continue;
          }
          kept.cpu_type[dst] = atom.cpu_type[n];
          kept.cpu_mass[dst] = atom.cpu_mass[n];
          kept.cpu_charge[dst] = atom.cpu_charge[n];
          kept.cpu_atom_symbol[dst] = atom.cpu_atom_symbol[n];
          kept.cpu_fixed[dst] = (n < (int)atom.cpu_fixed.size()) ? atom.cpu_fixed[n] : 0;
          kept.cpu_shell_frozen[dst] =
            (n < (int)atom.cpu_shell_frozen.size()) ? atom.cpu_shell_frozen[n] : 0;
          copy_soa3(atom.cpu_position_per_atom, N_old, n, kept.cpu_position_per_atom, n_keep, dst);
          copy_soa3(atom.cpu_velocity_per_atom, N_old, n, kept.cpu_velocity_per_atom, n_keep, dst);
          if (!old_image.empty()) {
            copy_soa3(old_image, N_old, n, image, n_keep, dst);
          }
          for (int m = 0; m < n_group; ++m) {
            new_labels[m][dst] = groups[m].cpu_label[n];
            if (!groups[m].cpu_label_user.empty()) {
              new_user_labels[m][dst] = groups[m].cpu_label_user[n];
            }
          }
        }
      });
    }
    for (auto& w : workers) {
      w.join();
    }
  }

  atom.cpu_type.swap(kept.cpu_type);
  atom.cpu_mass.swap(kept.cpu_mass);
  atom.cpu_charge.swap(kept.cpu_charge);
  atom.cpu_atom_symbol.swap(kept.cpu_atom_symbol);
  atom.cpu_fixed.swap(kept.cpu_fixed);
  atom.cpu_shell_frozen.swap(kept.cpu_shell_frozen);
  atom.cpu_position_per_atom.swap(kept.cpu_position_per_atom);
  atom.cpu_velocity_per_atom.swap(kept.cpu_velocity_per_atom);
  for (int m = 0; m < n_group; ++m) {
    groups[m].cpu_label.swap(new_labels[m]);
    groups[m].cpu_label_user.swap(new_user_labels[m]);
  }

  rebuild(atom, box, groups, thermo, force, image);
}
