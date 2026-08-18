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

#pragma once

#include "model/atom.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>

class Force;

// CPU-side atoms to append. position/velocity are SoA: [x0..xn-1, y0..yn-1, z0..zn-1].
struct NewAtoms
{
  int n = 0;
  std::vector<int> type;
  std::vector<double> mass;
  std::vector<float> charge;
  std::vector<std::string> symbol;
  std::vector<double> position;
  std::vector<double> velocity;
  std::vector<int> group_label; // n * groups.size(); empty means label 0
};

class AtomMutation
{
public:
  static void sync_cpu_from_gpu(Atom& atom);

  static void append_atoms(
    Atom& atom,
    std::vector<Group>& groups,
    GPU_Vector<double>& thermo,
    Force& force,
    const NewAtoms& added);

  // delete_mask[i] != 0 means atom i is removed.
  static void remove_atoms(
    Atom& atom,
    std::vector<Group>& groups,
    GPU_Vector<double>& thermo,
    Force& force,
    const std::vector<char>& delete_mask);

  // CPU arrays are the source of truth. Sets number_of_atoms from cpu_type.size().
  static void rebuild_after_mutation(
    Atom& atom, std::vector<Group>& groups, GPU_Vector<double>& thermo, Force& force);
};
