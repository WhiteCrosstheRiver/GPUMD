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
#include <string>
#include <vector>

struct Uf3Frame {
  int num_atoms = 0;
  std::vector<int> types;
  std::vector<float> x, y, z;
  std::vector<float> fx, fy, fz;
  float energy = 0.0f;

  // Lattice matrix (column-major, same convention as main_nep):
  //   box[0..8]  = H = [a|b|c], where a,b,c are lattice vectors as columns
  //   box_inv[0..8] = H^{-1} (for minimum-image convention)
  float box[9]     = {};
  float box_inv[9] = {};
  bool has_lattice = false;

  // Pre-built neighbor list within 3B cutoff (indices of atoms)
  std::vector<int> nn_counts;
  std::vector<int> nn_offset;
  std::vector<int> nn_list;
};

// elements: ordered list from UF3_Parameters (e.g. {"Si","Ge"}).
// nn_cutoff > 0 triggers neighbor-list construction with PBC.
std::vector<Uf3Frame> load_uf3_frames(
  const char* filename,
  const std::vector<std::string>& elements,
  float nn_cutoff = 0.0f);
