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

  // Pre-built neighbor list (indices of atoms within 3B cutoff)
  std::vector<int> nn_counts;    // per-atom neighbor count
  std::vector<int> nn_offset;    // per-atom offset in nn_list (cumsum)
  std::vector<int> nn_list;      // flat neighbor indices
};

std::vector<Uf3Frame> load_uf3_frames(const char* filename, float nn_cutoff = 0.0f);
