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

#include "dataset.cuh"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

std::vector<Uf3Frame> load_uf3_frames(const char* filename, float nn_cutoff)
{
  std::vector<Uf3Frame> frames;
  std::ifstream input(filename);
  if (!input.is_open()) {
    std::cerr << "Error: cannot open " << filename << std::endl;
    exit(1);
  }

  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    int natoms = std::stoi(line);

    Uf3Frame f;
    f.num_atoms = natoms;
    std::getline(input, line);
    size_t pos = line.find("energy=");
    if (pos != std::string::npos) f.energy = std::stof(line.substr(pos + 7));

    f.types.resize(natoms);
    f.x.resize(natoms); f.y.resize(natoms); f.z.resize(natoms);
    f.fx.resize(natoms); f.fy.resize(natoms); f.fz.resize(natoms);

    for (int i = 0; i < natoms; i++) {
      std::getline(input, line);
      std::istringstream iss(line);
      std::string elem;
      iss >> elem >> f.x[i] >> f.y[i] >> f.z[i] >> f.fx[i] >> f.fy[i] >> f.fz[i];
      if (elem == "Si") f.types[i] = 0;
      else if (elem == "Ge") f.types[i] = 1;
      else f.types[i] = 0;
    }

    // Build neighbor list within nn_cutoff (for 3B optimization)
    if (nn_cutoff > 0.0f) {
      float cutoff_sq = nn_cutoff * nn_cutoff;
      f.nn_counts.resize(natoms, 0);
      f.nn_offset.resize(natoms + 1, 0);
      // Pass 1: count neighbors
      for (int i = 0; i < natoms; i++) {
        for (int j = i + 1; j < natoms; j++) {
          float dx = f.x[i] - f.x[j], dy = f.y[i] - f.y[j], dz = f.z[i] - f.z[j];
          if (dx*dx + dy*dy + dz*dz < cutoff_sq) {
            f.nn_counts[i]++; f.nn_counts[j]++;
          }
        }
      }
      // Compute offsets
      for (int i = 0; i < natoms; i++) f.nn_offset[i+1] = f.nn_offset[i] + f.nn_counts[i];
      int total_nn = f.nn_offset[natoms];
      f.nn_list.resize(total_nn);
      // Pass 2: fill (use temp per-atom counters)
      std::vector<int> counters(natoms, 0);
      for (int i = 0; i < natoms; i++) {
        for (int j = i + 1; j < natoms; j++) {
          float dx = f.x[i] - f.x[j], dy = f.y[i] - f.y[j], dz = f.z[i] - f.z[j];
          if (dx*dx + dy*dy + dz*dz < cutoff_sq) {
            f.nn_list[f.nn_offset[i] + counters[i]++] = j;
            f.nn_list[f.nn_offset[j] + counters[j]++] = i;
          }
        }
      }
    }

    frames.push_back(f);
  }
  input.close();
  return frames;
}
