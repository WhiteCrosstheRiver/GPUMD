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

#pragma once

#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>

class Atom;
class Box;

// Per-atom ball-restricted Voronoi volume Vol[C_i intersected with B(r_i, R)], from equal-weight
// quadrature over 128 or 256 directions. Neighbors within 2R bound each cell.
class VoronoiVolume
{
public:
  bool enabled() const { return radius_ > 0.0; }
  double radius() const { return radius_; }
  int directions() const { return directions_; }

  // Parses "<R> <M>". Every keyword that uses it must give the same pair.
  void set(const std::string& radius_token, const std::string& direction_token);

  // Fills cpu_volume with one value per atom, in Angstrom^3.
  void compute(Box& box, Atom& atom, std::vector<double>& cpu_volume);

private:
  double radius_ = 0.0;
  int directions_ = 0;
  int max_neighbors_ = 64;
  GPU_Vector<double> directions_gpu_;
  GPU_Vector<double> volume_;
  GPU_Vector<int> cell_count_;
  GPU_Vector<int> cell_count_sum_;
  GPU_Vector<int> cell_contents_;
  GPU_Vector<int> NN_;
  GPU_Vector<int> NL_;
};
