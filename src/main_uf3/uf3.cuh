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
UF3 training model: manages B-spline coefficients, builds pre-computed cubic
polynomial tables, and launches GPU kernels for batch energy evaluation.
------------------------------------------------------------------------------*/

#pragma once
#include "dataset.cuh"
#include "parameters.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class Uf3Model
{
public:
  Uf3Model(UF3_Parameters& para);

  int num_parameters() const { return num_pairs_ * ncoeff_; }
  void get_parameters(float* params) const;
  void set_parameters(const float* params);

  // GPU batch energy evaluation: return per-frame energies in d_energy
  void evaluate(
    const std::vector<Uf3Frame>& frames,
    const std::vector<int>& batch_indices,
    GPU_Vector<float>& d_energy);

  // Access to knot info for file writing
  int ncoeff() const { return ncoeff_; }
  int nknots() const { return nknots_; }
  int num_pairs() const { return num_pairs_; }
  int num_types() const { return num_types_; }
  const std::vector<float>& knots() const { return knots_; }
  const std::vector<std::string>& elements() const { return elements_; }
  float rc() const { return rc_; }

  // Pre-allocated GPU buffers (reused across evaluate() calls)
  GPU_Vector<int> d_types, d_batch_idx, d_bnatoms, d_boffsets, d_type_map;
  GPU_Vector<float> d_x, d_y, d_z;
  GPU_Vector<float4> d_coeff_gpu;

private:
  void build_knots();
  void upload_coeffs_to_gpu();

  int ncoeff_, nknots_, nint_;
  int num_pairs_, num_types_;
  float rc_;
  std::vector<std::string> elements_;
  std::vector<float> knots_;
  std::vector<std::vector<float>> coeffs_;
};
