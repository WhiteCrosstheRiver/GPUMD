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
UF3 training model: manages 2-body and 3-body B-spline coefficients,
pre-computes cubic polynomial tables, and launches GPU kernels for batch
energy evaluation.  All GPU buffers are pre-allocated once; no per-call
resize to avoid heap fragmentation.
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

  int num_parameters() const { return num_params_total_; }
  void get_parameters(float* params) const;
  void set_parameters(const float* params);

  // GPU batch energy evaluation (2B + 3B).  d_energy is overwritten
  // with per-frame energies; must be pre-sized to match batch_indices.size().
  void evaluate(
    const std::vector<Uf3Frame>& frames,
    const std::vector<int>& batch_indices,
    GPU_Vector<float>& d_energy);

  // 2B accessors
  int ncoeff_2b() const { return ncoeff_2b_; }
  int nknots_2b() const { return nknots_2b_; }
  int num_pairs() const { return num_types_ * num_types_; }
  int num_types() const { return num_types_; }

  // 3B accessors
  bool has_3b() const { return has_3b_; }
  int ncoeff_3b(int d) const { return nc_3b_[d]; }
  int nknots_3b(int d) const { return nk_3b_[d]; }
  int num_triplets() const { return num_trips_; }

  const std::vector<float>& knots_2b() const { return knots_2b_; }
  const std::vector<float>& knots_3b(int d) const { return knots_3b_[d]; }
  const std::vector<std::string>& elements() const { return elements_; }
  float rc_2b() const { return rc_2b_; }
  float rc_3b(int d) const { return rc_3b_[d]; }

private:
  void build_knots();
  void prealloc_gpu(const UF3_Parameters& para);
  void upload_2b_coeffs();
  void upload_3b_coeffs();
  void init_3b_basis();
  void ensure_batch_buffers(int batch_atoms, int batch_size);

  // 2B
  int ncoeff_2b_, nknots_2b_, nint_2b_;
  int num_types_;
  float rc_2b_;
  std::vector<std::string> elements_;
  std::vector<float> knots_2b_;
  std::vector<std::vector<float>> coeffs_2b_;
  int num_params_2b_;

  // 3B
  bool has_3b_ = false;
  int nc_3b_[3];
  int nk_3b_[3], nint_3b_[3];
  int num_trips_;
  float rc_3b_[3];
  std::vector<float> knots_3b_[3];
  std::vector<float> coeffs_3b_;
  int num_params_3b_;
  int num_params_total_;

  // ---- Pre-allocated GPU buffers (never resized after init) ----
  int gpu_max_atoms_ = 0;       // current capacity
  int gpu_max_batch_ = 0;
  int gpu_max_tensor_ = 0;
  int gpu_max_coeff_2b_ = 0;

public:  // (optimizers access these directly)
  GPU_Vector<int>    d_types, d_batch_idx, d_bnatoms, d_boffsets;
  GPU_Vector<float>  d_x, d_y, d_z, d_energy_buf;
  GPU_Vector<float4> d_coeff_2b;
  GPU_Vector<float>  d_tensor_3b;
  GPU_Vector<float4> d_basis_3b_all;
  int basis_offsets_[3];
  GPU_Vector<int>    d_trip_map, d_type_map;
  GPU_Vector<int>    d_nn_off, d_nn_lst;   // pre-alloc'd 3B neighbor lists
};
