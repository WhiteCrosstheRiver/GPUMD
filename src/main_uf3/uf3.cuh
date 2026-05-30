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
#include "dataset_gpu.cuh"
#include "parameters.cuh"
#include "utilities/gpu_vector.cuh"
#include <cuda_runtime.h>
#include <vector>

class Uf3Model
{
public:
  Uf3Model(UF3_Parameters& para);
  ~Uf3Model();

  int num_parameters() const { return num_params_total_; }
  void get_parameters(float* params) const;

  // Async path: copies raw coeffs H2D and runs the float4 spline build on
  // the GPU using `stream`.  Caller must synchronize the stream (or use
  // events) before launching any kernel that consumes d_coeff_2b/d_tensor_3b.
  void set_parameters_async(const float* params, cudaStream_t stream);

  // Synchronous wrapper around set_parameters_async on stream_ (the model's
  // default compute stream).  Blocks until the GPU build kernel finishes.
  void set_parameters(const float* params);

  // Compute stream used for the training hot path (forward + loss reduction).
  cudaStream_t stream() const { return stream_; }

  // ---- Population (multi-individual) hot path -----------------------------
  // Evaluates `pop` parameter sets in a single launch chain.  Used by SNES to
  // amortize launch overhead and global memory traffic across an entire
  // generation.  Currently supports 2B-only; 3B falls back to the
  // per-individual loop in the caller (Uf3Fitness::compute_loss_population).
  //
  // host_pop_params: pop * nparam, row-major (individual p at offset p*nparam).
  // All launches go on `stream`.
  void set_population_parameters_async(
    const float* host_pop_params, int pop, cudaStream_t stream);

  // Fused forward + reduction for the entire population on one batch.
  // Replaces the old evaluate_batch_population + compute_loss_reduction_population
  // pair with a single kernel that:
  //   • computes energy and forces for all P individuals in one pass
  //   • immediately accumulates MSE residuals (no per-atom force storage)
  //   • writes d_loss_sum_pop[p*2+0] = energy MSE sum
  //           d_loss_sum_pop[p*2+1] = force MSE sum
  // Saves: d_fx_pop / d_fy_pop / d_fz_pop buffers and the redundant second
  // sweep over all pairs that the old separate-kernel design performed.
  void evaluate_and_reduce_population(
    const Uf3DatasetGPU& ds, int batch_id, int pop,
    GPU_Vector<float>& d_loss_sum_pop);   // size >= 2 * pop

  // Legacy separate-step API (kept for potential offline analysis).
  // compute_loss_population in Uf3Fitness uses evaluate_and_reduce_population
  // instead, so these are no longer on the SNES hot path.
  void evaluate_batch_population(
    const Uf3DatasetGPU& ds, int batch_id, int pop,
    GPU_Vector<float>& d_energy_pop);
  void compute_loss_reduction_population(
    const Uf3DatasetGPU& ds, int batch_id, int pop,
    GPU_Vector<float>& d_loss_sum_pop);

  // Lazy-allocate the population-side buffers to fit (pop, total_atoms).
  void ensure_pop_buffers(int pop, int dataset_total_atoms);

  // ---- Fast path: evaluate using pre-loaded GPU dataset (NO H2D transfers) ---
  // Uses dataset-cached GPU frame indices for batch_id — zero host->device traffic
  // on the hot path.  d_energy is auto-resized.
  void evaluate_batch(
    const Uf3DatasetGPU& ds,
    int batch_id,
    GPU_Vector<float>& d_energy);

  // GPU-side reduction of the loss: writes
  //   d_loss_sum[0] = sum_b (E_pred[b]/na_b - E_ref[fid]/na_b)^2          (energy MSE numerator)
  //   d_loss_sum[1] = sum_a ((Fx-Fx_ref)^2 + (Fy-Fy_ref)^2 + (Fz-Fz_ref)^2)  (force MSE numerator)
  // Must be called AFTER evaluate_batch(...) with the same batch_id.
  void compute_loss_reduction(
    const Uf3DatasetGPU& ds,
    int batch_id,
    GPU_Vector<float>& d_loss_sum);     // size >= 2

  // Gradient of the unified training loss (matches Uf3Fitness::compute_loss).
  // d_grad_ws (>= nparam) and d_ediff_ws (>= batch_size) are workspaces.
  void compute_loss_gradient(
    const Uf3DatasetGPU& ds,
    int batch_id,
    float loss_e,
    float loss_f,
    float lambda_e,
    float lambda_f,
    GPU_Vector<float>& d_grad_ws,
    GPU_Vector<float>& d_ediff_ws,
    std::vector<float>& host_gradient);

  // Ensure d_fx/fy/fz can hold every atom in the dataset (global-offset writes).
  void ensure_global_force_buffer(int dataset_total_atoms);

  // ---- Legacy: evaluate using host-side frames (per-call H2D upload) ---
  void evaluate(
    const std::vector<Uf3Frame>& frames,
    const std::vector<int>& batch_indices,
    GPU_Vector<float>& d_energy);

  void evaluate_forces(
    const std::vector<Uf3Frame>& frames,
    const std::vector<int>& batch_indices,
    GPU_Vector<float>& d_energy,
    GPU_Vector<float>& d_force_x,
    GPU_Vector<float>& d_force_y,
    GPU_Vector<float>& d_force_z);

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
  // Build float4 spline table on the GPU from the raw 2B coeffs already on
  // device (d_raw_coeffs_2b_).  Async on `stream`.
  void upload_2b_coeffs_gpu(cudaStream_t stream);
  // Host helper: copies coeffs_2b_ into a contiguous host array.
  void pack_2b_coeffs_host(std::vector<float>& flat) const;
  void upload_3b_coeffs();
  void init_3b_basis();
  void ensure_batch_buffers(int batch_atoms, int batch_size);
  cudaStream_t stream_ = 0;  // default compute stream; created in prealloc_gpu

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

  // 1-body: one constant energy offset per element type.  Stored at the END of
  // the parameter vector (offset e0_offset_) so 2B/3B parameter indices used by
  // the gradient kernels are unchanged.  A pure 2B/3B model cannot represent a
  // constant per-atom energy, so this term is essential for fitting raw DFT
  // energies (matches reference UF3 offset_1b).
  std::vector<float> coeffs_e0_;
  int e0_offset_;
  int num_params_total_;

  // ---- Pre-allocated GPU buffers (never resized after init) ----
  int gpu_max_atoms_ = 0;        // capacity for d_types/x/y/z (batch-local layout)
  int gpu_max_force_atoms_ = 0;  // capacity for d_fx/fy/fz (dataset-global layout)
  int gpu_max_batch_ = 0;
  int gpu_max_tensor_ = 0;
  int gpu_max_coeff_2b_ = 0;

  // ---- Population-mode capacity tracking ----
  int pop_capacity_ = 0;             // # individuals currently fit
  int pop_force_atoms_capacity_ = 0; // per-individual force atoms count

public:  // (optimizers access these directly)
  GPU_Vector<int>    d_types, d_batch_idx, d_bnatoms, d_boffsets;
  GPU_Vector<float>  d_x, d_y, d_z, d_energy_buf;
  GPU_Vector<float>  d_fx, d_fy, d_fz;         // per-atom forces
  GPU_Vector<float>  d_raw_coeffs_2b;          // [num_types_*num_types_ * ncoeff_2b_]
  GPU_Vector<float4> d_coeff_2b;
  GPU_Vector<float>  d_e0;                      // [num_types_] 1-body energy offsets
  GPU_Vector<float>  d_tensor_3b;
  GPU_Vector<float4> d_basis_3b_all;
  int basis_offsets_[3];
  GPU_Vector<int>    d_trip_map, d_type_map;
  GPU_Vector<int>    d_nn_off, d_nn_lst, d_nn_frame_off; // 3B neighbor lists

  // ---- Population-mode buffers (lazy alloc'd in ensure_pop_buffers) ----
  GPU_Vector<float>  d_raw_coeffs_2b_pop;  // [pop * np2 * ncoeff_2b_]
  GPU_Vector<float4> d_coeff_2b_pop;       // [pop * np2 * nint_2b_]
  GPU_Vector<float>  d_energy_buf_pop;     // [pop * batch_size]
  GPU_Vector<float>  d_fx_pop, d_fy_pop, d_fz_pop;  // [pop * total_atoms]
};
