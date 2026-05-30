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
GPU-resident training dataset.  All frames are pre-loaded to GPU once,
eliminating per-call H2D transfers.  Frames are sorted by energy and
interleaved into balanced batches (NEP-style round-robin).
------------------------------------------------------------------------------*/

#pragma once
#include "dataset.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

struct Uf3DatasetGPU
{
  int num_frames = 0;
  int total_atoms = 0;
  int num_batches = 0;
  int max_batch_size = 0;   // max number of frames in any batch
  int max_batch_atoms = 0;  // max sum-of-atoms across all batches

  // --- Per-frame metadata (GPU, size = num_frames) ---
  GPU_Vector<int> d_natoms;      // REAL atoms per frame (centers / loss / 1-body)
  GPU_Vector<int> d_natoms_tot;  // real + ghost atoms per frame (neighbor bound)
  GPU_Vector<int> d_offsets;     // cumulative EXPANDED atom offsets (num_frames+1)

  // --- Per-atom data (GPU, size = total_atoms = sum of expanded counts) ---
  // Layout per frame: real atoms [0,n_real) first, then ghost images.
  GPU_Vector<int> d_types;
  GPU_Vector<float> d_x, d_y, d_z;
  GPU_Vector<int> d_parent;      // frame-local real index a ghost images (self for real)

  // --- Reference data (GPU) ---
  GPU_Vector<float> d_energy_ref;  // per-frame total energy (num_frames)
  GPU_Vector<float> d_fx_ref, d_fy_ref, d_fz_ref;  // per-atom forces

  // --- 3B neighbor lists (GPU) ---
  bool has_3b = false;
  GPU_Vector<int> d_nn_off;          // per-atom offsets
  GPU_Vector<int> d_nn_lst;          // flat neighbor indices
  GPU_Vector<int> d_nn_frame_off;    // per-frame start in nn_off

  // --- CPU mirrors (for batch metadata building) ---
  std::vector<int> h_natoms;
  std::vector<int> h_offsets;

  // --- Round-robin batches (pre-sorted by energy, interleaved) ---
  std::vector<std::vector<int>> batches;  // batches[b][k] = global frame index
  std::vector<int> batch_total_atoms;     // [num_batches] sum of atoms per batch

  // --- GPU-cached batch frame-index lists (avoid per-step H2D) ---
  // d_batch_fidx_all[i] is the global frame index of slot i across all batches,
  // laid out as [batch0_size frames | batch1_size frames | ...].
  GPU_Vector<int> d_batch_fidx_all;
  std::vector<int> batch_fidx_off;        // [num_batches+1] offsets into d_batch_fidx_all

  // Load all frames to GPU, energy-sort, partition into batches
  void load(const std::vector<Uf3Frame>& frames, bool has_3b_flag, int batch_size);

  // Build per-batch metadata arrays for kernel launch (legacy host-build path).
  void build_batch_meta(int batch_id,
                        GPU_Vector<int>& d_fidx, GPU_Vector<int>& d_bnatoms,
                        GPU_Vector<int>& d_boffsets) const;

  // Fast accessor: device pointer to frame indices of a given batch (no copy).
  const int* batch_fidx_device_ptr(int batch_id) const {
    return d_batch_fidx_all.data() + batch_fidx_off[batch_id % num_batches];
  }
  int batch_size(int batch_id) const {
    return (int)batches[batch_id % num_batches].size();
  }
};
