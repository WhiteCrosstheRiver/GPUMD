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

#include "dataset_gpu.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

void Uf3DatasetGPU::load(const std::vector<Uf3Frame>& frames, bool has_3b_flag,
                         int batch_size)
{
  num_frames = (int)frames.size();
  has_3b = has_3b_flag;
  if (num_frames == 0) return;

  // Count total atoms & build flat host arrays
  total_atoms = 0;
  h_natoms.resize(num_frames);
  h_offsets.resize(num_frames + 1);
  h_offsets[0] = 0;
  for (int i = 0; i < num_frames; i++) {
    h_natoms[i] = frames[i].num_atoms;
    total_atoms += h_natoms[i];
    h_offsets[i + 1] = total_atoms;
  }

  std::vector<int> h_types(total_atoms);
  std::vector<float> h_x(total_atoms), h_y(total_atoms), h_z(total_atoms);
  std::vector<float> h_fx(total_atoms), h_fy(total_atoms), h_fz(total_atoms);
  std::vector<float> h_energy(num_frames);
  std::vector<int> h_nn_off, h_nn_lst, h_nn_foff;

  int nn_total_off = 0, nn_total_lst = 0;
  if (has_3b) {
    for (int i = 0; i < num_frames; i++) {
      nn_total_off += frames[i].num_atoms;
      nn_total_lst += (int)frames[i].nn_list.size();
    }
    h_nn_off.reserve(nn_total_off + num_frames);
    h_nn_lst.reserve(nn_total_lst);
    h_nn_foff.reserve(num_frames + 1);
  }

  int global_nn_off = 0;
  for (int i = 0; i < num_frames; i++) {
    const auto& f = frames[i];
    int off = h_offsets[i];
    std::copy(f.types.begin(), f.types.end(), h_types.begin() + off);
    std::copy(f.x.begin(), f.x.end(), h_x.begin() + off);
    std::copy(f.y.begin(), f.y.end(), h_y.begin() + off);
    std::copy(f.z.begin(), f.z.end(), h_z.begin() + off);
    std::copy(f.fx.begin(), f.fx.end(), h_fx.begin() + off);
    std::copy(f.fy.begin(), f.fy.end(), h_fy.begin() + off);
    std::copy(f.fz.begin(), f.fz.end(), h_fz.begin() + off);
    h_energy[i] = f.energy;

    if (has_3b) {
      h_nn_foff.push_back((int)h_nn_off.size());
      for (int a = 0; a < f.num_atoms; a++) {
        h_nn_off.push_back(global_nn_off);
        if (!f.nn_list.empty()) {
          for (int jj = 0; jj < f.nn_counts[a]; jj++)
            h_nn_lst.push_back(f.nn_list[f.nn_offset[a] + jj]);
          global_nn_off += f.nn_counts[a];
        }
      }
    }
  }
  if (has_3b) h_nn_foff.push_back((int)h_nn_off.size());

  // Upload to GPU
  d_natoms.resize(num_frames);
  d_natoms.copy_from_host(h_natoms.data());
  d_offsets.resize(num_frames + 1);
  d_offsets.copy_from_host(h_offsets.data());

  d_types.resize(total_atoms);
  d_types.copy_from_host(h_types.data());
  d_x.resize(total_atoms); d_x.copy_from_host(h_x.data());
  d_y.resize(total_atoms); d_y.copy_from_host(h_y.data());
  d_z.resize(total_atoms); d_z.copy_from_host(h_z.data());

  d_energy_ref.resize(num_frames);
  d_energy_ref.copy_from_host(h_energy.data());
  d_fx_ref.resize(total_atoms); d_fx_ref.copy_from_host(h_fx.data());
  d_fy_ref.resize(total_atoms); d_fy_ref.copy_from_host(h_fy.data());
  d_fz_ref.resize(total_atoms); d_fz_ref.copy_from_host(h_fz.data());

  if (has_3b) {
    d_nn_off.resize(h_nn_off.size());
    d_nn_off.copy_from_host(h_nn_off.data());
    d_nn_lst.resize(h_nn_lst.size());
    d_nn_lst.copy_from_host(h_nn_lst.data());
    d_nn_frame_off.resize(h_nn_foff.size());
    d_nn_frame_off.copy_from_host(h_nn_foff.data());
  }

  // Sort by energy per atom, interleave into balanced batches
  std::vector<int> order(num_frames);
  for (int i = 0; i < num_frames; i++) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return h_energy[a] / h_natoms[a] < h_energy[b] / h_natoms[b];
  });

  num_batches = std::max(1, num_frames / batch_size);
  batches.resize(num_batches);
  for (int i = 0; i < num_frames; i++)
    batches[i % num_batches].push_back(order[i]);

  // Pre-compute per-batch atom totals and cache batch frame indices on GPU
  // so optimizer hot-paths never need a per-step H2D for batch metadata.
  batch_total_atoms.assign(num_batches, 0);
  batch_fidx_off.assign(num_batches + 1, 0);
  max_batch_size = 0;
  max_batch_atoms = 0;
  for (int b = 0; b < num_batches; b++) {
    int sz = (int)batches[b].size();
    int sum_atoms = 0;
    for (int k = 0; k < sz; k++) sum_atoms += h_natoms[batches[b][k]];
    batch_total_atoms[b] = sum_atoms;
    batch_fidx_off[b + 1] = batch_fidx_off[b] + sz;
    if (sz > max_batch_size) max_batch_size = sz;
    if (sum_atoms > max_batch_atoms) max_batch_atoms = sum_atoms;
  }
  std::vector<int> h_batch_fidx_all(batch_fidx_off.back());
  for (int b = 0; b < num_batches; b++) {
    int base = batch_fidx_off[b];
    for (int k = 0; k < (int)batches[b].size(); k++)
      h_batch_fidx_all[base + k] = batches[b][k];
  }
  d_batch_fidx_all.resize(h_batch_fidx_all.size());
  d_batch_fidx_all.copy_from_host(h_batch_fidx_all.data());

  printf("GPU dataset: %d frames, %d atoms, %d batches (size=%d, max=%d frames / %d atoms), 3B=%s\n",
         num_frames, total_atoms, num_batches, batch_size,
         max_batch_size, max_batch_atoms, has_3b ? "yes" : "no");
}

void Uf3DatasetGPU::build_batch_meta(int batch_id,
                                     GPU_Vector<int>& d_fidx,
                                     GPU_Vector<int>& d_bnatoms,
                                     GPU_Vector<int>& d_boffsets) const
{
  const auto& bidx = batches[batch_id % num_batches];
  int B = (int)bidx.size();
  std::vector<int> h_fidx(B), h_nat(B), h_off(B + 1);
  h_off[0] = 0;
  for (int b = 0; b < B; b++) {
    h_fidx[b] = b;
    h_nat[b] = h_natoms[bidx[b]];
    h_off[b + 1] = h_off[b] + h_nat[b];
  }
  d_fidx.copy_from_host(h_fidx.data());
  d_bnatoms.copy_from_host(h_nat.data());
  d_boffsets.copy_from_host(h_off.data());
}
