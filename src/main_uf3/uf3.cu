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
UF3 training model — GPU-accelerated B-spline energy evaluation.
Pre-computes per-interval combined cubic polynomials (float4) from B-spline
coefficients, then evaluates energies in parallel on GPU.
------------------------------------------------------------------------------*/

#include "uf3.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>

// ---------------------------------------------------------------------------
// GPU device helpers
// ---------------------------------------------------------------------------
__device__ inline float uf3_eval_cubic(float4 c, float u)
{
  return c.x + u * (c.y + u * (c.z + u * c.w));
}

__device__ inline int uf3_find_interval(float r, float kmin, float kdelta, int nint)
{
  int i = (int)((r - kmin) / kdelta);
  if (i < 0) i = 0;
  if (i >= nint) i = nint - 1;
  return i;
}

// GPU kernel: evaluate 2-body energy for a batch of frames
static __global__ void uf3_eval_2b_batch(
  int num_frames,
  const int* __restrict__ d_frame_idx,
  const int* __restrict__ d_natoms,
  const int* __restrict__ d_offsets,
  const int* __restrict__ d_types,
  const float* __restrict__ d_x,
  const float* __restrict__ d_y,
  const float* __restrict__ d_z,
  const float4* __restrict__ d_coeff,
  int num_pairs, int nint,
  float knot_min, float knot_delta, float rc,
  const int* __restrict__ d_type_map,
  int num_types,
  float* __restrict__ d_energy)
{
  int b = blockIdx.x;
  if (b >= num_frames) return;

  int fidx = d_frame_idx[b];
  int n = d_natoms[fidx];
  int off = d_offsets[fidx];
  float pe = 0.0f;

  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      float dx = d_x[off + i] - d_x[off + j];
      float dy = d_y[off + i] - d_y[off + j];
      float dz = d_z[off + i] - d_z[off + j];
      float r = sqrtf(dx * dx + dy * dy + dz * dz);
      if (r >= rc) continue;

      int ti = d_types[off + i], tj = d_types[off + j];
      int pair_idx = d_type_map[ti * num_types + tj];
      int m = uf3_find_interval(r, knot_min, knot_delta, nint);
      float u = (r - (knot_min + m * knot_delta)) / knot_delta;
      float4 c = __ldg(&d_coeff[pair_idx * nint + m]);
      pe += uf3_eval_cubic(c, u);
    }
  }
  d_energy[b] = pe;
}

// ---------------------------------------------------------------------------
// Pre-computation: B-spline coeffs → combined cubic polynomial per interval
// ---------------------------------------------------------------------------
static void precompute_2b(
  const std::vector<float>& coeffs, std::vector<float4>& h_coeff)
{
  int nc = (int)coeffs.size();
  int nint = nc + 3; // nknots = nc + 4, nint = nknots - 1
  h_coeff.resize(nint);
  for (int m = 0; m < nint; m++) {
    int i0 = m - 3, i1 = m - 2, i2 = m - 1, i3 = m;
    if (i0 < 0) i0 = 0;
    if (i1 < 0) i1 = 0;
    if (i2 < 0) i2 = 0;
    if (i2 >= nc) i2 = nc - 1;
    if (i3 >= nc) i3 = nc - 1;
    float c0 = coeffs[i0], c1 = coeffs[i1], c2 = coeffs[i2], c3 = coeffs[i3];
    float A = (c0 + 4.0f * c1 + c2) / 6.0f;
    float B = (-3.0f * c0 + 3.0f * c2) / 6.0f;
    float C = (3.0f * c0 - 6.0f * c1 + 3.0f * c2) / 6.0f;
    float D = (-c0 + 3.0f * c1 - 3.0f * c2 + c3) / 6.0f;
    h_coeff[m] = make_float4(A, B, C, D);
  }
}

// ---------------------------------------------------------------------------
// Uf3Model implementation
// ---------------------------------------------------------------------------
Uf3Model::Uf3Model(UF3_Parameters& para)
{
  ncoeff_ = para.n_max_2b;
  nknots_ = ncoeff_ + 4;
  nint_ = nknots_ - 1;
  num_types_ = para.num_types;
  num_pairs_ = num_types_ * num_types_;
  rc_ = (float)para.rc_2b;
  elements_ = para.elements;

  build_knots();

  // Initialize coefficients randomly (host)
  coeffs_.resize(num_pairs_);
  srand(42);
  for (int p = 0; p < num_pairs_; p++) {
    coeffs_[p].resize(ncoeff_);
    for (int c = 0; c < ncoeff_; c++)
      coeffs_[p][c] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
  }

  // Pre-allocate GPU buffers (sized per-batch, reallocated as needed)
  int max_atoms = 200 * 128; // generous batch estimate
  d_types.resize(max_atoms);
  d_x.resize(max_atoms); d_y.resize(max_atoms); d_z.resize(max_atoms);

  // Type map
  std::vector<int> h_map(num_pairs_);
  for (int p = 0; p < num_pairs_; p++) h_map[p] = p;
  d_type_map.resize(num_pairs_);
  d_type_map.copy_from_host(h_map.data());

  d_coeff_gpu.resize(num_pairs_ * nint_);

  upload_coeffs_to_gpu();
}

void Uf3Model::build_knots()
{
  knots_.resize(nknots_);
  float delta = rc_ / (nknots_ - 1);
  for (int i = 0; i < nknots_; i++) knots_[i] = i * delta;
}

void Uf3Model::upload_coeffs_to_gpu()
{
  std::vector<float4> all;
  for (int p = 0; p < num_pairs_; p++) {
    std::vector<float4> hc;
    precompute_2b(coeffs_[p], hc);
    for (auto& c : hc) all.push_back(c);
  }
  d_coeff_gpu.resize(all.size());
  d_coeff_gpu.copy_from_host(all.data());
}

void Uf3Model::get_parameters(float* params) const
{
  for (int p = 0; p < num_pairs_; p++)
    for (int c = 0; c < ncoeff_; c++)
      params[p * ncoeff_ + c] = coeffs_[p][c];
}

void Uf3Model::set_parameters(const float* params)
{
  for (int p = 0; p < num_pairs_; p++)
    for (int c = 0; c < ncoeff_; c++)
      coeffs_[p][c] = params[p * ncoeff_ + c];
  upload_coeffs_to_gpu();
}

void Uf3Model::evaluate(
  const std::vector<Uf3Frame>& frames,
  const std::vector<int>& batch_indices,
  GPU_Vector<float>& d_energy)
{
  int batch_size = (int)batch_indices.size();

  // Build flat batch arrays on host
  int total = 0;
  for (int b = 0; b < batch_size; b++)
    total += frames[batch_indices[b]].num_atoms;

  std::vector<int> h_bnatoms(batch_size), h_boffsets(batch_size + 1);
  std::vector<int> h_btypes(total);
  std::vector<float> h_bx(total), h_by(total), h_bz(total);
  h_boffsets[0] = 0;
  {
    int off = 0;
    for (int b = 0; b < batch_size; b++) {
      const Uf3Frame& f = frames[batch_indices[b]];
      h_bnatoms[b] = f.num_atoms;
      h_boffsets[b + 1] = h_boffsets[b] + f.num_atoms;
      for (int i = 0; i < f.num_atoms; i++) {
        h_btypes[off + i] = f.types[i];
        h_bx[off + i] = f.x[i]; h_by[off + i] = f.y[i]; h_bz[off + i] = f.z[i];
      }
      off += f.num_atoms;
    }
  }

  // Upload to GPU
  d_types.resize(total);
  d_types.copy_from_host(h_btypes.data());
  d_x.resize(total); d_x.copy_from_host(h_bx.data());
  d_y.resize(total); d_y.copy_from_host(h_by.data());
  d_z.resize(total); d_z.copy_from_host(h_bz.data());

  std::vector<int> h_bidx(batch_size);
  for (int b = 0; b < batch_size; b++) h_bidx[b] = b;
  d_batch_idx.resize(batch_size);
  d_batch_idx.copy_from_host(h_bidx.data());
  d_bnatoms.resize(batch_size);
  d_bnatoms.copy_from_host(h_bnatoms.data());
  d_boffsets.resize(batch_size + 1);
  d_boffsets.copy_from_host(h_boffsets.data());
  d_energy.resize(batch_size);

  // Launch kernel
  float kmin = knots_[0], kdelta = (knots_.back() - knots_[0]) / nint_;
  uf3_eval_2b_batch<<<batch_size, 1>>>(
    batch_size, d_batch_idx.data(),
    d_bnatoms.data(), d_boffsets.data(),
    d_types.data(), d_x.data(), d_y.data(), d_z.data(),
    d_coeff_gpu.data(), num_pairs_, nint_,
    kmin, kdelta, rc_,
    d_type_map.data(), num_types_,
    d_energy.data());
  GPU_CHECK_KERNEL
}
