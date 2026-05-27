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
UF3 training model — GPU-accelerated B-spline energy evaluation (2B + 3B).
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

// GPU kernel: 2-body batch evaluation
static __global__ void uf3_eval_2b(
  int num_frames,
  const int* __restrict__ d_frame_idx,
  const int* __restrict__ d_natoms,
  const int* __restrict__ d_offsets,
  const int* __restrict__ d_types,
  const float* __restrict__ d_x, const float* __restrict__ d_y, const float* __restrict__ d_z,
  const float4* __restrict__ d_coeff,
  int num_pairs, int nint,
  float knot_min, float knot_delta, float rc,
  const int* __restrict__ d_type_map, int num_types,
  float* __restrict__ d_energy)
{
  int b = blockIdx.x;
  if (b >= num_frames) return;
  int fidx = d_frame_idx[b], n = d_natoms[fidx], off = d_offsets[fidx];
  float pe = 0.0f;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      float dx = d_x[off+i] - d_x[off+j], dy = d_y[off+i] - d_y[off+j], dz = d_z[off+i] - d_z[off+j];
      float r = sqrtf(dx*dx + dy*dy + dz*dz);
      if (r >= rc) continue;
      int m = uf3_find_interval(r, knot_min, knot_delta, nint);
      float u = (r - (knot_min + m * knot_delta)) / knot_delta;
      int ti = d_types[off+i], tj = d_types[off+j];
      float4 c = __ldg(&d_coeff[d_type_map[ti * num_types + tj] * nint + m]);
      pe += uf3_eval_cubic(c, u);
    }
  }
  d_energy[b] = pe;
}

// GPU kernel: 3-body batch evaluation (triplet energy)
static __global__ void uf3_eval_3b(
  int num_frames,
  const int* __restrict__ d_frame_idx,
  const int* __restrict__ d_natoms,
  const int* __restrict__ d_offsets,
  const int* __restrict__ d_types,
  const float* __restrict__ d_x, const float* __restrict__ d_y, const float* __restrict__ d_z,
  const float* __restrict__ d_tensor,
  int nc0, int nc1, int nc2,
  const float4* __restrict__ d_basis0, int nint0,
  const float4* __restrict__ d_basis1, int nint1,
  const float4* __restrict__ d_basis2, int nint2,
  float kmin0, float kd0, float rc0,
  float kmin1, float kd1, float rc1,
  float kmin2, float kd2, float rc2,
  const int* __restrict__ d_trip_map, int num_trips, int num_types,
  float* __restrict__ d_energy)
{
  int b = blockIdx.x;
  if (b >= num_frames) return;
  int fidx = d_frame_idx[b], n = d_natoms[fidx], off = d_offsets[fidx];
  float pe = 0.0f;

  for (int i = 0; i < n; i++) {
    int ti = d_types[off+i];
    for (int j = i + 1; j < n; j++) {
      float dx12 = d_x[off+j] - d_x[off+i], dy12 = d_y[off+j] - d_y[off+i], dz12 = d_z[off+j] - d_z[off+i];
      float r12 = sqrtf(dx12*dx12 + dy12*dy12 + dz12*dz12);
      if (r12 >= rc0) continue;
      int tj = d_types[off+j];

      for (int k = j + 1; k < n; k++) {
        float dx13 = d_x[off+k] - d_x[off+i], dy13 = d_y[off+k] - d_y[off+i], dz13 = d_z[off+k] - d_z[off+i];
        float r13 = sqrtf(dx13*dx13 + dy13*dy13 + dz13*dz13);
        if (r13 >= rc1) continue;

        float dx23 = d_x[off+k] - d_x[off+j], dy23 = d_y[off+k] - d_y[off+j], dz23 = d_z[off+k] - d_z[off+j];
        float r23 = sqrtf(dx23*dx23 + dy23*dy23 + dz23*dz23);
        if (r23 >= rc2) continue;

        int tk = d_types[off+k];
        int trip_idx = d_trip_map[(ti * num_types + tj) * num_types + tk];

        // Eval basis values on each dimension
        int m0 = uf3_find_interval(r12, kmin0, kd0, nint0);
        int m1 = uf3_find_interval(r13, kmin1, kd1, nint1);
        int m2 = uf3_find_interval(r23, kmin2, kd2, nint2);
        float u0 = (r12 - (kmin0 + m0*kd0)) / kd0;
        float u1 = (r13 - (kmin1 + m1*kd1)) / kd1;
        float u2 = (r23 - (kmin2 + m2*kd2)) / kd2;

        float b0[4], b1[4], b2[4];
        for (int p = 0; p < 4; p++) { b0[p] = uf3_eval_cubic(__ldg(&d_basis0[m0*4+p]), u0); }
        for (int p = 0; p < 4; p++) { b1[p] = uf3_eval_cubic(__ldg(&d_basis1[m1*4+p]), u1); }
        for (int p = 0; p < 4; p++) { b2[p] = uf3_eval_cubic(__ldg(&d_basis2[m2*4+p]), u2); }

        int p0 = m0 - 3; if (p0 < 0) p0 = 0;
        int p1 = m1 - 3; if (p1 < 0) p1 = 0;
        int p2 = m2 - 3; if (p2 < 0) p2 = 0;

        const float* C = &d_tensor[trip_idx * nc0 * nc1 * nc2];

        // Tensor contraction: 4x4x4 = 64 terms
        for (int dp = 0; dp < 4; dp++) {
          int q = p0 + dp; if (q >= nc0) continue;
          float bp = b0[dp];
          for (int dq = 0; dq < 4; dq++) {
            int r = p1 + dq; if (r >= nc1) continue;
            float bq = b1[dq];
            for (int dr = 0; dr < 4; dr++) {
              int s = p2 + dr; if (s >= nc2) continue;
              pe += C[q + r * nc0 + s * nc0 * nc1] * bp * bq * b2[dr];
            }
          }
        }
      }
    }
  }
  d_energy[b] += pe;
}

// ---------------------------------------------------------------------------
// Pre-computation helpers
// ---------------------------------------------------------------------------
static void precompute_2b(const std::vector<float>& coeffs, std::vector<float4>& out)
{
  int nc = (int)coeffs.size(), nint = nc + 3;
  out.resize(nint);
  for (int m = 0; m < nint; m++) {
    int i0 = m-3, i1 = m-2, i2 = m-1, i3 = m;
    if (i0 < 0) i0 = 0; if (i1 < 0) i1 = 0; if (i2 < 0) i2 = 0;
    if (i2 >= nc) i2 = nc-1; if (i3 >= nc) i3 = nc-1;
    float c0 = coeffs[i0], c1 = coeffs[i1], c2 = coeffs[i2], c3 = coeffs[i3];
    out[m] = make_float4(
      (c0 + 4.0f*c1 + c2) / 6.0f,
      (-3.0f*c0 + 3.0f*c2) / 6.0f,
      (3.0f*c0 - 6.0f*c1 + 3.0f*c2) / 6.0f,
      (-c0 + 3.0f*c1 - 3.0f*c2 + c3) / 6.0f);
  }
}

// Precompute per-interval basis cubic polynomials for 3B
static void precompute_3b_basis(int nint, std::vector<float4>& out)
{
  out.resize(nint * 4);
  float b[4][4] = {
    { 1.0f/6,  -3.0f/6,   3.0f/6,  -1.0f/6 },
    { 4.0f/6,   0.0f,     -6.0f/6,   3.0f/6 },
    { 1.0f/6,   3.0f/6,   3.0f/6,  -3.0f/6 },
    { 0.0f,     0.0f,     0.0f,     1.0f/6 }
  };
  for (int m = 0; m < nint; m++)
    for (int p = 0; p < 4; p++)
      out[m*4+p] = make_float4(b[p][0], b[p][1], b[p][2], b[p][3]);
}

// ---------------------------------------------------------------------------
// Uf3Model
// ---------------------------------------------------------------------------
Uf3Model::Uf3Model(UF3_Parameters& para)
{
  num_types_ = para.num_types;
  elements_ = para.elements;
  ncoeff_2b_ = para.n_max_2b;
  nknots_2b_ = ncoeff_2b_ + 4;
  nint_2b_ = nknots_2b_ - 1;
  rc_2b_ = (float)para.rc_2b;

  has_3b_ = (para.n_max_3b[0] > 0);
  if (has_3b_) {
    for (int d = 0; d < 3; d++) {
      nc_3b_[d] = para.n_max_3b[d];
      nk_3b_[d] = nc_3b_[d] + 4;
      nint_3b_[d] = nk_3b_[d] - 1;
      rc_3b_[d] = (float)(d < 2 ? para.rc_3b[d] : para.rc_3b[0]);
    }
    num_trips_ = num_types_ * num_types_ * num_types_;
    num_params_3b_ = num_trips_ * nc_3b_[0] * nc_3b_[1] * nc_3b_[2];
  } else {
    num_trips_ = 0;
    num_params_3b_ = 0;
  }

  num_params_2b_ = num_types_ * num_types_ * ncoeff_2b_;
  num_params_total_ = num_params_2b_ + num_params_3b_;

  build_knots();

  // Init 2B coeffs
  coeffs_2b_.resize(num_types_ * num_types_);
  srand(42);
  for (int p = 0; p < num_types_ * num_types_; p++) {
    coeffs_2b_[p].resize(ncoeff_2b_);
    for (int c = 0; c < ncoeff_2b_; c++)
      coeffs_2b_[p][c] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
  }

  // Init 3B coeffs
  if (has_3b_) {
    coeffs_3b_.resize(num_params_3b_);
    for (int i = 0; i < num_params_3b_; i++)
      coeffs_3b_[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.001f;
  }

  // GPU buffers
  int max_atoms = 200 * 128;
  d_types.resize(max_atoms);
  d_x.resize(max_atoms); d_y.resize(max_atoms); d_z.resize(max_atoms);

  // Type map
  std::vector<int> h_map(num_types_ * num_types_);
  for (int p = 0; p < num_types_ * num_types_; p++) h_map[p] = p;
  d_type_map.resize(num_types_ * num_types_);
  d_type_map.copy_from_host(h_map.data());

  upload_2b_coeffs();
  if (has_3b_) {
    init_3b_basis();
    upload_3b_coeffs();
    // Pre-build trip type map
    std::vector<int> h_trip(num_trips_);
    for (int t = 0; t < num_trips_; t++) h_trip[t] = t;
    d_trip_map.resize(num_trips_);
    d_trip_map.copy_from_host(h_trip.data());
  }
}

void Uf3Model::build_knots()
{
  knots_2b_.resize(nknots_2b_);
  float d2 = rc_2b_ / (nknots_2b_ - 1);
  for (int i = 0; i < nknots_2b_; i++) knots_2b_[i] = i * d2;

  for (int dim = 0; dim < 3; dim++) {
    if (!has_3b_) break;
    knots_3b_[dim].resize(nk_3b_[dim]);
    float d = rc_3b_[dim] / (nk_3b_[dim] - 1);
    for (int i = 0; i < nk_3b_[dim]; i++) knots_3b_[dim][i] = i * d;
  }
}

void Uf3Model::init_3b_basis()
{
  for (int d = 0; d < 3; d++) {
    std::vector<float4> hb;
    precompute_3b_basis(nint_3b_[d], hb);
    d_basis_3b[d].resize(hb.size());
    d_basis_3b[d].copy_from_host(hb.data());
  }
}

void Uf3Model::upload_2b_coeffs()
{
  std::vector<float4> all;
  for (size_t p = 0; p < coeffs_2b_.size(); p++) {
    std::vector<float4> hc;
    precompute_2b(coeffs_2b_[p], hc);
    for (auto& c : hc) all.push_back(c);
  }
  d_coeff_2b.resize(all.size());
  d_coeff_2b.copy_from_host(all.data());
}

void Uf3Model::upload_3b_coeffs()
{
  d_tensor_3b.resize(coeffs_3b_.size());
  d_tensor_3b.copy_from_host(coeffs_3b_.data());
}

void Uf3Model::get_parameters(float* params) const
{
  int idx = 0;
  for (size_t p = 0; p < coeffs_2b_.size(); p++)
    for (int c = 0; c < ncoeff_2b_; c++)
      params[idx++] = coeffs_2b_[p][c];
  for (size_t i = 0; i < coeffs_3b_.size(); i++)
    params[idx++] = coeffs_3b_[i];
}

void Uf3Model::set_parameters(const float* params)
{
  int idx = 0;
  for (size_t p = 0; p < coeffs_2b_.size(); p++)
    for (int c = 0; c < ncoeff_2b_; c++)
      coeffs_2b_[p][c] = params[idx++];
  for (size_t i = 0; i < coeffs_3b_.size(); i++)
    coeffs_3b_[i] = params[idx++];
  upload_2b_coeffs();
  if (has_3b_) upload_3b_coeffs();
}

void Uf3Model::evaluate(
  const std::vector<Uf3Frame>& frames,
  const std::vector<int>& batch_indices,
  GPU_Vector<float>& d_energy)
{
  int batch_size = (int)batch_indices.size();
  int total = 0;
  for (int b = 0; b < batch_size; b++)
    total += frames[batch_indices[b]].num_atoms;

  std::vector<int> h_bnatoms(batch_size), h_boffsets(batch_size+1);
  std::vector<int> h_btypes(total);
  std::vector<float> h_bx(total), h_by(total), h_bz(total);
  h_boffsets[0] = 0;
  { int off = 0;
    for (int b = 0; b < batch_size; b++) {
      const Uf3Frame& f = frames[batch_indices[b]];
      h_bnatoms[b] = f.num_atoms;
      h_boffsets[b+1] = h_boffsets[b] + f.num_atoms;
      for (int i = 0; i < f.num_atoms; i++) {
        h_btypes[off+i] = f.types[i];
        h_bx[off+i] = f.x[i]; h_by[off+i] = f.y[i]; h_bz[off+i] = f.z[i];
      }
      off += f.num_atoms;
    }
  }

  d_types.resize(total); d_types.copy_from_host(h_btypes.data());
  d_x.resize(total);     d_x.copy_from_host(h_bx.data());
  d_y.resize(total);     d_y.copy_from_host(h_by.data());
  d_z.resize(total);     d_z.copy_from_host(h_bz.data());

  std::vector<int> h_bidx(batch_size);
  for (int b = 0; b < batch_size; b++) h_bidx[b] = b;
  d_batch_idx.resize(batch_size); d_batch_idx.copy_from_host(h_bidx.data());
  d_bnatoms.resize(batch_size);   d_bnatoms.copy_from_host(h_bnatoms.data());
  d_boffsets.resize(batch_size+1); d_boffsets.copy_from_host(h_boffsets.data());
  d_energy.resize(batch_size);

  // 2B kernel
  float kmin2 = knots_2b_[0], kd2 = (knots_2b_.back() - knots_2b_[0]) / nint_2b_;
  uf3_eval_2b<<<batch_size, 1>>>(
    batch_size, d_batch_idx.data(), d_bnatoms.data(), d_boffsets.data(),
    d_types.data(), d_x.data(), d_y.data(), d_z.data(),
    d_coeff_2b.data(), num_types_ * num_types_, nint_2b_,
    kmin2, kd2, rc_2b_,
    d_type_map.data(), num_types_, d_energy.data());
  GPU_CHECK_KERNEL

  // 3B kernel (note: O(n^3) per frame, use small batches)
  if (has_3b_) {
    uf3_eval_3b<<<batch_size, 1>>>(
      batch_size, d_batch_idx.data(), d_bnatoms.data(), d_boffsets.data(),
      d_types.data(), d_x.data(), d_y.data(), d_z.data(),
      d_tensor_3b.data(),
      nc_3b_[0], nc_3b_[1], nc_3b_[2],
      d_basis_3b[0].data(), nint_3b_[0],
      d_basis_3b[1].data(), nint_3b_[1],
      d_basis_3b[2].data(), nint_3b_[2],
      knots_3b_[0][0], (knots_3b_[0].back() - knots_3b_[0][0]) / nint_3b_[0], rc_3b_[0],
      knots_3b_[1][0], (knots_3b_[1].back() - knots_3b_[1][0]) / nint_3b_[1], rc_3b_[1],
      knots_3b_[2][0], (knots_3b_[2].back() - knots_3b_[2][0]) / nint_3b_[2], rc_3b_[2],
      d_trip_map.data(), num_trips_, num_types_,
      d_energy.data());
    GPU_CHECK_KERNEL
  }
}
