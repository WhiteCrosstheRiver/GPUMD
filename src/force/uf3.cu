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
The UF3 (Ultra-Fast Force Field) potential — inference implementation.
Ref: S. R. Xie et al., npj Comput. Mater. 9, 143 (2023).

B-spline evaluation: cubic B-spline basis functions are pre-computed at
initialization as per-interval combined cubic polynomials (float4 per
interval).  At runtime, the potential is a simple Horner evaluation of a
cubic polynomial: V(u) = A + u*(B + u*(C + u*D)).
------------------------------------------------------------------------------*/

#include "neighbor.cuh"
#include "uf3.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>

// ---------------------------------------------------------------------------
// B-spline pre-computation helpers (host-side)
// ---------------------------------------------------------------------------

// de Boor evaluation: value of cubic B-spline at r on interval m
static float deboor_eval(float r, int m, const std::vector<float>& knots,
                          const std::vector<float>& coeffs)
{
  int nc = (int)coeffs.size(), nk = (int)knots.size();
  float d[4];
  for (int i = 0; i < 4; i++) {
    int idx = m - 3 + i;
    if (idx < 0) idx = 0; if (idx >= nc) idx = nc - 1;
    d[i] = coeffs[idx];
  }
  float t[8];
  for (int i = 0; i < 8; i++) {
    int ki = m - 3 + i;
    t[i] = knots[ki < 0 ? 0 : (ki >= nk ? nk-1 : ki)];
  }
  float d1[3], d2[2];
  for (int i = 0; i < 3; i++) {
    float den = t[i+4] - t[i+3];
    if (den < 1e-10f) d1[i] = 0; // repeated knot → 0/0 = 0
    else d1[i] = ((r - t[i+3]) * d[i] + (t[i+4] - r) * d[i+1]) / den;
  }
  for (int i = 0; i < 2; i++) {
    float den = t[i+5] - t[i+3];
    if (den < 1e-10f) d2[i] = 0;
    else d2[i] = ((r - t[i+3]) * d1[i] + (t[i+5] - r) * d1[i+1]) / den;
  }
  float den = t[6] - t[3];
  if (den < 1e-10f) return 0;
  return ((r - t[3]) * d2[0] + (t[6] - r) * d2[1]) / den;
}

// Uniform: combined cubic from 4 adjacent coefficients
static void precompute_2b_uniform(
  const std::vector<float>& knots,
  const std::vector<float>& coeffs,
  int leading_trim,
  std::vector<float4>& h_coeff)
{
  const int nk = (int)knots.size();
  const int nc = (int)coeffs.size();
  const int nint = nk - 1;

  h_coeff.resize(nint);
  for (int m = 0; m < nint; m++) {
    int i0 = m - 3, i1 = m - 2, i2 = m - 1, i3 = m;
    if (i0 < 0) i0 = 0;
    if (i1 < 0) i1 = 0;
    if (i2 < 0) i2 = 0;
    if (i2 >= nc) i2 = nc - 1;
    if (i3 >= nc) i3 = nc - 1;
    float c0 = coeffs[i0], c1 = coeffs[i1], c2 = coeffs[i2], c3 = coeffs[i3];
    // Uniform cubic B-spline basis on [0,1] with coeffs c0..c3
    float A = (c0 + 4.0f * c1 + c2) / 6.0f;
    float B = (-3.0f * c0 + 3.0f * c2) / 6.0f;
    float C = (3.0f * c0 - 6.0f * c1 + 3.0f * c2) / 6.0f;
    float D = (-c0 + 3.0f * c1 - 3.0f * c2 + c3) / 6.0f;
    h_coeff[m] = make_float4(A, B, C, D);
  }
}

// Non-uniform: fit cubic through 4 de Boor points on each interval
static void precompute_2b_nonuniform(
  const std::vector<float>& knots, const std::vector<float>& coeffs,
  std::vector<float4>& h_coeff)
{
  int nint = (int)knots.size() - 1;
  h_coeff.resize(nint);
  for (int m = 0; m < nint; m++) {
    float tm = knots[m], h = knots[m+1] - tm;
    if (h < 1e-10f) h = 1e-10f;
    float v[4];
    for (int s = 0; s < 4; s++) {
      v[s] = deboor_eval(tm + (s/3.0f)*h, m, knots, coeffs);
    }
    float A = v[0];
    float B = -5.5f*v[0] + 9.0f*v[1] - 4.5f*v[2] + v[3];
    float C = 9.0f*v[0] - 22.5f*v[1] + 18.0f*v[2] - 4.5f*v[3];
    float D = -4.5f*v[0] + 13.5f*v[1] - 13.5f*v[2] + 4.5f*v[3];
    h_coeff[m] = make_float4(A, B, C, D);
  }
}

// Build per-interval basis cubic polynomials for 3-body dimension.
// On each interval, the 4 active basis functions (cubic polynomials in u)
// are stored as float4 entries.  basis_coeff[m * 4 + p] for p=0,1,2,3.
static void precompute_3b_basis_uniform(
  const std::vector<float>& knots,
  int leading_trim,
  std::vector<float4>& h_basis)
{
  const int nk = (int)knots.size();
  const int nint = nk - 1;
  (void)leading_trim; // unused in uniform basis precomputation

  h_basis.resize(nint * 4);
  for (int m = 0; m < nint; m++) {
    // 4 active basis functions at interval m: B_{m-3}, B_{m-2}, B_{m-1}, B_m
    // In the uniform case they are simply the standard cubic B-spline basis:
    // B0(u)=(1-u)^3/6, B1(u)=(3u^3-6u^2+4)/6,
    // B2(u)=(-3u^3+3u^2+3u+1)/6, B3(u)=u^3/6
    // Pre-compute as A+Bu+Cu^2+Du^3
    float b[4][4] = {
      { 1.0f/6,  -3.0f/6,   3.0f/6,  -1.0f/6 },   // B0 = (1-u)^3
      { 4.0f/6,   0.0f,     -6.0f/6,   3.0f/6 },   // B1 = 3u^3-6u^2+4
      { 1.0f/6,   3.0f/6,   3.0f/6,  -3.0f/6 },   // B2 = -3u^3+3u^2+3u+1
      { 0.0f,     0.0f,     0.0f,     1.0f/6 }     // B3 = u^3
    };
    for (int p = 0; p < 4; p++) {
      h_basis[m * 4 + p] = make_float4(b[p][0], b[p][1], b[p][2], b[p][3]);
    }
  }
}

// ---------------------------------------------------------------------------
// Device helper: evaluate cubic polynomial (Horner form, FMA-friendly)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float eval_cubic(float4 c, float u)
{
  return __fmaf_rn(u, __fmaf_rn(u, __fmaf_rn(u, c.w, c.z), c.y), c.x);
}

__device__ __forceinline__ float eval_cubic_deriv(float4 c, float u)
{
  return __fmaf_rn(u, __fmaf_rn(u, 3.0f * c.w, 2.0f * c.z), c.y);
}

// ---------------------------------------------------------------------------
// Device helper: uniform cubic B-spline basis and derivatives — direct arithmetic.
//
// For uniform knots, the 4 active basis functions are the same constants on
// every interval (they depend only on the local coordinate u ∈ [0,1]).
// Storing and loading a per-interval GPU table wastes L2 bandwidth on
// identical values; direct evaluation costs ~10 FMAs and zero memory traffic.
//
// Basis (p=0..3):
//   B0 = (1-u)³/6,  B1 = (3u³-6u²+4)/6
//   B2 = (-3u³+3u²+3u+1)/6,  B3 = u³/6
// Derivatives d/du × inv_kd = d/dr:
//   dB0 = -(1-u)²/2,  dB1 = (3u²-4u)/2
//   dB2 = (-3u²+2u+1)/2,  dB3 = u²/2   (all × inv_kd)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void eval_bspline4(float u, float b[4])
{
  float u2 = u * u, u3 = u2 * u;
  const float inv6 = 1.0f / 6.0f;
  b[0] = (1.0f - 3.0f*u + 3.0f*u2 - u3) * inv6;
  b[1] = (4.0f - 6.0f*u2 + 3.0f*u3) * inv6;
  b[2] = (1.0f + 3.0f*u + 3.0f*u2 - 3.0f*u3) * inv6;
  b[3] = u3 * inv6;
}

__device__ __forceinline__ void eval_bspline4_deriv(float u, float inv_kd, float db[4])
{
  float om = 1.0f - u, u2 = u * u;
  db[0] = -om * om * 0.5f * inv_kd;
  db[1] = u * (3.0f*u - 4.0f) * 0.5f * inv_kd;
  db[2] = (-3.0f*u2 + 2.0f*u + 1.0f) * 0.5f * inv_kd;
  db[3] = u2 * 0.5f * inv_kd;
}

// ---------------------------------------------------------------------------
// Device helper: find knot interval (uniform grid, inv_delta pre-computed on host)
// ---------------------------------------------------------------------------
__device__ __forceinline__ int find_interval(float r, float knot_min, float inv_knot_delta, int nint)
{
  float t = (r - knot_min) * inv_knot_delta;
  int i = (int)t;
  if (i < 0) i = 0; if (i >= nint) i = nint - 1;
  return i;
}

// Maximum number of 2B float4 coefficients (num_pairs * nint) that fit in the
// shared-memory cache.  A single-element model uses ~10-40; a 2-element model
// (4 pairs) ~40-160.  1024 float4 = 16 KB covers realistic multi-element cases.
// When the table is larger the kernel falls back to L2-cached global reads.
static constexpr int UF3_2B_SHARED_COEFFS = 1024;

// Non-uniform: binary search
__device__ __forceinline__ int find_interval_nu(float r, const float* knots, int nk)
{
  int lo = 0, hi = nk - 2;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (r < knots[mid]) hi = mid - 1; else lo = mid;
  }
  return lo;
}

// ---------------------------------------------------------------------------
// Position packing: double SoA → float4 AoS  (run every step, fully coalesced)
// ---------------------------------------------------------------------------
static __global__ void pack_positions_float4(
  int N,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  float4* __restrict__ g_pos)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  g_pos[i] = make_float4((float)g_x[i], (float)g_y[i], (float)g_z[i], 0.0f);
}

// ---------------------------------------------------------------------------
// 2-body kernel (optimized v3)
//   - float4 packed positions (1×128-bit L2 read vs 3×64-bit double)
//   - cubic coefficient table in shared memory
//   - inv_knot_delta pre-computed, rsqrtf, __fmaf_rn
//   - full neighbor list (no atomics overhead)
// ---------------------------------------------------------------------------
// UNIFORM=true  -> O(1) interval lookup (r-knot_min)*inv_knot_delta (GPUMD trainer
//                  always emits uniform "uk" knots; this is the hot path).
// UNIFORM=false -> binary search over the (non-uniform) knot vector, per-interval
//                  width for the derivative scaling.  Lets the MD engine consume
//                  reference-UF3 models exported with lammps/non-uniform knots.
template <bool UNIFORM>
static __global__ void find_force_uf3_2b(
  const int N, const int N1, const int N2,
  const Box box,
  const float4* __restrict__ d_coeff,               // [num_pairs * nint]
  int num_pairs, int num_types,
  int nint,
  float knot_min, float inv_knot_delta,
  const float* __restrict__ d_knots, int nknots,    // used only when !UNIFORM
  float rc,
  const int* __restrict__ g_NN,
  const int* __restrict__ g_NL,
  const int* __restrict__ g_type,                   // atom types [N]
  const float* __restrict__ g_e0,                   // 1-body energy offsets [ntypes]
  const float4* __restrict__ g_pos,
  double* g_pe,
  double* g_fx, double* g_fy, double* g_fz,
  double* g_virial)
{
  // Cache the full per-pair coefficient table in shared memory when it fits;
  // otherwise read from L2-cached global memory (use_shared = false).
  __shared__ float4 s_coeff[UF3_2B_SHARED_COEFFS];
  const int ncoeff_total = num_pairs * nint;
  const bool use_shared = (ncoeff_total <= UF3_2B_SHARED_COEFFS);
  if (use_shared) {
    for (int idx = threadIdx.x; idx < ncoeff_total; idx += blockDim.x) {
      s_coeff[idx] = __ldg(&d_coeff[idx]);
    }
    __syncthreads();
  }
  const float4* coeff_tab = use_shared ? s_coeff : d_coeff;

  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;

  const int NN = g_NN[n1];
  const int type1 = g_type[n1];
  const float4 pos1 = g_pos[n1];
  const float x1 = pos1.x, y1 = pos1.y, z1 = pos1.z;
  const float rc2 = rc * rc;
  const int nint_minus_1 = nint - 1;

  float pe = 0.0f;
  if (g_e0) pe += g_e0[type1];                     // 1-body energy per atom
  float fx = 0.0f, fy = 0.0f, fz = 0.0f;
  float sxx = 0, sxy = 0, sxz = 0, syx = 0, syy = 0, syz = 0, szx = 0, szy = 0, szz = 0;

  for (int i1 = 0; i1 < NN; ++i1) {
    const int n2 = g_NL[n1 + N * i1];
    const float4 pos2 = g_pos[n2];
    float x12 = pos2.x - x1;
    float y12 = pos2.y - y1;
    float z12 = pos2.z - z1;
    apply_mic(box, x12, y12, z12);
    const float d2 = x12 * x12 + y12 * y12 + z12 * z12;
    if (d2 >= rc2) continue;

    const float rinv = rsqrtf(d2);
    const float r = d2 * rinv;

    int m;
    float u, inv_h;
    if (UNIFORM) {
      const float t = (r - knot_min) * inv_knot_delta;
      m = (int)t;
      if (m < 0) m = 0; else if (m > nint_minus_1) m = nint_minus_1;
      u = t - (float)m;
      inv_h = inv_knot_delta;
    } else {
      m = find_interval_nu(r, d_knots, nknots);    // largest m with knots[m] <= r
      const float lo = d_knots[m];
      inv_h = 1.0f / (d_knots[m + 1] - lo);
      u = (r - lo) * inv_h;
    }

    // Select the coefficient table for this ordered type pair (matches the
    // trainer's pair index = type1*num_types + type2).
    const int pair = type1 * num_types + g_type[n2];
    const float4 c = coeff_tab[pair * nint + m];
    const float val = eval_cubic(c, u);
    const float deriv = eval_cubic_deriv(c, u) * inv_h;

    // Force factor matches the trainer convention: each atom's force from a
    // pair is the full -dE/dr_i derivative — no 0.5 factor (the symmetric
    // neighbor list stores each pair once per center atom, not once total).
    const float fpair = deriv * rinv;
    const float f12x = fpair * x12;
    const float f12y = fpair * y12;
    const float f12z = fpair * z12;

    fx += f12x; fy += f12y; fz += f12z;
    pe += 0.5f * val;

    sxx -= f12x * x12; sxy -= f12x * y12; sxz -= f12x * z12;
    syx -= f12y * x12; syy -= f12y * y12; syz -= f12y * z12;
    szx -= f12z * x12; szy -= f12z * y12; szz -= f12z * z12;
  }

  g_pe[n1] += (double)pe;
  g_fx[n1] += (double)fx; g_fy[n1] += (double)fy; g_fz[n1] += (double)fz;
  g_virial[n1 + 0 * N] += (double)sxx;
  g_virial[n1 + 1 * N] += (double)syy;
  g_virial[n1 + 2 * N] += (double)szz;
  g_virial[n1 + 3 * N] += (double)sxy;
  g_virial[n1 + 4 * N] += (double)sxz;
  g_virial[n1 + 5 * N] += (double)syz;
  g_virial[n1 + 6 * N] += (double)syx;
  g_virial[n1 + 7 * N] += (double)szx;
  g_virial[n1 + 8 * N] += (double)szy;
}

// ---------------------------------------------------------------------------
// 3-body kernel v3 — optimized (Tersoff-style partial force accumulation)
//
// Key improvements over v1:
//   1. float4 packed positions — 1×128-bit read vs 3×64-bit double per atom.
//   2. rsqrtf for all three distances (saves one division each).
//   3. ij-basis hoisted out of inner k-loop — b_ij/db_ij depend only on r12
//      (fixed in the j-loop), saving 8 FMAs + 12 float4 L2 reads per inner iter.
//   4. Direct B-spline evaluation — uniform basis functions are identical for
//      every interval; no GPU table lookups needed.
//   5. Middle-loop product precompute (bpbq, dbpbq, bpdbq) + inner row-sum
//      (Rv = Σ C·b_jk, Rd23 = Σ C·db_jk) reduces inner-loop muls by ~50%.
//
// v3 force-accumulation rewrite — the dominant cost in 3B-heavy models was
// global atomic traffic: v2 issued 9 atomicAdds per triplet (~9·NN²/2 per
// centre), and the centre-atom (n1) legs all hit the *same* address from the
// *same* thread → fully serialized.  v3 keeps forces in registers as far as
// each leg's atom is constant:
//   • f1 (centre n1) — constant over the whole j,k loop → 1 atomicAdd/atom.
//   • f2 (neighbour n2) — constant over the inner k-loop → 1 atomicAdd/j.
//   • f3 (neighbour n3) — varies every inner iter → atomicAdd/triplet (but to
//     scattered addresses, so far less contended than the old n1 legs).
// This cuts atomic count ~3× and removes the worst serialization entirely.
//
// v3 tensor layout — d_tensor is stored jk-fastest (idx = r + (p + q·nc_ij)·nc_jk)
// so the innermost dr-loop reads 4 *contiguous* coefficients (one cache line)
// instead of 4 reads strided by nc_ij·nc_ik (4 separate lines).
// ---------------------------------------------------------------------------
static __global__ void find_force_uf3_3b(
  const int N, const int N1, const int N2,
  const Box box,
  const float* __restrict__ d_tensor,               // [num_trips * tensor_stride]
  int num_types, int tensor_stride,
  int nc_ij, int nc_ik, int nc_jk,
  int nint_ij, int nint_ik, int nint_jk,
  float knot_min_ij, float knot_delta_ij, float inv_knot_delta_ij,
  float knot_min_ik, float knot_delta_ik, float inv_knot_delta_ik,
  float knot_min_jk, float knot_delta_jk, float inv_knot_delta_jk,
  float rc_ij, float rc_ik, float rc_jk,
  const int* __restrict__ g_NN,
  const int* __restrict__ g_NL,
  const int* __restrict__ g_type,                   // atom types [N]
  const float* __restrict__ g_e0,                   // 1-body offsets
  const float4* __restrict__ g_pos,
  double* g_pe,                                     // per-atom energy
  double* g_fx, double* g_fy, double* g_fz)         // per-atom forces (atomicAdd)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;

  const int NN = g_NN[n1];
  const int type1 = g_type[n1];
  const float4 pos1 = g_pos[n1];
  const float x1 = pos1.x, y1 = pos1.y, z1 = pos1.z;
  float pe = 0.0f;
  if (g_e0) pe += g_e0[type1];                     // 1-body energy per atom

  // Centre-atom (n1) force accumulates in registers across the whole j,k
  // double loop — its address is constant, so a single atomicAdd at the end
  // replaces the ~NN²/2 same-address (serialized) atomics of the v2 kernel.
  float f1x = 0.0f, f1y = 0.0f, f1z = 0.0f;

  for (int j1 = 0; j1 < NN; ++j1) {
    const int n2 = g_NL[n1 + N * j1];
    const int type2 = g_type[n2];
    const float4 pos2 = __ldg(&g_pos[n2]);
    float x12 = pos2.x - x1, y12 = pos2.y - y1, z12 = pos2.z - z1;
    apply_mic(box, x12, y12, z12);
    const float d12sq = x12*x12 + y12*y12 + z12*z12;
    const float inv_r12 = rsqrtf(d12sq);
    const float r12 = d12sq * inv_r12;
    if (r12 >= rc_ij) continue;

    int mi = (int)((r12 - knot_min_ij) * inv_knot_delta_ij);
    if (mi < 0) mi = 0; else if (mi >= nint_ij) mi = nint_ij - 1;
    const float ui = (r12 - knot_min_ij - mi * knot_delta_ij) * inv_knot_delta_ij;
    float b_ij[4], db_ij[4];
    eval_bspline4(ui, b_ij);
    eval_bspline4_deriv(ui, inv_knot_delta_ij, db_ij);
    const int p0 = (mi >= 3) ? mi - 3 : 0;

    // Neighbour n2 force accumulates over the inner k-loop (its address is
    // constant for fixed j) — 1 atomicAdd per j instead of per triplet.
    float f2x = 0.0f, f2y = 0.0f, f2z = 0.0f;

    for (int k1 = j1 + 1; k1 < NN; ++k1) {
      const int n3 = g_NL[n1 + N * k1];
      const int type3 = g_type[n3];
      // Coefficient tensor for this type triplet (matches the trainer's index
      // (type1*num_types + type2)*num_types + type3).  The j/k loop visits each
      // unordered neighbour pair once with n2 on the ij leg and n3 on the ik leg.
      const float* __restrict__ tensor_t =
        d_tensor + (size_t)((type1 * num_types + type2) * num_types + type3) * tensor_stride;
      const float4 pos3 = __ldg(&g_pos[n3]);
      float x13 = pos3.x - x1, y13 = pos3.y - y1, z13 = pos3.z - z1;
      apply_mic(box, x13, y13, z13);
      const float d13sq = x13*x13 + y13*y13 + z13*z13;
      const float inv_r13 = rsqrtf(d13sq);
      const float r13 = d13sq * inv_r13;
      if (r13 >= rc_ik) continue;

      float x23 = pos3.x - pos2.x, y23 = pos3.y - pos2.y, z23 = pos3.z - pos2.z;
      apply_mic(box, x23, y23, z23);
      const float d23sq = x23*x23 + y23*y23 + z23*z23;
      const float inv_r23 = rsqrtf(d23sq);
      const float r23 = d23sq * inv_r23;
      if (r23 >= rc_jk) continue;

      int mk = (int)((r13 - knot_min_ik) * inv_knot_delta_ik);
      if (mk < 0) mk = 0; else if (mk >= nint_ik) mk = nint_ik - 1;
      const float uk = (r13 - knot_min_ik - mk * knot_delta_ik) * inv_knot_delta_ik;
      float b_ik[4], db_ik[4];
      eval_bspline4(uk, b_ik);
      eval_bspline4_deriv(uk, inv_knot_delta_ik, db_ik);
      const int q0 = (mk >= 3) ? mk - 3 : 0;

      int mj = (int)((r23 - knot_min_jk) * inv_knot_delta_jk);
      if (mj < 0) mj = 0; else if (mj >= nint_jk) mj = nint_jk - 1;
      const float uj = (r23 - knot_min_jk - mj * knot_delta_jk) * inv_knot_delta_jk;
      float b_jk[4], db_jk[4];
      eval_bspline4(uj, b_jk);
      eval_bspline4_deriv(uj, inv_knot_delta_jk, db_jk);
      const int r0 = (mj >= 3) ? mj - 3 : 0;

      float val = 0, dv12 = 0, dv13 = 0, dv23 = 0;
      #pragma unroll 4
      for (int dp = 0; dp < 4; dp++) {
        const int p = p0 + dp; if (p >= nc_ij) break;
        const float bp = b_ij[dp], dbp = db_ij[dp];
        #pragma unroll 4
        for (int dq = 0; dq < 4; dq++) {
          const int q = q0 + dq; if (q >= nc_ik) break;
          const float bpbq = bp * b_ik[dq], dbpbq = dbp * b_ik[dq], bpdbq = bp * db_ik[dq];
          // jk-fastest layout: 4 inner coeffs are contiguous (one cache line).
          const float* __restrict__ C4 = &tensor_t[(p + q * nc_ij) * nc_jk + r0];
          float Rv = 0, Rd23 = 0;
          #pragma unroll 4
          for (int dr = 0; dr < 4; dr++) {
            if (r0 + dr >= nc_jk) break;
            const float C = __ldg(&C4[dr]);
            Rv += C * b_jk[dr]; Rd23 += C * db_jk[dr];
          }
          val   += bpbq  * Rv;
          dv12  += dbpbq * Rv;
          dv13  += bpdbq * Rv;
          dv23  += bpbq  * Rd23;
        }
      }

      pe += val;   // per-centre (no /3 — matches trainer convention)

      // Force legs (sign convention FD-verified against trainer):
      //   f1 = +a12 +a13      f2 = -a12 +a23      f3 = -a13 -a23
      // where a12 = dv12·inv_r12·r12vec, etc.  Identical algebra to v2, only
      // the accumulation target (register vs atomic) differs per leg.
      const float t12 = dv12 * inv_r12, t13 = dv13 * inv_r13, t23 = dv23 * inv_r23;
      const float a12x = t12 * x12, a12y = t12 * y12, a12z = t12 * z12;
      const float a13x = t13 * x13, a13y = t13 * y13, a13z = t13 * z13;
      const float a23x = t23 * x23, a23y = t23 * y23, a23z = t23 * z23;

      f1x += a12x + a13x; f1y += a12y + a13y; f1z += a12z + a13z;
      f2x += a23x - a12x; f2y += a23y - a12y; f2z += a23z - a12z;

      atomicAdd(&g_fx[n3], (double)(-a13x - a23x));
      atomicAdd(&g_fy[n3], (double)(-a13y - a23y));
      atomicAdd(&g_fz[n3], (double)(-a13z - a23z));
    }
    atomicAdd(&g_fx[n2], (double)f2x);
    atomicAdd(&g_fy[n2], (double)f2y);
    atomicAdd(&g_fz[n2], (double)f2z);
  }
  atomicAdd(&g_fx[n1], (double)f1x);
  atomicAdd(&g_fy[n1], (double)f1y);
  atomicAdd(&g_fz[n1], (double)f1z);
  g_pe[n1] += (double)pe;
}

// ---------------------------------------------------------------------------
// UF3 class implementation
// ---------------------------------------------------------------------------

UF3::UF3(const char* filename, const int number_of_atoms, const int max_neighbor)
{
  max_neighbor_ = max_neighbor;
  has_2b = false;
  has_3b = false;
  initialize(filename, number_of_atoms);
}

UF3::~UF3(void) {}

void UF3::initialize(const char* filename, const int number_of_atoms)
{
  // ---- Parse the .uf3 file ----
  std::ifstream input(filename);
  if (!input.is_open()) {
    std::cout << "Failed to open " << filename << std::endl;
    exit(1);
  }

  std::vector<std::string> lines;
  {
    std::string line;
    while (std::getline(input, line)) {
      lines.push_back(line);
    }
  }
  input.close();

  // Map an element symbol to its type index (position in the header element
  // list).  Returns -1 if not found.
  auto symbol_to_type = [this](const std::string& s) -> int {
    for (int t = 0; t < (int)elements_.size(); t++)
      if (elements_[t] == s) return t;
    return -1;
  };

  // Host accumulators for the full multi-type coefficient tables.  Allocated
  // lazily on the first 2B / 3B block (once nint / tensor dims are known).
  std::vector<float4> h_coeff_all;   // [num_pairs * nint]
  std::vector<float> h_tensor_all;   // [num_trips * tensor_stride]

  double rc_max = 0.0;
  size_t li = 0;
  while (li < lines.size()) {
    // Skip comments and empty lines.
    if (lines[li].empty() || lines[li][0] == '#') { li++; continue; }
    {
      std::istringstream iss_test(lines[li]);
      std::string first;
      iss_test >> first;
      if (first == "uf3") {
        // Header: "uf3 N elem1 elem2 ...".  The element order defines the atom
        // type indices used by the kernels and by model.xyz / run.in.
        int nt_header;
        iss_test >> nt_header;
        num_types_ = nt_header;
        elements_.clear();
        for (int n = 0; n < nt_header; n++) {
          std::string el;
          if (iss_test >> el) elements_.push_back(el);
        }
        if ((int)elements_.size() != nt_header) {
          std::cout << "UF3 header lists " << elements_.size()
                    << " elements but declares " << nt_header << "." << std::endl;
          exit(1);
        }
        if (!d_e0.size()) d_e0.resize(nt_header); // allocate e0 if no 1B line
        li++; continue;
      } // GPUMD header line
    }

    std::istringstream iss(lines[li]);
    std::string body_type;
    iss >> body_type;

    if (body_type == "1B") {
      int ntypes = (int)d_e0.size();
      std::vector<float> e0(ntypes);
      for (int n = 0; n < ntypes; n++) iss >> e0[n];
      d_e0.copy_from_host(e0.data());
    }
    else if (body_type == "2B") {
      // Format: 2B elem1 elem2 leading_trim trailing_trim knot_type
      std::string e1, e2;
      int leading_trim, trailing_trim;
      std::string kt_str;
      iss >> e1 >> e2 >> leading_trim >> trailing_trim >> kt_str;
      int knot_type = (kt_str == "uk") ? 1 : 0;
      (void)trailing_trim; // always 3 for cubic
      int ti = symbol_to_type(e1), tj = symbol_to_type(e2);
      if (ti < 0 || tj < 0) {
        std::cout << "UF3 2B block references unknown element (" << e1 << ","
                  << e2 << ")." << std::endl;
        exit(1);
      }

      li++;
      // cutoff and knot count
      {
        std::istringstream iss2(lines[li]);
        double rc_val; int nk;
        iss2 >> rc_val >> nk;
        two_body.rc = rc_val;
        two_body.nknots = nk;
        two_body.nint = nk - 1;
        two_body.knot_type = knot_type;
        if (rc_val > rc_max) rc_max = rc_val;
      }

      li++;
      // knot vector
      std::vector<float> knots(two_body.nknots);
      {
        std::istringstream iss2(lines[li]);
        for (int i = 0; i < two_body.nknots; i++) iss2 >> knots[i];
      }

      li++;
      // coefficient count
      int ncoeff;
      {
        std::istringstream iss2(lines[li]);
        iss2 >> ncoeff;
      }

      li++;
      // coefficient vector
      std::vector<float> coeffs(ncoeff);
      {
        std::istringstream iss2(lines[li]);
        for (int i = 0; i < ncoeff; i++) iss2 >> coeffs[i];
      }

      // Pre-compute per-interval combined cubic polynomials for this pair.
      // knot_type 1 ("uk") -> uniform basis (GPUMD trainer default, hot path);
      // knot_type 0 ("nk") -> general non-uniform basis via de Boor sampling
      // (lets the MD engine consume reference-UF3 models with lammps knots).
      std::vector<float4> h_coeff;
      if (knot_type == 1)
        precompute_2b_uniform(knots, coeffs, leading_trim, h_coeff);
      else
        precompute_2b_nonuniform(knots, coeffs, h_coeff);

      // Lazily size the full per-pair table on the first 2B block.
      two_body.num_pairs = num_types_ * num_types_;
      if (h_coeff_all.empty()) {
        h_coeff_all.assign((size_t)two_body.num_pairs * two_body.nint,
                           make_float4(0.f, 0.f, 0.f, 0.f));
      }
      int pair = ti * num_types_ + tj;
      for (int m = 0; m < two_body.nint; m++)
        h_coeff_all[(size_t)pair * two_body.nint + m] = h_coeff[m];

      // Store knot info for GPU (shared uniform grid across all pairs).
      two_body.knot_min = knots[0];
      two_body.knot_delta = (knots[two_body.nknots - 1] - knots[0]) / (two_body.nknots - 1);
      two_body.inv_knot_delta = 1.0f / two_body.knot_delta;
      {
        two_body.d_knots.resize(two_body.nknots);
        two_body.d_knots.copy_from_host(knots.data());
      }

      has_2b = true;
    }
    else if (body_type == "3B") {
      // Format: 3B elem1 elem2 elem3 leading_trim trailing_trim knot_type
      std::string e1, e2, e3;
      int leading_trim, trailing_trim;
      std::string kt_str;
      iss >> e1 >> e2 >> e3 >> leading_trim >> trailing_trim >> kt_str;
      int knot_type = (kt_str == "uk") ? 1 : 0;
      (void)trailing_trim;
      int ti = symbol_to_type(e1), tj = symbol_to_type(e2), tk = symbol_to_type(e3);
      if (ti < 0 || tj < 0 || tk < 0) {
        std::cout << "UF3 3B block references unknown element (" << e1 << ","
                  << e2 << "," << e3 << ")." << std::endl;
        exit(1);
      }
      // The 3B kernel evaluates the uniform cubic B-spline basis directly
      // (eval_bspline4), which is only valid for uniform knots.  Non-uniform 3B
      // would need per-interval basis tables — refuse rather than silently
      // produce wrong energies/forces.
      if (knot_type != 1) {
        std::cout << "UF3: non-uniform (nk) 3B knots are not supported by the MD "
                     "engine (uniform basis only). Re-export with uniform knots."
                  << std::endl;
        exit(1);
      }

      li++;
      // cutoffs and knot counts, forward order: rc_ij rc_ik rc_jk nk_ij nk_ik nk_jk
      {
        std::istringstream iss2(lines[li]);
        iss2 >> three_body.rc_ij >> three_body.rc_ik >> three_body.rc_jk
             >> three_body.nk_ij >> three_body.nk_ik >> three_body.nk_jk;
        three_body.nint_ij = three_body.nk_ij - 1;
        three_body.nint_ik = three_body.nk_ik - 1;
        three_body.nint_jk = three_body.nk_jk - 1;
        three_body.knot_type = knot_type;
        double rc3 = three_body.rc_ij;
        if (three_body.rc_ik > rc3) rc3 = three_body.rc_ik;
        if (rc3 > rc_max) rc_max = rc3;
      }

      // Read 3 knot vectors, forward order: ij, ik, jk
      std::vector<float> k_ij(three_body.nk_ij);
      std::vector<float> k_ik(three_body.nk_ik);
      std::vector<float> k_jk(three_body.nk_jk);

      li++; { std::istringstream iss2(lines[li]);
        for (int i = 0; i < three_body.nk_ij; i++) iss2 >> k_ij[i]; }
      li++; { std::istringstream iss2(lines[li]);
        for (int i = 0; i < three_body.nk_ik; i++) iss2 >> k_ik[i]; }
      li++; { std::istringstream iss2(lines[li]);
        for (int i = 0; i < three_body.nk_jk; i++) iss2 >> k_jk[i]; }

      li++;
      // coefficient dimensions: nc_ij nc_ik nc_jk
      {
        std::istringstream iss2(lines[li]);
        iss2 >> three_body.nc_ij >> three_body.nc_ik >> three_body.nc_jk;
      }

      // Read coefficient tensor.  File rows are (ij outer, ik inner) with jk
      // along each row; store in the kernel's jk-fastest layout
      // idx = r + (p + q*nc_ij)*nc_jk  (jk fastest, then ij, then ik) so the
      // innermost dr-loop in find_force_uf3_3b reads 4 contiguous coefficients.
      int nci = three_body.nc_ij, nck = three_body.nc_ik, ncj = three_body.nc_jk;
      three_body.tensor_stride = nci * nck * ncj;
      three_body.num_trips = num_types_ * num_types_ * num_types_;
      // Lazily size the full per-triplet tensor on the first 3B block.
      if (h_tensor_all.empty()) {
        h_tensor_all.assign((size_t)three_body.num_trips * three_body.tensor_stride, 0.0f);
      }
      int trip = (ti * num_types_ + tj) * num_types_ + tk;
      float* tdst = h_tensor_all.data() + (size_t)trip * three_body.tensor_stride;
      for (int row = 0; row < nci * nck; row++) {
        int p = row / nck;   // ij index
        int q = row % nck;   // ik index
        li++;
        std::istringstream iss2(lines[li]);
        for (int c = 0; c < ncj; c++)            // c == jk index (r)
          iss2 >> tdst[(size_t)c + ((size_t)p + (size_t)q * nci) * ncj];
      }

      // Store knot info
      three_body.knot_min_ij = k_ij[0];
      three_body.knot_min_ik = k_ik[0];
      three_body.knot_min_jk = k_jk[0];
      three_body.knot_delta_ij = (k_ij[three_body.nk_ij - 1] - k_ij[0]) / (three_body.nk_ij - 1);
      three_body.knot_delta_ik = (k_ik[three_body.nk_ik - 1] - k_ik[0]) / (three_body.nk_ik - 1);
      three_body.knot_delta_jk = (k_jk[three_body.nk_jk - 1] - k_jk[0]) / (three_body.nk_jk - 1);
      three_body.inv_knot_delta_ij = 1.0f / three_body.knot_delta_ij;
      three_body.inv_knot_delta_ik = 1.0f / three_body.knot_delta_ik;
      three_body.inv_knot_delta_jk = 1.0f / three_body.knot_delta_jk;

      has_3b = true;
    }
    li++;
  }

  rc = rc_max;
  if (!has_2b && !has_3b) {
    std::cout << "UF3 potential file has no 2B or 3B blocks." << std::endl;
    exit(1);
  }

  // Upload the assembled multi-type tables to the device.
  if (has_2b) {
    if ((size_t)two_body.num_pairs * two_body.nint != h_coeff_all.size()) {
      std::cout << "UF3: missing 2B blocks (expected " << two_body.num_pairs
                << " type pairs)." << std::endl;
      exit(1);
    }
    two_body.d_coeff.resize(h_coeff_all.size());
    two_body.d_coeff.copy_from_host(h_coeff_all.data());
  }
  if (has_3b) {
    if ((size_t)three_body.num_trips * three_body.tensor_stride != h_tensor_all.size()) {
      std::cout << "UF3: missing 3B blocks (expected " << three_body.num_trips
                << " type triplets)." << std::endl;
      exit(1);
    }
    three_body.d_tensor.resize(h_tensor_all.size());
    three_body.d_tensor.copy_from_host(h_tensor_all.data());
  }

  // If no 1B line was present, initialize e0 to zeros.
  if (!d_e0.size()) d_e0.resize(1); // at least 1 type as fallback
  cudaMemset(d_e0.data(), 0, d_e0.size() * sizeof(float));

  // Allocate neighbor list and position buffer.
  neighbor.initialize(rc, number_of_atoms, max_neighbor_);
  d_pos_packed.resize(number_of_atoms);

  printf("Use UF3 potential with %d atom type%s.\n", num_types_,
         num_types_ > 1 ? "s" : "");
  for (int t = 0; t < (int)elements_.size(); t++)
    printf("    type %d (%s).\n", t, elements_[t].c_str());
  if (has_2b) {
    printf("    2B: rc=%.1f A, %d intervals, %d knots, %d type pairs\n",
           two_body.rc, two_body.nint, two_body.nknots, two_body.num_pairs);
  }
  if (has_3b) {
    printf("    3B: rc(ij,ik,jk)=(%.1f,%.1f,%.1f) A, coeff dims=%dx%dx%d, %d type triplets\n",
           three_body.rc_ij, three_body.rc_ik, three_body.rc_jk,
           three_body.nc_ij, three_body.nc_ik, three_body.nc_jk, three_body.num_trips);
  }
}

void UF3::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 128;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  neighbor.find_neighbor_global(rc, box, type, position_per_atom);

  // Pack positions double SoA → float4 AoS (fully coalesced, O(N))
  {
    const int grid_pack = (N - 1) / BLOCK_SIZE + 1;
    pack_positions_float4<<<grid_pack, BLOCK_SIZE>>>(
      N,
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      d_pos_packed.data());
    GPU_CHECK_KERNEL
  }

  if (has_2b) {
    auto launch_2b = [&](auto uniform_tag) {
      constexpr bool UNI = decltype(uniform_tag)::value;
      find_force_uf3_2b<UNI><<<grid_size, BLOCK_SIZE>>>(
        N, N1, N2, box,
        two_body.d_coeff.data(), two_body.num_pairs, num_types_, two_body.nint,
        two_body.knot_min, two_body.inv_knot_delta,
        two_body.d_knots.data(), two_body.nknots,
        (float)two_body.rc,
        neighbor.NN.data(), neighbor.NL.data(),
        type.data(), d_e0.data(),
        d_pos_packed.data(),
        potential_per_atom.data(),
        force_per_atom.data(),
        force_per_atom.data() + N,
        force_per_atom.data() + N * 2,
        virial_per_atom.data());
    };
    if (two_body.knot_type == 1) launch_2b(std::true_type{});
    else                         launch_2b(std::false_type{});
    GPU_CHECK_KERNEL
  }

  if (has_3b) {
    find_force_uf3_3b<<<grid_size, BLOCK_SIZE>>>(
      N, N1, N2, box,
      three_body.d_tensor.data(),
      num_types_, three_body.tensor_stride,
      three_body.nc_ij, three_body.nc_ik, three_body.nc_jk,
      three_body.nint_ij, three_body.nint_ik, three_body.nint_jk,
      three_body.knot_min_ij, three_body.knot_delta_ij, three_body.inv_knot_delta_ij,
      three_body.knot_min_ik, three_body.knot_delta_ik, three_body.inv_knot_delta_ik,
      three_body.knot_min_jk, three_body.knot_delta_jk, three_body.inv_knot_delta_jk,
      (float)three_body.rc_ij, (float)three_body.rc_ik, (float)three_body.rc_jk,
      neighbor.NN.data(), neighbor.NL.data(),
      type.data(), nullptr,        // 1-body already added by the 2B kernel (avoid double count)
      d_pos_packed.data(),
      potential_per_atom.data(),
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2);
    GPU_CHECK_KERNEL
  }
}
