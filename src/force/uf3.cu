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
    // The four control points active on interval m are coeffs[m-3 .. m], each
    // clamped to the valid range [0, nc-1].  All four upper clamps are required:
    // on the final intervals m-2 and m-1 exceed nc-1 and would otherwise read
    // out of bounds, corrupting the edge cubic (spurious force near rc).
    int i0 = m - 3, i1 = m - 2, i2 = m - 1, i3 = m;
    if (i0 < 0) i0 = 0; else if (i0 >= nc) i0 = nc - 1;
    if (i1 < 0) i1 = 0; else if (i1 >= nc) i1 = nc - 1;
    if (i2 < 0) i2 = 0; else if (i2 >= nc) i2 = nc - 1;
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
// Periodic image-shift encoding for the multi-image (ghost) neighbour list.
//
// When the simulation cell is smaller than 2*rc in a periodic direction the
// minimum-image convention is ambiguous (an atom has several images within rc).
// Instead of materialising ghost atoms we tag every neighbour entry with the
// integer lattice shift (sx,sy,sz) of the interacting image.  The force kernels
// reconstruct the image position as  pos[n2] + sx*a + sy*b + sz*c  (no MIC),
// which reproduces the trainer's ghost-supercell distances exactly and keeps the
// 3-body triangle (r12,r13,r23) self-consistent.  code 0 == no shift.
// ---------------------------------------------------------------------------
constexpr int UF3_SHIFT_BIAS = 15;
constexpr int UF3_SHIFT_PAYLOAD_MASK = 0x7FFF;
constexpr int UF3_SHIFT_FLAG = 1 << 15;

__host__ __device__ __forceinline__ int uf3_encode_shift(int sx, int sy, int sz)
{
  if (sx == 0 && sy == 0 && sz == 0) return 0;
  const int bx = sx + UF3_SHIFT_BIAS, by = sy + UF3_SHIFT_BIAS, bz = sz + UF3_SHIFT_BIAS;
  return ((bx & 0x1F) | ((by & 0x1F) << 5) | ((bz & 0x1F) << 10)) | UF3_SHIFT_FLAG;
}

__device__ __forceinline__ void uf3_decode_shift(int code, int& sx, int& sy, int& sz)
{
  if (code == 0) { sx = sy = sz = 0; return; }
  const int p = code & UF3_SHIFT_PAYLOAD_MASK;
  sx = (p & 0x1F) - UF3_SHIFT_BIAS;
  sy = ((p >> 5) & 0x1F) - UF3_SHIFT_BIAS;
  sz = ((p >> 10) & 0x1F) - UF3_SHIFT_BIAS;
}

// Image-shifted neighbour position (float4 packed positions + lattice shift).
__device__ __forceinline__ float4 uf3_image_pos(const float4 p, const Box& box, int code)
{
  if (code == 0) return p;
  int sx, sy, sz;
  uf3_decode_shift(code, sx, sy, sz);
  // box.cpu_h rows are (a_x b_x c_x | a_y b_y c_y | a_z b_z c_z): column j = vector j.
  float ox = sx * (float)box.cpu_h[0] + sy * (float)box.cpu_h[1] + sz * (float)box.cpu_h[2];
  float oy = sx * (float)box.cpu_h[3] + sy * (float)box.cpu_h[4] + sz * (float)box.cpu_h[5];
  float oz = sx * (float)box.cpu_h[6] + sy * (float)box.cpu_h[7] + sz * (float)box.cpu_h[8];
  return make_float4(p.x + ox, p.y + oy, p.z + oz, 0.0f);
}

// Absolute image position of a neighbour.  With an explicit shift table the
// image is pos2 + shift; otherwise fall back to minimum image about pos1.  Both
// return an *absolute* position so the 3-body triangle stays self-consistent.
__device__ __forceinline__ float4 neighbor_image(
  float4 pos2, const float4 pos1, const Box& box, const int* g_shift, int idx)
{
  if (g_shift) return uf3_image_pos(pos2, box, g_shift[idx]);
  float dx = pos2.x - pos1.x, dy = pos2.y - pos1.y, dz = pos2.z - pos1.z;
  apply_mic(box, dx, dy, dz);
  return make_float4(pos1.x + dx, pos1.y + dy, pos1.z + dz, 0.0f);
}

// ---------------------------------------------------------------------------
// Multi-image O(N^2) neighbour build (host-driven, used for small boxes).
// For each real atom n1, enumerate every periodic image of every atom within rc
// and store (neighbour index, encoded shift).  Produces the same neighbour set
// as the trainer's generate_ghosts (cutoff = global rc), but as index+shift
// pairs.  Each physical image is one entry, so there are no degenerate (r=0)
// duplicates and Newton's third law holds exactly.
// ---------------------------------------------------------------------------
static __global__ void build_neighbor_uf3_multiimage(
  const Box box, const int N, const float rc,
  const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
  int* __restrict__ NN, int* __restrict__ NL, int* __restrict__ NL_shift,
  const int max_neighbors)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) return;

  const double x1 = x[n1], y1 = y[n1], z1 = z[n1];
  const double rc2 = (double)rc * rc;

  // Image range per lattice direction: ceil(rc / perpendicular_spacing).
  const double ax = box.cpu_h[0], ay = box.cpu_h[3], az = box.cpu_h[6];
  const double bx = box.cpu_h[1], by = box.cpu_h[4], bz = box.cpu_h[7];
  const double cx = box.cpu_h[2], cy = box.cpu_h[5], cz = box.cpu_h[8];
  auto cross = [](double ux, double uy, double uz, double vx, double vy, double vz,
                  double& rx, double& ry, double& rz) {
    rx = uy * vz - uz * vy; ry = uz * vx - ux * vz; rz = ux * vy - uy * vx;
  };
  double nx, ny, nz; cross(bx, by, bz, cx, cy, cz, nx, ny, nz);
  double vol = fabs(ax * nx + ay * ny + az * nz);
  auto perp = [&](double ux, double uy, double uz, double vx, double vy, double vz) {
    double rx, ry, rz; cross(ux, uy, uz, vx, vy, vz, rx, ry, rz);
    double area = sqrt(rx * rx + ry * ry + rz * rz);
    return vol / (area > 1e-6 ? area : 1e-6);
  };
  int P1 = box.pbc_x ? (int)ceil(rc / perp(bx, by, bz, cx, cy, cz)) : 0;
  int P2 = box.pbc_y ? (int)ceil(rc / perp(ax, ay, az, cx, cy, cz)) : 0;
  int P3 = box.pbc_z ? (int)ceil(rc / perp(ax, ay, az, bx, by, bz)) : 0;

  int count = 0;
  for (int n2 = 0; n2 < N; ++n2) {
    const double bx2 = x[n2], by2 = y[n2], bz2 = z[n2];
    for (int i1 = -P1; i1 <= P1; ++i1) {
      for (int i2 = -P2; i2 <= P2; ++i2) {
        for (int i3 = -P3; i3 <= P3; ++i3) {
          if (n2 == n1 && i1 == 0 && i2 == 0 && i3 == 0) continue;
          double sx = i1 * ax + i2 * bx + i3 * cx;
          double sy = i1 * ay + i2 * by + i3 * cy;
          double sz = i1 * az + i2 * bz + i3 * cz;
          double dx = (bx2 + sx) - x1, dy = (by2 + sy) - y1, dz = (bz2 + sz) - z1;
          double d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < rc2 && d2 > 1e-12 && count < max_neighbors) {
            NL[count * N + n1] = n2;
            NL_shift[count * N + n1] = uf3_encode_shift(i1, i2, i3);
            ++count;
          }
        }
      }
    }
  }
  NN[n1] = count;
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
  const int* __restrict__ g_shift,                  // per-neighbour image shift (nullptr -> MIC)
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
    const int idx = n1 + N * i1;
    const int n2 = g_NL[idx];
    float4 pos2 = g_pos[n2];
    float x12, y12, z12;
    if (g_shift) {
      pos2 = uf3_image_pos(pos2, box, g_shift[idx]);  // explicit periodic image
      x12 = pos2.x - x1; y12 = pos2.y - y1; z12 = pos2.z - z1;
    } else {
      x12 = pos2.x - x1; y12 = pos2.y - y1; z12 = pos2.z - z1;
      apply_mic(box, x12, y12, z12);
    }
    const float d2 = x12 * x12 + y12 * y12 + z12 * z12;
    if (d2 >= rc2 || d2 < 1e-12f) continue;   // d2~0: overlapped atoms -> inf/NaN

    const float rinv = rsqrtf(d2);
    const float r = d2 * rinv;

    // Below the knot grid the cubic diverges in an arbitrary (possibly
    // attractive) direction; extrapolate linearly from the first knot instead
    // so a hard close approach always sees a continuous, finite potential.
    int m;
    float u, inv_h, ext = 0.0f;   // ext: (r - knots[0]) in u units, <= 0
    if (UNIFORM) {
      float t = (r - knot_min) * inv_knot_delta;
      if (t < 0.0f) { ext = t; t = 0.0f; }
      m = (int)t;
      if (m > nint_minus_1) m = nint_minus_1;
      u = t - (float)m;
      inv_h = inv_knot_delta;
    } else {
      m = find_interval_nu(r, d_knots, nknots);    // largest m with knots[m] <= r
      const float lo = d_knots[m];
      inv_h = 1.0f / (d_knots[m + 1] - lo);
      u = (r - lo) * inv_h;
      if (u < 0.0f) { ext = u; u = 0.0f; }         // r < knots[0] (m clamps to 0)
    }

    // Select the coefficient table for this ordered type pair (matches the
    // trainer's pair index = type1*num_types + type2).
    const int pair = type1 * num_types + g_type[n2];
    const float4 c = coeff_tab[pair * nint + m];
    const float deriv_u = eval_cubic_deriv(c, u);
    const float val = eval_cubic(c, u) + deriv_u * ext;
    const float deriv = deriv_u * inv_h;

    // Force factor matches the trainer convention: each atom's force from a
    // pair is the full -dE/dr_i derivative — no 0.5 factor (the symmetric
    // neighbor list stores each pair once per center atom, not once total).
    const float fpair = deriv * rinv;
    const float f12x = fpair * x12;
    const float f12y = fpair * y12;
    const float f12z = fpair * z12;

    fx += f12x; fy += f12y; fz += f12z;
    pe += 0.5f * val;

    // Per-atom virial.  The symmetric neighbor list visits each physical pair
    // twice (once from each end), so each visit carries half the pair virial —
    // same convention as lj.cu / nep.cu.  Without the 0.5 the total virial (and
    // pressure) is exactly doubled.
    const float hx = 0.5f * f12x, hy = 0.5f * f12y, hz = 0.5f * f12z;
    sxx -= hx * x12; sxy -= hx * y12; sxz -= hx * z12;
    syx -= hy * x12; syy -= hy * y12; syz -= hy * z12;
    szx -= hz * x12; szy -= hz * y12; szz -= hz * z12;
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
// Evaluate one ordered 3-body triplet (centre p1; pa on the ij leg, pb on the
// ik leg; pa/pb are absolute image positions) and accumulate weighted energy
// and the three force legs.  Returns silently if any leg is outside its cutoff.
//   f1 += a12 + a13      fa += a23 - a12      fb += -a13 - a23
// (sign convention FD-verified against the trainer).  `w` scales every output
// so the caller can average the two neighbour orderings (w = 0.5 each).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void uf3_eval_triplet(
  const float4 p1, const float4 pa, const float4 pb,
  const float* __restrict__ tensor_t,
  int nc_ij, int nc_ik, int nc_jk,
  int nint_ij, int nint_ik, int nint_jk,
  float kmin_ij, float kd_ij, float ikd_ij,
  float kmin_ik, float kd_ik, float ikd_ik,
  float kmin_jk, float kd_jk, float ikd_jk,
  float rc_ij, float rc_ik, float rc_jk,
  float w, float& pe, float3& f1, float3& fa, float3& fb, float vir[6])
{
  float xij = pa.x - p1.x, yij = pa.y - p1.y, zij = pa.z - p1.z;
  float dij2 = xij*xij + yij*yij + zij*zij;
  if (dij2 >= rc_ij*rc_ij || dij2 < 1e-12f) return;
  float xik = pb.x - p1.x, yik = pb.y - p1.y, zik = pb.z - p1.z;
  float dik2 = xik*xik + yik*yik + zik*zik;
  if (dik2 >= rc_ik*rc_ik || dik2 < 1e-12f) return;
  float xjk = pb.x - pa.x, yjk = pb.y - pa.y, zjk = pb.z - pa.z;
  float djk2 = xjk*xjk + yjk*yjk + zjk*zjk;
  if (djk2 >= rc_jk*rc_jk || djk2 < 1e-12f) return;

  const float inv_ij = rsqrtf(dij2), r_ij = dij2 * inv_ij;
  const float inv_ik = rsqrtf(dik2), r_ik = dik2 * inv_ik;
  const float inv_jk = rsqrtf(djk2), r_jk = djk2 * inv_jk;

  int mi = (int)((r_ij - kmin_ij) * ikd_ij);
  if (mi < 0) mi = 0; else if (mi >= nint_ij) mi = nint_ij - 1;
  const float ui = (r_ij - kmin_ij - mi * kd_ij) * ikd_ij;
  float b_ij[4], db_ij[4]; eval_bspline4(ui, b_ij); eval_bspline4_deriv(ui, ikd_ij, db_ij);
  const int p0 = (mi >= 3) ? mi - 3 : 0;

  int mk = (int)((r_ik - kmin_ik) * ikd_ik);
  if (mk < 0) mk = 0; else if (mk >= nint_ik) mk = nint_ik - 1;
  const float uk = (r_ik - kmin_ik - mk * kd_ik) * ikd_ik;
  float b_ik[4], db_ik[4]; eval_bspline4(uk, b_ik); eval_bspline4_deriv(uk, ikd_ik, db_ik);
  const int q0 = (mk >= 3) ? mk - 3 : 0;

  int mj = (int)((r_jk - kmin_jk) * ikd_jk);
  if (mj < 0) mj = 0; else if (mj >= nint_jk) mj = nint_jk - 1;
  const float uj = (r_jk - kmin_jk - mj * kd_jk) * ikd_jk;
  float b_jk[4], db_jk[4]; eval_bspline4(uj, b_jk); eval_bspline4_deriv(uj, ikd_jk, db_jk);
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
      // Trainer layout: idx = p + q*nc_ij + r*nc_ij*nc_ik (matches write_uf3_file / CPU ref).
      const float* __restrict__ Crow = &tensor_t[p + q * nc_ij + r0 * nc_ij * nc_ik];
      float Rv = 0, Rd23 = 0;
      #pragma unroll 4
      for (int dr = 0; dr < 4; dr++) {
        if (r0 + dr >= nc_jk) break;
        const float C = __ldg(&Crow[dr * nc_ij * nc_ik]);
        Rv += C * b_jk[dr]; Rd23 += C * db_jk[dr];
      }
      val  += bpbq  * Rv;
      dv12 += dbpbq * Rv;
      dv13 += bpdbq * Rv;
      dv23 += bpbq  * Rd23;
    }
  }

  pe += w * val;
  const float t12 = w * dv12 * inv_ij, t13 = w * dv13 * inv_ik, t23 = w * dv23 * inv_jk;
  const float a12x = t12*xij, a12y = t12*yij, a12z = t12*zij;
  const float a13x = t13*xik, a13y = t13*yik, a13z = t13*zik;
  const float a23x = t23*xjk, a23y = t23*yjk, a23z = t23*zjk;
  f1.x += a12x + a13x;   f1.y += a12y + a13y;   f1.z += a12z + a13z;
  fa.x += a23x - a12x;   fa.y += a23y - a12y;   fa.z += a23z - a12z;
  fb.x += -a13x - a23x;  fb.y += -a13y - a23y;  fb.z += -a13z - a23z;

  // Triplet virial W = -Σ_edges t_e (r_e ⊗ r_e); each edge term is symmetric
  // (a ∥ r), so 6 components suffice.  The a-vectors already carry w.
  vir[0] -= a12x * xij + a13x * xik + a23x * xjk;   // xx
  vir[1] -= a12y * yij + a13y * yik + a23y * yjk;   // yy
  vir[2] -= a12z * zij + a13z * zik + a23z * zjk;   // zz
  vir[3] -= a12x * yij + a13x * yik + a23x * yjk;   // xy
  vir[4] -= a12x * zij + a13x * zik + a23x * zjk;   // xz
  vir[5] -= a12y * zij + a13y * zik + a23y * zjk;   // yz
}

// ---------------------------------------------------------------------------
// 3-body dual-order kernel (legacy/asymmetric models, thread-per-atom).
//
// Periodic images come from g_shift (explicit lattice shift; nullptr -> minimum
// image), so the (r12,r13,r23) triangle is self-consistent even when the cell is
// smaller than 2*rc and a neighbour pair has several images within the cutoff —
// reproducing the trainer's ghost-supercell distances exactly.
//
// When the ij and ik legs use different knot grids / cutoffs the triplet is NOT
// symmetric under j<->k, so the single-ordering result depends on which
// neighbour the (index-sorted) loop places on each leg.  Averaging both
// assignments — 0.5*[V(n2 on ij, n3 on ik) + V(n3 on ij, n2 on ik)] — restores
// order independence.  Models whose grids match (the common case; tensor
// symmetrised on load) take the warp-parallel kernel below instead.
// ---------------------------------------------------------------------------
static __global__ void find_force_uf3_3b_dual(
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
  const int* __restrict__ g_shift,                  // per-neighbour image shift (nullptr -> MIC)
  const int* __restrict__ g_type,                   // atom types [N]
  const float* __restrict__ g_e0,                   // 1-body offsets
  const float4* __restrict__ g_pos,
  double* g_pe,                                     // per-atom energy
  double* g_fx, double* g_fy, double* g_fz,         // per-atom forces (atomicAdd)
  double* g_virial)                                 // per-atom virial [9*N]
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;

  const int NN = g_NN[n1];
  const int type1 = g_type[n1];
  const float4 pos1 = g_pos[n1];
  float pe = 0.0f;
  if (g_e0) pe += g_e0[type1];                     // 1-body energy per atom

  float3 f1 = make_float3(0.0f, 0.0f, 0.0f);       // centre force, 1 atomicAdd/atom
  // Whole-triplet virial assigned to the centre atom (total stress exact; the
  // per-atom split is the per-centre decomposition, like the trainer's energy).
  float vir[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int j1 = 0; j1 < NN; ++j1) {
    const int idx2 = n1 + N * j1;
    const int n2 = g_NL[idx2];
    const int type2 = g_type[n2];
    const float4 pos2 = neighbor_image(__ldg(&g_pos[n2]), pos1, box, g_shift, idx2);
    float3 f2 = make_float3(0.0f, 0.0f, 0.0f);      // n2 force, 1 atomicAdd/j

    for (int k1 = j1 + 1; k1 < NN; ++k1) {
      const int idx3 = n1 + N * k1;
      const int n3 = g_NL[idx3];
      const int type3 = g_type[n3];
      const float4 pos3 = neighbor_image(__ldg(&g_pos[n3]), pos1, box, g_shift, idx3);
      float3 f3 = make_float3(0.0f, 0.0f, 0.0f);

      // Assignment A: n2 on the ij leg, n3 on the ik leg → tensor[t1][t2][t3].
      const float* __restrict__ tA =
        d_tensor + (size_t)((type1 * num_types + type2) * num_types + type3) * tensor_stride;
      uf3_eval_triplet(
        pos1, pos2, pos3, tA, nc_ij, nc_ik, nc_jk, nint_ij, nint_ik, nint_jk,
        knot_min_ij, knot_delta_ij, inv_knot_delta_ij,
        knot_min_ik, knot_delta_ik, inv_knot_delta_ik,
        knot_min_jk, knot_delta_jk, inv_knot_delta_jk,
        rc_ij, rc_ik, rc_jk, 0.5f, pe, f1, f2, f3, vir);

      // Assignment B: n3 on the ij leg, n2 on the ik leg → tensor[t1][t3][t2].
      const float* __restrict__ tB =
        d_tensor + (size_t)((type1 * num_types + type3) * num_types + type2) * tensor_stride;
      uf3_eval_triplet(
        pos1, pos3, pos2, tB, nc_ij, nc_ik, nc_jk, nint_ij, nint_ik, nint_jk,
        knot_min_ij, knot_delta_ij, inv_knot_delta_ij,
        knot_min_ik, knot_delta_ik, inv_knot_delta_ik,
        knot_min_jk, knot_delta_jk, inv_knot_delta_jk,
        rc_ij, rc_ik, rc_jk, 0.5f, pe, f1, f3, f2, vir);

      atomicAdd(&g_fx[n3], (double)f3.x);
      atomicAdd(&g_fy[n3], (double)f3.y);
      atomicAdd(&g_fz[n3], (double)f3.z);
    }
    atomicAdd(&g_fx[n2], (double)f2.x);
    atomicAdd(&g_fy[n2], (double)f2.y);
    atomicAdd(&g_fz[n2], (double)f2.z);
  }
  atomicAdd(&g_fx[n1], (double)f1.x);
  atomicAdd(&g_fy[n1], (double)f1.y);
  atomicAdd(&g_fz[n1], (double)f1.z);
  g_pe[n1] += (double)pe;
  // Layout matches the 2B kernel: xx yy zz xy xz yz yx zx zy.  Only this
  // thread writes row n1, so plain += is safe.
  g_virial[n1 + 0 * N] += (double)vir[0];
  g_virial[n1 + 1 * N] += (double)vir[1];
  g_virial[n1 + 2 * N] += (double)vir[2];
  g_virial[n1 + 3 * N] += (double)vir[3];
  g_virial[n1 + 4 * N] += (double)vir[4];
  g_virial[n1 + 5 * N] += (double)vir[5];
  g_virial[n1 + 6 * N] += (double)vir[3];
  g_virial[n1 + 7 * N] += (double)vir[4];
  g_virial[n1 + 8 * N] += (double)vir[5];
}

// ---------------------------------------------------------------------------
// Compact 3-body neighbour list: copy the entries of the active neighbour list
// (built with the global rc, usually rc_2b > rc_3b) that lie within rc_keep =
// max(rc_ij, rc_ik).  The jk leg connects two neighbours and never constrains
// the centre's list.  Candidate pair count in the 3B kernel scales with NN², so
// rc_2b=5.5 vs rc_3b=4.25 cuts the triplet loop ~(5.5/4.25)^6 ≈ 4.7×.
// The image-shift code of each kept entry is preserved (g_shift_out is non-null
// exactly when g_shift_in is, so the MIC-vs-explicit-shift convention of the
// source list carries over unchanged).
// ---------------------------------------------------------------------------
static __global__ void filter_neighbor_3b(
  const int N, const float rc2_keep, const Box box,
  const int* __restrict__ g_NN_in,
  const int* __restrict__ g_NL_in,
  const int* __restrict__ g_shift_in,               // nullptr -> MIC list
  const float4* __restrict__ g_pos,
  int* __restrict__ g_NN_out,
  int* __restrict__ g_NL_out,
  int* __restrict__ g_shift_out)                    // nullptr when g_shift_in is
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) return;
  const float4 pos1 = g_pos[n1];
  const int NN = g_NN_in[n1];
  int count = 0;
  for (int i = 0; i < NN; ++i) {
    const int idx = n1 + N * i;
    const int n2 = g_NL_in[idx];
    const float4 pos2 = neighbor_image(__ldg(&g_pos[n2]), pos1, box, g_shift_in, idx);
    const float dx = pos2.x - pos1.x, dy = pos2.y - pos1.y, dz = pos2.z - pos1.z;
    const float d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < rc2_keep && d2 > 1e-12f) {
      const int out = n1 + N * count;
      g_NL_out[out] = n2;
      if (g_shift_out) g_shift_out[out] = g_shift_in[idx];
      ++count;
    }
  }
  g_NN_out[n1] = count;
}

// ---------------------------------------------------------------------------
// 3-body warp-parallel kernel (symmetric models — the hot path).
//
// Parallel granularity: one warp per centre atom.  The triangular (j,k) pair
// loop (k > j) is flattened into a single index t ∈ [0, NN(NN-1)/2) and strided
// across the 32 lanes, so all lanes of a warp work on the same atom's pair list
// — no warp divergence from per-atom NN variation, and ~32× more parallelism
// than thread-per-atom for the same grid of atoms.
//
// The 3B coefficient tensor (all type triplets) is staged in dynamic shared
// memory when it fits (smem_count > 0); each triplet evaluation gathers 16 rows
// of 4 floats from it, which otherwise all goes through L2.
//
// Accumulation is float throughout: per-lane registers for the centre atom's
// energy/force/virial, and native float atomics into the per-atom scratch
// buffer for neighbour forces (double atomics serialize far harder).  The
// scratch is folded into the double-precision global arrays once per step by
// uf3_3b_collect_scratch.  Scratch layout: [0,3N) fx fy fz, [3N,4N) pe,
// [4N,10N) virial xx yy zz xy xz yz.
// ---------------------------------------------------------------------------
static __global__ void find_force_uf3_3b_warp(
  const int N, const int N1, const int N2,
  const Box box,
  const float* __restrict__ d_tensor,               // [num_trips * tensor_stride]
  int num_types, int tensor_stride,
  int smem_count,                                   // floats staged in shared (0 = global reads)
  int nc_ij, int nc_ik, int nc_jk,
  int nint_ij, int nint_ik, int nint_jk,
  float knot_min_ij, float knot_delta_ij, float inv_knot_delta_ij,
  float knot_min_ik, float knot_delta_ik, float inv_knot_delta_ik,
  float knot_min_jk, float knot_delta_jk, float inv_knot_delta_jk,
  float rc_ij, float rc_ik, float rc_jk,
  const int* __restrict__ g_NN,
  const int* __restrict__ g_NL,
  const int* __restrict__ g_shift,                  // per-neighbour image shift (nullptr -> MIC)
  const int* __restrict__ g_type,
  const float4* __restrict__ g_pos,
  float* __restrict__ g_scratch)                    // [10*N] float accumulators
{
  // Cooperative tensor staging must involve every thread of the block, so it
  // runs before any early-out.
  extern __shared__ float s_tensor[];
  for (int i = threadIdx.x; i < smem_count; i += blockDim.x) {
    s_tensor[i] = d_tensor[i];
  }
  if (smem_count > 0) {
    __syncthreads();
  }
  const float* __restrict__ tensor = (smem_count > 0) ? s_tensor : d_tensor;

  const int lane = threadIdx.x & 31;
  const int n1 = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5) + N1;
  if (n1 >= N2) return;

  const int NN = g_NN[n1];
  if (NN < 2) return;                               // scratch row stays zero
  const int type1 = g_type[n1];
  const float4 pos1 = g_pos[n1];

  float pe = 0.0f;
  float3 f1 = make_float3(0.0f, 0.0f, 0.0f);
  float vir[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  // Unrank the flattened pair index: pairs with first index < j number
  // C(j) = j*(2*NN-1-j)/2 (one of the factors is always even — exact in int).
  // The float sqrt gives j up to ±1; the two correction loops pin it down.
  const int T = (NN * (NN - 1)) >> 1;
  const float a = (float)(2 * NN - 1);
  for (int t = lane; t < T; t += 32) {
    int j = (int)((a - sqrtf(a * a - 8.0f * (float)t)) * 0.5f);
    if (j < 0) j = 0; else if (j > NN - 2) j = NN - 2;
    while (j > 0 && j * (2 * NN - 1 - j) / 2 > t) --j;
    while (j < NN - 2 && (j + 1) * (2 * NN - 2 - j) / 2 <= t) ++j;
    const int k = t - j * (2 * NN - 1 - j) / 2 + j + 1;

    const int idx2 = n1 + N * j;
    const int idx3 = n1 + N * k;
    const int n2 = g_NL[idx2];
    const int n3 = g_NL[idx3];
    const float4 pos2 = neighbor_image(__ldg(&g_pos[n2]), pos1, box, g_shift, idx2);
    const float4 pos3 = neighbor_image(__ldg(&g_pos[n3]), pos1, box, g_shift, idx3);
    const int type2 = g_type[n2];
    const int type3 = g_type[n3];

    float3 f2 = make_float3(0.0f, 0.0f, 0.0f);
    float3 f3 = make_float3(0.0f, 0.0f, 0.0f);
    const float* __restrict__ tT =
      tensor + (size_t)((type1 * num_types + type2) * num_types + type3) * tensor_stride;
    uf3_eval_triplet(
      pos1, pos2, pos3, tT, nc_ij, nc_ik, nc_jk, nint_ij, nint_ik, nint_jk,
      knot_min_ij, knot_delta_ij, inv_knot_delta_ij,
      knot_min_ik, knot_delta_ik, inv_knot_delta_ik,
      knot_min_jk, knot_delta_jk, inv_knot_delta_jk,
      rc_ij, rc_ik, rc_jk, 1.0f, pe, f1, f2, f3, vir);

    // A triplet outside any leg's cutoff leaves f2/f3 exactly zero — skip the
    // atomics for it (most candidate pairs fail the jk-leg check).
    if (f2.x != 0.0f || f2.y != 0.0f || f2.z != 0.0f) {
      atomicAdd(&g_scratch[n2], f2.x);
      atomicAdd(&g_scratch[n2 + N], f2.y);
      atomicAdd(&g_scratch[n2 + 2 * N], f2.z);
    }
    if (f3.x != 0.0f || f3.y != 0.0f || f3.z != 0.0f) {
      atomicAdd(&g_scratch[n3], f3.x);
      atomicAdd(&g_scratch[n3 + N], f3.y);
      atomicAdd(&g_scratch[n3 + 2 * N], f3.z);
    }
  }

  // Fold the per-lane partials for the centre atom.  Plain float atomics keep
  // this portable (no warp shuffles, which GPUMD avoids for HIP builds); 10
  // atomics per lane per atom is negligible next to the triplet loop.
  atomicAdd(&g_scratch[n1], f1.x);
  atomicAdd(&g_scratch[n1 + N], f1.y);
  atomicAdd(&g_scratch[n1 + 2 * N], f1.z);
  atomicAdd(&g_scratch[n1 + 3 * N], pe);
  for (int c = 0; c < 6; ++c) {
    atomicAdd(&g_scratch[n1 + (4 + c) * N], vir[c]);
  }
}

// ---------------------------------------------------------------------------
// Fold the 3B float scratch into the double-precision global arrays.  Runs over
// all N atoms (neighbour forces can land outside [N1,N2)); the 1-body offsets
// belong to the centre atoms only, and are added here when no 2B kernel ran.
// Virial layout matches the 2B kernel: xx yy zz xy xz yz yx zx zy (symmetric).
// ---------------------------------------------------------------------------
static __global__ void uf3_3b_collect_scratch(
  const int N, const int N1, const int N2,
  const float* __restrict__ g_scratch,
  const int* __restrict__ g_type,
  const float* __restrict__ g_e0,                   // nullptr if 2B already added it
  double* __restrict__ g_pe,
  double* __restrict__ g_fx,
  double* __restrict__ g_fy,
  double* __restrict__ g_fz,
  double* __restrict__ g_virial)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  g_fx[i] += (double)g_scratch[i];
  g_fy[i] += (double)g_scratch[i + N];
  g_fz[i] += (double)g_scratch[i + 2 * N];
  double pe = (double)g_scratch[i + 3 * N];
  if (g_e0 && i >= N1 && i < N2) pe += (double)g_e0[g_type[i]];
  g_pe[i] += pe;
  const double vxx = (double)g_scratch[i + 4 * N];
  const double vyy = (double)g_scratch[i + 5 * N];
  const double vzz = (double)g_scratch[i + 6 * N];
  const double vxy = (double)g_scratch[i + 7 * N];
  const double vxz = (double)g_scratch[i + 8 * N];
  const double vyz = (double)g_scratch[i + 9 * N];
  g_virial[i + 0 * N] += vxx;
  g_virial[i + 1 * N] += vyy;
  g_virial[i + 2 * N] += vzz;
  g_virial[i + 3 * N] += vxy;
  g_virial[i + 4 * N] += vxz;
  g_virial[i + 5 * N] += vyz;
  g_virial[i + 6 * N] += vxy;
  g_virial[i + 7 * N] += vxz;
  g_virial[i + 8 * N] += vyz;
}

// ---------------------------------------------------------------------------
// 3B neighbour-swap symmetrization (host, jk-fastest tensor layout).
// C[ti,tj,tk][p,q,r] := 0.5*(C[ti,tj,tk][p,q,r] + C[ti,tk,tj][q,p,r]).
// Tensor index: r + (p + q*nc_ij)*nc_jk.  Only valid when ij and ik legs share
// the same knot grid (nc_ij==nc_ik and rc_ij==rc_ik).
// ---------------------------------------------------------------------------
void UF3::project_3b_symmetric(std::vector<float>& tensor) const
{
  if (!sym_3b_) return;
  const int ncij = three_body.nc_ij;
  const int ncik = three_body.nc_ik;
  const int ncjk = three_body.nc_jk;
  const int stride = three_body.tensor_stride;
  const int nt = num_types_;
  for (int ti = 0; ti < nt; ti++)
    for (int tj = 0; tj < nt; tj++)
      for (int tk = tj; tk < nt; tk++) {
        int trip  = (ti * nt + tj) * nt + tk;
        int part  = (ti * nt + tk) * nt + tj;
        float* base  = tensor.data() + (size_t)trip * stride;
        float* pbase = tensor.data() + (size_t)part * stride;
        for (int p = 0; p < ncij; p++)
          for (int q = 0; q < ncik; q++)
            for (int r = 0; r < ncjk; r++) {
              int a = p + q * ncij + r * ncij * ncik;
              int b = q + p * ncij + r * ncij * ncik;
              float avg = 0.5f * (base[a] + pbase[b]);
              base[a] = avg;
              pbase[b] = avg;
            }
      }
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
      // innermost dr-loop in uf3_eval_triplet reads 4 contiguous coefficients.
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
          iss2 >> tdst[(size_t)p + (size_t)q * nci + (size_t)c * nci * nck];
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
    sym_3b_ = (three_body.nc_ij == three_body.nc_ik) &&
              (std::fabs(three_body.rc_ij - three_body.rc_ik) < 1e-4);
    if (sym_3b_) {
      project_3b_symmetric(h_tensor_all);
    } else {
      std::cout << "Warning: 3B ij/ik grids differ (nc=" << three_body.nc_ij << ","
                << three_body.nc_ik << " rc=" << three_body.rc_ij << ","
                << three_body.rc_ik << "); neighbour-swap symmetrization skipped. "
                << "Energies may depend on neighbour-list ordering.\n";
    }
    three_body.d_tensor.resize(h_tensor_all.size());
    three_body.d_tensor.copy_from_host(h_tensor_all.data());

    // Stage the whole tensor table in dynamic shared memory when it fits the
    // portable 48 KB per-block limit (no opt-in attribute needed on any arch).
    // Single-element models (~13³ floats ≈ 8.8 KB) always fit; larger tables
    // fall back to L2-cached global reads.
    const size_t tensor_floats = h_tensor_all.size();
    smem_floats_3b_ =
      (tensor_floats * sizeof(float) <= 48 * 1024) ? (int)tensor_floats : 0;
  }

  // If no 1B line was present, initialize e0 to zeros.
  if (!d_e0.size()) d_e0.resize(1); // at least 1 type as fallback
  cudaMemset(d_e0.data(), 0, d_e0.size() * sizeof(float));

  // Allocate neighbor list and position buffer.
  neighbor.initialize(rc, number_of_atoms, max_neighbor_);
  d_pos_packed.resize(number_of_atoms);

  // Multi-image neighbour list (small-box path).  A thin cell with rc>half-box
  // can list many periodic images per atom, so size generously.
  neighbor_MN_ = max_neighbor_;
  d_NN.resize(number_of_atoms);
  d_NL.resize((size_t)number_of_atoms * neighbor_MN_);
  d_NL_shift.resize((size_t)number_of_atoms * neighbor_MN_);

  if (has_3b) {
    // Compact 3B list pays off only when the 3B cutoff is actually below the
    // global rc (i.e. a 2B block with a larger cutoff exists).
    const double rc_keep =
      three_body.rc_ij > three_body.rc_ik ? three_body.rc_ij : three_body.rc_ik;
    use_3b_list_ = rc_keep < rc - 1e-4;
    if (use_3b_list_) {
      d_NN_3b.resize(number_of_atoms);
      d_NL_3b.resize((size_t)number_of_atoms * neighbor_MN_);
      d_NL_shift_3b.resize((size_t)number_of_atoms * neighbor_MN_);
    }
    // Float accumulators for the warp-parallel symmetric kernel.
    if (sym_3b_) {
      d_scratch_3b.resize((size_t)number_of_atoms * 10);
    }
  }

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
    printf("    3B: %s neighbour list, tensor in %s, %s kernel\n",
           use_3b_list_ ? "compact (rc_3b-filtered)" : "shared (global-rc)",
           smem_floats_3b_ > 0 ? "shared memory" : "global memory (L2)",
           sym_3b_ ? "warp-parallel symmetric" : "thread-per-atom dual-order");
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

  // Decide between the cell-list (minimum-image) path and the multi-image
  // (ghost) path.  When any periodic thickness is below ~2*rc the minimum image
  // is ambiguous; get_num_bins(rc) flags exactly that regime (a direction with
  // < 3 cells of width rc).  The multi-image path then enumerates every image
  // within rc, reproducing the trainer's ghost-supercell distances.
  int nbins[3];
  const bool small_box = box.get_num_bins(rc, nbins);

  const int* nl_NN = nullptr;
  const int* nl_NL = nullptr;
  const int* nl_shift = nullptr;
  if (small_box) {
    const int grid_nb = (N - 1) / BLOCK_SIZE + 1;
    build_neighbor_uf3_multiimage<<<grid_nb, BLOCK_SIZE>>>(
      box, N, (float)rc,
      position_per_atom.data(), position_per_atom.data() + N, position_per_atom.data() + N * 2,
      d_NN.data(), d_NL.data(), d_NL_shift.data(), neighbor_MN_);
    GPU_CHECK_KERNEL
    nl_NN = d_NN.data();
    nl_NL = d_NL.data();
    nl_shift = d_NL_shift.data();
  } else {
    neighbor.find_neighbor_global(rc, box, type, position_per_atom);
    nl_NN = neighbor.NN.data();
    nl_NL = neighbor.NL.data();
    nl_shift = nullptr;     // minimum-image path: kernels apply MIC internally
  }

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
        nl_NN, nl_NL, nl_shift,
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
    // The 1-body offsets are added by the 2B kernel when it runs; pass them to
    // the 3B path only in the (unusual) 3B-only case so they are never counted
    // twice.
    const float* e0_for_3b = has_2b ? nullptr : d_e0.data();

    // Compact 3B neighbour list: filter the active list down to the 3B cutoff
    // so the O(NN²) pair loop below runs on ~rc_3b-sized neighbourhoods instead
    // of rc_2b-sized ones.
    const int* nl3_NN = nl_NN;
    const int* nl3_NL = nl_NL;
    const int* nl3_shift = nl_shift;
    if (use_3b_list_) {
      const float rc_keep = (float)(three_body.rc_ij > three_body.rc_ik
                                      ? three_body.rc_ij : three_body.rc_ik);
      int* shift_out = small_box ? d_NL_shift_3b.data() : nullptr;
      const int grid_nb = (N - 1) / BLOCK_SIZE + 1;
      filter_neighbor_3b<<<grid_nb, BLOCK_SIZE>>>(
        N, rc_keep * rc_keep, box,
        nl_NN, nl_NL, nl_shift,
        d_pos_packed.data(),
        d_NN_3b.data(), d_NL_3b.data(), shift_out);
      GPU_CHECK_KERNEL
      nl3_NN = d_NN_3b.data();
      nl3_NL = d_NL_3b.data();
      nl3_shift = shift_out;
    }

    if (sym_3b_) {
      // Symmetric models (ij/ik grids match, tensor symmetrised on load): a
      // single j<k ordering is exact — run the warp-parallel kernel with float
      // scratch accumulation, then fold into the double-precision arrays.
      CHECK(cudaMemset(d_scratch_3b.data(), 0, (size_t)N * 10 * sizeof(float)));
      const int warps_per_block = BLOCK_SIZE / 32;
      const int grid_3b = (N2 - N1 + warps_per_block - 1) / warps_per_block;
      const size_t smem_bytes = (size_t)smem_floats_3b_ * sizeof(float);
      find_force_uf3_3b_warp<<<grid_3b, BLOCK_SIZE, smem_bytes>>>(
        N, N1, N2, box,
        three_body.d_tensor.data(),
        num_types_, three_body.tensor_stride, smem_floats_3b_,
        three_body.nc_ij, three_body.nc_ik, three_body.nc_jk,
        three_body.nint_ij, three_body.nint_ik, three_body.nint_jk,
        three_body.knot_min_ij, three_body.knot_delta_ij, three_body.inv_knot_delta_ij,
        three_body.knot_min_ik, three_body.knot_delta_ik, three_body.inv_knot_delta_ik,
        three_body.knot_min_jk, three_body.knot_delta_jk, three_body.inv_knot_delta_jk,
        (float)three_body.rc_ij, (float)three_body.rc_ik, (float)three_body.rc_jk,
        nl3_NN, nl3_NL, nl3_shift,
        type.data(),
        d_pos_packed.data(),
        d_scratch_3b.data());
      GPU_CHECK_KERNEL
      const int grid_collect = (N - 1) / BLOCK_SIZE + 1;
      uf3_3b_collect_scratch<<<grid_collect, BLOCK_SIZE>>>(
        N, N1, N2,
        d_scratch_3b.data(),
        type.data(), e0_for_3b,
        potential_per_atom.data(),
        force_per_atom.data(),
        force_per_atom.data() + N,
        force_per_atom.data() + N * 2,
        virial_per_atom.data());
      GPU_CHECK_KERNEL
    } else {
      // Asymmetric legacy models: average both neighbour orderings
      // (thread-per-atom kernel, double accumulation — correctness path).
      find_force_uf3_3b_dual<<<grid_size, BLOCK_SIZE>>>(
        N, N1, N2, box,
        three_body.d_tensor.data(),
        num_types_, three_body.tensor_stride,
        three_body.nc_ij, three_body.nc_ik, three_body.nc_jk,
        three_body.nint_ij, three_body.nint_ik, three_body.nint_jk,
        three_body.knot_min_ij, three_body.knot_delta_ij, three_body.inv_knot_delta_ij,
        three_body.knot_min_ik, three_body.knot_delta_ik, three_body.inv_knot_delta_ik,
        three_body.knot_min_jk, three_body.knot_delta_jk, three_body.inv_knot_delta_jk,
        (float)three_body.rc_ij, (float)three_body.rc_ik, (float)three_body.rc_jk,
        nl3_NN, nl3_NL, nl3_shift,
        type.data(), e0_for_3b,
        d_pos_packed.data(),
        potential_per_atom.data(),
        force_per_atom.data(),
        force_per_atom.data() + N,
        force_per_atom.data() + N * 2,
        virial_per_atom.data());
      GPU_CHECK_KERNEL
    }
  }
}
