/*
    Copyright 2026 MUF development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    MUF-C GPU kernels — multi-element alloy MD inference
    Key fixes over v1:
      1. GPU neighbor list via Neighbor class (NOT CPU O(N^2))
      2. MUF-C multi-element: type-channel moments, cross-type W, per-JJ-pair 2B
      3. No GPU↔CPU roundtrips in compute path
      4. Per-type e0, per-JJ-pair 2B coefficients
*/

#include "muf.cuh"
#include "neighbor.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

// ============================================================================
// Device: C3B metric (80 elements for L=1..8, copied from NEP)
// ============================================================================
__constant__ double muf_C3B[80] = {
  0.238732414637843, 0.119366207318922, 0.119366207318922, 0.099471839432435,
  0.596831036594608, 0.596831036594608, 0.149207759148652, 0.149207759148652,
  0.139260575205408, 0.104445431404056, 0.104445431404056, 1.044454314040563,
  1.044454314040563, 0.174075719006761, 0.174075719006761, 0.011190581936149,
  0.223811638722978, 0.223811638722978, 0.111905819361489, 0.111905819361489,
  1.566681471060845, 1.566681471060845, 0.195835183882606, 0.195835183882606,
  0.013677377921960, 0.102580334414698, 0.102580334414698, 2.872249363611549,
  2.872249363611549, 0.119677056817148, 0.119677056817148, 2.154187022708661,
  2.154187022708661, 0.215418702270866, 0.215418702270866, 0.004041043476943,
  0.169723826031592, 0.169723826031592, 0.106077391269745, 0.106077391269745,
  0.424309565078979, 0.424309565078979, 0.127292869523694, 0.127292869523694,
  2.800443129521260, 2.800443129521260, 0.233370260793438, 0.233370260793438,
  0.004662742473395, 0.004079899664221, 0.004079899664221, 0.024479397985326,
  0.024479397985326, 0.012239698992663, 0.012239698992663, 0.538546755677165,
  0.538546755677165, 0.134636688919291, 0.134636688919291, 3.500553911901575,
  3.500553911901575, 0.250039565135827, 0.250039565135827, 0.000082569397966,
  0.005944996653579, 0.005944996653579, 0.104037441437634, 0.104037441437634,
  0.762941237209318, 0.762941237209318, 0.114441185581398, 0.114441185581398,
  5.950941650232678, 5.950941650232678, 0.141689086910302, 0.141689086910302,
  4.250672607309055, 4.250672607309055, 0.265667037956816, 0.265667037956816};

// ============================================================================
// Device: Z_COEFFICIENT matrices for SH (L=1..4, supports L_max up to 4)
// ============================================================================
__constant__ double muf_Z_COEFF_1[4]  = {0.0, 1.0, 1.0, 0.0};
__constant__ double muf_Z_COEFF_2[9]  = {-1.0,0.0,3.0,0.0, 1.0,0.0,0.0, 1.0,0.0};
__constant__ double muf_Z_COEFF_3[16] = {0.0,-3.0,0.0,5.0, -1.0,0.0,5.0,0.0, 0.0,1.0,0.0,0.0, 1.0,0.0,0.0,0.0};
__constant__ double muf_Z_COEFF_4[25] = {3.0,0.0,-30.0,0.0,35.0, 0.0,-3.0,0.0,7.0,0.0, -1.0,0.0,7.0,0.0,0.0, 0.0,1.0,0.0,0.0,0.0, 1.0,0.0,0.0,0.0,0.0};

__device__ inline double muf_get_z_coeff(int L, int n1, int n2) {
  switch (L) {
    case 1: return muf_Z_COEFF_1[n1 * 2 + n2];
    case 2: return muf_Z_COEFF_2[n1 * 3 + n2];
    case 3: return muf_Z_COEFF_3[n1 * 4 + n2];
    case 4: return muf_Z_COEFF_4[n1 * 5 + n2];
    default: return 0.0;
  }
}

__device__ inline void muf_complex_product_inplace(
    double b_real, double b_imag, double& a_real, double& a_imag)
{
  double r = a_real * b_real - a_imag * b_imag;
  double i = a_real * b_imag + a_imag * b_real;
  a_real = r; a_imag = i;
}

// ============================================================================
// Device: Index helpers
// ============================================================================
__device__ inline int muf_JJ_idx(int I, int J, int num_types) {
  // I <= J assumed; returns upper-triangular row-major index
  return I * (2 * num_types - I - 1) / 2 + J;
}

__device__ inline int muf_ab_idx(int a, int b, int K) {
  // a <= b assumed
  return a * (2 * K - a - 1) / 2 + b;
}

__device__ inline double muf_C_ab(int a, int b) {
  return (a == b) ? 1.0 : 2.0;
}

__device__ inline double muf_C_jj(int J1, int J2) {
  return (J1 == J2) ? 1.0 : 2.0;
}

// ============================================================================
// Device: B-spline with smooth cutoff
// ============================================================================
__device__ inline void muf_eval_bspline_smooth(
    double r, double r_min, double rc, double inv_kdelta, int K,
    double* btilde, int& p0)
{
  if (r >= rc) {
    btilde[0] = btilde[1] = btilde[2] = btilde[3] = 0.0;
    p0 = K - 4;
    return;
  }
  double fc = 0.5 * (cos(M_PI * r / rc) + 1.0);
  double t = (r - r_min) * inv_kdelta;
  p0 = (int)floor(t);
  if (p0 < 0) p0 = 0;
  if (p0 > K - 4) p0 = K - 4;
  double x = t - p0;
  double omx = 1.0 - x;
  double inv6 = 1.0 / 6.0;
  btilde[0] = inv6 * omx * omx * omx * fc;
  btilde[1] = inv6 * (3.0*x*x*x - 6.0*x*x + 4.0) * fc;
  btilde[2] = inv6 * (-3.0*x*x*x + 3.0*x*x + 3.0*x + 1.0) * fc;
  btilde[3] = inv6 * x * x * x * fc;
}

__device__ inline void muf_eval_bspline_smooth_deriv(
    double r, double r_min, double rc, double inv_kdelta, int K,
    double* dbtilde, int& p0)
{
  if (r >= rc) {
    dbtilde[0] = dbtilde[1] = dbtilde[2] = dbtilde[3] = 0.0;
    p0 = K - 4;
    return;
  }
  double fc = 0.5 * (cos(M_PI * r / rc) + 1.0);
  double fcp = -0.5 * (M_PI / rc) * sin(M_PI * r / rc);
  double t = (r - r_min) * inv_kdelta;
  p0 = (int)floor(t);
  if (p0 < 0) p0 = 0;
  if (p0 > K - 4) p0 = K - 4;
  double x = t - p0;
  double dinv = inv_kdelta;
  double inv6 = 1.0 / 6.0;
  double omx = 1.0 - x;
  double b0 = inv6 * omx * omx * omx;
  double b1 = inv6 * (3.0*x*x*x - 6.0*x*x + 4.0);
  double b2 = inv6 * (-3.0*x*x*x + 3.0*x*x + 3.0*x + 1.0);
  double b3 = inv6 * x*x*x;
  dbtilde[0] = (-0.5 * omx * omx * dinv) * fc + b0 * fcp;
  dbtilde[1] = ((1.5*x*x - 2.0*x) * dinv) * fc + b1 * fcp;
  dbtilde[2] = ((-1.5*x*x + x + 0.5) * dinv) * fc + b2 * fcp;
  dbtilde[3] = (0.5*x*x * dinv) * fc + b3 * fcp;
}

// Combined value+derivative: saves one cos() per neighbor
__device__ inline void muf_eval_bspline_smooth_both(
    double r, double r_min, double rc, double inv_kdelta, int K,
    double* btilde, double* dbtilde, int& p0)
{
  if (r >= rc) {
    btilde[0] = btilde[1] = btilde[2] = btilde[3] = 0.0;
    dbtilde[0] = dbtilde[1] = dbtilde[2] = dbtilde[3] = 0.0;
    p0 = K - 4;
    return;
  }
  double arg = M_PI * r / rc;
  double fc = 0.5 * (cos(arg) + 1.0);
  double fcp = -0.5 * (M_PI / rc) * sin(arg);
  double t = (r - r_min) * inv_kdelta;
  p0 = (int)floor(t);
  if (p0 < 0) p0 = 0;
  if (p0 > K - 4) p0 = K - 4;
  double x = t - p0;
  double dinv = inv_kdelta;
  double inv6 = 1.0 / 6.0;
  double omx = 1.0 - x;
  double b0 = inv6 * omx * omx * omx;
  double b1 = inv6 * (3.0*x*x*x - 6.0*x*x + 4.0);
  double b2 = inv6 * (-3.0*x*x*x + 3.0*x*x + 3.0*x + 1.0);
  double b3 = inv6 * x*x*x;
  btilde[0] = b0 * fc;  btilde[1] = b1 * fc;
  btilde[2] = b2 * fc;  btilde[3] = b3 * fc;
  dbtilde[0] = (-0.5 * omx * omx * dinv) * fc + b0 * fcp;
  dbtilde[1] = ((1.5*x*x - 2.0*x) * dinv) * fc + b1 * fcp;
  dbtilde[2] = ((-1.5*x*x + x + 0.5) * dinv) * fc + b2 * fcp;
  dbtilde[3] = (0.5*x*x * dinv) * fc + b3 * fcp;
}

// ============================================================================
// Device: SH accumulation (spherical harmonics, NEP convention)
// ============================================================================
template <int L>
__device__ void muf_accumulate_s_one(double x12, double y12, double z12, double fn, double* s)
{
  int s_index = L * L - 1;
  double z_pow[9];
  z_pow[0] = 1.0;
  for (int n = 1; n <= L; ++n) z_pow[n] = z12 * z_pow[n - 1];
  double real_part = x12, imag_part = y12;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    double z_factor = 0.0;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2)
      z_factor += muf_get_z_coeff(L, n1, n2) * z_pow[n2];
    z_factor *= fn;
    if (n1 == 0) {
      s[s_index++] += z_factor;
    } else {
      s[s_index++] += z_factor * real_part;
      s[s_index++] += z_factor * imag_part;
      muf_complex_product_inplace(x12, y12, real_part, imag_part);
    }
  }
}

__device__ void muf_accumulate_s(int L_max, double d12, double x12, double y12, double z12,
                                  double fn, double* s)
{
  double d12inv = 1.0 / d12;
  x12 *= d12inv; y12 *= d12inv; z12 *= d12inv;
  if (L_max >= 1) muf_accumulate_s_one<1>(x12, y12, z12, fn, s);
  if (L_max >= 2) muf_accumulate_s_one<2>(x12, y12, z12, fn, s);
  if (L_max >= 3) muf_accumulate_s_one<3>(x12, y12, z12, fn, s);
  if (L_max >= 4) muf_accumulate_s_one<4>(x12, y12, z12, fn, s);
}

// ============================================================================
// Device: SH force backprop (accumulate_f12)
// ============================================================================
template <int L>
__device__ void muf_accumulate_f12_one(double d12inv, double fn, double fnp,
                                        const double* s, const double* r12, double* f12)
{
  const double dx[3] = {(1.0-r12[0]*r12[0])*d12inv, -r12[0]*r12[1]*d12inv, -r12[0]*r12[2]*d12inv};
  const double dy[3] = {-r12[0]*r12[1]*d12inv, (1.0-r12[1]*r12[1])*d12inv, -r12[1]*r12[2]*d12inv};
  const double dz[3] = {-r12[0]*r12[2]*d12inv, -r12[1]*r12[2]*d12inv, (1.0-r12[2]*r12[2])*d12inv};

  double z_pow[9]; z_pow[0] = 1.0;
  for (int n = 1; n <= L; ++n) z_pow[n] = r12[2] * z_pow[n - 1];
  double real_part = 1.0, imag_part = 0.0;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    double z_factor = 0.0, dz_factor = 0.0;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      z_factor += muf_get_z_coeff(L, n1, n2) * z_pow[n2];
      if (n2 > 0) dz_factor += muf_get_z_coeff(L, n1, n2) * n2 * z_pow[n2 - 1];
    }
    if (n1 == 0) {
      for (int d = 0; d < 3; ++d)
        f12[d] += s[0] * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
    } else {
      double real_part_n1 = n1 * real_part, imag_part_n1 = n1 * imag_part;
      for (int d = 0; d < 3; ++d) {
        double real_part_dx = dx[d], imag_part_dy = dy[d];
        muf_complex_product_inplace(real_part_n1, imag_part_n1, real_part_dx, imag_part_dy);
        f12[d] += (s[2*n1-1]*real_part_dx + s[2*n1-0]*imag_part_dy) * z_factor * fn;
      }
      double xy_real = r12[0]*real_part - r12[1]*imag_part;
      double xy_imag = r12[0]*imag_part + r12[1]*real_part;
      double xy_temp = s[2*n1-1]*xy_real + s[2*n1-0]*xy_imag;
      for (int d = 0; d < 3; ++d)
        f12[d] += xy_temp * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
      muf_complex_product_inplace(r12[0], r12[1], real_part, imag_part);
    }
  }
}

// ============================================================================
// Device: SH force backprop — FP32 for Blackwell (128 FP32 units vs 2 FP64 per SM)
// ============================================================================
template <int L>
__device__ void muf_accumulate_f12_one_f32(float d12inv, float fn, float fnp,
                                            const float* s, const float* r12, float* f12)
{
  float dx[3] = {(1.0f-r12[0]*r12[0])*d12inv, -r12[0]*r12[1]*d12inv, -r12[0]*r12[2]*d12inv};
  float dy[3] = {-r12[0]*r12[1]*d12inv, (1.0f-r12[1]*r12[1])*d12inv, -r12[1]*r12[2]*d12inv};
  float dz[3] = {-r12[0]*r12[2]*d12inv, -r12[1]*r12[2]*d12inv, (1.0f-r12[2]*r12[2])*d12inv};

  float z_pow[9]; z_pow[0] = 1.0f;
  for (int n = 1; n <= L; ++n) z_pow[n] = r12[2] * z_pow[n - 1];
  float real_part = 1.0f, imag_part = 0.0f;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    float z_factor = 0.0f, dz_factor = 0.0f;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      float zc = (float)muf_get_z_coeff(L, n1, n2);
      z_factor += zc * z_pow[n2];
      if (n2 > 0) dz_factor += zc * (float)n2 * z_pow[n2 - 1];
    }
    if (n1 == 0) {
      for (int d = 0; d < 3; ++d)
        f12[d] += s[0] * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
    } else {
      float real_part_n1 = (float)n1 * real_part, imag_part_n1 = (float)n1 * imag_part;
      for (int d = 0; d < 3; ++d) {
        float real_part_dx = dx[d], imag_part_dy = dy[d];
        float cp_real = real_part_n1 * real_part_dx - imag_part_n1 * imag_part_dy;
        float cp_imag = real_part_n1 * imag_part_dy + imag_part_n1 * real_part_dx;
        f12[d] += (s[2*n1-1]*cp_real + s[2*n1-0]*cp_imag) * z_factor * fn;
      }
      float xy_real = r12[0]*real_part - r12[1]*imag_part;
      float xy_imag = r12[0]*imag_part + r12[1]*real_part;
      float xy_temp = s[2*n1-1]*xy_real + s[2*n1-0]*xy_imag;
      for (int d = 0; d < 3; ++d)
        f12[d] += xy_temp * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
      float rp_real = r12[0]*real_part - r12[1]*imag_part;
      float rp_imag = r12[0]*imag_part + r12[1]*real_part;
      real_part = rp_real; imag_part = rp_imag;
    }
  }
}

// ============================================================================
// Kernel: MUF-C 2B energy + force + virial (type-pair aware)
// ============================================================================
static __global__ void muf_kernel_2b(
    const double* __restrict__ g_x,
    const double* __restrict__ g_y,
    const double* __restrict__ g_z,
    const int* __restrict__ g_type,
    int N, int num_types,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    int MN,
    double rc_2b, double r_min_2b, double inv_kdelta_2b, int K,
    const double* __restrict__ g_coeff_2b,  // [num_JJ_pairs * K]
    double* __restrict__ g_energy,
    double* __restrict__ g_fx,
    double* __restrict__ g_fy,
    double* __restrict__ g_fz,
    double* __restrict__ g_virial)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  int I = g_type[i];
  int n_neigh = g_NN[i];
  double xi = g_x[i], yi = g_y[i], zi = g_z[i];
  double e2_i = 0.0;
  double fx_i = 0.0, fy_i = 0.0, fz_i = 0.0;
  double virial_xx = 0.0, virial_yy = 0.0, virial_zz = 0.0;
  double virial_xy = 0.0, virial_xz = 0.0, virial_yz = 0.0;

  for (int k = 0; k < n_neigh; ++k) {
    int j = g_NL[i * MN + k];
    double dx = xi - g_x[j];
    double dy = yi - g_y[j];
    double dz = zi - g_z[j];
    double r2 = dx*dx + dy*dy + dz*dz;
    double r = sqrt(r2);
    if (r >= rc_2b || r < 1e-12) continue;

    // Combined B-spline value + derivative (single cos/sin call)
    double btilde[4], dbtilde[4];
    int p0;
    muf_eval_bspline_smooth_both(r, r_min_2b, rc_2b, inv_kdelta_2b, K,
                                  btilde, dbtilde, p0);

    int J = g_type[j];
    int I_eff = min(I, J), J_eff = max(I, J);
    int jj = muf_JJ_idx(I_eff, J_eff, num_types);
    const double* coeff_jj = g_coeff_2b + jj * K;

    double v2 = 0.0, dv2 = 0.0;
    for (int p = 0; p < 4; ++p) {
      int a = p0 + p;
      if (a >= K) continue;
      v2 += coeff_jj[a] * btilde[p];
      dv2 += coeff_jj[a] * dbtilde[p];
    }

    // Energy with 0.5 factor (undirected neighbor list)
    e2_i += 0.5 * v2;

    // Force: accumulate atom i in registers (one write at end)
    double rinv = 1.0 / r;
    double f_mag = 0.5 * (-dv2);
    double fx = f_mag * dx * rinv;
    double fy = f_mag * dy * rinv;
    double fz = f_mag * dz * rinv;

    fx_i += fx;
    fy_i += fy;
    fz_i += fz;

    // Atom j still needs atomicAdd (other threads may target j)
    atomicAdd(&g_fx[j], -fx);
    atomicAdd(&g_fy[j], -fy);
    atomicAdd(&g_fz[j], -fz);

    // 2B virial: accumulate in registers (written once at end)
    double f_raw = -dv2;
    double frx = f_raw * dx * rinv;
    double fry = f_raw * dy * rinv;
    double frz = f_raw * dz * rinv;
    virial_xx += 0.5 * dx * frx;
    virial_yy += 0.5 * dy * fry;
    virial_zz += 0.5 * dz * frz;
    virial_xy += 0.5 * dx * fry;
    virial_xz += 0.5 * dx * frz;
    virial_yz += 0.5 * dy * frz;
  }

  g_energy[i] = e2_i;

  // Write atom i's force (non-atomic: one thread per atom)
  g_fx[i] = fx_i;
  g_fy[i] = fy_i;
  g_fz[i] = fz_i;

  // GPUMD virial format: [xx, yy, zz, xy, xz, yz, yx, zx, zy] per atom
  g_virial[i * 9 + 0] = virial_xx;
  g_virial[i * 9 + 1] = virial_yy;
  g_virial[i * 9 + 2] = virial_zz;
  g_virial[i * 9 + 3] = virial_xy;
  g_virial[i * 9 + 4] = virial_xz;
  g_virial[i * 9 + 5] = virial_yz;
  g_virial[i * 9 + 6] = virial_xy;
  g_virial[i * 9 + 7] = virial_xz;
  g_virial[i * 9 + 8] = virial_yz;
}

// ============================================================================
// Kernel: MUF-C 3B type-channel descriptors
// A_{a,l,k}^{I→J,i} = sum_{j,type(j)=J} B̃_a(r_ij) * Y_{l,k}(r̂_ij)
// Layout: moments[i * stride + J*K*num_sh + a*num_sh + sh_index]
// stride = num_types * K * num_sh_terms
// ============================================================================
static __global__ void muf_kernel_descriptors(
    const double* __restrict__ g_x,
    const double* __restrict__ g_y,
    const double* __restrict__ g_z,
    const int* __restrict__ g_type,
    int N, int num_types,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    int MN,
    double rc_3b, double r_min_3b, double inv_kdelta_3b, int K, int L_max,
    int num_sh_terms,
    float* __restrict__ g_moments)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  int stride = num_types * K * num_sh_terms;
  float* A_i = g_moments + i * stride;
  int n_neigh = g_NN[i];
  double xi = g_x[i], yi = g_y[i], zi = g_z[i];

  for (int k = 0; k < n_neigh; ++k) {
    int j = g_NL[i * MN + k];
    double dx = xi - g_x[j];
    double dy = yi - g_y[j];
    double dz = zi - g_z[j];
    double r2 = dx*dx + dy*dy + dz*dz;
    double r = sqrt(r2);
    if (r >= rc_3b || r < 1e-12) continue;

    double btilde[4];
    int p0;
    muf_eval_bspline_smooth(r, r_min_3b, rc_3b, inv_kdelta_3b, K, btilde, p0);

    int J = g_type[j];  // neighbor type
    double s_local[80] = {0.0};
    muf_accumulate_s(L_max, r, dx, dy, dz, 1.0, s_local);

    for (int p = 0; p < 4; ++p) {
      int a = p0 + p;
      if (a >= K) continue;
      float weight = (float)btilde[p];  // FP32 for accumulation
      // Offset for type channel J, B-spline a
      float* A_aJ = A_i + J * K * num_sh_terms + a * num_sh_terms;
      for (int sh = 0; sh < num_sh_terms; ++sh)
        A_aJ[sh] += weight * (float)s_local[sh];
    }
  }
}

// ============================================================================
// Kernel: MUF-C 3B contraction V3 — scatter-optimized dE/dA
//
// Optimizations:
//   1. Energy: register-hoist C3B-weighted A_J1a[k] before inner b-loop
//   2. dE/dA: precompute W_eff[J1][a][J2][b] for all combos, then scatter:
//      each A[J2][b][k] is read ONCE and scattered to all (J1,a) targets.
//      Eliminates 20x redundant A reads in dE/dA pass.
//   3. W access cached in L1 (only ~10KB per type, accessed regularly)
//
// Launch: grid=(N+127)/128, block=128 (thread-per-atom, full parallelism)
// ============================================================================
static __global__ void muf_kernel_3b_contract_v2(
    int Nmin, int num_types, int K, int L_max, int num_sh_terms,
    int num_JJ_pairs, int num_ab_pairs, int band_width,
    const int* __restrict__ g_type,
    const double* __restrict__ g_moments,
    const double* __restrict__ g_W,
    double* __restrict__ g_energy_3b,
    double* __restrict__ g_dE_dA)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= Nmin) return;

  int I = g_type[i];
  int stride = num_types * K * num_sh_terms;
  const double* A_i = g_moments + i * stride;
  double* dA_i = g_dE_dA + i * stride;
  double e3_i = 0.0;

  int W_stride_I = I * num_JJ_pairs * num_ab_pairs * L_max;

  for (int L = 1; L <= L_max; ++L) {
    int start = L * L - 1;
    int ncomp = 2 * L + 1;
    int l_slot = L - 1;

    double C3B_L[17];
    for (int k = 0; k < ncomp; ++k)
      C3B_L[k] = muf_C3B[start + k];

    // ====== Energy pass (same as V2: register-hoisted) ======
    for (int J1 = 0; J1 < num_types; ++J1) {
      int J1_base = J1 * K * num_sh_terms;
      for (int J2 = J1; J2 < num_types; ++J2) {
        int jj = muf_JJ_idx(J1, J2, num_types);
        double c_jj = (J1 == J2) ? 1.0 : 2.0;
        int W_jj_base = W_stride_I + jj * num_ab_pairs * L_max;
        int J2_base = J2 * K * num_sh_terms;

        for (int a = 0; a < K; ++a) {
          double A_J1a_w[17];
          int off_J1a = J1_base + a * num_sh_terms + start;
          for (int k = 1; k < ncomp; ++k)
            A_J1a_w[k] = C3B_L[k] * (double)A_i[off_J1a + k];

          int b_lo = max(a, a - band_width);
          int b_hi = min(K - 1, a + band_width);
          for (int b = b_lo; b <= b_hi; ++b) {
            int aa = min(a, b), bb = max(a, b);
            if (aa != a) continue;
            int ab = muf_ab_idx(a, b, K);
            double W_val = g_W[W_jj_base + ab * L_max + l_slot];
            if (W_val == 0.0) continue;
            double c_ab = (a == b) ? 1.0 : 2.0;

            double S = 0.0;
            int off_J2b = J2_base + b * num_sh_terms + start;
            for (int k = 1; k < ncomp; ++k)
              S += A_J1a_w[k] * (double)A_i[off_J2b + k];
            e3_i += W_val * c_jj * c_ab * S;
          }
        }
      }
    }

    // ====== dE/dA pass: W_eff precompute + scatter ======
    // Step A: Precompute W_eff[J1][a][J2][b] for ALL combinations
    double W_eff[2][10][2][10];  // [num_types][K][num_types][K]
    for (int j1 = 0; j1 < num_types; ++j1)
      for (int aa = 0; aa < K; ++aa)
        for (int j2 = 0; j2 < num_types; ++j2)
          for (int bb = 0; bb < K; ++bb)
            W_eff[j1][aa][j2][bb] = 0.0;

    for (int J1 = 0; J1 < num_types; ++J1) {
      for (int a = 0; a < K; ++a) {
        // J2 >= J1
        for (int J2 = J1; J2 < num_types; ++J2) {
          int jj = muf_JJ_idx(J1, J2, num_types);
          double c_jj = (J1 == J2) ? 1.0 : 2.0;
          int W_jj_base = W_stride_I + jj * num_ab_pairs * L_max;

          int b_lo = max(0, a - band_width);
          int b_hi = min(K - 1, a + band_width);
          for (int b = b_lo; b <= b_hi; ++b) {
            int aa = min(a, b), bb = max(a, b);
            int ab = muf_ab_idx(aa, bb, K);
            double w = g_W[W_jj_base + ab * L_max + l_slot];
            if (w == 0.0) continue;
            double factor = 2.0 * c_jj;
            double c_ab = (aa == bb) ? 1.0 : 2.0;
            W_eff[J1][a][J2][b] += factor * c_ab * w;
          }
        }
        // J2 < J1
        for (int J2 = 0; J2 < J1; ++J2) {
          int jj = muf_JJ_idx(J2, J1, num_types);
          int W_jj_base = W_stride_I + jj * num_ab_pairs * L_max;

          int b_lo = max(0, a - band_width);
          int b_hi = min(K - 1, a + band_width);
          for (int b = b_lo; b <= b_hi; ++b) {
            int aa = min(a, b), bb = max(a, b);
            int ab = muf_ab_idx(aa, bb, K);
            double w = g_W[W_jj_base + ab * L_max + l_slot];
            if (w == 0.0) continue;
            double c_ab = (aa == bb) ? 1.0 : 2.0;
            W_eff[J1][a][J2][b] += 4.0 * c_ab * w;  // 2*c_jj, c_jj=2
          }
        }
      }
    }

    // Step B: Scatter: read each A[J2][b][k] once, scatter to all (J1,a)
    for (int J2 = 0; J2 < num_types; ++J2) {
      int J2_base = J2 * K * num_sh_terms;
      for (int b = 0; b < K; ++b) {
        int off_J2b = J2_base + b * num_sh_terms + start;
        for (int k = 1; k < ncomp; ++k) {
          double A_val = C3B_L[k] * (double)A_i[off_J2b + k];
          if (A_val == 0.0) continue;
          // Scatter to all (J1,a)
          for (int J1 = 0; J1 < num_types; ++J1) {
            int J1_base = J1 * K * num_sh_terms;
            for (int a = 0; a < K; ++a) {
              double we = W_eff[J1][a][J2][b];
              if (we != 0.0)
                dA_i[J1_base + a * num_sh_terms + start + k] += we * A_val;
            }
          }
        }
      }
    }
  }

  g_energy_3b[i] = e3_i;
}

// ============================================================================
// Kernel: MUF-C 3B contract + force + energy fused (V4)
//
// Fuses contract (energy + dE/dA) + 3B force backprop + energy addition
// into a single kernel. Eliminates dE_dA global memory (514MB write+read).
//
// Key design:
//   - Per-L outer loop: dE_dA_L fits in ~180 doubles (L=4 worst case)
//   - Scatter: each A[J2][b][k] read once, scatter to all (J1,a) targets
//     with g_W read directly (L1-cached, no W_eff table needed)
//   - Force backprop: per-L neighbor iteration, SH backprop immediately
//   - 4x neighbor iterations (once per L), but B-spline eval is cheap
//
// Launch: grid=(N+127)/128, block=128 (thread-per-atom, full parallelism)
// ============================================================================
static __global__ void muf_kernel_3b_contract_force_v4(
    const double* __restrict__ g_x,
    const double* __restrict__ g_y,
    const double* __restrict__ g_z,
    const int* __restrict__ g_type,
    int N, int num_types,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    int MN,
    double rc_3b, double r_min_3b, double inv_kdelta_3b,
    int K, int L_max, int num_sh_terms,
    int num_JJ_pairs, int num_ab_pairs, int band_width,
    const double* __restrict__ g_moments,
    const double* __restrict__ g_W,
    const double* __restrict__ g_e0,
    double* __restrict__ g_potential,  // in: 2B energy, out: 2B+3B+e0
    double* __restrict__ g_fx,
    double* __restrict__ g_fy,
    double* __restrict__ g_fz,
    double* __restrict__ g_virial)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  int I = g_type[i];
  int stride = num_types * K * num_sh_terms;
  const double* A_i = g_moments + i * stride;
  double e3_i = 0.0;

  int W_stride_I = I * num_JJ_pairs * num_ab_pairs * L_max;
  double xi = g_x[i], yi = g_y[i], zi = g_z[i];

  // Virial accumulators
  double virial_xx = 0.0, virial_yy = 0.0, virial_zz = 0.0;
  double virial_xy = 0.0, virial_xz = 0.0, virial_yz = 0.0;

  // ====== Per-L outer loop ======
  for (int L = 1; L <= L_max; ++L) {
    int start = L * L - 1;
    int ncomp = 2 * L + 1;
    int l_slot = L - 1;

    // Load C3B for this L
    double C3B_L[17];
    for (int k = 0; k < ncomp; ++k)
      C3B_L[k] = muf_C3B[start + k];

    // ====== Part 1: Energy (register-hoisted, same as V3) ======
    for (int J1 = 0; J1 < num_types; ++J1) {
      int J1_base = J1 * K * num_sh_terms;
      for (int J2 = J1; J2 < num_types; ++J2) {
        int jj = muf_JJ_idx(J1, J2, num_types);
        double c_jj = (J1 == J2) ? 1.0 : 2.0;
        int W_jj_base = W_stride_I + jj * num_ab_pairs * L_max;
        int J2_base = J2 * K * num_sh_terms;

        for (int a = 0; a < K; ++a) {
          // Register-hoist C3B-weighted A[J1][a] values
          double A_J1a_w[17];
          int off_J1a = J1_base + a * num_sh_terms + start;
          for (int k = 1; k < ncomp; ++k)
            A_J1a_w[k] = C3B_L[k] * (double)A_i[off_J1a + k];

          int b_lo = max(a, a - band_width);
          int b_hi = min(K - 1, a + band_width);
          for (int b = b_lo; b <= b_hi; ++b) {
            int aa = min(a, b), bb = max(a, b);
            if (aa != a) continue;
            int ab = muf_ab_idx(a, b, K);
            double W_val = g_W[W_jj_base + ab * L_max + l_slot];
            if (W_val == 0.0) continue;
            double c_ab = (a == b) ? 1.0 : 2.0;

            double S = 0.0;
            int off_J2b = J2_base + b * num_sh_terms + start;
            for (int k = 1; k < ncomp; ++k)
              S += A_J1a_w[k] * (double)A_i[off_J2b + k];
            e3_i += W_val * c_jj * c_ab * S;
          }
        }
      }
    }

    // ====== Part 2: dE_dA scatter (per-L, in registers/stack) ======
    // dE_dA_L[J1][a][k] — each thread's private copy for this L
    // Compiler manages register spillage to L1 for L=4 (180 doubles)
    double dE_dA_L[2][10][17];  // [num_types][K][2L+1], max 17 for L=4
    for (int j1 = 0; j1 < num_types; ++j1)
      for (int aa = 0; aa < K; ++aa)
        for (int kk = 0; kk < ncomp; ++kk)
          dE_dA_L[j1][aa][kk] = 0.0;

    // Scatter: read each A[J2][b][k] once, distribute to all (J1,a)
    for (int J2 = 0; J2 < num_types; ++J2) {
      int J2_base = J2 * K * num_sh_terms;
      for (int b = 0; b < K; ++b) {
        int off_J2b = J2_base + b * num_sh_terms + start;
        for (int k = 1; k < ncomp; ++k) {
          double A_val = C3B_L[k] * (double)A_i[off_J2b + k];
          if (A_val == 0.0) continue;

          for (int J1 = 0; J1 < num_types; ++J1) {
            for (int a = 0; a < K; ++a) {
              if (abs(a - b) > band_width) continue;

              double w_eff;
              if (J1 <= J2) {
                int jj = muf_JJ_idx(J1, J2, num_types);
                double c_jj = (J1 == J2) ? 1.0 : 2.0;
                int ab = muf_ab_idx(min(a, b), max(a, b), K);
                double w = g_W[W_stride_I + jj * num_ab_pairs * L_max
                               + ab * L_max + l_slot];
                double c_ab = (a == b) ? 1.0 : 2.0;
                w_eff = 2.0 * c_jj * c_ab * w;
              } else {
                int jj = muf_JJ_idx(J2, J1, num_types);
                int ab = muf_ab_idx(min(a, b), max(a, b), K);
                double w = g_W[W_stride_I + jj * num_ab_pairs * L_max
                               + ab * L_max + l_slot];
                double c_ab = (a == b) ? 1.0 : 2.0;
                w_eff = 4.0 * c_ab * w;
              }

              if (w_eff != 0.0)
                dE_dA_L[J1][a][k] += w_eff * A_val;
            }
          }
        }
      }
    }

    // ====== Part 3: Force backprop for this L ======
    int n_neigh = g_NN[i];
    for (int n = 0; n < n_neigh; ++n) {
      int j = g_NL[i * MN + n];
      double dx = xi - g_x[j];
      double dy = yi - g_y[j];
      double dz = zi - g_z[j];
      double r2 = dx*dx + dy*dy + dz*dz;
      double r = sqrt(r2);
      if (r >= rc_3b || r < 1e-12) continue;

      double rinv = 1.0 / r;
      double r12[3] = {dx * rinv, dy * rinv, dz * rinv};

      double btilde[4], dbtilde[4];
      int p0;
      muf_eval_bspline_smooth(r, r_min_3b, rc_3b, inv_kdelta_3b, K, btilde, p0);
      muf_eval_bspline_smooth_deriv(r, r_min_3b, rc_3b, inv_kdelta_3b,
                                     K, dbtilde, p0);

      int J = g_type[j];
      double f12_total[3] = {0.0, 0.0, 0.0};

      for (int p = 0; p < 4; ++p) {
        int a = p0 + p;
        if (a >= K) continue;
        double fn = btilde[p], fnp = dbtilde[p];

        // Extract dE_dA_L for this (J,a) into s_L array
        double s_L[17];
        for (int kk = 0; kk < ncomp; ++kk)
          s_L[kk] = dE_dA_L[J][a][kk];

        double f12[3] = {0.0, 0.0, 0.0};
        switch (L) {
          case 1: muf_accumulate_f12_one<1>(rinv, fn, fnp, s_L, r12, f12); break;
          case 2: muf_accumulate_f12_one<2>(rinv, fn, fnp, s_L, r12, f12); break;
          case 3: muf_accumulate_f12_one<3>(rinv, fn, fnp, s_L, r12, f12); break;
          case 4: muf_accumulate_f12_one<4>(rinv, fn, fnp, s_L, r12, f12); break;
        }
        f12_total[0] += f12[0];
        f12_total[1] += f12[1];
        f12_total[2] += f12[2];
      }

      // Newton's 3rd law: dA_i contributes to f_i → -f12, f_j → +f12
      atomicAdd(&g_fx[i], -f12_total[0]);
      atomicAdd(&g_fy[i], -f12_total[1]);
      atomicAdd(&g_fz[i], -f12_total[2]);
      atomicAdd(&g_fx[j], f12_total[0]);
      atomicAdd(&g_fy[j], f12_total[1]);
      atomicAdd(&g_fz[j], f12_total[2]);

      // 3B virial
      virial_xx += -f12_total[0] * dx;
      virial_yy += -f12_total[1] * dy;
      virial_zz += -f12_total[2] * dz;
      virial_xy += -f12_total[0] * dy;
      virial_xz += -f12_total[0] * dz;
      virial_yz += -f12_total[1] * dz;
    }

    // End of L loop
  }

  // ====== Write results (after all L processed) ======
  // 3B energy + e0 added to potential (2B already there)
  g_potential[i] += e3_i + g_e0[I];

  // 3B virial (atomic: 2B kernel already wrote there)
  atomicAdd(&g_virial[i * 9 + 0], virial_xx);
  atomicAdd(&g_virial[i * 9 + 1], virial_yy);
  atomicAdd(&g_virial[i * 9 + 2], virial_zz);
  atomicAdd(&g_virial[i * 9 + 3], virial_xy);
  atomicAdd(&g_virial[i * 9 + 4], virial_xz);
  atomicAdd(&g_virial[i * 9 + 5], virial_yz);
  atomicAdd(&g_virial[i * 9 + 6], virial_xy);
  atomicAdd(&g_virial[i * 9 + 7], virial_xz);
  atomicAdd(&g_virial[i * 9 + 8], virial_yz);
}

// ============================================================================
// Kernel: MUF-C 3B contract + force + energy fused (V5)
//
// Fuses contract (energy + dE/dA) + 3B force backprop + energy addition.
// Eliminates dE_dA global memory (514MB write+read).
//
// Key design:
//   - W_eff precomputed in SHARED memory (2 types × 400 doubles = 6.4KB)
//     Filled by 2 threads per block, reused by all threads during scatter
//   - dE_dA accumulated across ALL L in per-thread local memory (480 doubles)
//     Then ONE neighbor pass backprops all L simultaneously
//   - Eliminates both scatter g_W reads AND multiple neighbor iterations
//
// Launch: grid=(N+127)/128, block=128 (thread-per-atom, full parallelism)
// ============================================================================
static __global__ void muf_kernel_3b_contract_force_v7(
    const double* __restrict__ g_x,
    const double* __restrict__ g_y,
    const double* __restrict__ g_z,
    const int* __restrict__ g_type,
    int N, int num_types,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    int MN,
    double rc_3b, double r_min_3b, double inv_kdelta_3b,
    int K, int L_max, int num_sh_terms,
    int num_JJ_pairs, int num_ab_pairs, int band_width,
    const double* __restrict__ g_W,
    const double* __restrict__ g_e0,
    double* __restrict__ g_potential,
    double* __restrict__ g_fx,
    double* __restrict__ g_fy,
    double* __restrict__ g_fz,
    double* __restrict__ g_virial)
{
  // V9: Descriptor+Contract+Force fused — A_i computed in local memory (no global rountrip)
  // s_W_eff[l_slot][type_I][J1][a][J2][b], 4×2×2×10×2×10 = 3200 doubles = 25.6KB
  __shared__ double s_W_eff[4][2][2][10][2][10];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  int I = g_type[i];
  float dE_dA_all[480];  // num_types*K*num_sh_terms = 480
  for (int idx = 0; idx < num_types * K * num_sh_terms; ++idx)
    dE_dA_all[idx] = 0.0f;

  double xi = g_x[i], yi = g_y[i], zi = g_z[i];
  double virial_xx = 0.0, virial_yy = 0.0, virial_zz = 0.0;
  double virial_xy = 0.0, virial_xz = 0.0, virial_yz = 0.0;

  // ====== Phase 0: Compute A_i descriptors in local memory (V9: fused from descriptor kernel) ======
  float A_i_local[480];
  for (int idx = 0; idx < num_types * K * num_sh_terms; ++idx)
    A_i_local[idx] = 0.0f;

  int n_neigh = g_NN[i];
  for (int k = 0; k < n_neigh; ++k) {
    int j = g_NL[i * MN + k];
    double dx = xi - g_x[j];
    double dy = yi - g_y[j];
    double dz = zi - g_z[j];
    double r2 = dx*dx + dy*dy + dz*dz;
    double r = sqrt(r2);
    if (r >= rc_3b || r < 1e-12) continue;

    double btilde[4];
    int p0;
    muf_eval_bspline_smooth(r, r_min_3b, rc_3b, inv_kdelta_3b, K, btilde, p0);

    int J = g_type[j];
    double s_local[80] = {0.0};
    muf_accumulate_s(L_max, r, dx, dy, dz, 1.0, s_local);

    for (int p = 0; p < 4; ++p) {
      int a = p0 + p;
      if (a >= K) continue;
      float weight = (float)btilde[p];
      float* A_aJ = A_i_local + J * K * num_sh_terms + a * num_sh_terms;
      for (int sh = 0; sh < num_sh_terms; ++sh)
        A_aJ[sh] += weight * (float)s_local[sh];
    }
  }

  double e3_i = 0.0;
  const float* A_i = A_i_local;

  // ====== Phase 1: Fill s_W_eff for ALL L (all 128 threads, single barrier) ======
  // Total elements: L_max * 2 * 2 * 10 * 2 * 10 = 4*800 = 3200
  // Each thread handles ~25 elements. Use stride over linear index.
  int total_we = L_max * 2 * num_types * K * num_types * K; // 4*2*2*10*2*10 = 3200
  // First, zero the shared memory (all threads)
  for (int idx = threadIdx.x; idx < total_we; idx += blockDim.x)
    ((double*)s_W_eff)[idx] = 0.0;
  __syncthreads();

  // Fill W_eff for all L, all type pairs — parallelized across all threads
  for (int idx = threadIdx.x; idx < total_we; idx += blockDim.x) {
    // Decode linear index: [l_slot][It][J1][a][J2][b]
    int tmp = idx;
    int b = tmp % K;        tmp /= K;
    int J2 = tmp % num_types; tmp /= num_types;
    int a = tmp % K;         tmp /= K;
    int J1 = tmp % num_types; tmp /= num_types;
    int It = tmp % 2;        tmp /= 2;
    int l_slot = tmp;        // 0..L_max-1

    int W_stride_It = It * num_JJ_pairs * num_ab_pairs * L_max;
    double val = 0.0;

    if (J2 >= J1) {
      int jj = muf_JJ_idx(J1, J2, num_types);
      int bmin = max(0, a - band_width), bmax = min(K - 1, a + band_width);
      if (b >= bmin && b <= bmax) {
        int aa = (a < b) ? a : b, bb = (a < b) ? b : a;
        int ab = muf_ab_idx(aa, bb, K);
        double w = g_W[W_stride_It + jj * num_ab_pairs * L_max + ab * L_max + l_slot];
        if (w != 0.0) {
          double c_jj = (J1 == J2) ? 1.0 : 2.0;
          double c_ab = (aa == bb) ? 1.0 : 2.0;
          val = 2.0 * c_jj * c_ab * w;
        }
      }
    } else {
      // J2 < J1: cross-type term, factor 4 instead of 2
      int jj = muf_JJ_idx(J2, J1, num_types);
      int bmin = max(0, a - band_width), bmax = min(K - 1, a + band_width);
      if (b >= bmin && b <= bmax) {
        int aa = (a < b) ? a : b, bb = (a < b) ? b : a;
        int ab = muf_ab_idx(aa, bb, K);
        double w = g_W[W_stride_It + jj * num_ab_pairs * L_max + ab * L_max + l_slot];
        if (w != 0.0) {
          double c_ab = (aa == bb) ? 1.0 : 2.0;
          val = 4.0 * c_ab * w;
        }
      }
    }
    ((double*)s_W_eff)[idx] = val;
  }
  __syncthreads();

  // ====== Phase 2+3: Energy + scatter for all L (no barriers needed) ======
  for (int L = 1; L <= L_max; ++L) {
    int start = L * L - 1;
    int ncomp = 2 * L + 1;
    int l_slot = L - 1;

    double C3B_L[17];
    for (int k = 0; k < ncomp; ++k)
      C3B_L[k] = muf_C3B[start + k];

    // Phase 2: Energy (register-hoisted, reads W directly from global)
    int W_stride_I = I * num_JJ_pairs * num_ab_pairs * L_max;
    for (int J1 = 0; J1 < num_types; ++J1) {
      int J1_base = J1 * K * num_sh_terms;
      for (int J2 = J1; J2 < num_types; ++J2) {
        int jj = muf_JJ_idx(J1, J2, num_types);
        double c_jj = (J1 == J2) ? 1.0 : 2.0;
        int W_jj_base = W_stride_I + jj * num_ab_pairs * L_max;
        int J2_base = J2 * K * num_sh_terms;
        for (int a = 0; a < K; ++a) {
          double A_J1a_w[17];
          int off_J1a = J1_base + a * num_sh_terms + start;
          for (int k = 1; k < ncomp; ++k)
            A_J1a_w[k] = C3B_L[k] * (double)A_i[off_J1a + k];
          for (int b = max(a, a - band_width);
               b <= min(K - 1, a + band_width); ++b) {
            int aa = (a < b) ? a : b, bb = (a < b) ? b : a;
            if (aa != a) continue;
            int ab = muf_ab_idx(a, b, K);
            double W_val = g_W[W_jj_base + ab * L_max + l_slot];
            if (W_val == 0.0) continue;
            double c_ab = (a == b) ? 1.0 : 2.0;
            double S = 0.0;
            int off_J2b = J2_base + b * num_sh_terms + start;
            for (int k = 1; k < ncomp; ++k)
              S += A_J1a_w[k] * (double)A_i[off_J2b + k];
            e3_i += W_val * c_jj * c_ab * S;
          }
        }
      }
    }

    // Phase 3: Scatter dE_dA for this L (uses ALL-L s_W_eff from shared memory)
    for (int J2 = 0; J2 < num_types; ++J2) {
      int J2_base = J2 * K * num_sh_terms;
      for (int b = 0; b < K; ++b) {
        int off_J2b = J2_base + b * num_sh_terms + start;
        for (int k = 1; k < ncomp; ++k) {
          double A_val = C3B_L[k] * (double)A_i[off_J2b + k];
          if (A_val == 0.0) continue;
          for (int J1 = 0; J1 < num_types; ++J1) {
            int dA_J1_base = J1 * K * num_sh_terms;
            for (int a = 0; a < K; ++a) {
              double we = s_W_eff[l_slot][I][J1][a][J2][b];
              if (we != 0.0)
                dE_dA_all[dA_J1_base + a * num_sh_terms + start + k] += (float)(we * A_val);
            }
          }
        }
      }
    }
  }  // end of L loop

  // ====== Single neighbor pass: force backprop for ALL L ======
  // V9: Accumulate atom i's force in registers (3 atomicAdd at end, not per-neighbor)
  double fx_i = 0.0, fy_i = 0.0, fz_i = 0.0;
  // n_neigh already declared in Phase 0 (V9: fused descriptor)
  for (int n = 0; n < n_neigh; ++n) {
    int j = g_NL[i * MN + n];
    double dx = xi - g_x[j];
    double dy = yi - g_y[j];
    double dz = zi - g_z[j];
    double r2 = dx*dx + dy*dy + dz*dz;
    double r = sqrt(r2);
    if (r >= rc_3b || r < 1e-12) continue;

    double rinv = 1.0 / r;
    // Precompute FP32 unit vector + rinv for fast SH backprop
    float rinv_f = (float)rinv;
    float r12_f[3] = {(float)(dx * rinv), (float)(dy * rinv), (float)(dz * rinv)};

    double btilde[4], dbtilde[4];
    int p0;
    muf_eval_bspline_smooth_both(r, r_min_3b, rc_3b, inv_kdelta_3b, K,
                                  btilde, dbtilde, p0);

    int J = g_type[j];
    int dA_J_off = J * K * num_sh_terms;
    double f12_total[3] = {0.0, 0.0, 0.0};

    for (int p = 0; p < 4; ++p) {
      int a = p0 + p;
      if (a >= K) continue;
      float fn_f = (float)btilde[p], fnp_f = (float)dbtilde[p];
      const float* dA_aJ = dE_dA_all + dA_J_off + a * num_sh_terms;

      for (int L = 1; L <= L_max; ++L) {
        int start = L * L - 1;
        int ncomp = 2 * L + 1;
        float s_L_f[17];
        for (int kk = 0; kk < ncomp; ++kk)
          s_L_f[kk] = dA_aJ[start + kk];

        float f12_f[3] = {0.0f, 0.0f, 0.0f};
        switch (L) {
          case 1: muf_accumulate_f12_one_f32<1>(rinv_f, fn_f, fnp_f, s_L_f, r12_f, f12_f); break;
          case 2: muf_accumulate_f12_one_f32<2>(rinv_f, fn_f, fnp_f, s_L_f, r12_f, f12_f); break;
          case 3: muf_accumulate_f12_one_f32<3>(rinv_f, fn_f, fnp_f, s_L_f, r12_f, f12_f); break;
          case 4: muf_accumulate_f12_one_f32<4>(rinv_f, fn_f, fnp_f, s_L_f, r12_f, f12_f); break;
        }
        f12_total[0] += (double)f12_f[0];
        f12_total[1] += (double)f12_f[1];
        f12_total[2] += (double)f12_f[2];
      }
    }

    // Accumulate atom i's force in registers (one atomicAdd at end)
    fx_i -= f12_total[0];
    fy_i -= f12_total[1];
    fz_i -= f12_total[2];
    // Atom j still needs per-neighbor atomicAdd (other threads may target j)
    atomicAdd(&g_fx[j], f12_total[0]);
    atomicAdd(&g_fy[j], f12_total[1]);
    atomicAdd(&g_fz[j], f12_total[2]);

    virial_xx += -f12_total[0] * dx;
    virial_yy += -f12_total[1] * dy;
    virial_zz += -f12_total[2] * dz;
    virial_xy += -f12_total[0] * dy;
    virial_xz += -f12_total[0] * dz;
    virial_yz += -f12_total[1] * dz;
  }

  // One atomicAdd for atom i (instead of per-neighbor)
  atomicAdd(&g_fx[i], fx_i);
  atomicAdd(&g_fy[i], fy_i);
  atomicAdd(&g_fz[i], fz_i);

  // Non-atomic stores: one thread per atom i (safe for virial and potential)
  g_potential[i] += e3_i + g_e0[I];
  g_virial[i * 9 + 0] += virial_xx;
  g_virial[i * 9 + 1] += virial_yy;
  g_virial[i * 9 + 2] += virial_zz;
  g_virial[i * 9 + 3] += virial_xy;
  g_virial[i * 9 + 4] += virial_xz;
  g_virial[i * 9 + 5] += virial_yz;
  g_virial[i * 9 + 6] += virial_xy;
  g_virial[i * 9 + 7] += virial_xz;
  g_virial[i * 9 + 8] += virial_yz;
}

// ============================================================================
// Kernel: MUF-C 3B force + virial from dE/dA (for testing/fallback)
// ============================================================================
static __global__ void muf_kernel_3b_force(
    const double* __restrict__ g_x,
    const double* __restrict__ g_y,
    const double* __restrict__ g_z,
    const int* __restrict__ g_type,
    int N, int num_types,
    const int* __restrict__ g_NN,
    const int* __restrict__ g_NL,
    int MN,
    double rc_3b, double r_min_3b, double inv_kdelta_3b, int K, int L_max,
    int num_sh_terms,
    const double* __restrict__ g_dE_dA,
    double* __restrict__ g_fx,
    double* __restrict__ g_fy,
    double* __restrict__ g_fz,
    double* __restrict__ g_virial)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  int stride = num_types * K * num_sh_terms;
  const double* dA_i = g_dE_dA + i * stride;
  int n_neigh = g_NN[i];
  double xi = g_x[i], yi = g_y[i], zi = g_z[i];
  double virial_xx = 0.0, virial_yy = 0.0, virial_zz = 0.0;
  double virial_xy = 0.0, virial_xz = 0.0, virial_yz = 0.0;

  for (int k = 0; k < n_neigh; ++k) {
    int j = g_NL[i * MN + k];
    double dx = xi - g_x[j];
    double dy = yi - g_y[j];
    double dz = zi - g_z[j];
    double r2 = dx*dx + dy*dy + dz*dz;
    double r = sqrt(r2);
    if (r >= rc_3b || r < 1e-12) continue;

    double rinv = 1.0 / r;
    double btilde[4], dbtilde[4];
    int p0;
    muf_eval_bspline_smooth(r, r_min_3b, rc_3b, inv_kdelta_3b, K, btilde, p0);
    muf_eval_bspline_smooth_deriv(r, r_min_3b, rc_3b, inv_kdelta_3b, K, dbtilde, p0);

    double r12[3] = {dx * rinv, dy * rinv, dz * rinv};
    int J = g_type[j];
    double f12_total[3] = {0.0, 0.0, 0.0};

    for (int p = 0; p < 4; ++p) {
      int a = p0 + p;
      if (a >= K) continue;
      double fn = btilde[p], fnp = dbtilde[p];

      // Access dE_dA for type channel J, B-spline a
      const double* dA_aJ = dA_i + J * K * num_sh_terms + a * num_sh_terms;

      for (int L = 1; L <= L_max; ++L) {
        int start = L * L - 1;
        int ncomp = 2 * L + 1;
        double s_L[17] = {0.0};
        for (int kk = 0; kk < ncomp; ++kk)
          s_L[kk] = dA_aJ[start + kk];

        double f12[3] = {0.0, 0.0, 0.0};
        switch (L) {
          case 1: muf_accumulate_f12_one<1>(rinv, fn, fnp, s_L, r12, f12); break;
          case 2: muf_accumulate_f12_one<2>(rinv, fn, fnp, s_L, r12, f12); break;
          case 3: muf_accumulate_f12_one<3>(rinv, fn, fnp, s_L, r12, f12); break;
          case 4: muf_accumulate_f12_one<4>(rinv, fn, fnp, s_L, r12, f12); break;
        }
        f12_total[0] += f12[0];
        f12_total[1] += f12[1];
        f12_total[2] += f12[2];
      }
    }

    // Force: Newton's 3rd law (dA_i contributes to f_i → -f12, f_j → +f12)
    atomicAdd(&g_fx[i], -f12_total[0]);
    atomicAdd(&g_fy[i], -f12_total[1]);
    atomicAdd(&g_fz[i], -f12_total[2]);
    atomicAdd(&g_fx[j], f12_total[0]);
    atomicAdd(&g_fy[j], f12_total[1]);
    atomicAdd(&g_fz[j], f12_total[2]);

    // 3B virial
    virial_xx += -f12_total[0] * dx;
    virial_yy += -f12_total[1] * dy;
    virial_zz += -f12_total[2] * dz;
    virial_xy += -f12_total[0] * dy;
    virial_xz += -f12_total[0] * dz;
    virial_yz += -f12_total[1] * dz;
  }

  atomicAdd(&g_virial[i * 9 + 0], virial_xx);
  atomicAdd(&g_virial[i * 9 + 1], virial_yy);
  atomicAdd(&g_virial[i * 9 + 2], virial_zz);
  atomicAdd(&g_virial[i * 9 + 3], virial_xy);
  atomicAdd(&g_virial[i * 9 + 4], virial_xz);
  atomicAdd(&g_virial[i * 9 + 5], virial_yz);
  atomicAdd(&g_virial[i * 9 + 6], virial_xy);
  atomicAdd(&g_virial[i * 9 + 7], virial_xz);
  atomicAdd(&g_virial[i * 9 + 8], virial_yz);
}

// ============================================================================
// Kernel: Add 3B energy + e0 to per-atom potential
// ============================================================================
static __global__ void muf_kernel_add_energy(
    int N, int num_types,
    const int* __restrict__ g_type,
    const double* __restrict__ g_e0,      // [num_types]
    const double* __restrict__ g_energy_3b,
    double* __restrict__ g_potential)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  int I = g_type[i];
  g_potential[i] += g_energy_3b[i] + g_e0[I];
}

// ============================================================================
// MUF class implementation
// ============================================================================

MUF::MUF(FILE* fid, const int num_types, const int number_of_atoms)
{
  fprintf(stderr, "MUF constructor START: num_types=%d N=%d\n", num_types, number_of_atoms);
  number_of_atoms_ = number_of_atoms;
  parse_model_file(fid, num_types);

  // Set maximum cutoff for neighbor list
  rc = std::max(param.rc_2b, param.rc_3b);

  // Initialize GPU neighbor list (same as NEP)
  neighbor.initialize(rc, number_of_atoms, param.MN);

  // Allocate MUF-C GPU buffers
  int moments_size = number_of_atoms * num_types * param.K * param.num_sh_terms;

  muf_data.moments.resize(moments_size);
  muf_data.dE_dA.resize(moments_size);
  muf_data.energy_3b.resize(number_of_atoms);

  printf("MUF-C initialized: N=%d, K=%d, L_max=%d, num_types=%d, rc=%.2f\n",
         number_of_atoms, param.K, param.L_max, num_types, rc);
}

void MUF::parse_model_file(FILE* fid, int num_types)
{
  char line[4096];
  param.num_types = num_types;
  param.num_JJ_pairs = num_types * (num_types + 1) / 2;
  param.num_ab_pairs = param.K > 0 ? param.K * (param.K + 1) / 2 : 0;

  // Skip header line: "muf <num_types> <elem1> ..."
  if (!fgets(line, sizeof(line), fid)) {
    PRINT_INPUT_ERROR("MUF: cannot read header");
  }

  // 1B line: "1B <e0_1> <e0_2> ..."
  if (!fgets(line, sizeof(line), fid)) {
    PRINT_INPUT_ERROR("MUF: cannot read 1B line");
  }
  {
    std::vector<double> e0_host(num_types);
    char* p = line;
    while (*p && *p != ' ') ++p;  // skip "1B"
    for (int t = 0; t < num_types; ++t) {
      while (*p == ' ') ++p;
      e0_host[t] = strtod(p, &p);
    }
    param.e0.resize(num_types);
    param.e0.copy_from_host(e0_host.data());
  }

  // 2B header: "2B <rc_2b> <r_min_2b> <K>"
  if (!fgets(line, sizeof(line), fid)) {
    PRINT_INPUT_ERROR("MUF: cannot read 2B header");
  }
  {
    int K_2b;
    sscanf(line, "2B %lf %lf %d", &param.rc_2b, &param.r_min_2b, &K_2b);
    param.K = K_2b;
    param.kdelta_2b = (param.rc_2b - param.r_min_2b) / (param.K - 3);
    param.inv_kdelta_2b = 1.0 / param.kdelta_2b;
  }

  // 2B coefficients: per JJ-pair, K values each, with comment lines
  int num_JJ = num_types * (num_types + 1) / 2;
  std::vector<double> coeff_2b_host(num_JJ * param.K);
  for (int I = 0; I < num_types; ++I) {
    for (int J = I; J < num_types; ++J) {
      int jj = I * (2 * num_types - I - 1) / 2 + J;
      // Skip comment line "# Pair (I,J)"
      if (!fgets(line, sizeof(line), fid)) {
        PRINT_INPUT_ERROR("MUF: cannot read 2B pair comment");
      }
      for (int a = 0; a < param.K; ++a) {
        if (!fgets(line, sizeof(line), fid)) {
          PRINT_INPUT_ERROR("MUF: cannot read 2B coeff");
        }
        coeff_2b_host[jj * param.K + a] = strtod(line, nullptr);
      }
    }
  }
  param.coeff_2b.resize(num_JJ * param.K);
  param.coeff_2b.copy_from_host(coeff_2b_host.data());

  // Skip "#" separator
  if (!fgets(line, sizeof(line), fid)) {
    PRINT_INPUT_ERROR("MUF: cannot read separator");
  }

  // 3B header: "3B <rc_3b> <r_min_3b> <K> <L_max> <band_width>"
  if (!fgets(line, sizeof(line), fid)) {
    PRINT_INPUT_ERROR("MUF: cannot read 3B header");
  }
  {
    int K_3b, L_max, band_width;
    sscanf(line, "3B %lf %lf %d %d %d",
           &param.rc_3b, &param.r_min_3b, &K_3b, &L_max, &band_width);
    param.L_max = L_max;
    param.num_sh_terms = (L_max + 1) * (L_max + 1) - 1;
    param.band_width = band_width;
    param.kdelta_3b = (param.rc_3b - param.r_min_3b) / (K_3b - 3);
    param.inv_kdelta_3b = 1.0 / param.kdelta_3b;
    param.num_ab_pairs = param.K * (param.K + 1) / 2;
  }

  // 3B weights: [num_types * num_JJ_pairs * num_ab_pairs * L_max]
  int nW = num_types * num_JJ * param.num_ab_pairs * param.L_max;
  std::vector<double> W_host(nW);
  for (int i = 0; i < nW; ++i) {
    if (!fgets(line, sizeof(line), fid)) {
      PRINT_INPUT_ERROR("MUF: cannot read 3B weight");
    }
    W_host[i] = strtod(line, nullptr);
  }
  param.W.resize(nW);
  param.W.copy_from_host(W_host.data());

  printf("MUF model: K=%d L_max=%d num_sh=%d rc_2b=%.2f rc_3b=%.2f bw=%d\n",
         param.K, param.L_max, param.num_sh_terms,
         param.rc_2b, param.rc_3b, param.band_width);
  printf("  e0: [%d], 2B coeffs: [%d], 3B W: [%d]\n",
         num_types, num_JJ * param.K, nW);
}

void MUF::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int N = number_of_atoms_;

  // Profile events
  static bool first_call = true;
  static cudaEvent_t evt_nl_start, evt_nl_stop, evt_2b_stop, evt_desc_stop,
                     evt_cont_stop, evt_force_stop, evt_add_stop;
  if (first_call) {
    cudaEventCreate(&evt_nl_start); cudaEventCreate(&evt_nl_stop);
    cudaEventCreate(&evt_2b_stop); cudaEventCreate(&evt_desc_stop);
    cudaEventCreate(&evt_cont_stop); cudaEventCreate(&evt_force_stop);
    cudaEventCreate(&evt_add_stop);
    first_call = false;
  }

  // Step 0: Build GPU neighbor list (cell-list based, same as NEP)
  cudaEventRecord(evt_nl_start);
  neighbor.find_neighbor_global(rc, box, type, position_per_atom);
  cudaEventRecord(evt_nl_stop);

  const double* g_x = position_per_atom.data();
  const double* g_y = position_per_atom.data() + N;
  const double* g_z = position_per_atom.data() + 2 * N;
  const int* g_type = type.data();  // already on GPU

  const int block_size = 128;
  const int grid_size = (N + block_size - 1) / block_size;

  // Zero outputs
  force_per_atom.fill(0.0);
  virial_per_atom.fill(0.0);
  potential_per_atom.fill(0.0);
  muf_data.moments.fill(0.0);

  // Step 1: 2B energy + force + virial
  muf_kernel_2b<<<grid_size, block_size>>>(
    g_x, g_y, g_z, g_type, N, param.num_types,
    neighbor.NN.data(), neighbor.NL.data(), param.MN,
    param.rc_2b, param.r_min_2b, param.inv_kdelta_2b, param.K,
    param.coeff_2b.data(),
    potential_per_atom.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + 2 * N,
    virial_per_atom.data());
  cudaEventRecord(evt_2b_stop);

  // Step 2: 3B descriptors+contract+force FUSED (V9: A_i in local memory)
  cudaEventRecord(evt_desc_stop);  // V9: descriptor fused inline

  // 3B contract + force + energy FUSED (V7: all-L W_eff, full parallelism)
  // W_eff in shared memory, dE_dA accumulated across all L,
  // single neighbor pass. Eliminates dE_dA+energy_3b global arrays.
  muf_kernel_3b_contract_force_v7<<<grid_size, block_size>>>(
    g_x, g_y, g_z, g_type, N, param.num_types,
    neighbor.NN.data(), neighbor.NL.data(), param.MN,
    param.rc_3b, param.r_min_3b, param.inv_kdelta_3b,
    param.K, param.L_max, param.num_sh_terms,
    param.num_JJ_pairs, param.num_ab_pairs, param.band_width,
    param.W.data(), param.e0.data(),
    potential_per_atom.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + 2 * N,
    virial_per_atom.data());
  cudaEventRecord(evt_cont_stop);
  cudaEventRecord(evt_force_stop);  // fused: no-op
  cudaEventRecord(evt_add_stop);    // fused: no-op

  GPU_CHECK_KERNEL

  // Print profile every 100 calls
  static int call_count = 0;
  if (++call_count % 100 == 1) {
    cudaEventSynchronize(evt_add_stop);
    float t_nl, t_2b, t_desc, t_cont, t_force, t_add;
    cudaEventElapsedTime(&t_nl, evt_nl_start, evt_nl_stop);
    cudaEventElapsedTime(&t_2b, evt_nl_stop, evt_2b_stop);
    cudaEventElapsedTime(&t_desc, evt_2b_stop, evt_desc_stop);
    cudaEventElapsedTime(&t_cont, evt_desc_stop, evt_cont_stop);
    cudaEventElapsedTime(&t_force, evt_cont_stop, evt_force_stop);
    cudaEventElapsedTime(&t_add, evt_force_stop, evt_add_stop);
    float total = t_nl + t_2b + t_desc + t_cont + t_force + t_add;
    fprintf(stderr, "\n=== MUF GPU Profile (ms) ===\n");
    fprintf(stderr, "  Neighbor:   %7.3f ms (%5.1f%%)\n", t_nl, 100*t_nl/total);
    fprintf(stderr, "  2B kernel:  %7.3f ms (%5.1f%%)\n", t_2b, 100*t_2b/total);
    fprintf(stderr, "  Descriptor: %7.3f ms (%5.1f%%)\n", t_desc, 100*t_desc/total);
    fprintf(stderr, "  Cont+Force+Add(fused): %7.3f ms (%5.1f%%)\n", t_cont, 100*t_cont/total);
    fprintf(stderr, "  Total GPU:  %7.3f ms\n", total);
  }
}
