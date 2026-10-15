// SPDX-License-Identifier: GPL-3.0-or-later
/* Analytical derivatives of the NEP q_222 and q_1111 invariants.
   Included after utilities/gnep_utilities.cuh. */
#pragma once

namespace gnep_higher_body {

// These coefficients and harmonic ordering match utilities/nep_utilities.cuh.
constexpr float c40 = -0.007499480826664f;
constexpr float c41 = -0.134990654879954f;
constexpr float c42 = 0.067495327439977f;
constexpr float c43 = 0.404971964639861f;
constexpr float c44 = -0.809943929279723f;
constexpr float c50 = 0.026596810706114f;
constexpr float c51 = 0.053193621412227f;
constexpr float c52 = 0.026596810706114f;

struct Direction {
  float value;
  float derivative[3];
  __device__ Direction(float v = 0.0f) : value(v), derivative{0.0f, 0.0f, 0.0f} {}
};

static __device__ __forceinline__ Direction operator+(const Direction& a, const Direction& b)
{
  Direction result(a.value + b.value);
  for (int d = 0; d < 3; ++d) result.derivative[d] = a.derivative[d] + b.derivative[d];
  return result;
}

static __device__ __forceinline__ Direction operator-(const Direction& a, const Direction& b)
{
  Direction result(a.value - b.value);
  for (int d = 0; d < 3; ++d) result.derivative[d] = a.derivative[d] - b.derivative[d];
  return result;
}

static __device__ __forceinline__ Direction operator*(const Direction& a, const Direction& b)
{
  Direction result(a.value * b.value);
  for (int d = 0; d < 3; ++d)
    result.derivative[d] = a.derivative[d] * b.value + a.value * b.derivative[d];
  return result;
}

template <typename T>
static __device__ __forceinline__ void gradient(const int body, const T* s, T* g)
{
  for (int a = 0; a < 8; ++a) g[a] = T(0.0f);
  if (body == 4) {
    g[3] = T(3.0f * c40) * s[3] * s[3] + T(c41) * (s[4] * s[4] + s[5] * s[5]) +
           T(c42) * (s[6] * s[6] + s[7] * s[7]);
    g[4] = T(2.0f * c41) * s[3] * s[4] - T(2.0f * c43) * s[6] * s[4] + T(c44) * s[5] * s[7];
    g[5] = T(2.0f * c41) * s[3] * s[5] + T(2.0f * c43) * s[6] * s[5] + T(c44) * s[4] * s[7];
    g[6] = T(2.0f * c42) * s[3] * s[6] + T(c43) * (s[5] * s[5] - s[4] * s[4]);
    g[7] = T(2.0f * c42) * s[3] * s[7] + T(c44) * s[4] * s[5];
  } else {
    T transverse = s[1] * s[1] + s[2] * s[2];
    g[0] = T(4.0f * c50) * s[0] * s[0] * s[0] + T(2.0f * c51) * s[0] * transverse;
    g[1] = T(2.0f * c51) * s[0] * s[0] * s[1] + T(4.0f * c52) * transverse * s[1];
    g[2] = T(2.0f * c51) * s[0] * s[0] * s[2] + T(4.0f * c52) * transverse * s[2];
  }
}

static __device__ __forceinline__ void values(
  const GNEP::ParaMB& p, const int n, const float* s, float* q)
{
  const int stride = p.n_max_angular + 1;
  if (p.L_max_4body == 2) {
    q[p.L_max * stride + n] =
      c40 * s[3] * s[3] * s[3] + c41 * s[3] * (s[4] * s[4] + s[5] * s[5]) +
      c42 * s[3] * (s[6] * s[6] + s[7] * s[7]) + c43 * s[6] * (s[5] * s[5] - s[4] * s[4]) +
      c44 * s[4] * s[5] * s[7];
  }
  if (p.L_max_5body == 1) {
    const float longitudinal = s[0] * s[0];
    const float transverse = s[1] * s[1] + s[2] * s[2];
    q[(p.L_max + 1) * stride + n] =
      c50 * longitudinal * longitudinal + c51 * longitudinal * transverse + c52 * transverse * transverse;
  }
}

static __device__ __forceinline__ void load_s(const int N, const int n, const float* sum, float* s)
{
  for (int a = 0; a < 8; ++a) s[a] = sum[(n * NUM_OF_ABC + a) * N];
}

static __device__ __forceinline__ void edge(
  const float distance, const float* r, const float fn, const float fnp, float* b, float* db)
{
  const float inverse = 1.0f / distance;
  const float unit[3] = {r[0] * inverse, r[1] * inverse, r[2] * inverse};
  // The existing harmonic evaluator returns B_a and its Cartesian derivative.
  calculate_s_i1_one<1>(inverse, fn, fnp, b, unit, b, db);
  calculate_s_i1_one<2>(inverse, fn, fnp, b, unit, b, db);
}

static __device__ __forceinline__ void qc(
  const GNEP::ParaMB& p, const int N, const int channel, const int n,
  const float distance, const float* r, const float fn, const float* sum, float* result)
{
  if (channel < p.L_max) {
    accumulate_qc(N, channel + 1, n, p.n_max_angular + 1, p.basis_size_angular + 1,
                  distance, r, fn, sum, result);
    return;
  }
  float s[8], b[8], db[24], g[8];
  load_s(N, n, sum, s);
  edge(distance, r, fn, 0.0f, b, db);
  gradient(channel == p.L_max ? 4 : 5, s, g);
  *result = 0.0f;
  for (int a = 0; a < 8; ++a) *result += g[a] * b[a];
}

static __device__ __forceinline__ void feature_derivatives(
  const GNEP::ParaMB& p, const int N, const int n, const float distance, const float* r,
  const float gn, const float gnp, const float* sum, float* fx, float* fy, float* fz)
{
  if (p.num_L == p.L_max) return;
  float s[8], b[8], db[24], g[8];
  load_s(N, n, sum, s);
  edge(distance, r, gn, gnp, b, db);
  for (int l = p.L_max; l < p.num_L; ++l) {
    gradient(l == p.L_max ? 4 : 5, s, g);
    const int offset = l * (p.n_max_angular + 1);
    fx[offset] = fy[offset] = fz[offset] = 0.0f;
    for (int a = 0; a < 8; ++a) {
      fx[offset] += g[a] * db[3 * a];
      fy[offset] += g[a] * db[3 * a + 1];
      fz[offset] += g[a] * db[3 * a + 2];
    }
  }
}

static __device__ __forceinline__ float energy_derivative(
  const GNEP::ParaMB& p, const int N, const int n, const float distance, const float* r,
  const float fn, const float* sum, const float* fp)
{
  float result = 0.0f;
  for (int l = p.L_max; l < p.num_L; ++l) {
    float q_c;
    qc(p, N, l, n, distance, r, fn, sum, &q_c);
    result += fp[l * (p.n_max_angular + 1) + n] * q_c;
  }
  return result;
}

static __device__ __forceinline__ void mixed_derivative(
  const GNEP::ParaMB& p, const int N, const int n, const float distance_i, const float* r_i,
  const float gn_i, const float gnp_i, const float distance_j, const float* r_j,
  const float fn_j, const float fnp_j, const bool same_edge, const float* sum,
  const float* fp, float* result)
{
  result[0] = result[1] = result[2] = 0.0f;
  if (p.num_L == p.L_max) return;
  float s[8], bi[8], dbi[24], bj[8], dbj[24];
  load_s(N, n, sum, s);
  edge(distance_i, r_i, gn_i, gnp_i, bi, dbi);
  edge(distance_j, r_j, fn_j, fnp_j, bj, dbj);
  Direction directional_s[8], directional_g[8];
  for (int a = 0; a < 8; ++a) {
    directional_s[a].value = s[a];
    for (int d = 0; d < 3; ++d) directional_s[a].derivative[d] = dbi[3 * a + d];
  }
  // d(q_r_i)/dc_j = Hess(q)[B'_i, B_j] + delta_ij grad(q).B'_j.
  for (int l = p.L_max; l < p.num_L; ++l) {
    gradient(l == p.L_max ? 4 : 5, directional_s, directional_g);
    const float weight = fp[l * (p.n_max_angular + 1) + n];
    for (int a = 0; a < 8; ++a) {
      for (int d = 0; d < 3; ++d) {
        result[d] += weight * (directional_g[a].derivative[d] * bj[a] +
          (same_edge ? directional_g[a].value * dbj[3 * a + d] : 0.0f));
      }
    }
  }
}

static __device__ __forceinline__ void force(
  const GNEP::ParaMB& p, const int N, const int n, const float distance, const float* r,
  const float gn, const float gnp, const float* sum, const float* fp, float* result)
{
  if (p.num_L == p.L_max) return;
  float s[8], b[8], db[24], g[8];
  load_s(N, n, sum, s);
  edge(distance, r, gn, gnp, b, db);
  for (int l = p.L_max; l < p.num_L; ++l) {
    gradient(l == p.L_max ? 4 : 5, s, g);
    const float weight = fp[l * (p.n_max_angular + 1) + n];
    for (int a = 0; a < 8; ++a)
      for (int d = 0; d < 3; ++d) result[d] += weight * g[a] * db[3 * a + d];
  }
}

} // namespace gnep_higher_body
