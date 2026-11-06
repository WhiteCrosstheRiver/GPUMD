// SPDX-License-Identifier: GPL-3.0-or-later

/*----------------------------------------------------------------------------80
Moment-space Jacobian rows for the native GNEP parameter layout.
The harmonic polynomials, invariant coefficients, and radial basis follow
utilities/gnep_utilities.cuh.
------------------------------------------------------------------------------*/

#pragma once
#include "utilities/common.cuh"
#include "utilities/gnep_utilities.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

namespace lm
{

constexpr int MAX_TYPES = 94;
constexpr int MAX_DIM = ::MAX_DIM;
constexpr int MAX_NEURON = 200;
constexpr int MAX_KR = ::MAX_NUM_N;
constexpr int MAX_KA = ::MAX_NUM_N;
constexpr int MAX_LM = 80;
__device__ __constant__ float C4B[5] = {
  -0.007499480826664f,
  -0.134990654879954f,
  0.067495327439977f,
  0.404971964639861f,
  -0.809943929279723f};
__device__ __constant__ float C5B[3] = {0.026596810706114f, 0.053193621412227f, 0.026596810706114f};

struct Dual {
  float v;
  float d;
  __host__ __device__ Dual(float value = 0.0f, float derivative = 0.0f) : v(value), d(derivative)
  {
  }
};

__host__ __device__ inline Dual operator+(Dual a, Dual b)
{
  return {a.v + b.v, a.d + b.d};
}

__host__ __device__ inline Dual operator-(Dual a, Dual b)
{
  return {a.v - b.v, a.d - b.d};
}

__host__ __device__ inline Dual operator*(Dual a, Dual b)
{
  return {a.v * b.v, a.d * b.v + a.v * b.d};
}

__host__ __device__ inline Dual operator/(Dual a, Dual b)
{
  return {a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v)};
}

__host__ __device__ inline Dual& operator+=(Dual& a, Dual b)
{
  a = a + b;
  return a;
}

__host__ __device__ inline Dual sqrt(Dual a)
{
  const float value = sqrtf(a.v);
  return {value, 0.5f * a.d / value};
}

struct Arch {
  int num_types = 0;
  int dim = 0;
  int dim_R = 0;
  int n_R = 0;
  int n_A = 0;
  int K_R = 0;
  int K_A = 0;
  int n_lm = 0;
  int neurons = 0;
  float rc_R[MAX_TYPES] = {};
  float rc_A[MAX_TYPES] = {};
  int L_max = 0;
  int L_max_4body = 0;
  int L_max_5body = 0;
};

inline void validate_arch(const Arch& a)
{
  const int channels = a.L_max + (a.L_max_4body == 2) + (a.L_max_5body == 1);
  if (
    a.num_types < 1 || a.num_types > MAX_TYPES || a.dim < 1 || a.dim > MAX_DIM || a.neurons < 1 ||
    a.neurons > MAX_NEURON || a.n_R < 1 || a.n_R > 20 || a.n_A < 1 || a.n_A > 20 || a.K_R < 1 ||
    a.K_R > MAX_KR || a.K_A < 1 || a.K_A > MAX_KA || a.L_max < 0 || a.L_max > 8 ||
    (a.L_max_4body != 0 && a.L_max_4body != 2) || a.L_max_4body > a.L_max ||
    (a.L_max_5body != 0 && a.L_max_5body != 1) || (a.L_max_5body && !a.L_max_4body) ||
    a.dim_R != a.n_R || a.dim != a.n_R + channels * a.n_A || a.n_lm != a.L_max * (a.L_max + 2)) {
    throw std::invalid_argument("Invalid or unsupported native NEP-LM architecture.");
  }
  for (int t = 0; t < a.num_types; ++t) {
    if (!(a.rc_R[t] > 0.0f) || !(a.rc_A[t] > 0.0f)) {
      throw std::invalid_argument("NEP-LM cutoffs must be positive.");
    }
  }
}

struct Shard {
  int N;
  int num_structures;
  const int* type;
  const int* structure_of;
  const int* atom_start;
  const int* NN_R;
  const int* NL_R;
  const float* x12R;
  const float* y12R;
  const float* z12R;
  const int* NN_A;
  const int* NL_A;
  const float* x12A;
  const float* y12A;
  const float* z12A;
  const int* global_atom_ids = nullptr;
};

struct ANNView {
  const float* parameters;
  const float* cR;
  const float* cA;
};

__host__ __device__ inline int ann_block_size(const Arch& a)
{
  return (a.dim + 2) * a.neurons + 1;
}

inline ANNView make_ann_view(const Arch& a, const float* parameters)
{
  const int ann_count = a.num_types * ann_block_size(a);
  const int radial_count = a.n_R * a.K_R * a.num_types * a.num_types;
  return {parameters, parameters + ann_count, parameters + ann_count + radial_count};
}

__host__ __device__ inline size_t phi_index(const Arch& a, int i, int tj, int k)
{
  return (size_t(i) * a.num_types + tj) * a.K_R + k;
}

__host__ __device__ inline size_t A_index(const Arch& a, int i, int tj, int k, int h)
{
  return ((size_t(i) * a.num_types + tj) * a.K_A + k) * a.n_lm + h;
}

__host__ __device__ inline int c_index(const Arch& a, int ti, int tj, int n, int k, int K)
{
  return (n * K + k) * a.num_types * a.num_types + ti * a.num_types + tj;
}

// Matches GNEP: a pair cutoff is the mean of the two per-element cutoffs.
__device__ inline float cutoff(const Arch& a, int ti, int tj, bool angular)
{
  const float* rc = angular ? a.rc_A : a.rc_R;
  return (rc[ti] + rc[tj]) * 0.5f;
}

__device__ inline void basis(int K, float rc, float distance, float* fn, float* fnp)
{
  float fc;
  float fcp;
  const float inverse = 1.0f / rc;
  ::find_fc_and_fcp(rc, inverse, distance, fc, fcp);
  ::find_fn_and_fnp(K - 1, inverse, distance, fc, fcp, fn, fnp);
}

template <int L, typename T>
__device__ inline void harmonic_one(T x, T y, T z, T* values)
{
  T z_pow[L + 1];
  z_pow[0] = T(1.0f);
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = z * z_pow[n - 1];
  }
  T real_part = x;
  T imag_part = y;
  int index = L * L - 1;
  for (int m = 0; m <= L; ++m) {
    T factor(0.0f);
    for (int n = (L + m) % 2 ? 1 : 0; n <= L - m; n += 2) {
      factor += T(::get_z_coefficient<L>(m, n)) * z_pow[n];
    }
    if (m == 0) {
      values[index++] = factor;
    } else {
      values[index++] = factor * real_part;
      values[index++] = factor * imag_part;
      const T previous = real_part;
      real_part = x * real_part - y * imag_part;
      imag_part = x * imag_part + y * previous;
    }
  }
}

template <typename T>
__device__ inline void harmonics(int L, T x, T y, T z, T* values)
{
  if (L >= 1) {
    harmonic_one<1>(x, y, z, values);
  }
  if (L >= 2) {
    harmonic_one<2>(x, y, z, values);
  }
  if (L >= 3) {
    harmonic_one<3>(x, y, z, values);
  }
  if (L >= 4) {
    harmonic_one<4>(x, y, z, values);
  }
  if (L >= 5) {
    harmonic_one<5>(x, y, z, values);
  }
  if (L >= 6) {
    harmonic_one<6>(x, y, z, values);
  }
  if (L >= 7) {
    harmonic_one<7>(x, y, z, values);
  }
  if (L >= 8) {
    harmonic_one<8>(x, y, z, values);
  }
}

// Output arrays are owned by the atom thread and must initially contain zero.
static __global__ void moments_kernel(Arch a, Shard sh, float* Phi, float* A)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= sh.N) {
    return;
  }
  for (int jj = 0; jj < sh.NN_R[i]; ++jj) {
    const int edge = i + jj * sh.N;
    const int tj = sh.type[sh.NL_R[edge]];
    const float x = sh.x12R[edge];
    const float y = sh.y12R[edge];
    const float z = sh.z12R[edge];
    const float distance = sqrtf(x * x + y * y + z * z);
    float fn[MAX_KR];
    float fnp[MAX_KR];
    basis(a.K_R, cutoff(a, sh.type[i], tj, false), distance, fn, fnp);
    for (int k = 0; k < a.K_R; ++k) {
      Phi[phi_index(a, i, tj, k)] += fn[k];
    }
  }
  for (int jj = 0; jj < sh.NN_A[i]; ++jj) {
    const int edge = i + jj * sh.N;
    const int tj = sh.type[sh.NL_A[edge]];
    const float x = sh.x12A[edge];
    const float y = sh.y12A[edge];
    const float z = sh.z12A[edge];
    const float distance = sqrtf(x * x + y * y + z * z);
    float fn[MAX_KA];
    float fnp[MAX_KA];
    float Y[MAX_LM];
    basis(a.K_A, cutoff(a, sh.type[i], tj, true), distance, fn, fnp);
    harmonics(a.L_max, x / distance, y / distance, z / distance, Y);
    for (int k = 0; k < a.K_A; ++k) {
      for (int h = 0; h < a.n_lm; ++h) {
        A[A_index(a, i, tj, k, h)] += fn[k] * Y[h];
      }
    }
  }
}

__host__ __device__ inline uint64_t splitmix64(uint64_t x)
{
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

__host__ __device__ inline float
rademacher(uint64_t seed, int iter, int atom, int probe, int component)
{
  const uint64_t key =
    seed ^ (uint64_t(iter) << 40) ^ (uint64_t(probe) << 24) ^ (uint64_t(atom) * 3 + component);
  return splitmix64(key) & 1ull ? 1.0f : -1.0f;
}

enum DenseKind { DENSE_RADEMACHER = 0, DENSE_STRAIN = 1, DENSE_RESIDUAL = 2 };

struct DenseProbe {
  int kind;
  int index;
  int structure;
  uint64_t seed;
  int iter;
};

__device__ inline float3 dense_dv(
  const DenseProbe& pr,
  const Shard& sh,
  int i,
  int j,
  float x,
  float y,
  float z,
  const float* residual_field,
  float weight,
  const float* force_weights)
{
  if (pr.kind == DENSE_RADEMACHER) {
    const int key_i = sh.global_atom_ids ? sh.global_atom_ids[i] : i;
    const int key_j = sh.global_atom_ids ? sh.global_atom_ids[j] : j;
    const float wi = force_weights ? force_weights[i] : 1.0f;
    const float wj = force_weights ? force_weights[j] : 1.0f;
    return make_float3(
      wj * rademacher(pr.seed, pr.iter, key_j, pr.index, 0) -
        wi * rademacher(pr.seed, pr.iter, key_i, pr.index, 0),
      wj * rademacher(pr.seed, pr.iter, key_j, pr.index, 1) -
        wi * rademacher(pr.seed, pr.iter, key_i, pr.index, 1),
      wj * rademacher(pr.seed, pr.iter, key_j, pr.index, 2) -
        wi * rademacher(pr.seed, pr.iter, key_i, pr.index, 2));
  }
  if (pr.kind == DENSE_STRAIN) {
    switch (pr.index) {
      case 0:
        return make_float3(x, 0.0f, 0.0f);
      case 1:
        return make_float3(0.0f, y, 0.0f);
      case 2:
        return make_float3(0.0f, 0.0f, z);
      case 3:
        return make_float3(0.5f * y, 0.5f * x, 0.0f);
      case 4:
        return make_float3(0.0f, 0.5f * z, 0.5f * y);
      default:
        return make_float3(0.5f * z, 0.0f, 0.5f * x);
    }
  }
  // The host supplies v_i = residual_i times the complete squared atom weight.
  return make_float3(
    weight * (residual_field[3 * j] - residual_field[3 * i]),
    weight * (residual_field[3 * j + 1] - residual_field[3 * i + 1]),
    weight * (residual_field[3 * j + 2] - residual_field[3 * i + 2]));
}

__device__ inline void pair_tangent(
  const Arch& a, int ti, int tj, float x, float y, float z, float3 dv, bool angular, float* output)
{
  const Dual X(x, dv.x);
  const Dual Y(y, dv.y);
  const Dual Z(z, dv.z);
  const Dual distance = sqrt(X * X + Y * Y + Z * Z);
  const int K = angular ? a.K_A : a.K_R;
  float fn[MAX_KR];
  float fnp[MAX_KR];
  basis(K, cutoff(a, ti, tj, angular), distance.v, fn, fnp);
  if (angular) {
    Dual harmonic[MAX_LM];
    harmonics(a.L_max, X / distance, Y / distance, Z / distance, harmonic);
    for (int k = 0; k < K; ++k) {
      const Dual f(fn[k], fnp[k] * distance.d);
      for (int h = 0; h < a.n_lm; ++h) {
        output[(tj * K + k) * a.n_lm + h] += (f * harmonic[h]).d;
      }
    }
  } else {
    for (int k = 0; k < K; ++k) {
      output[tj * K + k] += fnp[k] * distance.d;
    }
  }
}

static __global__ void dense_tangent_kernel(
  Arch a,
  Shard sh,
  const DenseProbe* probes,
  const int2* records,
  int num_records,
  const float* residual_field,
  float weight,
  float* dPhi,
  float* dA,
  const float* force_weights = nullptr)
{
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= num_records) {
    return;
  }
  const int i = records[r].y;
  const DenseProbe pr = probes[records[r].x];
  float* dphi = dPhi + size_t(r) * a.num_types * a.K_R;
  float* da = dA + size_t(r) * a.num_types * a.K_A * a.n_lm;
  for (int jj = 0; jj < sh.NN_R[i]; ++jj) {
    const int edge = i + jj * sh.N;
    const int j = sh.NL_R[edge];
    const float x = sh.x12R[edge];
    const float y = sh.y12R[edge];
    const float z = sh.z12R[edge];
    const float3 dv = dense_dv(pr, sh, i, j, x, y, z, residual_field, weight, force_weights);
    pair_tangent(a, sh.type[i], sh.type[j], x, y, z, dv, false, dphi);
  }
  for (int jj = 0; jj < sh.NN_A[i]; ++jj) {
    const int edge = i + jj * sh.N;
    const int j = sh.NL_A[edge];
    const float x = sh.x12A[edge];
    const float y = sh.y12A[edge];
    const float z = sh.z12A[edge];
    const float3 dv = dense_dv(pr, sh, i, j, x, y, z, residual_field, weight, force_weights);
    pair_tangent(a, sh.type[i], sh.type[j], x, y, z, dv, true, da);
  }
}

template <typename T>
__device__ inline T invariant(const Arch& a, int channel, const T* s)
{
  if (channel < a.L_max) {
    const int L = channel + 1;
    const int begin = L * L - 1;
    T value(0.0f);
    for (int h = begin + 1; h < begin + 2 * L + 1; ++h) {
      value += T(C3B[h]) * s[h] * s[h];
    }
    return T(2.0f) * value + T(C3B[begin]) * s[begin] * s[begin];
  }
  if (channel == a.L_max) {
    return T(C4B[0]) * s[3] * s[3] * s[3] + T(C4B[1]) * s[3] * (s[4] * s[4] + s[5] * s[5]) +
           T(C4B[2]) * s[3] * (s[6] * s[6] + s[7] * s[7]) +
           T(C4B[3]) * s[6] * (s[5] * s[5] - s[4] * s[4]) + T(C4B[4]) * s[4] * s[5] * s[7];
  }
  const T longitudinal = s[0] * s[0];
  const T transverse = s[1] * s[1] + s[2] * s[2];
  return T(C5B[0]) * longitudinal * longitudinal + T(C5B[1]) * longitudinal * transverse +
         T(C5B[2]) * transverse * transverse;
}

template <typename T>
__device__ inline void invariant_reverse(const Arch& a, int n, const T* s, const float* seed, T* g)
{
  for (int h = 0; h < a.n_lm; ++h) {
    g[h] = T(0.0f);
  }
  for (int L = 1; L <= a.L_max; ++L) {
    const int begin = L * L - 1;
    const float scale = seed[a.n_R + (L - 1) * a.n_A + n];
    for (int h = begin; h < begin + 2 * L + 1; ++h) {
      g[h] += T((h == begin ? 2.0f : 4.0f) * C3B[h] * scale) * s[h];
    }
  }
  if (a.L_max_4body == 2) {
    const T w(seed[a.n_R + a.L_max * a.n_A + n]);
    g[3] += w * (T(3.0f * C4B[0]) * s[3] * s[3] + T(C4B[1]) * (s[4] * s[4] + s[5] * s[5]) +
                 T(C4B[2]) * (s[6] * s[6] + s[7] * s[7]));
    g[4] += w * (T(2.0f * C4B[1]) * s[3] * s[4] - T(2.0f * C4B[3]) * s[6] * s[4] +
                 T(C4B[4]) * s[5] * s[7]);
    g[5] += w * (T(2.0f * C4B[1]) * s[3] * s[5] + T(2.0f * C4B[3]) * s[6] * s[5] +
                 T(C4B[4]) * s[4] * s[7]);
    g[6] += w * (T(2.0f * C4B[2]) * s[3] * s[6] + T(C4B[3]) * (s[5] * s[5] - s[4] * s[4]));
    g[7] += w * (T(2.0f * C4B[2]) * s[3] * s[7] + T(C4B[4]) * s[4] * s[5]);
  }
  if (a.L_max_5body == 1) {
    const T w(seed[a.n_R + (a.L_max + 1) * a.n_A + n]);
    const T transverse = s[1] * s[1] + s[2] * s[2];
    g[0] += w * (T(4.0f * C5B[0]) * s[0] * s[0] * s[0] + T(2.0f * C5B[1]) * s[0] * transverse);
    g[1] += w * (T(2.0f * C5B[1]) * s[0] * s[0] * s[1] + T(4.0f * C5B[2]) * transverse * s[1]);
    g[2] += w * (T(2.0f * C5B[1]) * s[0] * s[0] * s[2] + T(4.0f * C5B[2]) * transverse * s[2]);
  }
}

__device__ inline void angular_s(
  const Arch& a, int ti, int n, const float* coefficients, const float* A, const float* dA, Dual* s)
{
  for (int h = 0; h < a.n_lm; ++h) {
    s[h] = Dual(0.0f);
    for (int tj = 0; tj < a.num_types; ++tj) {
      for (int k = 0; k < a.K_A; ++k) {
        const int index = (tj * a.K_A + k) * a.n_lm + h;
        const float c = coefficients[c_index(a, ti, tj, n, k, a.K_A)];
        s[h] += Dual(c) * Dual(A[index], dA ? dA[index] : 0.0f);
      }
    }
  }
}

__device__ inline void descriptor_and_tangent(
  const Arch& a,
  int ti,
  const ANNView& ann,
  const float* scaler,
  const float* Phi,
  const float* A,
  const float* dPhi,
  const float* dA,
  float* q,
  float* dq)
{
  for (int n = 0; n < a.n_R; ++n) {
    Dual value(0.0f);
    for (int tj = 0; tj < a.num_types; ++tj) {
      for (int k = 0; k < a.K_R; ++k) {
        const int index = tj * a.K_R + k;
        const float c = ann.cR[c_index(a, ti, tj, n, k, a.K_R)];
        value += Dual(c) * Dual(Phi[index], dPhi ? dPhi[index] : 0.0f);
      }
    }
    q[n] = value.v * scaler[n];
    dq[n] = value.d * scaler[n];
  }
  const int channels = a.L_max + (a.L_max_4body == 2) + (a.L_max_5body == 1);
  for (int n = 0; n < a.n_A; ++n) {
    Dual s[MAX_LM];
    angular_s(a, ti, n, ann.cA, A, dA, s);
    for (int channel = 0; channel < channels; ++channel) {
      const int index = a.n_R + channel * a.n_A + n;
      const Dual value = invariant(a, channel, s);
      q[index] = value.v * scaler[index];
      dq[index] = value.d * scaler[index];
    }
  }
}

// Phi/A use atom-major layouts; q output uses the native dimension-major
// layout.
static __global__ void descriptors_kernel(
  Arch a,
  ANNView ann,
  Shard sh,
  const float* scaler,
  const float* Phi,
  const float* A,
  float* output)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= sh.N) {
    return;
  }
  float q[MAX_DIM];
  float dq[MAX_DIM];
  descriptor_and_tangent(
    a,
    sh.type[i],
    ann,
    scaler,
    Phi + size_t(i) * a.num_types * a.K_R,
    A + size_t(i) * a.num_types * a.K_A * a.n_lm,
    nullptr,
    nullptr,
    q,
    dq);
  for (int d = 0; d < a.dim; ++d) {
    output[i + size_t(d) * sh.N] = q[d];
  }
}

// Each record contributes to its row with atomics. Zero rows and values first.
// ENERGY scale = weight/N_s; force scale = -1; strain scale = -weight/N_s.
// ZBL is parameter independent: include it in host residuals, not Jacobian
// rows.
template <bool ENERGY>
static __global__ void rows_kernel(
  Arch a,
  ANNView ann,
  Shard sh,
  const int2* records,
  int num_records,
  const float* row_scale,
  const float* scaler,
  const float* Phi,
  const float* A,
  const float* dPhi,
  const float* dA,
  float* rows,
  int n_params,
  float* values = nullptr)
{
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= num_records) {
    return;
  }
  const int row = records[r].x;
  const int i = records[r].y;
  const int ti = sh.type[i];
  const int block = ann_block_size(a);
  const int offset = ti * block;
  const int offset_b0 = offset + a.neurons * a.dim;
  const int offset_w1 = offset_b0 + a.neurons;
  const int offset_cR = a.num_types * block;
  const int offset_cA = offset_cR + a.n_R * a.K_R * a.num_types * a.num_types;
  const float* w0 = ann.parameters + offset;
  const float* b0 = ann.parameters + offset_b0;
  const float* w1 = ann.parameters + offset_w1;
  const float* phi = Phi + size_t(i) * a.num_types * a.K_R;
  const float* angular = A + size_t(i) * a.num_types * a.K_A * a.n_lm;
  const float* dphi = dPhi ? dPhi + size_t(r) * a.num_types * a.K_R : nullptr;
  const float* dangular = dA ? dA + size_t(r) * a.num_types * a.K_A * a.n_lm : nullptr;
  float* output = rows + size_t(row) * n_params;
  const float scale = row_scale[row];
  float q[MAX_DIM];
  float dq[MAX_DIM];
  float gq[MAX_DIM] = {};
  float gdq[MAX_DIM] = {};
  descriptor_and_tangent(a, ti, ann, scaler, phi, angular, dphi, dangular, q, dq);
  float value = ENERGY ? -w1[a.neurons] : 0.0f;
  for (int k = 0; k < a.neurons; ++k) {
    float z = -b0[k];
    float u = 0.0f;
    for (int d = 0; d < a.dim; ++d) {
      z += w0[k * a.dim + d] * q[d];
      if (!ENERGY) {
        u += w0[k * a.dim + d] * dq[d];
      }
    }
    const float h = tanhf(z);
    const float derivative = 1.0f - h * h;
    const float ws = w1[k] * derivative;
    if (ENERGY) {
      value += w1[k] * h;
      atomicAdd(output + offset_b0 + k, -scale * ws);
      atomicAdd(output + offset_w1 + k, scale * h);
      for (int d = 0; d < a.dim; ++d) {
        atomicAdd(output + offset + k * a.dim + d, scale * ws * q[d]);
        gq[d] += ws * w0[k * a.dim + d];
      }
    } else {
      const float beta = w1[k] * u * (-2.0f * h * derivative);
      value += ws * u;
      atomicAdd(output + offset_b0 + k, -scale * beta);
      atomicAdd(output + offset_w1 + k, scale * derivative * u);
      for (int d = 0; d < a.dim; ++d) {
        atomicAdd(output + offset + k * a.dim + d, scale * (ws * dq[d] + beta * q[d]));
        gq[d] += beta * w0[k * a.dim + d];
        gdq[d] += ws * w0[k * a.dim + d];
      }
    }
  }
  if (ENERGY) {
    atomicAdd(output + offset_w1 + a.neurons, -scale);
  }
  if (values) {
    atomicAdd(values + row, scale * value);
  }
  for (int d = 0; d < a.dim; ++d) {
    gq[d] *= scaler[d];
    gdq[d] *= scaler[d];
  }
  for (int n = 0; n < a.n_R; ++n) {
    for (int tj = 0; tj < a.num_types; ++tj) {
      for (int k = 0; k < a.K_R; ++k) {
        const int index = tj * a.K_R + k;
        const float derivative = gq[n] * phi[index] + (dphi ? gdq[n] * dphi[index] : 0.0f);
        atomicAdd(output + offset_cR + c_index(a, ti, tj, n, k, a.K_R), scale * derivative);
      }
    }
  }
  for (int n = 0; n < a.n_A; ++n) {
    Dual s[MAX_LM];
    Dual reverse_tangent[MAX_LM];
    float reverse_value[MAX_LM];
    float plain_s[MAX_LM];
    angular_s(a, ti, n, ann.cA, angular, dangular, s);
    for (int h = 0; h < a.n_lm; ++h) {
      plain_s[h] = s[h].v;
    }
    invariant_reverse(a, n, plain_s, gq, reverse_value);
    invariant_reverse(a, n, s, gdq, reverse_tangent);
    for (int tj = 0; tj < a.num_types; ++tj) {
      for (int k = 0; k < a.K_A; ++k) {
        float derivative = 0.0f;
        for (int h = 0; h < a.n_lm; ++h) {
          const int index = (tj * a.K_A + k) * a.n_lm + h;
          derivative += (reverse_value[h] + reverse_tangent[h].d) * angular[index] +
                        (dangular ? reverse_tangent[h].v * dangular[index] : 0.0f);
        }
        atomicAdd(output + offset_cA + c_index(a, ti, tj, n, k, a.K_A), scale * derivative);
      }
    }
  }
}

} // namespace lm
