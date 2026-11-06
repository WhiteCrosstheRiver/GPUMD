// SPDX-License-Identifier: GPL-3.0-or-later

/*----------------------------------------------------------------------------80
Per-update atom intermediates for the native NEP-LM Jacobian rows.
The cache belongs to one geometry batch and one parameter vector only.
------------------------------------------------------------------------------*/

#pragma once
#include "nep_lm_kernels.cuh"

namespace lm
{

struct AtomCache {
  float* q;
  float* h;
  float* ws;
  float* basegdq;
  float* angular_s;
};

inline size_t atom_cache_floats(const Arch& a, int num_atoms)
{
  return size_t(num_atoms) * (2 * a.dim + 2 * a.neurons + a.n_A * a.n_lm);
}

inline AtomCache make_atom_cache_view(const Arch& a, int num_atoms, float* storage)
{
  const size_t count_q = size_t(num_atoms) * a.dim;
  const size_t count_h = size_t(num_atoms) * a.neurons;
  return {
    storage,
    storage + count_q,
    storage + count_q + count_h,
    storage + count_q + 2 * count_h,
    storage + 2 * count_q + 2 * count_h};
}

static __global__ void atom_cache_kernel(
  Arch a,
  ANNView ann,
  Shard sh,
  const float* scaler,
  const float* Phi,
  const float* A,
  AtomCache cache)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= sh.N) {
    return;
  }
  const int ti = sh.type[i];
  float* q = cache.q + size_t(i) * a.dim;
  float* h = cache.h + size_t(i) * a.neurons;
  float* ws = cache.ws + size_t(i) * a.neurons;
  float* basegdq = cache.basegdq + size_t(i) * a.dim;
  float* angular_s = cache.angular_s + size_t(i) * a.n_A * a.n_lm;
  const float* phi = Phi + size_t(i) * a.num_types * a.K_R;
  const float* angular = A + size_t(i) * a.num_types * a.K_A * a.n_lm;
  for (int n = 0; n < a.n_R; ++n) {
    float value = 0.0f;
    for (int tj = 0; tj < a.num_types; ++tj) {
      for (int k = 0; k < a.K_R; ++k) {
        value += ann.cR[c_index(a, ti, tj, n, k, a.K_R)] * phi[tj * a.K_R + k];
      }
    }
    q[n] = value * scaler[n];
  }
  const int channels = a.L_max + (a.L_max_4body == 2) + (a.L_max_5body == 1);
  for (int n = 0; n < a.n_A; ++n) {
    float s[MAX_LM];
    for (int l = 0; l < a.n_lm; ++l) {
      s[l] = 0.0f;
      for (int tj = 0; tj < a.num_types; ++tj) {
        for (int k = 0; k < a.K_A; ++k) {
          const int index = (tj * a.K_A + k) * a.n_lm + l;
          s[l] += ann.cA[c_index(a, ti, tj, n, k, a.K_A)] * angular[index];
        }
      }
      angular_s[n * a.n_lm + l] = s[l];
    }
    for (int channel = 0; channel < channels; ++channel) {
      const int index = a.n_R + channel * a.n_A + n;
      q[index] = invariant(a, channel, s) * scaler[index];
    }
  }
  const int offset = ti * ann_block_size(a);
  const float* w0 = ann.parameters + offset;
  const float* b0 = w0 + a.neurons * a.dim;
  const float* w1 = b0 + a.neurons;
  float adjoint[MAX_DIM] = {};
  for (int k = 0; k < a.neurons; ++k) {
    float z = -b0[k];
    for (int d = 0; d < a.dim; ++d) {
      z += w0[k * a.dim + d] * q[d];
    }
    h[k] = tanhf(z);
    ws[k] = w1[k] * (1.0f - h[k] * h[k]);
    for (int d = 0; d < a.dim; ++d) {
      adjoint[d] += ws[k] * w0[k * a.dim + d];
    }
  }
  for (int d = 0; d < a.dim; ++d) {
    basegdq[d] = adjoint[d] * scaler[d];
  }
}

__device__ inline void cached_angular_tangent(
  const Arch& a,
  int ti,
  int n,
  const float* coefficients,
  const float* primal_s,
  const float* dA,
  Dual* s)
{
  for (int l = 0; l < a.n_lm; ++l) {
    float tangent = 0.0f;
    for (int tj = 0; tj < a.num_types; ++tj) {
      for (int k = 0; k < a.K_A; ++k) {
        const int index = (tj * a.K_A + k) * a.n_lm + l;
        tangent += coefficients[c_index(a, ti, tj, n, k, a.K_A)] * (dA ? dA[index] : 0.0f);
      }
    }
    s[l] = Dual(primal_s[n * a.n_lm + l], tangent);
  }
}

template <bool ENERGY>
static __global__ void rows_cached_kernel(
  Arch a,
  ANNView ann,
  AtomCache cache,
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
  const float* w1 = ann.parameters + offset_w1;
  const float* phi = Phi + size_t(i) * a.num_types * a.K_R;
  const float* angular = A + size_t(i) * a.num_types * a.K_A * a.n_lm;
  const float* dphi = dPhi ? dPhi + size_t(r) * a.num_types * a.K_R : nullptr;
  const float* dangular = dA ? dA + size_t(r) * a.num_types * a.K_A * a.n_lm : nullptr;
  const float* q = cache.q + size_t(i) * a.dim;
  const float* h = cache.h + size_t(i) * a.neurons;
  const float* ws = cache.ws + size_t(i) * a.neurons;
  const float* basegdq = cache.basegdq + size_t(i) * a.dim;
  const float* primal_s = cache.angular_s + size_t(i) * a.n_A * a.n_lm;
  float* output = rows + size_t(row) * n_params;
  const float scale = row_scale[row];
  float dq[MAX_DIM];
  float gq[MAX_DIM] = {};
  if (!ENERGY) {
    for (int n = 0; n < a.n_R; ++n) {
      float tangent = 0.0f;
      for (int tj = 0; tj < a.num_types; ++tj) {
        for (int k = 0; k < a.K_R; ++k) {
          tangent += ann.cR[c_index(a, ti, tj, n, k, a.K_R)] * (dphi ? dphi[tj * a.K_R + k] : 0.0f);
        }
      }
      dq[n] = tangent * scaler[n];
    }
    const int channels = a.L_max + (a.L_max_4body == 2) + (a.L_max_5body == 1);
    for (int n = 0; n < a.n_A; ++n) {
      Dual s[MAX_LM];
      cached_angular_tangent(a, ti, n, ann.cA, primal_s, dangular, s);
      for (int channel = 0; channel < channels; ++channel) {
        const int index = a.n_R + channel * a.n_A + n;
        dq[index] = invariant(a, channel, s).d * scaler[index];
      }
    }
  }
  float value = ENERGY ? -w1[a.neurons] : 0.0f;
  for (int k = 0; k < a.neurons; ++k) {
    if (ENERGY) {
      value += w1[k] * h[k];
      atomicAdd(output + offset_b0 + k, -scale * ws[k]);
      atomicAdd(output + offset_w1 + k, scale * h[k]);
      for (int d = 0; d < a.dim; ++d) {
        atomicAdd(output + offset + k * a.dim + d, scale * ws[k] * q[d]);
      }
    } else {
      float u = 0.0f;
      for (int d = 0; d < a.dim; ++d) {
        u += w0[k * a.dim + d] * dq[d];
      }
      const float derivative = 1.0f - h[k] * h[k];
      const float beta = w1[k] * u * (-2.0f * h[k] * derivative);
      value += ws[k] * u;
      atomicAdd(output + offset_b0 + k, -scale * beta);
      atomicAdd(output + offset_w1 + k, scale * derivative * u);
      for (int d = 0; d < a.dim; ++d) {
        atomicAdd(output + offset + k * a.dim + d, scale * (ws[k] * dq[d] + beta * q[d]));
        gq[d] += beta * w0[k * a.dim + d];
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
    gq[d] = ENERGY ? basegdq[d] : gq[d] * scaler[d];
  }
  for (int n = 0; n < a.n_R; ++n) {
    for (int tj = 0; tj < a.num_types; ++tj) {
      for (int k = 0; k < a.K_R; ++k) {
        const int index = tj * a.K_R + k;
        const float derivative =
          gq[n] * phi[index] + (ENERGY || !dphi ? 0.0f : basegdq[n] * dphi[index]);
        atomicAdd(output + offset_cR + c_index(a, ti, tj, n, k, a.K_R), scale * derivative);
      }
    }
  }
  for (int n = 0; n < a.n_A; ++n) {
    float reverse_value[MAX_LM];
    invariant_reverse(a, n, primal_s + n * a.n_lm, gq, reverse_value);
    Dual reverse_tangent[MAX_LM];
    if (!ENERGY) {
      Dual s[MAX_LM];
      cached_angular_tangent(a, ti, n, ann.cA, primal_s, dangular, s);
      invariant_reverse(a, n, s, basegdq, reverse_tangent);
    }
    for (int tj = 0; tj < a.num_types; ++tj) {
      for (int k = 0; k < a.K_A; ++k) {
        float derivative = 0.0f;
        for (int l = 0; l < a.n_lm; ++l) {
          const int index = (tj * a.K_A + k) * a.n_lm + l;
          if (ENERGY) {
            derivative += reverse_value[l] * angular[index];
          } else {
            derivative += (reverse_value[l] + reverse_tangent[l].d) * angular[index] +
                          reverse_tangent[l].v * (dangular ? dangular[index] : 0.0f);
          }
        }
        atomicAdd(output + offset_cA + c_index(a, ti, tj, n, k, a.K_A), scale * derivative);
      }
    }
  }
}

} // namespace lm
