// SPDX-License-Identifier: GPL-3.0-or-later

/*----------------------------------------------------------------------------80
Warp-cooperative native NEP-LM rows with one shared row per structure.
Each cooperative group handles one atom; model dimensions remain runtime values.
------------------------------------------------------------------------------*/

#pragma once
#include "nep_lm_cached.cuh"

namespace lm
{

constexpr int ROW_LANES = 16;
constexpr int ROW_THREADS = 128;

inline size_t rows_warp_shared_bytes(const Arch& a, int n_params)
{
  return sizeof(float) *
         (size_t(n_params) + size_t(ROW_THREADS / ROW_LANES) * (2 * a.dim + a.neurons));
}

// Subgroups need independent thread scheduling: build for sm_70 or newer.
// Tangent records must use the canonical r=i order.
template <bool ENERGY>
static __global__ void rows_warp_kernel(
  Arch a,
  ANNView ann,
  AtomCache cache,
  Shard sh,
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
  const int structure = blockIdx.x;
  if (structure >= sh.num_structures) {
    return;
  }
  extern __shared__ float shared[];
  float* output = shared;
  for (int p = threadIdx.x; p < n_params; p += blockDim.x) {
    output[p] = 0.0f;
  }
  __syncthreads();
  const int group = threadIdx.x / ROW_LANES;
  const int lane = threadIdx.x % ROW_LANES;
  const int groups = blockDim.x / ROW_LANES;
  const unsigned mask = (0xffffffffu >> (32 - ROW_LANES)) << ((threadIdx.x % 32) / ROW_LANES * ROW_LANES);
  float* dq = shared + n_params + group * (2 * a.dim + a.neurons);
  float* gq = dq + a.dim;
  float* beta = gq + a.dim;
  const int begin = sh.atom_start[structure];
  const int end = structure + 1 < sh.num_structures ? sh.atom_start[structure + 1] : sh.N;
  const float scale = row_scale[structure];
  for (int i = begin + group; i < end; i += groups) {
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
    const float* dphi = dPhi ? dPhi + size_t(i) * a.num_types * a.K_R : nullptr;
    const float* dangular = dA ? dA + size_t(i) * a.num_types * a.K_A * a.n_lm : nullptr;
    const float* q = cache.q + size_t(i) * a.dim;
    const float* h = cache.h + size_t(i) * a.neurons;
    const float* ws = cache.ws + size_t(i) * a.neurons;
    const float* basegdq = cache.basegdq + size_t(i) * a.dim;
    const float* primal_s = cache.angular_s + size_t(i) * a.n_A * a.n_lm;
    if (!ENERGY) {
      for (int n = lane; n < a.n_R; n += ROW_LANES) {
        float tangent = 0.0f;
        for (int tj = 0; tj < a.num_types; ++tj) {
          for (int k = 0; k < a.K_R; ++k) {
            tangent +=
              ann.cR[c_index(a, ti, tj, n, k, a.K_R)] * (dphi ? dphi[tj * a.K_R + k] : 0.0f);
          }
        }
        dq[n] = tangent * scaler[n];
      }
      const int channels = a.L_max + (a.L_max_4body == 2) + (a.L_max_5body == 1);
      for (int n = lane; n < a.n_A; n += ROW_LANES) {
        Dual s[MAX_LM];
        cached_angular_tangent(a, ti, n, ann.cA, primal_s, dangular, s);
        for (int channel = 0; channel < channels; ++channel) {
          const int index = a.n_R + channel * a.n_A + n;
          dq[index] = invariant(a, channel, s).d * scaler[index];
        }
      }
    }
    __syncwarp(mask);
    float value = ENERGY && lane == 0 ? -w1[a.neurons] : 0.0f;
    for (int k = lane; k < a.neurons; k += ROW_LANES) {
      if (ENERGY) {
        value += w1[k] * h[k];
        atomicAdd(output + offset_b0 + k, -scale * ws[k]);
        atomicAdd(output + offset_w1 + k, scale * h[k]);
      } else {
        float u = 0.0f;
        for (int d = 0; d < a.dim; ++d) {
          u += w0[k * a.dim + d] * dq[d];
        }
        const float derivative = 1.0f - h[k] * h[k];
        beta[k] = w1[k] * u * (-2.0f * h[k] * derivative);
        value += ws[k] * u;
        atomicAdd(output + offset_b0 + k, -scale * beta[k]);
        atomicAdd(output + offset_w1 + k, scale * derivative * u);
      }
    }
    __syncwarp(mask);
    for (int index = lane; index < a.neurons * a.dim; index += ROW_LANES) {
      const int k = index / a.dim;
      const int d = index % a.dim;
      const float derivative = ENERGY ? ws[k] * q[d] : ws[k] * dq[d] + beta[k] * q[d];
      atomicAdd(output + offset + index, scale * derivative);
    }
    for (int d = lane; d < a.dim; d += ROW_LANES) {
      float adjoint = 0.0f;
      if (!ENERGY) {
        for (int k = 0; k < a.neurons; ++k) {
          adjoint += beta[k] * w0[k * a.dim + d];
        }
      }
      gq[d] = ENERGY ? basegdq[d] : adjoint * scaler[d];
    }
    if (ENERGY && lane == 0) {
      atomicAdd(output + offset_w1 + a.neurons, -scale);
    }
    if (values) {
      for (int offset = ROW_LANES / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(mask, value, offset, ROW_LANES);
      }
      if (lane == 0) {
        atomicAdd(values + structure, scale * value);
      }
    }
    __syncwarp(mask);
    for (int index = lane; index < a.n_R * a.num_types * a.K_R; index += ROW_LANES) {
      const int k = index % a.K_R;
      const int tj = index / a.K_R % a.num_types;
      const int n = index / (a.K_R * a.num_types);
      const int moment = tj * a.K_R + k;
      const float derivative =
        gq[n] * phi[moment] + (ENERGY || !dphi ? 0.0f : basegdq[n] * dphi[moment]);
      atomicAdd(output + offset_cR + c_index(a, ti, tj, n, k, a.K_R), scale * derivative);
    }
    for (int n = lane; n < a.n_A; n += ROW_LANES) {
      float reverse_value[MAX_LM];
      invariant_reverse(a, n, primal_s + n * a.n_lm, gq, reverse_value);
      Dual reverse_tangent[MAX_LM];
      if (!ENERGY) {
        Dual s[MAX_LM];
        cached_angular_tangent(a, ti, n, ann.cA, primal_s, dangular, s);
        invariant_reverse(a, n, s, basegdq, reverse_tangent);
      }
      for (int index = 0; index < a.num_types * a.K_A; ++index) {
        const int tj = index / a.K_A;
        const int k = index % a.K_A;
        float derivative = 0.0f;
        for (int l = 0; l < a.n_lm; ++l) {
          const int moment = index * a.n_lm + l;
          if (ENERGY) {
            derivative += reverse_value[l] * angular[moment];
          } else {
            derivative += (reverse_value[l] + reverse_tangent[l].d) * angular[moment] +
                          reverse_tangent[l].v * (dangular ? dangular[moment] : 0.0f);
          }
        }
        atomicAdd(output + offset_cA + c_index(a, ti, tj, n, k, a.K_A), scale * derivative);
      }
    }
    __syncwarp(mask);
  }
  __syncthreads();
  for (int p = threadIdx.x; p < n_params; p += blockDim.x) {
    rows[size_t(structure) * n_params + p] = output[p];
  }
}

inline cudaError_t configure_rows_warp_kernels(size_t shared_bytes)
{
  cudaError_t error = cudaFuncSetAttribute(
    rows_warp_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared_bytes));
  if (error != cudaSuccess) {
    return error;
  }
  return cudaFuncSetAttribute(
    rows_warp_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared_bytes));
}

template <bool ENERGY>
inline void launch_rows_warp_kernel(
  Arch a,
  ANNView ann,
  AtomCache cache,
  Shard sh,
  const float* row_scale,
  const float* scaler,
  const float* Phi,
  const float* A,
  const float* dPhi,
  const float* dA,
  float* rows,
  int n_params)
{
  rows_warp_kernel<ENERGY><<<
    sh.num_structures,
    ROW_THREADS,
    rows_warp_shared_bytes(a, n_params)>>>(
    a, ann, cache, sh, row_scale, scaler, Phi, A, dPhi, dA, rows, n_params);
}

} // namespace lm
