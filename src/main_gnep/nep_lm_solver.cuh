/*----------------------------------------------------------------------------80
  Experimental damped Gauss-Newton solver for NEP training.
  Channel curvature is accumulated with cuBLAS; candidates use FP64 Cholesky.
------------------------------------------------------------------------------*/

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef NEP_LM_USE_NCCL
#include <nccl.h>
#endif

#define LM_CUDA(expression)                                                                        \
  do {                                                                                             \
    const cudaError_t error = (expression);                                                        \
    if (error != cudaSuccess) {                                                                    \
      fprintf(stderr, "NEP-LM CUDA %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

#define LM_CUBLAS(expression)                                                                      \
  do {                                                                                             \
    const cublasStatus_t error = (expression);                                                     \
    if (error != CUBLAS_STATUS_SUCCESS) {                                                          \
      fprintf(stderr, "NEP-LM cuBLAS %d at %s:%d\n", int(error), __FILE__, __LINE__);              \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

#define LM_CUSOLVER(expression)                                                                    \
  do {                                                                                             \
    const cusolverStatus_t error = (expression);                                                   \
    if (error != CUSOLVER_STATUS_SUCCESS) {                                                        \
      fprintf(stderr, "NEP-LM cuSOLVER %d at %s:%d\n", int(error), __FILE__, __LINE__);            \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

#ifdef NEP_LM_USE_NCCL
#define LM_NCCL(expression)                                                                        \
  do {                                                                                             \
    const ncclResult_t error = (expression);                                                       \
    if (error != ncclSuccess) {                                                                    \
      fprintf(stderr, "NEP-LM NCCL %s at %s:%d\n", ncclGetErrorString(error), __FILE__, __LINE__); \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)
#endif

enum LMChannel { LM_ENERGY = 0, LM_FORCE = 1, LM_VIRIAL = 2, LM_NUM_CHANNELS = 3 };

struct LMConfig {
  int num_candidates = 8;
  double lambda0 = 1e-2;
  double lambda_ratio = 4.0;
  double lambda_min = 1e-9;
  double lambda_max = 1e9;
  double ema[LM_NUM_CHANNELS] = {1.0, 0.2, 1.0};
  double diag_floor = 1e-6;
  double diag_absolute_floor = 1e-12;
  double l2 = 0.0;
};

namespace lm_detail
{

inline void require(bool condition, const char* message)
{
  if (!condition) {
    fprintf(stderr, "NEP-LM input error: %s\n", message);
    std::exit(1);
  }
}

__global__ void ema_convert(int count, const float* h32, double* h64, double rho, bool first)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    h64[i] = first ? double(h32[i]) : (1.0 - rho) * h64[i] + rho * double(h32[i]);
  }
}

__global__ void combine_channels(
  int count,
  const double* energy,
  const double* force,
  const double* virial,
  double weight_energy,
  double weight_force,
  double weight_virial,
  double* out)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] = weight_energy * energy[i] + weight_force * force[i] + weight_virial * virial[i];
  }
}

__global__ void extract_diag(int n, const double* matrix, double* diagonal)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    diagonal[i] = matrix[size_t(i) * n + i];
  }
}

__global__ void floor_diag(int n, double* diagonal, double floor_value)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    diagonal[i] = fmax(diagonal[i], floor_value);
  }
}

__global__ void
add_scaled_diag(int n, double* matrix, const double* diagonal, double lambda, double l2)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    matrix[size_t(i) * n + i] += lambda * diagonal[i] + l2;
  }
}

__global__ void negate_to_double(int n, const double* gradient, double* out)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = -gradient[i];
  }
}

// cuSOLVER writes info on the same stream before this conversion executes.
__global__ void to_float(int n, const double* in, float* out, int* info)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float value = float(in[i]);
    out[i] = value;
    if (!isfinite(in[i]) || !isfinite(value)) {
      atomicExch(info, 1);
    }
  }
}

inline int blocks(size_t n)
{
  return int((n + 255) / 256);
}

} // namespace lm_detail

class LMSolver
{
public:
  LMSolver(int n, const LMConfig& cfg, cudaStream_t stream = 0) : n_(n), cfg_(cfg), stream_(stream)
  {
    lm_detail::require(n > 0 && size_t(n) * n <= INT_MAX, "Invalid matrix dimension.");
    lm_detail::require(cfg.num_candidates > 0, "The candidate count must be positive.");
    lm_detail::require(
      std::isfinite(cfg.lambda_min) && std::isfinite(cfg.lambda_max) && cfg.lambda_min > 0.0 &&
        cfg.lambda_max >= cfg.lambda_min && std::isfinite(cfg.lambda0) &&
        cfg.lambda0 >= cfg.lambda_min && cfg.lambda0 <= cfg.lambda_max,
      "Invalid damping bounds or initial damping.");
    lm_detail::require(
      std::isfinite(cfg.lambda_ratio) && cfg.lambda_ratio > 1.0,
      "The damping ratio must be finite and greater than one.");
    lm_detail::require(
      std::isfinite(cfg.diag_floor) && cfg.diag_floor >= 0.0 &&
        std::isfinite(cfg.diag_absolute_floor) && cfg.diag_absolute_floor > 0.0,
      "Invalid diagonal floor.");
    for (int channel = 0; channel < LM_NUM_CHANNELS; ++channel) {
      lm_detail::require(
        std::isfinite(cfg.ema[channel]) && cfg.ema[channel] > 0.0 && cfg.ema[channel] <= 1.0,
        "EMA weights must be finite and in (0, 1].");
    }
    set_l2(cfg.l2);
    lambda_ = cfg.lambda0;
    const size_t count = size_t(n) * n;
    for (int channel = 0; channel < LM_NUM_CHANNELS; ++channel) {
      LM_CUDA(cudaMalloc(&h32_[channel], count * sizeof(float)));
      LM_CUDA(cudaMalloc(&h64_[channel], count * sizeof(double)));
      LM_CUDA(cudaMemsetAsync(h64_[channel], 0, count * sizeof(double), stream_));
      first_[channel] = true;
    }
    LM_CUDA(cudaMalloc(&combined_, count * sizeof(double)));
    LM_CUDA(cudaMalloc(&factor_, count * sizeof(double)));
    LM_CUDA(cudaMalloc(&diag_, n * sizeof(double)));
    LM_CUDA(cudaMalloc(&rhs_, n * sizeof(double)));
    LM_CUDA(cudaMalloc(&info_, sizeof(int)));
#ifdef NEP_LM_USE_NCCL
    LM_CUDA(cudaMalloc(&rows_seen_device_, LM_NUM_CHANNELS * sizeof(long long)));
#endif
    LM_CUBLAS(cublasCreate(&blas_));
    LM_CUBLAS(cublasSetStream(blas_, stream_));
    LM_CUSOLVER(cusolverDnCreate(&solver_));
    LM_CUSOLVER(cusolverDnSetStream(solver_, stream_));
    LM_CUSOLVER(
      cusolverDnDpotrf_bufferSize(solver_, CUBLAS_FILL_MODE_LOWER, n_, factor_, n_, &lwork_));
    LM_CUDA(cudaMalloc(&work_, size_t(lwork_) * sizeof(double)));
  }

  LMSolver(const LMSolver&) = delete;
  LMSolver& operator=(const LMSolver&) = delete;

  ~LMSolver()
  {
    for (int channel = 0; channel < LM_NUM_CHANNELS; ++channel) {
      cudaFree(h32_[channel]);
      cudaFree(h64_[channel]);
    }
    cudaFree(combined_);
    cudaFree(factor_);
    cudaFree(diag_);
    cudaFree(rhs_);
    cudaFree(info_);
    cudaFree(work_);
#ifdef NEP_LM_USE_NCCL
    cudaFree(rows_seen_device_);
#endif
    cublasDestroy(blas_);
    cusolverDnDestroy(solver_);
  }

  void set_l2(double weight)
  {
    lm_detail::require(std::isfinite(weight) && weight >= 0.0, "Invalid L2 IRLS weight.");
    cfg_.l2 = weight;
  }

  void begin_curvature()
  {
    for (int channel = 0; channel < LM_NUM_CHANNELS; ++channel) {
      LM_CUDA(cudaMemsetAsync(h32_[channel], 0, size_t(n_) * n_ * sizeof(float), stream_));
      rows_seen_[channel] = 0;
    }
  }

  // Row-major rows x n is column-major n x rows, so A * A^T forms J^T J.
  // Sampling corrections must scale rows by the square root of the weight.
  void accumulate(LMChannel channel, const float* rows_device, int rows)
  {
    lm_detail::require(channel >= 0 && channel < LM_NUM_CHANNELS, "Invalid residual channel.");
    lm_detail::require(rows >= 0, "The Jacobian row count must be nonnegative.");
    if (rows == 0) {
      return;
    }
    lm_detail::require(rows_device != nullptr, "Missing Jacobian rows.");
    const float one = 1.0f;
    LM_CUBLAS(cublasGemmEx(
      blas_,
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      n_,
      n_,
      rows,
      &one,
      rows_device,
      CUDA_R_32F,
      n_,
      rows_device,
      CUDA_R_32F,
      n_,
      &one,
      h32_[channel],
      CUDA_R_32F,
      n_,
      CUBLAS_COMPUTE_32F_FAST_TF32,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    rows_seen_[channel] += rows;
  }

#ifdef NEP_LM_USE_NCCL
  void allreduce_curvature(ncclComm_t communicator)
  {
    for (int channel = 0; channel < LM_NUM_CHANNELS; ++channel) {
      LM_NCCL(ncclAllReduce(
        h32_[channel], h32_[channel], size_t(n_) * n_, ncclFloat, ncclSum, communicator, stream_));
    }
    LM_CUDA(cudaMemcpyAsync(
      rows_seen_device_, rows_seen_, sizeof(rows_seen_), cudaMemcpyHostToDevice, stream_));
    LM_NCCL(ncclAllReduce(
      rows_seen_device_,
      rows_seen_device_,
      LM_NUM_CHANNELS,
      ncclInt64,
      ncclSum,
      communicator,
      stream_));
    LM_CUDA(cudaMemcpyAsync(
      rows_seen_, rows_seen_device_, sizeof(rows_seen_), cudaMemcpyDeviceToHost, stream_));
    LM_CUDA(cudaStreamSynchronize(stream_));
  }
#endif

  void end_curvature()
  {
    const size_t count = size_t(n_) * n_;
    for (int channel = 0; channel < LM_NUM_CHANNELS; ++channel) {
      if (rows_seen_[channel] == 0) {
        continue;
      }
      lm_detail::ema_convert<<<lm_detail::blocks(count), 256, 0, stream_>>>(
        int(count), h32_[channel], h64_[channel], cfg_.ema[channel], first_[channel]);
      LM_CUDA(cudaGetLastError());
      first_[channel] = false;
    }
  }

  std::vector<double> solve_candidates(
    const double weights[LM_NUM_CHANNELS],
    const double* gradient,
    float* deltas,
    std::vector<bool>& valid)
  {
    lm_detail::require(
      weights != nullptr && gradient != nullptr && deltas != nullptr,
      "Missing solver input or output.");
    for (int channel = 0; channel < LM_NUM_CHANNELS; ++channel) {
      lm_detail::require(
        std::isfinite(weights[channel]) && weights[channel] >= 0.0,
        "Residual channel weights must be finite and nonnegative.");
    }
    const size_t count = size_t(n_) * n_;
    lm_detail::combine_channels<<<lm_detail::blocks(count), 256, 0, stream_>>>(
      int(count), h64_[0], h64_[1], h64_[2], weights[0], weights[1], weights[2], combined_);
    LM_CUDA(cudaGetLastError());
    lm_detail::extract_diag<<<lm_detail::blocks(n_), 256, 0, stream_>>>(n_, combined_, diag_);
    LM_CUDA(cudaGetLastError());
    std::vector<double> diagonal(n_);
    LM_CUDA(cudaMemcpyAsync(
      diagonal.data(), diag_, n_ * sizeof(double), cudaMemcpyDeviceToHost, stream_));
    LM_CUDA(cudaStreamSynchronize(stream_));
    for (const double value : diagonal) {
      lm_detail::require(std::isfinite(value) && value >= 0.0, "Invalid curvature diagonal.");
    }
    const double maximum = *std::max_element(diagonal.begin(), diagonal.end());
    const double floor_value = std::max(cfg_.diag_absolute_floor, cfg_.diag_floor * maximum);
    lm_detail::floor_diag<<<lm_detail::blocks(n_), 256, 0, stream_>>>(n_, diag_, floor_value);
    LM_CUDA(cudaGetLastError());

    const int candidates = cfg_.num_candidates;
    std::vector<double> lambdas(candidates);
    valid.assign(candidates, false);
    for (int j = 0; j < candidates; ++j) {
      lambdas[j] = std::max(
        cfg_.lambda_min,
        std::min(cfg_.lambda_max, lambda_ * std::pow(cfg_.lambda_ratio, j - candidates / 2)));
      LM_CUDA(cudaMemcpyAsync(
        factor_, combined_, count * sizeof(double), cudaMemcpyDeviceToDevice, stream_));
      lm_detail::add_scaled_diag<<<lm_detail::blocks(n_), 256, 0, stream_>>>(
        n_, factor_, diag_, lambdas[j], cfg_.l2);
      LM_CUDA(cudaGetLastError());
      LM_CUSOLVER(
        cusolverDnDpotrf(solver_, CUBLAS_FILL_MODE_LOWER, n_, factor_, n_, work_, lwork_, info_));
      int info = 0;
      LM_CUDA(cudaMemcpyAsync(&info, info_, sizeof(int), cudaMemcpyDeviceToHost, stream_));
      LM_CUDA(cudaStreamSynchronize(stream_));
      if (info != 0) {
        continue;
      }
      lm_detail::negate_to_double<<<lm_detail::blocks(n_), 256, 0, stream_>>>(n_, gradient, rhs_);
      LM_CUDA(cudaGetLastError());
      LM_CUSOLVER(
        cusolverDnDpotrs(solver_, CUBLAS_FILL_MODE_LOWER, n_, 1, factor_, n_, rhs_, n_, info_));
      lm_detail::to_float<<<lm_detail::blocks(n_), 256, 0, stream_>>>(
        n_, rhs_, deltas + size_t(j) * n_, info_);
      LM_CUDA(cudaGetLastError());
      LM_CUDA(cudaMemcpyAsync(&info, info_, sizeof(int), cudaMemcpyDeviceToHost, stream_));
      LM_CUDA(cudaStreamSynchronize(stream_));
      valid[j] = info == 0;
    }
    last_lambdas_ = lambdas;
    last_valid_ = valid;
    return lambdas;
  }

  void report(int best, bool improved)
  {
    lm_detail::require(!last_lambdas_.empty(), "No candidates have been solved.");
    if (improved) {
      lm_detail::require(
        best >= 0 && size_t(best) < last_lambdas_.size() && last_valid_[best],
        "The accepted candidate is invalid.");
      lambda_ = last_lambdas_[best];
      rejects_ = 0;
    } else {
      lambda_ =
        std::min(cfg_.lambda_max, last_lambdas_.back() * cfg_.lambda_ratio * cfg_.lambda_ratio);
      ++rejects_;
    }
  }

  double lambda() const
  {
    return lambda_;
  }

  int consecutive_rejects() const
  {
    return rejects_;
  }

private:
  int n_;
  LMConfig cfg_;
  cudaStream_t stream_;
  cublasHandle_t blas_;
  cusolverDnHandle_t solver_;
  float* h32_[LM_NUM_CHANNELS];
  double* h64_[LM_NUM_CHANNELS];
  bool first_[LM_NUM_CHANNELS];
  long long rows_seen_[LM_NUM_CHANNELS] = {0, 0, 0};
#ifdef NEP_LM_USE_NCCL
  long long* rows_seen_device_;
#endif
  double* combined_;
  double* factor_;
  double* diag_;
  double* rhs_;
  double* work_;
  int* info_;
  int lwork_ = 0;
  double lambda_;
  int rejects_ = 0;
  std::vector<double> last_lambdas_;
  std::vector<bool> last_valid_;
};
