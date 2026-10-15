// SPDX-License-Identifier: GPL-3.0-or-later
#include "dataset.cuh"
#include "fcvp.cuh"
#include "fcvp_solver.cuh"
#include "gnep.cuh"
#include "parameters.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cublas_v2.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace
{
void check_blas(cublasStatus_t status)
{
  if (status != CUBLAS_STATUS_SUCCESS)
    PRINT_INPUT_ERROR("FCVP cuBLAS operation failed.");
}
struct Blas {
  cublasHandle_t handle;
  Blas() { check_blas(cublasCreate(&handle)); }
  ~Blas() { cublasDestroy(handle); }
};
struct Shard_Basis {
  int device, rows, tile_rows, columns;
  GPU_Vector<double> matrix, reconstructed, device_beta, h[3], g[3], yy, dot;
  Blas blas;
  Shard_Basis(int device_id, int row_count, int capacity, int columns, bool check)
    : device(device_id), rows(row_count), tile_rows(capacity), columns(columns),
      matrix(size_t(capacity) * (columns + 1)), yy(3, 0.0), dot(1)
  {
    if (check) {
      reconstructed.resize(capacity);
      device_beta.resize(columns);
    }
    for (int k = 0; k < 3; ++k) {
      h[k].resize(size_t(columns) * columns, 0.0);
      g[k].resize(columns, 0.0);
    }
  }
  ~Shard_Basis() { CHECK(cudaSetDevice(device)); }
};
// Column-major row tile with E, F, V physical row coordinates.
static __global__ void pack_column(
  int N,
  int Nc,
  int row_start,
  int row_count,
  bool target,
  float force_delta,
  float shear,
  const int* na,
  const int* start,
  const int* batch,
  const int* types,
  const int* has_virial,
  const float* weights,
  const float* type_weights,
  const float* energy,
  const float* force,
  const float* virial,
  const float* energy_ref,
  const float* force_ref,
  const float* virial_ref,
  double* out)
{
  const int local_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_row >= row_count)
    return;
  const int row = row_start + local_row;
  double value = 0, weight = 1;
  if (row < Nc) {
    for (int i = start[row]; i < start[row] + na[row]; ++i)
      value += energy[i];
    value /= na[row];
    if (target)
      value = energy_ref[row] - value;
    weight = weights[row];
  } else if (row < Nc + 3 * N) {
    const int f = row - Nc, i = f % N;
    value = target ? force_ref[f] - double(force[f]) : force[f];
    weight = double(weights[batch[i]]) * type_weights[types[i]];
    if (force_delta > 0) {
      double x = force_ref[i], y = force_ref[N + i], z = force_ref[2 * N + i];
      weight *= sqrt(force_delta / (force_delta + sqrt(x * x + y * y + z * z)));
    }
  } else {
    const int v = row - Nc - 3 * N, c = v % Nc, d = v / Nc;
    if (has_virial[c]) {
      for (int i = start[c]; i < start[c] + na[c]; ++i)
        value += virial[d * N + i];
      value /= na[c];
      if (target)
        value = virial_ref[d * Nc + c] - value;
      weight = double(weights[c]) * (d >= 3 ? shear : 1.0f);
    }
  }
  out[local_row] = value * weight;
}
void pack(Parameters& p, Dataset& d, int row_start, int row_count, bool target, double* output)
{
  pack_column<<<(row_count - 1) / 256 + 1, 256>>>(
    d.N,
    d.Nc,
    row_start,
    row_count,
    target,
    p.force_delta,
    p.lambda_shear,
    d.Na.data(),
    d.Na_sum.data(),
    d.batch_idx.data(),
    d.type.data(),
    d.has_virial_gpu.data(),
    d.weight_gpu.data(),
    d.type_weight_gpu.data(),
    d.energy.data(),
    d.force.data(),
    d.virial.data(),
    d.energy_ref_gpu.data(),
    d.force_ref_gpu.data(),
    d.virial_ref_gpu.data(),
    output);
  GPU_CHECK_KERNEL
}

void check_reconstruction(
  Shard_Basis& shard,
  const std::vector<double>& expected,
  int row,
  int length,
  std::vector<double>& actual,
  double& worst)
{
  const double* target = shard.matrix.data() + size_t(shard.tile_rows) * shard.columns;
  shard.reconstructed.copy_from_device(target, length);
  const double minus_one = -1, one = 1;
  check_blas(cublasDgemv(
    shard.blas.handle,
    CUBLAS_OP_N,
    length,
    shard.columns,
    &minus_one,
    shard.matrix.data(),
    shard.tile_rows,
    shard.device_beta.data(),
    1,
    &one,
    shard.reconstructed.data(),
    1));
  shard.reconstructed.copy_to_host(actual.data(), length);
  for (int i = 0; i < length; ++i) {
    const double reference = expected[row + i];
    if (!std::isfinite(actual[i]) || !std::isfinite(reference))
      PRINT_INPUT_ERROR("FCVP nonfinite basis reconstruction.");
    worst = std::max(worst, std::abs(actual[i] - reference) / (1 + std::abs(reference)));
  }
}

double native_objective(Parameters& p, std::vector<Dataset>& dataset, const float* parameters)
{
  const auto mse = get_batch_mse(p, dataset, true);
  double eps2 = p.fcvp_epsilon * p.fcvp_epsilon;
  double value =
    p.lambda_e * std::sqrt(mse.energy + eps2) + p.lambda_f * std::sqrt(mse.force + eps2);
  if (mse.virial_configurations)
    value += p.lambda_v * std::sqrt(mse.virial + eps2);
  double l1 = 0, l2 = 0;
  for (int i = 0; i < p.number_of_variables; ++i) {
    l1 += std::abs(parameters[i]);
    l2 += double(parameters[i]) * parameters[i];
  }
  return value + p.lambda_1 * l1 / p.number_of_variables +
         p.lambda_2 * std::sqrt(l2 / p.number_of_variables + eps2);
}
} // namespace

void fcvp_project(
  Parameters& para, GNEP& potential, std::vector<Dataset>& dataset, float* parameters, int step)
{
  CHECK(cudaSetDevice(0));
  auto start_time = std::chrono::steady_clock::now();
  const std::vector<int> indices = para.output_parameter_indices();
  if (indices.empty() || indices.size() > size_t(std::numeric_limits<int>::max()))
    PRINT_INPUT_ERROR("FCVP output parameter count is invalid.");
  const int p = int(indices.size());
  if (p > std::numeric_limits<int>::max() / p)
    PRINT_INPUT_ERROR("FCVP Gram dimension exceeds the solver's 32-bit indexing range.");
  const int devices = dataset.size();
  uint64_t counts[3] = {0, 0, 0};
  size_t entries = 0;
  if (para.fcvp_memory_mib <= 0 ||
      uint64_t(para.fcvp_memory_mib) > std::numeric_limits<size_t>::max() / (1024 * 1024))
    PRINT_INPUT_ERROR("FCVP memory budget must be a positive number of MiB.");
  const size_t budget_elements =
    size_t(para.fcvp_memory_mib) * 1024 * 1024 / sizeof(double);
  const size_t columns = size_t(p);
  const size_t fixed_elements = 3 * columns * columns + 3 * columns + 4 +
                                (step == 0 ? columns : 0);
  const size_t elements_per_row = columns + 1 + (step == 0 ? 1 : 0);
  if (fixed_elements >= budget_elements ||
      budget_elements - fixed_elements < elements_per_row)
    PRINT_INPUT_ERROR(
      "FCVP memory budget cannot hold its three Grams and one basis row; increase "
      "fcvp_memory or reduce output parameters.");
  std::vector<std::unique_ptr<Shard_Basis>> shards;
  std::vector<std::vector<double>> original_residuals;
  if (step == 0)
    original_residuals.resize(devices);
  for (int device = 0; device < devices; ++device) {
    const auto& data = dataset[device];
    const int64_t physical_rows = int64_t(7) * data.Nc + int64_t(3) * data.N;
    if (physical_rows <= 0 || physical_rows > std::numeric_limits<int>::max() ||
        int64_t(3) * data.N > std::numeric_limits<int>::max())
      PRINT_INPUT_ERROR("FCVP batch row count exceeds cuBLAS limits.");
    const int rows = int(physical_rows);
    const int capacity = int(std::min(
      size_t(rows), (budget_elements - fixed_elements) / elements_per_row));
    CHECK(cudaSetDevice(device));
    shards.emplace_back(new Shard_Basis(device, rows, capacity, p, step == 0));
    if (step == 0)
      original_residuals[device].resize(rows);
    entries += size_t(capacity) * (p + 1);
    counts[0] += data.Nc;
    counts[1] += uint64_t(3) * data.N;
    counts[2] += uint64_t(6) * data.sum_virial_Nc;
  }
  std::vector<double> beta(p);
  std::vector<float> zero(parameters, parameters + para.number_of_variables);
  fcvp::Options options;
  options.iterations = para.fcvp_iterations;
  options.epsilon = para.fcvp_epsilon;
  options.tolerance = para.fcvp_tolerance;
  options.total_parameters = para.number_of_variables;
  options.lambda1 = para.lambda_1;
  options.lambda2 = para.lambda_2;
  // The native basis is FP32 even though its Grams are FP64. Damp fragile
  // near-null directions at that precision; this proximal term is about
  // the current iterate and does not add a ridge penalty to the objective.
  options.proximal_relative = std::numeric_limits<float>::epsilon();
  for (int j = 0; j < p; ++j) {
    beta[j] = parameters[indices[j]];
    zero[indices[j]] = 0;
  }
  for (float v : zero) {
    options.fixed_l1 += std::abs(v);
    options.fixed_l2_squared += double(v) * v;
  }
  potential.find_force(para, parameters, false, dataset, false, true, devices);
  const double before = native_objective(para, dataset, parameters);
  if (step == 0) {
    for (int device = 0; device < devices; ++device) {
      CHECK(cudaSetDevice(device));
      auto& shard = *shards[device];
      double* tile = shard.matrix.data() + size_t(shard.tile_rows) * p;
      for (int row = 0; row < shard.rows; row += shard.tile_rows) {
        const int length = std::min(shard.tile_rows, shard.rows - row);
        pack(para, dataset[device], row, length, true, tile);
        CHECK(cudaMemcpy(
          original_residuals[device].data() + row,
          tile,
          size_t(length) * sizeof(double),
          cudaMemcpyDeviceToHost));
      }
    }
  }
  // beta=0 measures the fixed affine offset, including ZBL. No test labels used.
  potential.find_force(para, zero.data(), false, dataset, false, false, devices);
  std::vector<std::vector<double>> zero_residuals(devices);
  for (int device = 0; device < devices; ++device) {
    CHECK(cudaSetDevice(device));
    auto& shard = *shards[device];
    double* tile = shard.matrix.data() + size_t(shard.tile_rows) * p;
    if (shard.tile_rows == shard.rows) {
      pack(para, dataset[device], 0, shard.rows, true, tile);
      continue;
    }
    zero_residuals[device].resize(shard.rows);
    for (int row = 0; row < shard.rows; row += shard.tile_rows) {
      const int length = std::min(shard.tile_rows, shard.rows - row);
      pack(para, dataset[device], row, length, true, tile);
      CHECK(cudaMemcpy(
        zero_residuals[device].data() + row,
        tile,
        size_t(length) * sizeof(double),
        cudaMemcpyDeviceToHost));
    }
  }
  const double lambdas[3] = {para.lambda_e, para.lambda_f, para.lambda_v};
  double reconstruction_worst = 0;
  // Reuse one physical-row tile across the E/F/V blocks. Each device retains
  // only globally normalized sufficient statistics for all output columns.
  for (int device = 0; device < devices; ++device) {
    CHECK(cudaSetDevice(device));
    auto& data = dataset[device];
    auto& shard = *shards[device];
    const int offsets[3] = {0, data.Nc, data.Nc + 3 * data.N};
    const int lengths[3] = {data.Nc, 3 * data.N, 6 * data.Nc};
    if (step == 0)
      shard.device_beta.copy_from_host(beta.data());
    double* target = shard.matrix.data() + size_t(shard.tile_rows) * p;
    std::vector<double> actual;
    if (step == 0)
      actual.resize(shard.tile_rows);
    for (int row = 0; row < shard.rows; row += shard.tile_rows) {
      const int length = std::min(shard.tile_rows, shard.rows - row);
      if (shard.tile_rows != shard.rows)
        CHECK(cudaMemcpy(
          target,
          zero_residuals[device].data() + row,
          size_t(length) * sizeof(double),
          cudaMemcpyHostToDevice));
      for (int j = 0; j < p; ++j) {
        potential.find_output_basis(para, data, j, device);
        pack(
          para,
          data,
          row,
          length,
          false,
          shard.matrix.data() + size_t(shard.tile_rows) * j);
      }
      for (int k = 0; k < 3; ++k) {
        const int first = std::max(row, offsets[k]);
        const int last = std::min(row + length, offsets[k] + lengths[k]);
        if (first < last && counts[k] && lambdas[k]) {
          const int group_rows = last - first;
          const int local_offset = first - row;
          const double alpha = 1.0 / double(counts[k]), one = 1.0;
          const double* a = shard.matrix.data() + local_offset;
          const double* y = target + local_offset;
          check_blas(cublasDgemm(
            shard.blas.handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            p,
            p,
            group_rows,
            &alpha,
            a,
            shard.tile_rows,
            a,
            shard.tile_rows,
            &one,
            shard.h[k].data(),
            p));
          check_blas(cublasDgemv(
            shard.blas.handle,
            CUBLAS_OP_T,
            group_rows,
            p,
            &alpha,
            a,
            shard.tile_rows,
            y,
            1,
            &one,
            shard.g[k].data(),
            1));
          check_blas(cublasSetPointerMode(shard.blas.handle, CUBLAS_POINTER_MODE_DEVICE));
          check_blas(cublasDdot(shard.blas.handle, group_rows, y, 1, y, 1, shard.dot.data()));
          check_blas(cublasSetPointerMode(shard.blas.handle, CUBLAS_POINTER_MODE_HOST));
          check_blas(cublasDaxpy(
            shard.blas.handle,
            1,
            &one,
            shard.dot.data(),
            1,
            shard.yy.data() + k,
            1));
        }
      }
      if (step == 0 && shard.tile_rows != shard.rows)
        check_reconstruction(
          shard, original_residuals[device], row, length, actual, reconstruction_worst);
    }
  }
  if (step == 0) {
    for (int device = 0; device < devices; ++device) {
      CHECK(cudaSetDevice(device));
      auto& shard = *shards[device];
      if (shard.tile_rows != shard.rows)
        continue;
      std::vector<double> actual(shard.rows);
      check_reconstruction(
        shard,
        original_residuals[device],
        0,
        shard.rows,
        actual,
        reconstruction_worst);
    }
    printf(
      "FCVP native E/F/V basis reconstruction normalized max error: %.9g\n",
      reconstruction_worst);
    if (reconstruction_worst > 5e-4)
      PRINT_INPUT_ERROR("FCVP native linearity check failed.");
  }
  original_residuals.clear();
  zero_residuals.clear();
  std::vector<fcvp::Group> groups;
  for (int k = 0; k < 3; ++k) {
    if (!counts[k] || !lambdas[k])
      continue;
    fcvp::Group group;
    group.lambda = lambdas[k];
    group.h.resize(size_t(p) * p);
    group.g.resize(p);
    for (int device = 0; device < devices; ++device) {
      CHECK(cudaSetDevice(device));
      std::vector<double> h(size_t(p) * p), g(p), yy(3);
      shards[device]->h[k].copy_to_host(h.data());
      shards[device]->g[k].copy_to_host(g.data());
      shards[device]->yy.copy_to_host(yy.data());
      for (size_t j = 0; j < size_t(p) * p; ++j)
        group.h[j] += h[j];
      for (int j = 0; j < p; ++j)
        group.g[j] += g[j];
      group.yy += yy[k] / double(counts[k]);
    }
    groups.push_back(std::move(group));
  }
  fcvp::Result result;
  try {
    result = fcvp::solve(groups, beta, options);
  } catch (const std::exception& e) {
    PRINT_INPUT_ERROR(e.what());
  }
  // Round to the stored FP32 model, then check the real predictor's objective.
  // This protects against both Gram cancellation and basis/forward roundoff.
  double after = before;
  bool accepted = false;
  for (int attempt = 0; attempt < 12; ++attempt) {
    for (int j = 0; j < p; ++j)
      parameters[indices[j]] = float(result.beta[j]);
    potential.find_force(para, parameters, false, dataset, false, false, devices);
    after = native_objective(para, dataset, parameters);
    if (std::isfinite(after) && after <= before) {
      accepted = true;
      break;
    }
    for (int j = 0; j < p; ++j)
      result.beta[j] = 0.5 * (result.beta[j] + beta[j]);
  }
  if (!accepted) {
    for (int j = 0; j < p; ++j)
      parameters[indices[j]] = float(beta[j]);
    after = before;
  }
  shards.clear();
  CHECK(cudaSetDevice(0));
  CHECK(cudaDeviceSynchronize());
  double seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
  FILE* log = my_fopen("fcvp.out", step == 0 ? "w" : "a");
  if (step == 0)
    fprintf(
      log,
      "# step before after accepted mm_iterations inner_converged seconds basis_bytes "
      "proximal_shift\n");
  fprintf(
    log,
    "%d %.12g %.12g %d %d %d %.9g %zu %.9g\n",
    step,
    before,
    after,
    accepted,
    result.iterations,
    result.converged,
    seconds,
    entries * sizeof(double),
    result.proximal_shift);
  fclose(log);
}

// A GPU gate, separate from CPU tests: compare the inherited analytical
// derivatives (with the opt-in loss normalization) against native finite differences.
void fcvp_verify_gradient(
  Parameters& para, GNEP& potential, std::vector<Dataset>& dataset, float* parameters)
{
  potential.find_force(para, parameters, true, dataset, false, true, dataset.size());
  std::vector<float> analytical(para.number_of_variables);
  potential.gradients.grad_sum.copy_to_host(analytical.data());
  double norm2 = 0;
  for (int i = 0; i < para.number_of_variables; ++i)
    norm2 += double(parameters[i]) * parameters[i];
  const double norm =
    std::sqrt(norm2 / para.number_of_variables + para.fcvp_epsilon * para.fcvp_epsilon);
  std::vector<int> indices;
  if (para.fcvp_check == 2) {
    indices.reserve(para.number_of_variables);
    for (int i = 0; i < para.number_of_variables; ++i)
      indices.push_back(i);
  } else {
    indices = {
      0,
      para.dim * para.num_neurons1,
      (para.dim + 1) * para.num_neurons1,
      para.number_of_variables_ann - 1,
      para.number_of_variables_ann,
      para.number_of_variables - 1};
    for (int i = 0; i < 8; ++i)
      indices.push_back((uint64_t(i) * 7919 + para.seed) % para.number_of_variables);
  }
  double worst = 0;
  for (int i : indices) {
    float x = parameters[i], delta = 1e-3f * std::max(1.0f, std::abs(x));
    parameters[i] = x + delta;
    const double plus_x = parameters[i];
    potential.find_force(para, parameters, false, dataset, false, false, dataset.size());
    double plus = native_objective(para, dataset, parameters);
    parameters[i] = x - delta;
    const double minus_x = parameters[i];
    potential.find_force(para, parameters, false, dataset, false, false, dataset.size());
    double minus = native_objective(para, dataset, parameters);
    parameters[i] = x;
    double finite_difference = (plus - minus) / (plus_x - minus_x);
    double derivative = analytical[i] +
                        para.lambda_1 * ((x > 0) - (x < 0)) / para.number_of_variables +
                        para.lambda_2 * x / (para.number_of_variables * norm);
    double error = std::abs(derivative - finite_difference) / (1 + std::abs(finite_difference));
    printf(
      "FCVP gradient check parameter %d analytical %.9g finite_difference %.9g error %.9g\n",
      i,
      derivative,
      finite_difference,
      error);
    if (!std::isfinite(error))
      PRINT_INPUT_ERROR("Nonfinite FCVP gradient check.");
    worst = std::max(worst, error);
  }
  if (worst > 5e-3)
    PRINT_INPUT_ERROR("FCVP analytical gradient gate failed; do not benchmark.");
  printf("FCVP analytical gradient gate passed (normalized max error %.9g).\n", worst);
}
