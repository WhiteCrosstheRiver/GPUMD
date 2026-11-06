// SPDX-License-Identifier: GPL-3.0-or-later

/*----------------------------------------------------------------------------80
Single-device LM loop using native E/F/V predictions and moment rows.
Included by gnep.cu so the native harmonic constants have one definition.
------------------------------------------------------------------------------*/

#pragma once
#include "nep_lm.cuh"
#include "nep_lm_cached.cuh"
#include "nep_lm_candidates.cuh"
#include "nep_lm_kernels.cuh"
#include "nep_lm_solver.cuh"
#include "nep_lm_warp.cuh"
#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <type_traits>

namespace lm_training
{

static __global__ void add_rows_double(
  int parameters, int count, const float* rows, const float* coefficients, double* gradient)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < parameters) {
    double sum = 0.0;
    for (int r = 0; r < count; ++r) {
      sum += double(rows[size_t(r) * parameters + i]) * coefficients[r];
    }
    gradient[i] += sum;
  }
}

static lm::Arch architecture(const Parameters& p)
{
  lm::Arch a;
  a.num_types = p.num_types;
  a.dim = p.dim;
  a.dim_R = p.dim_radial;
  a.n_R = p.n_max_radial + 1;
  a.n_A = p.n_max_angular + 1;
  a.K_R = p.basis_size_radial + 1;
  a.K_A = p.basis_size_angular + 1;
  a.L_max = p.L_max;
  a.L_max_4body = p.L_max_4body;
  a.L_max_5body = p.L_max_5body;
  a.n_lm = a.L_max * (a.L_max + 2);
  a.neurons = p.num_neurons1;
  for (int t = 0; t < p.num_types; ++t) {
    a.rc_R[t] = p.rc_radial[t];
    a.rc_A[t] = p.rc_angular[t];
  }
  lm::validate_arch(a);
  return a;
}

static void evaluate(
  Parameters& p,
  GNEP& potential,
  const float* theta,
  bool require_grad,
  std::vector<Dataset>& datasets,
  bool calculate_neighbor)
{
  Dataset& data = datasets[0];
  potential.find_force(
    p,
    theta,
    require_grad,
    datasets,
    false,
    calculate_neighbor,
    0,
    1,
    data.Nc,
    data.N,
    6 * data.sum_virial_Nc);
}

static double loss(const Parameters& p, const Batch_MSE& mse, const float* theta)
{
  const double eps2 = p.fcvp_epsilon * p.fcvp_epsilon;
  double value = p.lambda_e * std::sqrt(mse.energy + eps2) +
                 p.lambda_f * std::sqrt(mse.force + eps2) +
                 p.lambda_v * std::sqrt(mse.virial + eps2);
  if (p.lambda_2 > 0) {
    double norm = 0.0;
    for (int i = 0; i < p.number_of_variables; ++i) {
      norm += double(theta[i]) * theta[i];
    }
    value += p.lambda_2 * std::sqrt(norm / p.number_of_variables + eps2);
  }
  return value;
}

struct Cache {
  GPU_Vector<float> phi;
  GPU_Vector<float> angular;
  GPU_Vector<float> force_weights;
  GPU_Vector<int2> records;
  std::vector<int> atom_structure;
  std::vector<float> reference_force;

  lm::Shard shard(Dataset& data, GNEP_Data& native)
  {
    return {
      data.N,
      data.Nc,
      data.type.data(),
      data.batch_idx.data(),
      data.Na_sum.data(),
      native.NN_radial.data(),
      native.NL_radial.data(),
      native.x12_radial.data(),
      native.y12_radial.data(),
      native.z12_radial.data(),
      native.NN_angular.data(),
      native.NL_angular.data(),
      native.x12_angular.data(),
      native.y12_angular.data(),
      native.z12_angular.data()};
  }

  Cache(Parameters& p, const lm::Arch& a, Dataset& data, GNEP_Data& native)
  {
    phi.resize(size_t(data.N) * a.num_types * a.K_R, 0.0f);
    angular.resize(std::max(size_t(1), size_t(data.N) * a.num_types * a.K_A * a.n_lm), 0.0f);
    std::vector<int2> record_host(data.N);
    atom_structure.resize(data.N);
    for (int s = 0; s < data.Nc; ++s) {
      for (int i = data.Na_sum_cpu[s]; i < data.Na_sum_cpu[s] + data.Na_cpu[s]; ++i) {
        record_host[i] = make_int2(s, i);
        atom_structure[i] = s;
      }
    }
    records.resize(data.N);
    records.copy_from_host(record_host.data());
    reference_force.resize(3 * data.N);
    data.force_ref_gpu.copy_to_host(reference_force.data());
    std::vector<int> types(data.N);
    data.type.copy_to_host(types.data());
    std::vector<float> weight_host(data.N);
    for (int i = 0; i < data.N; ++i) {
      double weight = p.type_weight_cpu[types[i]];
      if (p.force_delta > 0) {
        double magnitude = 0.0;
        for (int d = 0; d < 3; ++d) {
          const double f = reference_force[i + d * data.N];
          magnitude += f * f;
        }
        weight *= std::sqrt(p.force_delta / (p.force_delta + std::sqrt(magnitude)));
      }
      weight_host[i] = weight * data.weight_cpu[atom_structure[i]];
    }
    force_weights.resize(data.N);
    force_weights.copy_from_host(weight_host.data());
    lm::moments_kernel<<<(data.N + 127) / 128, 128>>>(
      a, shard(data, native), phi.data(), angular.data());
    GPU_CHECK_KERNEL
    GPU_Vector<float> q(size_t(data.N) * p.dim);
    lm::descriptors_kernel<<<(data.N + 127) / 128, 128>>>(
      a,
      lm::make_ann_view(a, native.parameters.data()),
      shard(data, native),
      p.q_scaler_gpu(0).data(),
      phi.data(),
      angular.data(),
      q.data());
    GPU_CHECK_KERNEL
    std::vector<float> actual(size_t(data.N) * p.dim);
    std::vector<float> expected(actual.size());
    q.copy_to_host(actual.data());
    native.descriptors.copy_to_host(expected.data(), expected.size());
    double error = 0.0;
    for (int d = 0; d < p.dim; ++d) {
      for (int i = 0; i < data.N; ++i) {
        const size_t index = size_t(d) * data.N + i;
        const double reference = expected[index] * p.q_scaler_cpu[d];
        if (!std::isfinite(actual[index]) || !std::isfinite(reference)) {
          PRINT_INPUT_ERROR("Nonfinite LM/native descriptor in the moment gate.");
        }
        error = std::max(error, std::abs(actual[index] - reference) / (1.0 + std::abs(reference)));
      }
    }
    printf("LM moment descriptor gate max_scaled_error=%.9g atoms=%d\n", error, data.N);
    if (error > 2e-5) {
      PRINT_INPUT_ERROR("LM moment descriptors disagree with the native predictor.");
    }
  }
};

} // namespace lm_training

struct NEP_LM::State {
  lm::Arch arch;
  int probes;
  bool use_guard;
  bool check_gradient;
  bool fast = false;
  bool warp_rows = false;
  bool warned_atom_cache = false;
  std::vector<float> guard_parameters;
  double guard_value = 0.0;
  double guard_ratio;
  bool checked = false;
  double best_guard = std::numeric_limits<double>::infinity();
  int n;
  LMConfig config;
  std::unique_ptr<LMSolver> solver;
  std::map<Dataset*, std::unique_ptr<lm_training::Cache>> caches;
  GPU_Vector<float> rows;
  GPU_Vector<float> atom_cache;
  GPU_Vector<float> dphi;
  GPU_Vector<float> dangular;
  GPU_Vector<float> scales;
  GPU_Vector<float> coefficients;
  GPU_Vector<double> gradient64;
  GPU_Vector<float> residual;
  GPU_Vector<lm::DenseProbe> probe_gpu;
  GPU_Vector<float> deltas;
  FILE* output;

  explicit State(Parameters& p) : arch(lm_training::architecture(p)), n(p.number_of_variables)
  {
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices));
    if (
      devices != 1 || !p.loss_rmse || p.lambda_1 != 0 || p.weight_decay != 0) {
      PRINT_INPUT_ERROR("LM requires one visible GPU, loss_rmse, lambda_1 0, "
                        "and weight_decay 0.");
    }
    probes = p.lm_probes;
    use_guard = true;
    check_gradient = p.lm_check;
    fast = p.lm_fast;
    if (fast) {
      int major = 0;
      int maximum_shared = 0;
      CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
      CHECK(cudaDeviceGetAttribute(&maximum_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
      const size_t shared_bytes = lm::rows_warp_shared_bytes(arch, n);
      warp_rows = major >= 7 && shared_bytes <= size_t(maximum_shared);
      if (warp_rows) {
        CHECK(lm::configure_rows_warp_kernels(shared_bytes));
      }
    }
    guard_ratio = p.lm_guard_ratio;
    config.num_candidates = p.lm_candidates;
    config.ema[LM_FORCE] = p.lm_force_ema;
    config.lambda0 = p.lm_lambda0;
    solver.reset(new LMSolver(n, config));
    gradient64.resize(n);
    deltas.resize(size_t(config.num_candidates) * n);
    output = my_fopen("lm.out", "w");
    fprintf(
      output,
      "# step before after accepted lambda build_seconds candidate_seconds guard_before "
      "guard_selected guard_best\n");
    printf(
      "LM parameters=%d probes=%d candidates=%d guard=%d\n",
      n,
      probes,
      config.num_candidates,
      int(use_guard));
    if (fast) {
      printf(
        "LM fast path: cached atom intermediates, %s rows, contiguous candidate passes.\n",
        warp_rows ? "cooperative shared" : "atomic");
    }
    if (use_guard) {
      printf(
        "LM uses %s for step selection; test.xyz is not used for selection.\n",
        p.lm_guard_filename.c_str());
    }
  }

  ~State()
  {
    fclose(output);
  }
};

NEP_LM::NEP_LM(Parameters& para, GNEP&) : state(new State(para))
{
}

NEP_LM::~NEP_LM() = default;

Batch_MSE NEP_LM::update(
  Parameters& p,
  GNEP& potential,
  std::vector<Dataset>& datasets,
  std::vector<Dataset>& guard,
  float* theta,
  int step)
{
  auto& s = *state;
  if (datasets.size() != 1 || (s.use_guard && guard.empty())) {
    PRINT_INPUT_ERROR("LM needs a single training shard and an available guard set.");
  }
  auto& data = datasets[0];
  const auto begin = std::chrono::steady_clock::now();
  lm_training::evaluate(p, potential, theta, false, datasets, true);
  const Batch_MSE mse = get_batch_mse(p, datasets, true);
  const double before = lm_training::loss(p, mse, theta);
  if (!std::isfinite(before)) {
    PRINT_INPUT_ERROR("Nonfinite initial LM objective.");
  }
  auto& native = potential.cached_data();
  auto& pointer = s.caches[&data];
  if (!pointer) {
    pointer.reset(new lm_training::Cache(p, s.arch, data, native));
  }
  auto& cache = *pointer;
  const auto shard = cache.shard(data, native);
  const auto ann = lm::make_ann_view(s.arch, native.parameters.data());
  const int blocks = (data.N + 127) / 128;
  s.rows.resize(size_t(data.Nc) * s.n);
  s.dphi.resize(cache.phi.size());
  s.dangular.resize(cache.angular.size());
  s.scales.resize(data.Nc);
  s.coefficients.resize(data.Nc);
  s.probe_gpu.resize(data.Nc);
  lm::AtomCache atom_cache = {};
  bool use_atom_cache = s.fast;
  if (use_atom_cache) {
    const size_t count = lm::atom_cache_floats(s.arch, data.N);
    if (s.atom_cache.size() != count) {
      size_t free_bytes;
      size_t total_bytes;
      CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
      use_atom_cache = count * sizeof(float) <= free_bytes / 2;
      if (use_atom_cache) {
        s.atom_cache.resize(count);
      }
    }
    if (use_atom_cache) {
      atom_cache = lm::make_atom_cache_view(s.arch, data.N, s.atom_cache.data());
      lm::atom_cache_kernel<<<blocks, 128>>>(
        s.arch,
        ann,
        shard,
        p.q_scaler_gpu(0).data(),
        cache.phi.data(),
        cache.angular.data(),
        atom_cache);
      GPU_CHECK_KERNEL
    } else if (!s.warned_atom_cache) {
      printf("LM atom cache exceeds the memory budget; using reference rows.\n");
      s.warned_atom_cache = true;
    }
  }
  const bool use_warp_rows = s.warp_rows && use_atom_cache;
  s.gradient64.fill(0.0);
  s.solver->begin_curvature();
  const double eps2 = p.fcvp_epsilon * p.fcvp_epsilon;
  const double omega[3] = {
    p.lambda_e / (std::sqrt(mse.energy + eps2) * data.Nc),
    p.lambda_f / (std::sqrt(mse.force + eps2) * 3.0 * data.N),
    data.sum_virial_Nc ? p.lambda_v / (std::sqrt(mse.virial + eps2) * 6.0 * data.sum_virial_Nc)
                       : 0.0};
  std::vector<float> scale(data.Nc);
  std::vector<float> coeff(data.Nc);
  std::vector<float> diff_e(data.Nc);
  std::vector<float> diff_v(6 * data.Nc);
  data.diff_gpu_e.copy_to_host(diff_e.data());
  data.diff_gpu_v.copy_to_host(diff_v.data());
  std::vector<lm::DenseProbe> probes(data.Nc);
  auto launch_rows = [&](auto energy, const float* dphi, const float* dangular) {
    constexpr bool ENERGY = decltype(energy)::value;
    if (use_warp_rows) {
      lm::launch_rows_warp_kernel<ENERGY>(
        s.arch,
        ann,
        atom_cache,
        shard,
        s.scales.data(),
        p.q_scaler_gpu(0).data(),
        cache.phi.data(),
        cache.angular.data(),
        dphi,
        dangular,
        s.rows.data(),
        s.n);
    } else if (use_atom_cache) {
      lm::rows_cached_kernel<ENERGY><<<blocks, 128>>>(
        s.arch,
        ann,
        atom_cache,
        shard,
        cache.records.data(),
        data.N,
        s.scales.data(),
        p.q_scaler_gpu(0).data(),
        cache.phi.data(),
        cache.angular.data(),
        dphi,
        dangular,
        s.rows.data(),
        s.n);
    } else {
      lm::rows_kernel<ENERGY><<<blocks, 128>>>(
        s.arch,
        ann,
        shard,
        cache.records.data(),
        data.N,
        s.scales.data(),
        p.q_scaler_gpu(0).data(),
        cache.phi.data(),
        cache.angular.data(),
        dphi,
        dangular,
        s.rows.data(),
        s.n);
    }
  };
  auto rows = [&](bool energy, int kind, int index) {
    s.scales.copy_from_host(scale.data());
    s.rows.fill(0.0f);
    if (energy) {
      launch_rows(std::true_type{}, nullptr, nullptr);
    } else {
      for (int i = 0; i < data.Nc; ++i) {
        probes[i] = {kind, index, i, uint64_t(p.seed), step};
      }
      s.probe_gpu.copy_from_host(probes.data());
      s.dphi.fill(0.0f);
      s.dangular.fill(0.0f);
      lm::dense_tangent_kernel<<<blocks, 128>>>(
        s.arch,
        shard,
        s.probe_gpu.data(),
        cache.records.data(),
        data.N,
        kind == lm::DENSE_RESIDUAL ? s.residual.data() : nullptr,
        1.0f,
        s.dphi.data(),
        s.dangular.data(),
        cache.force_weights.data());
      GPU_CHECK_KERNEL
      launch_rows(std::false_type{}, s.dphi.data(), s.dangular.data());
    }
    GPU_CHECK_KERNEL
  };
  auto add_gradient = [&]() {
    s.coefficients.copy_from_host(coeff.data());
    lm_training::add_rows_double<<<(s.n + 127) / 128, 128>>>(
      s.n, data.Nc, s.rows.data(), s.coefficients.data(), s.gradient64.data());
    GPU_CHECK_KERNEL
  };
  for (int i = 0; i < data.Nc; ++i) {
    scale[i] = data.weight_cpu[i] / data.Na_cpu[i];
    coeff[i] = omega[LM_ENERGY] * data.weight_cpu[i] * diff_e[i];
  }
  rows(true, 0, 0);
  add_gradient();
  s.solver->accumulate(LM_ENERGY, s.rows.data(), data.Nc);
  if (p.lambda_f > 0) {
    std::vector<float> predicted(3 * data.N);
    std::vector<float> weights(data.N);
    std::vector<float> field(3 * data.N);
    data.force.copy_to_host(predicted.data());
    cache.force_weights.copy_to_host(weights.data());
    for (int i = 0; i < data.N; ++i) {
      for (int d = 0; d < 3; ++d) {
        const int index = i + d * data.N;
        field[3 * i + d] = omega[LM_FORCE] * weights[i] * weights[i] *
                           (predicted[index] - cache.reference_force[index]);
      }
    }
    s.residual.resize(field.size());
    s.residual.copy_from_host(field.data());
    std::fill(scale.begin(), scale.end(), -1.0f);
    std::fill(coeff.begin(), coeff.end(), 1.0f);
    rows(false, lm::DENSE_RESIDUAL, 0);
    add_gradient();
    std::fill(scale.begin(), scale.end(), -1.0f / std::sqrt(double(s.probes)));
    for (int j = 0; j < s.probes; ++j) {
      rows(false, lm::DENSE_RADEMACHER, j);
      s.solver->accumulate(LM_FORCE, s.rows.data(), data.Nc);
    }
  }
  if (p.lambda_v > 0 && data.sum_virial_Nc > 0) {
    for (int d = 0; d < 6; ++d) {
      const double shear = d >= 3 ? p.lambda_shear : 1.0;
      for (int i = 0; i < data.Nc; ++i) {
        scale[i] = data.has_virial[i] ? -data.weight_cpu[i] * shear / data.Na_cpu[i] : 0.0f;
        coeff[i] = shear ? omega[LM_VIRIAL] * data.weight_cpu[i] * diff_v[6 * i + d] / shear : 0.0f;
      }
      rows(false, lm::DENSE_STRAIN, d);
      add_gradient();
      s.solver->accumulate(LM_VIRIAL, s.rows.data(), data.Nc);
    }
  }
  s.solver->end_curvature();
  std::vector<double> gradient(s.n);
  s.gradient64.copy_to_host(gradient.data());
  for (double value : gradient) {
    if (!std::isfinite(value)) {
      PRINT_INPUT_ERROR("Nonfinite LM gradient.");
    }
  }
  if (s.check_gradient && !s.checked) {
    // Scale the native MSE gradient into the gradient of lambda*sqrt(MSE + epsilon^2).
    const float lambda_e = p.lambda_e;
    const float lambda_f = p.lambda_f;
    const float lambda_v = p.lambda_v;
    p.lambda_e /= 2.0 * std::sqrt(mse.energy + eps2);
    p.lambda_f /= 2.0 * std::sqrt(mse.force + eps2);
    p.lambda_v /= 2.0 * std::sqrt(mse.virial + eps2);
    lm_training::evaluate(p, potential, theta, true, datasets, true);
    p.lambda_e = lambda_e;
    p.lambda_f = lambda_f;
    p.lambda_v = lambda_v;
    std::vector<double> configuration_gradients(size_t(data.Nc) * s.n);
    potential.getGradients(0).grad_sum.copy_to_host(configuration_gradients.data());
    std::vector<double> reference(s.n, 0.0);
    for (int c = 0; c < data.Nc; ++c) {
      for (int i = 0; i < s.n; ++i) {
        reference[i] += configuration_gradients[size_t(c) * s.n + i];
      }
    }
    double error = 0.0;
    double difference_norm = 0.0;
    double reference_norm = 0.0;
    int worst = 0;
    for (int i = 0; i < s.n; ++i) {
      if (!std::isfinite(reference[i])) {
        PRINT_INPUT_ERROR("Nonfinite native gradient in the LM gate.");
      }
      const double difference =
        std::abs(double(gradient[i]) - reference[i]) / (1.0 + std::abs(reference[i]));
      difference_norm += (gradient[i] - reference[i]) * (gradient[i] - reference[i]);
      reference_norm += double(reference[i]) * reference[i];
      if (difference > error) {
        error = difference;
        worst = i;
      }
    }
    const double relative = std::sqrt(difference_norm / std::max(reference_norm, 1e-24));
    printf(
      "LM native gradient gate max_scaled_error=%.9g relative_norm_error=%.9g parameter=%d new=%g "
      "native=%g\n",
      error,
      relative,
      worst,
      gradient[worst],
      reference[worst]);
    if (error > 0.002 || relative > 0.002) {
      PRINT_INPUT_ERROR("LM dense gradient disagrees with native analytical gradient.");
    }
    s.checked = true;
  }
  double norm2 = 0.0;
  for (int i = 0; i < s.n; ++i) {
    norm2 += double(theta[i]) * theta[i];
  }
  const double l2 = p.lambda_2 / (s.n * std::sqrt(norm2 / s.n + eps2));
  std::vector<double> gradient64(s.n);
  for (int i = 0; i < s.n; ++i) {
    gradient64[i] = gradient[i] + l2 * theta[i];
  }
  s.gradient64.copy_from_host(gradient64.data());
  s.solver->set_l2(l2);
  std::vector<bool> valid;
  s.solver->solve_candidates(omega, s.gradient64.data(), s.deltas.data(), valid);
  const auto built = std::chrono::steady_clock::now();
  std::vector<float> deltas(s.deltas.size());
  s.deltas.copy_to_host(deltas.data());
  std::vector<float> candidate(s.n);
  std::vector<lm_training::Candidate_Loss> candidate_losses;
  if (s.fast) {
    candidate_losses = lm_training::evaluate_candidate_losses(
      theta,
      deltas.data(),
      s.n,
      valid,
      before,
      candidate,
      [&](float* trial, bool calculate_neighbor, Batch_MSE& candidate_mse) {
        lm_training::evaluate(p, potential, trial, false, datasets, calculate_neighbor);
        candidate_mse = get_batch_mse(p, datasets, true);
        return lm_training::loss(p, candidate_mse, trial);
      });
  }
  double guard_before = 0.0;
  bool guard_neighbors_current = false;
  if (s.use_guard) {
    const bool unchanged = s.fast && s.guard_parameters.size() == size_t(s.n) &&
                           std::memcmp(s.guard_parameters.data(), theta, s.n * sizeof(float)) == 0;
    if (unchanged) {
      guard_before = s.guard_value;
    } else {
      lm_training::evaluate(p, potential, theta, false, guard, true);
      guard_before = lm_training::loss(p, get_batch_mse(p, guard, true), theta);
      guard_neighbors_current = true;
    }
    if (!std::isfinite(guard_before)) {
      PRINT_INPUT_ERROR("Nonfinite LM guard objective.");
    }
    s.best_guard = std::min(s.best_guard, guard_before);
  }
  std::vector<float> best(theta, theta + s.n);
  double best_loss = before;
  Batch_MSE best_mse = mse;
  int selected = -1;
  double selected_guard = guard_before;
  if (s.fast) {
    for (const auto& entry : candidate_losses) {
      for (int i = 0; i < s.n; ++i) {
        candidate[i] = theta[i] + deltas[size_t(entry.index) * s.n + i];
      }
      lm_training::evaluate(p, potential, candidate.data(), false, guard, !guard_neighbors_current);
      guard_neighbors_current = true;
      const double guard_loss =
        lm_training::loss(p, get_batch_mse(p, guard, true), candidate.data());
      if (!std::isfinite(guard_loss) || guard_loss > s.guard_ratio * s.best_guard) {
        continue;
      }
      selected = entry.index;
      best_loss = entry.loss;
      best_mse = entry.mse;
      selected_guard = guard_loss;
      best = candidate;
      break;
    }
  } else {
    for (int j = 0; j < s.config.num_candidates; ++j) {
      if (!valid[j]) {
        continue;
      }
      for (int i = 0; i < s.n; ++i) {
        candidate[i] = theta[i] + deltas[size_t(j) * s.n + i];
      }
      lm_training::evaluate(p, potential, candidate.data(), false, datasets, true);
      const Batch_MSE candidate_mse = get_batch_mse(p, datasets, true);
      const double value = lm_training::loss(p, candidate_mse, candidate.data());
      if (!std::isfinite(value) || value >= best_loss) {
        continue;
      }
      if (s.use_guard) {
        lm_training::evaluate(p, potential, candidate.data(), false, guard, true);
        const double guard_loss =
          lm_training::loss(p, get_batch_mse(p, guard, true), candidate.data());
        // Anchor the tolerance to accepted history so stepwise increases cannot compound.
        if (!std::isfinite(guard_loss) || guard_loss > s.guard_ratio * s.best_guard) {
          continue;
        }
        selected_guard = guard_loss;
      }
      selected = j;
      best_loss = value;
      best_mse = candidate_mse;
      best = candidate;
    }
  }
  s.solver->report(selected, selected >= 0);
  if (s.use_guard && selected >= 0) {
    s.best_guard = std::min(s.best_guard, selected_guard);
  }
  std::copy(best.begin(), best.end(), theta);
  if (s.fast) {
    s.guard_parameters = best;
    s.guard_value = selected >= 0 ? selected_guard : guard_before;
  }
  const auto end = std::chrono::steady_clock::now();
  fprintf(
    s.output,
    "%d %.12g %.12g %d %.12g %.6f %.6f %.12g %.12g %.12g\n",
    step,
    before,
    best_loss,
    selected >= 0,
    s.solver->lambda(),
    std::chrono::duration<double>(built - begin).count(),
    std::chrono::duration<double>(end - built).count(),
    guard_before,
    selected >= 0 ? selected_guard : guard_before,
    s.use_guard ? s.best_guard : 0.0);
  fflush(s.output);
  return best_mse;
}
