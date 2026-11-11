/*----------------------------------------------------------------------------80
Experimental optimizers for gnep (included at the end of fitness.cu).

GNEP_EXPERIMENT=snes|adam|ivon replaces the training loop. All methods share one
evaluator, so they minimize the same loss (lambda-weighted RMSE, as loss_rmse) on the
same mini-batch sequence, and they are timed the same way. Test-set reports and the
premise diagnostics are excluded from the reported algorithm time.

  snes  separable NES with ranking, as in nep (global ranking only).
  adam  Adam on the analytical gradient.
  ivon  improved variational online Newton (Shen et al., ICML 2024): Adam-like cost,
        but the iterate is a Gaussian N(m, sigma^2) whose precision tracks a diagonal
        Hessian estimate, so it optimizes the same smoothed loss as SNES.

Environment (all optional): GNEP_BUDGET seconds of algorithm time, GNEP_REPORT seconds
between test reports, GNEP_POP, GNEP_MIRROR, GNEP_SIGMA0, GNEP_PREMISE (diagnose every
k generations), GNEP_LR, GNEP_LR_END, GNEP_IVON_ESS, GNEP_IVON_H0, GNEP_IVON_WD,
GNEP_IVON_BETA2, GNEP_IVON_CLIP, GNEP_GRADCHECK.
------------------------------------------------------------------------------*/

#include <algorithm>
#include <cstdlib>
#include <numeric>

namespace
{
double env_double(const char* name, double fallback)
{
  const char* value = std::getenv(name);
  return value ? std::atof(value) : fallback;
}

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0)
{
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

double dot(const std::vector<double>& a, const std::vector<double>& b)
{
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    s += a[i] * b[i];
  }
  return s;
}

double cosine(const std::vector<double>& a, const std::vector<double>& b)
{
  const double na = std::sqrt(dot(a, a));
  const double nb = std::sqrt(dot(b, b));
  return (na > 0 && nb > 0) ? dot(a, b) / (na * nb) : 0.0;
}

class Experiment_Evaluator
{
public:
  Experiment_Evaluator(Parameters& para, GNEP& gnep, int n) : para(para), gnep(gnep), n(n)
  {
    theta.resize(n);
  }

  double loss(const std::vector<double>& x, GNEPTrainingBatch& batch)
  {
    const auto t0 = Clock::now();
    run(x, batch, false);
    const double value = combine(get_batch_mse(para, batch.shards, true));
    forward_seconds += seconds_since(t0);
    ++forwards;
    return value;
  }

  double gradient(const std::vector<double>& x, GNEPTrainingBatch& batch, std::vector<double>& g)
  {
    const auto t0 = Clock::now();
    if (fast) {
      // Energy-only directional-derivative gradient (fast_gradient.cuh).
      run(x, batch, false);
      gnep.fast_rmse_gradient(
        para, batch.shards, batch.num_configurations, batch.num_atoms, batch.virial_components, g);
      const double value = combine(get_batch_mse(para, batch.shards, true));
      gradient_seconds += seconds_since(t0);
      ++gradients;
      return value;
    }
    run(x, batch, true);
    gnep.accumulate_rmse_gradients(
      para, batch.shards, batch.num_configurations, batch.num_atoms, batch.virial_components);
    g.assign(n, 0.0);
    std::vector<double> local(n);
    for (int device = 0; device < static_cast<int>(batch.shards.size()); ++device) {
      CHECK(cudaSetDevice(device));
      Gradients& gradients = gnep.getGradients(device);
      reduce_configuration_gradients<<<(n + 255) / 256, 256>>>(
        batch.shards[device].Nc, n, gradients.grad_sum.data(), gradients.local_sum.data());
      GPU_CHECK_KERNEL
      gradients.local_sum.copy_to_host(local.data());
      for (int i = 0; i < n; ++i) {
        g[i] += local[i];
      }
    }
    CHECK(cudaSetDevice(0));
    const double value = combine(get_batch_mse(para, batch.shards, true));
    gradient_seconds += seconds_since(t0);
    ++gradients;
    return value;
  }

  double combine(const Batch_MSE& mse) const
  {
    const double eps2 = double(para.fcvp_epsilon) * para.fcvp_epsilon;
    double value =
      para.lambda_e * std::sqrt(mse.energy + eps2) + para.lambda_f * std::sqrt(mse.force + eps2);
    if (mse.virial_configurations) {
      value += para.lambda_v * std::sqrt(mse.virial + eps2);
    }
    return value;
  }

  bool fast = std::getenv("GNEP_FASTGRAD") != nullptr;
  long forwards = 0;
  long gradients = 0;
  double forward_seconds = 0.0;
  double gradient_seconds = 0.0;

private:
  Parameters& para;
  GNEP& gnep;
  int n;
  std::vector<float> theta;

  void run(const std::vector<double>& x, GNEPTrainingBatch& batch, bool require_grad)
  {
    for (int i = 0; i < n; ++i) {
      theta[i] = static_cast<float>(x[i]);
    }
    std::vector<std::thread> workers;
    for (int device = 0; device < static_cast<int>(batch.shards.size()); ++device) {
      workers.emplace_back([&, device]() {
        gnep.find_force(
          para,
          theta.data(),
          require_grad,
          batch.shards,
          false,
          true,
          device,
          1,
          batch.num_configurations,
          batch.num_atoms,
          batch.virial_components);
      });
    }
    for (std::thread& worker : workers) {
      worker.join();
    }
  }
};
} // namespace

void Fitness::run_experiment(Parameters& para, float* initial_parameters)
{
  const std::string method = std::getenv("GNEP_EXPERIMENT");
  if (method != "snes" && method != "adam" && method != "ivon") {
    PRINT_INPUT_ERROR("GNEP_EXPERIMENT must be snes, adam or ivon.");
  }
  if (!para.loss_rmse) {
    PRINT_INPUT_ERROR("GNEP_EXPERIMENT requires loss_rmse.");
  }
  const int n = number_of_variables;
  GNEP& gnep = *static_cast<GNEP*>(potential.get());
  Experiment_Evaluator eval(para, gnep, n);
  std::mt19937 rng(para.is_seed_set ? para.seed : 12345);
  std::normal_distribution<double> normal(0.0, 1.0);

  const double budget = env_double("GNEP_BUDGET", 900.0);
  const double report_every = env_double("GNEP_REPORT", 60.0);
  std::vector<double> mu(initial_parameters, initial_parameters + n);

  // Mini-batches in reshuffled passes, as in the Adam loop.
  int cursor = 0;
  auto next_batch = [&]() -> GNEPTrainingBatch& {
    if (cursor % num_batches == 0) {
      std::shuffle(batch_indices.begin(), batch_indices.end(), rng);
    }
    return train_set[batch_indices[cursor++ % num_batches]];
  };
  auto peek_batch = [&]() -> GNEPTrainingBatch& {
    return train_set[batch_indices[cursor % num_batches]];
  };

  double excluded = 0.0;
  const auto start = Clock::now();
  auto algorithm_time = [&]() { return seconds_since(start) - excluded; };
  double next_report = 0.0;
  long step = 0;
  std::vector<float> host(n);
  auto report = [&](const char* extra) {
    const auto t0 = Clock::now();
    const double t = algorithm_time();
    float rmse[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < n; ++i) {
      host[i] = static_cast<float>(mu[i]);
    }
    if (has_test_set) {
      potential->find_force(
        para, host.data(), false, test_set, false, true, 0, 1, test_set[0].Nc, test_set[0].N,
        test_set[0].sum_virial_Nc * 6);
      rmse[0] = sqrt(test_set[0].get_mse_energy(para, false, 0).back());
      rmse[1] = sqrt(test_set[0].get_mse_force(para, false, 0).back());
      rmse[2] = sqrt(test_set[0].get_mse_virial(para, false, 0).back());
    }
    printf(
      "EXP %s step %ld time %.1f test_E %.5f test_F %.5f test_V %.5f forwards %ld gradients %ld"
      " t_forward %.4f t_gradient %.4f %s\n",
      method.c_str(), step, t, rmse[0], rmse[1], rmse[2], eval.forwards, eval.gradients,
      eval.forwards ? eval.forward_seconds / eval.forwards : 0.0,
      eval.gradients ? eval.gradient_seconds / eval.gradients : 0.0, extra);
    fflush(stdout);
    excluded += seconds_since(t0);
  };

  // Fast gradient against the exact analytical gradient at the same point.
  auto fast_check = [&](GNEPTrainingBatch& batch, const char* label) {
    const auto t0 = Clock::now();
    const bool saved = eval.fast;
    std::vector<double> exact, quick;
    auto ta = Clock::now();
    eval.fast = false;
    eval.gradient(mu, batch, exact);
    const double t_exact = seconds_since(ta);
    ta = Clock::now();
    eval.fast = true;
    eval.gradient(mu, batch, quick);
    const double t_fast = seconds_since(ta);
    eval.fast = saved;
    const int num_ann = para.number_of_variables_ann;
    auto part = [&](int begin, int end, double& rel, double& cos) {
      double ee = 0, qq = 0, eq = 0, dd = 0;
      for (int i = begin; i < end; ++i) {
        ee += exact[i] * exact[i];
        qq += quick[i] * quick[i];
        eq += exact[i] * quick[i];
        dd += (exact[i] - quick[i]) * (exact[i] - quick[i]);
      }
      rel = ee > 0 ? std::sqrt(dd / ee) : 0.0;
      cos = (ee > 0 && qq > 0) ? eq / std::sqrt(ee * qq) : 0.0;
    };
    double rel_all, cos_all, rel_ann, cos_ann, rel_c, cos_c;
    part(0, n, rel_all, cos_all);
    part(0, num_ann, rel_ann, cos_ann);
    part(num_ann, n, rel_c, cos_c);
    printf("FASTCHECK %s step %ld rel_err %.3e cos %.8f | ann rel %.3e cos %.8f | c rel %.3e cos %.8f"
      " | |g| %.4e | t_exact %.3f t_fast %.4f\n",
      label, step, rel_all, cos_all, rel_ann, cos_ann, rel_c, cos_c, std::sqrt(dot(exact, exact)),
      t_exact, t_fast);
    fflush(stdout);
    excluded += seconds_since(t0);
  };
  const long fastcheck_every = static_cast<long>(env_double("GNEP_FASTCHECK_EVERY", 0));
  if (env_double("GNEP_FASTCHECK", 0) > 0) {
    fast_check(train_set[0], "start");
  }

  // Optional check of the analytical gradient against a central difference.
  if (env_double("GNEP_GRADCHECK", 0) > 0) {
    GNEPTrainingBatch& batch = train_set[0];
    std::vector<double> g;
    eval.gradient(mu, batch, g);
    for (int trial = 0; trial < 3; ++trial) {
      std::vector<double> v(n), plus(mu), minus(mu);
      for (int i = 0; i < n; ++i) {
        v[i] = normal(rng);
      }
      const double scale = 1.0 / std::sqrt(dot(v, v));
      for (const double h : {1e-2, 3e-3, 1e-3}) {
        for (int i = 0; i < n; ++i) {
          plus[i] = mu[i] + h * v[i] * scale;
          minus[i] = mu[i] - h * v[i] * scale;
        }
        const double fd = (eval.loss(plus, batch) - eval.loss(minus, batch)) / (2 * h);
        printf("GRADCHECK trial %d h %g analytic %.6e central %.6e\n", trial, h,
          dot(g, v) * scale, fd);
      }
    }
    fflush(stdout);
  }

  if (method == "snes") {
    const int P = static_cast<int>(env_double("GNEP_POP", 50));
    const bool mirrored = env_double("GNEP_MIRROR", 0) > 0;
    const int premise_every = static_cast<int>(env_double("GNEP_PREMISE", 0));
    std::vector<double> sigma(n, env_double("GNEP_SIGMA0", 0.1));
    std::vector<double> utility(P);
    double usum = 0.0;
    for (int p = 0; p < P; ++p) {
      utility[p] = std::max(0.0, std::log(P * 0.5 + 1.0) - std::log(p + 1.0));
      usum += utility[p];
    }
    for (int p = 0; p < P; ++p) {
      utility[p] = utility[p] / usum - 1.0 / P;
    }
    const double num = double(n) / para.num_types;
    const double eta_sigma = (3.0 + std::log(num)) / (5.0 * std::sqrt(num)) / 2.0;
    std::vector<double> s(size_t(P) * n), x(n), loss(P), delta(n), grad_sigma(n);
    std::vector<int> order(P);
    while (algorithm_time() < budget) {
      if (algorithm_time() >= next_report) {
        report("");
        next_report += report_every;
      }
      GNEPTrainingBatch& batch = next_batch();
      for (int p = 0; p < P; ++p) {
        for (int i = 0; i < n; ++i) {
          s[size_t(p) * n + i] =
            (mirrored && p >= P / 2) ? -s[size_t(p - P / 2) * n + i] : normal(rng);
          x[i] = mu[i] + sigma[i] * s[size_t(p) * n + i];
        }
        loss[p] = eval.loss(x, batch);
      }
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](int a, int b) { return loss[a] < loss[b]; });
      std::fill(delta.begin(), delta.end(), 0.0);
      std::fill(grad_sigma.begin(), grad_sigma.end(), 0.0);
      for (int k = 0; k < P; ++k) {
        const double* sp = &s[size_t(order[k]) * n];
        for (int i = 0; i < n; ++i) {
          delta[i] += utility[k] * sp[i];
          grad_sigma[i] += utility[k] * (sp[i] * sp[i] - 1.0);
        }
      }
      for (int i = 0; i < n; ++i) {
        delta[i] *= sigma[i];
      }

      if (premise_every > 0 && step % premise_every == 0) {
        const auto t0 = Clock::now();
        GNEPTrainingBatch& held = peek_batch();
        std::vector<double> g0, ga(n, 0.0), gb(n, 0.0), gj;
        const double L_batch = eval.gradient(mu, batch, g0);
        const int J = 8;
        for (int j = 0; j < J; ++j) {
          for (int i = 0; i < n; ++i) {
            x[i] = mu[i] + sigma[i] * normal(rng);
          }
          eval.gradient(x, batch, gj);
          std::vector<double>& half = j < J / 2 ? ga : gb;
          for (int i = 0; i < n; ++i) {
            half[i] += gj[i] / (J / 2);
          }
        }
        // SNES's expected mean step is proportional to -sigma^2 * (smoothed gradient).
        std::vector<double> d0(n), ds(n), da(n), db(n), dplain(n);
        for (int i = 0; i < n; ++i) {
          const double s2 = sigma[i] * sigma[i];
          d0[i] = -s2 * g0[i];
          da[i] = -s2 * ga[i];
          db[i] = -s2 * gb[i];
          ds[i] = 0.5 * (da[i] + db[i]);
          dplain[i] = -g0[i];
        }
        const double step_norm = std::sqrt(dot(delta, delta));
        auto held_loss_along = [&](const std::vector<double>& d, double length) {
          const double scale = length / std::sqrt(dot(d, d));
          for (int i = 0; i < n; ++i) {
            x[i] = mu[i] + scale * d[i];
          }
          return eval.loss(x, held);
        };
        const double L0 = eval.loss(mu, held);
        for (int i = 0; i < n; ++i) {
          x[i] = mu[i] + delta[i];
        }
        const double L_snes = eval.loss(x, held);
        const double L_grad = held_loss_along(ds, step_norm);
        const double L_plain = held_loss_along(dplain, step_norm);
        double L_best = L_grad, best_scale = 1.0, L_pbest = L_plain, pbest_scale = 1.0;
        for (const double scale :
             {1e-3, 3e-3, 1e-2, 0.03, 0.1, 0.25, 0.5, 2.0, 4.0, 8.0, 16.0}) {
          const double value = held_loss_along(ds, scale * step_norm);
          if (value < L_best) {
            L_best = value;
            best_scale = scale;
          }
          const double plain = held_loss_along(dplain, scale * step_norm);
          if (plain < L_pbest) {
            L_pbest = plain;
            pbest_scale = scale;
          }
        }
        std::vector<double> sorted_sigma(sigma);
        std::nth_element(
          sorted_sigma.begin(), sorted_sigma.begin() + n / 2, sorted_sigma.end());
        printf(
          "PREMISE gen %ld batch_loss %.5f cos(snes,-s2*g_mean) %.4f cos(snes,-s2*g_smooth) %.4f "
          "cos(halfA,halfB) %.4f | held L0 %.5f snes %.5f grad_same_len %.5f plain_same_len %.5f "
          "grad_best %.5f (x%.3g) plain_best %.5f (x%.3g) | step_norm %.3e sigma_median %.3e\n",
          step, L_batch, cosine(delta, d0), cosine(delta, ds), cosine(da, db), L0, L_snes,
          L_grad, L_plain, L_best, best_scale, L_pbest, pbest_scale, step_norm, sorted_sigma[n / 2]);
        fflush(stdout);
        excluded += seconds_since(t0);
      }

      for (int i = 0; i < n; ++i) {
        mu[i] += delta[i];
        sigma[i] = std::min(sigma[i] * std::exp(eta_sigma * grad_sigma[i]), 1.0);
      }
      ++step;
    }
  } else {
    // adam and ivon share the time-based cosine learning-rate schedule.
    const double lr0 = env_double("GNEP_LR", method == "adam" ? 1e-3 : 0.1);
    const double lr1 = env_double("GNEP_LR_END", lr0 * 1e-3);
    const double beta1 = 0.9;
    const double beta2 = env_double("GNEP_IVON_BETA2", method == "adam" ? 0.999 : 0.99999);
    const double ess = env_double("GNEP_IVON_ESS", 1e6);
    const double wd = env_double("GNEP_IVON_WD", 1e-4);
    const double clip = env_double("GNEP_IVON_CLIP", 1e-3);
    std::vector<double> m1(n, 0.0), h(n, env_double("GNEP_IVON_H0", 0.1)), g, x(n), eps(n);
    std::vector<double> v2(n, 0.0);
    while (algorithm_time() < budget) {
      if (algorithm_time() >= next_report) {
        char extra[128] = "";
        if (method == "ivon") {
          std::vector<double> sigma(n);
          for (int i = 0; i < n; ++i) {
            sigma[i] = 1.0 / std::sqrt(ess * (h[i] + wd));
          }
          std::nth_element(sigma.begin(), sigma.begin() + n / 2, sigma.end());
          snprintf(extra, sizeof(extra), "sigma_median %.3e", sigma[n / 2]);
        }
        report(extra);
        next_report += report_every;
      }
      const double progress = std::min(1.0, algorithm_time() / budget);
      const double lr = lr1 + (lr0 - lr1) * 0.5 * (1.0 + std::cos(PI * progress));
      GNEPTrainingBatch& batch = next_batch();
      const double bc1 = 1.0 - std::pow(beta1, double(step + 1));
      if (method == "adam") {
        if (fastcheck_every > 0 && step > 0 && step % fastcheck_every == 0) {
          fast_check(batch, "train");
        }
        eval.gradient(mu, batch, g);
        const double bc2 = 1.0 - std::pow(beta2, double(step + 1));
        for (int i = 0; i < n; ++i) {
          m1[i] = beta1 * m1[i] + (1 - beta1) * g[i];
          v2[i] = beta2 * v2[i] + (1 - beta2) * g[i] * g[i];
          mu[i] -= lr * (m1[i] / bc1) / (std::sqrt(v2[i] / bc2) + 1e-8);
        }
      } else {
        for (int i = 0; i < n; ++i) {
          const double sigma = 1.0 / std::sqrt(ess * (h[i] + wd));
          eps[i] = normal(rng);
          x[i] = mu[i] + sigma * eps[i];
        }
        eval.gradient(x, batch, g);
        for (int i = 0; i < n; ++i) {
          const double sigma = 1.0 / std::sqrt(ess * (h[i] + wd));
          const double hhat = g[i] * eps[i] / sigma;
          m1[i] = beta1 * m1[i] + (1 - beta1) * g[i];
          const double hn = beta2 * h[i] + (1 - beta2) * hhat;
          h[i] = hn + 0.5 * (1 - beta2) * (1 - beta2) * (h[i] - hhat) * (h[i] - hhat) / (h[i] + wd);
          double update = (m1[i] / bc1 + wd * mu[i]) / (h[i] + wd);
          update = std::max(-clip, std::min(clip, update));
          mu[i] -= lr * update;
        }
      }
      ++step;
    }
  }
  report("final");
  for (int i = 0; i < n; ++i) {
    host[i] = static_cast<float>(mu[i]);
  }
  FILE* fid = my_fopen("nep.txt", "w");
  write_gnep_txt(fid, para, host.data());
  fclose(fid);
}
