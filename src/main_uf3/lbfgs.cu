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

#include "lbfgs.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

void run_lbfgs(
  const UF3_Parameters& para,
  const UF3_OptimizerStage& stage,
  int stage_id,
  int gen_offset,
  Uf3Fitness& fitness)
{
  int nparam = fitness.num_parameters();
  int gen = stage.generation;
  const auto& ds = fitness.dataset();
  const int M = 5;

  std::vector<float> x(nparam), grad(nparam), grad_old(nparam);
  fitness.model()->get_parameters(x.data());

  std::vector<float> s(M * nparam, 0), y(M * nparam, 0);
  std::vector<float> rho(M, 0), alpha(M, 0);
  std::vector<float> q(nparam), dir(nparam);
  int hist_idx = 0, hist_count = 0;

  fitness.begin_stage(stage_id, "lbfgs", "lbfgs_step");
  auto t0 = std::chrono::high_resolution_clock::now();
  float best_loss = 1e30f;

  int batch_id0 = 0 % ds.num_batches;
  fitness.compute_gradient(batch_id0, gen_offset, grad);
  for (int i = 0; i < nparam; i++) {
    dir[i] = -grad[i];
  }

  for (int g = 0; g < gen; g++) {
    int batch_id = g % ds.num_batches;
    int global_gen = gen_offset + g;

    float step = 0.001f;
    std::vector<float> x_trial(nparam);
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    float loss_old = fitness.compute_loss(batch_id, global_gen, stage_id, g + 1, (float)dt);
    float loss_new = loss_old;
    bool found = false;
    for (int ls = 0; ls < 10; ls++) {
      for (int i = 0; i < nparam; i++) {
        x_trial[i] = x[i] + step * dir[i];
      }
      fitness.model()->set_parameters(x_trial.data());
      loss_new = fitness.compute_loss(batch_id, global_gen, stage_id, g + 1, (float)dt);
      if (loss_new < loss_old) {
        found = true;
        break;
      }
      step *= 0.5f;
    }
    if (found) {
      for (int i = 0; i < nparam; i++) x[i] = x_trial[i];
    } else {
      // All backtrack steps failed — revert to previous params, skip update
      fitness.model()->set_parameters(x.data());
      loss_new = loss_old;
      step = 0.001f;
    }
    if (loss_new < best_loss) {
      best_loss = loss_new;
    }

    int grad_batch_id = (g + 1) % ds.num_batches;
    for (int i = 0; i < nparam; i++) {
      grad_old[i] = grad[i];
    }
    fitness.compute_gradient(grad_batch_id, global_gen + 1, grad);

    float gnorm = 0;
    for (int i = 0; i < nparam; i++) {
      gnorm += grad[i] * grad[i];
    }
    gnorm = sqrtf(gnorm);
    if (gnorm > 10.0f) {
      float scl = 10.0f / gnorm;
      for (int i = 0; i < nparam; i++) {
        grad[i] *= scl;
      }
    }

    float ys = 0;
    for (int i = 0; i < nparam; i++) {
      s[hist_idx * nparam + i] = step * dir[i];
      y[hist_idx * nparam + i] = grad[i] - grad_old[i];
      ys += s[hist_idx * nparam + i] * y[hist_idx * nparam + i];
    }
    rho[hist_idx] = (ys > 1e-8f) ? 1.0f / ys : 0.0f;
    if (ys > 1e-8f) {
      hist_idx = (hist_idx + 1) % M;
      hist_count = std::min(hist_count + 1, M);
    }

    for (int i = 0; i < nparam; i++) {
      q[i] = grad[i];
    }
    for (int j = hist_count - 1; j >= 0; j--) {
      int idx = (hist_idx - 1 - j + M) % M;
      float dot_sq = 0;
      for (int i = 0; i < nparam; i++) {
        dot_sq += s[idx * nparam + i] * q[i];
      }
      alpha[j] = rho[idx] * dot_sq;
      for (int i = 0; i < nparam; i++) {
        q[i] -= alpha[j] * y[idx * nparam + i];
      }
    }
    float gamma = 1.0f;
    if (hist_count > 0) {
      int last = (hist_idx - 1 + M) % M;
      float yy = 0, sy = 0;
      for (int i = 0; i < nparam; i++) {
        yy += y[last * nparam + i] * y[last * nparam + i];
        sy += s[last * nparam + i] * y[last * nparam + i];
      }
      if (yy > 1e-10f) {
        gamma = sy / yy;
      }
    }
    for (int i = 0; i < nparam; i++) {
      dir[i] = gamma * q[i];
    }
    for (int j = 0; j < hist_count; j++) {
      int idx = (hist_idx - hist_count + j + M) % M;
      float dot_y = 0;
      for (int i = 0; i < nparam; i++) {
        dot_y += y[idx * nparam + i] * dir[i];
      }
      float beta = rho[idx] * dot_y;
      for (int i = 0; i < nparam; i++) {
        dir[i] += s[idx * nparam + i] * (alpha[j] - beta);
      }
    }
    for (int i = 0; i < nparam; i++) {
      dir[i] = -dir[i];
    }

    if (g % 5 == 0 || g == gen - 1) {
      printf("  LBFGS step %5d: loss=%.4f eV, best=%.4f eV (%.1fs)\n",
             g + 1, loss_new, best_loss, dt);
    }
  }
}
