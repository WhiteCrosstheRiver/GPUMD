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

#include "adam.cuh"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Finite-difference gradient (central difference, O(h^2))
static void compute_gradient(
  Uf3Fitness& fitness,
  const std::vector<float>& x,
  const std::vector<int>& batch_indices,
  std::vector<float>& grad,
  int nparam)
{
  float h = 1e-4f;
  std::vector<float> xp(nparam), xm(nparam);
  for (int i = 0; i < nparam; i++) {
    xp[i] = x[i] + h;
    xm[i] = x[i] - h;
    float fp = fitness.compute_loss_for_params(xp.data(), batch_indices);
    float fm = fitness.compute_loss_for_params(xm.data(), batch_indices);
    grad[i] = (fp - fm) / (2.0f * h);
    // Restore for next iteration (compute_loss_for_params modifies model params)
    xp[i] = x[i]; xm[i] = x[i];
  }
  fitness.model()->set_parameters(x.data());
}

void run_adam(UF3_Parameters& para, Uf3Fitness& fitness)
{
  int nparam = fitness.num_parameters();
  int gen = para.generation;
  int batch = para.batch;

  const auto& train_set = fitness.train_set();
  int nframes = (int)train_set.size();

  std::vector<float> x(nparam);
  fitness.model()->get_parameters(x.data());

  // Adam state
  std::vector<float> m(nparam, 0.0f), v(nparam, 0.0f);
  std::vector<float> grad(nparam);
  float lr = 0.01f;
  float beta1 = 0.9f, beta2 = 0.999f;
  float eps = 1e-8f;

  srand(12345);
  auto t0 = std::chrono::high_resolution_clock::now();
  float best_rmse = 1e30f;

  for (int g = 0; g < gen; g++) {
    // Random batch
    std::vector<int> bidx(batch);
    for (int b = 0; b < batch; b++) bidx[b] = rand() % nframes;

    compute_gradient(fitness, x, bidx, grad, nparam);

    // Adam update
    for (int i = 0; i < nparam; i++) {
      m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
      v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
      float m_hat = m[i] / (1.0f - powf(beta1, g + 1));
      float v_hat = v[i] / (1.0f - powf(beta2, g + 1));
      x[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }

    fitness.model()->set_parameters(x.data());
    float rmse = fitness.compute_loss(bidx);

    if (rmse < best_rmse) best_rmse = rmse;

    if (g % 5 == 0 || g == gen - 1) {
      auto t1 = std::chrono::high_resolution_clock::now();
      double dt = std::chrono::duration<double>(t1 - t0).count();
      printf("  Adam gen %5d: loss=%.3f eV, best=%.3f eV (%.1fs)\n",
             g, rmse, best_rmse, dt);
    }
  }
}
