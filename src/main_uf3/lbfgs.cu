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

// Simple L-BFGS with backtracking line search
void run_lbfgs(UF3_Parameters& para, Uf3Fitness& fitness)
{
  int nparam = fitness.num_parameters();
  int gen = para.generation;
  int batch = para.batch;
  const auto& train_set = fitness.train_set();
  int nframes = (int)train_set.size();
  const int M = 5; // L-BFGS memory

  std::vector<float> x(nparam), grad(nparam), grad_old(nparam);
  fitness.model()->get_parameters(x.data());

  // L-BFGS history
  std::vector<float> s(M * nparam, 0);  // x_{k+1} - x_k
  std::vector<float> y(M * nparam, 0);  // grad_{k+1} - grad_k
  std::vector<float> rho(M, 0);
  std::vector<float> alpha(M, 0);
  std::vector<float> q(nparam), dir(nparam);
  GPU_Vector<float> d_ediff(batch);
  std::vector<int> bidx(batch);
  int hist_idx = 0, hist_count = 0;

  srand(12345);
  auto t0 = std::chrono::high_resolution_clock::now();
  float best_loss = 1e30f;

  // Initial gradient
  for (int b = 0; b < batch; b++) bidx[b] = rand() % nframes;
  fitness.model()->compute_energy_gradient(train_set, bidx, d_ediff, grad);
  for (int i = 0; i < nparam; i++) dir[i] = -grad[i]; // initial search direction = -grad

  for (int g = 0; g < gen; g++) {
    // Line search (simple backtracking)
    float step = 0.01f;
    std::vector<float> x_trial(nparam);
    float loss_old = fitness.compute_loss(bidx, g);
    float loss_new;
    for (int ls = 0; ls < 10; ls++) {
      for (int i = 0; i < nparam; i++) x_trial[i] = x[i] + step * dir[i];
      fitness.model()->set_parameters(x_trial.data());
      loss_new = fitness.compute_loss(bidx, g);
      if (loss_new < loss_old) break;
      step *= 0.5f;
    }
    if (loss_new >= loss_old) step = 0.001f; // fallback
    for (int i = 0; i < nparam; i++) x[i] += step * dir[i];
    fitness.model()->set_parameters(x.data());
    if (loss_new < best_loss) best_loss = loss_new;

    // New batch + gradient
    for (int b = 0; b < batch; b++) bidx[b] = rand() % nframes;
    for (int i = 0; i < nparam; i++) grad_old[i] = grad[i];
    fitness.model()->compute_energy_gradient(train_set, bidx, d_ediff, grad);

    // Update L-BFGS history: s = dx, y = dgrad
    float ys = 0;
    for (int i = 0; i < nparam; i++) {
      s[hist_idx * nparam + i] = step * dir[i];
      y[hist_idx * nparam + i] = grad[i] - grad_old[i];
      ys += s[hist_idx * nparam + i] * y[hist_idx * nparam + i];
    }
    rho[hist_idx] = (ys > 1e-10f) ? 1.0f / ys : 0.0f;
    hist_idx = (hist_idx + 1) % M;
    hist_count = std::min(hist_count + 1, M);

    // Two-loop recursion for search direction
    for (int i = 0; i < nparam; i++) q[i] = grad[i];
    for (int j = hist_count - 1; j >= 0; j--) {
      int idx = (hist_idx - 1 - j + M) % M;
      float dot_sq = 0;
      for (int i = 0; i < nparam; i++) dot_sq += s[idx*nparam+i] * q[i];
      alpha[j] = rho[idx] * dot_sq;
      for (int i = 0; i < nparam; i++) q[i] -= alpha[j] * y[idx*nparam+i];
    }
    // Scale initial Hessian
    float gamma = (hist_count > 0 && rho[(hist_idx-1+M)%M] > 0) ? 1.0f / (rho[(hist_idx-1+M)%M] * rho[(hist_idx-1+M)%M] + 1e-10f) : 1.0f;
    for (int i = 0; i < nparam; i++) dir[i] = gamma * q[i];
    for (int j = 0; j < hist_count; j++) {
      int idx = (hist_idx - hist_count + j + M) % M;
      float dot_y = 0;
      for (int i = 0; i < nparam; i++) dot_y += y[idx*nparam+i] * dir[i];
      float beta = rho[idx] * dot_y;
      for (int i = 0; i < nparam; i++) dir[i] += s[idx*nparam+i] * (alpha[j] - beta);
    }
    for (int i = 0; i < nparam; i++) dir[i] = -dir[i]; // descent direction

    if (g % 5 == 0 || g == gen - 1) {
      auto t1 = std::chrono::high_resolution_clock::now();
      printf("  LBFGS gen %5d: loss=%.4f eV, best=%.4f eV (%.1fs)\n",
             g, loss_new, best_loss, std::chrono::duration<double>(t1-t0).count());
    }
  }
}
