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
#include "fitness.cuh"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

void run_adam(
  const UF3_Parameters& para,
  const UF3_OptimizerStage& stage,
  int stage_id,
  int gen_offset,
  Uf3Fitness& fitness)
{
  int nparam = fitness.num_parameters();
  int gen = stage.generation;
  const auto& ds = fitness.dataset();

  std::vector<float> x(nparam), grad(nparam);
  fitness.model()->get_parameters(x.data());

  std::vector<float> m(nparam, 0.0f), v(nparam, 0.0f);
  // lr0 tunable via env (UF3_LR); default higher than before since the warm
  // start from lstsq sits near the convex optimum and needs a real step to
  // refine.  Cosine decay to ~0 over the run for clean convergence.
  float lr0 = 0.01f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
  if (const char* e = getenv("UF3_LR")) lr0 = atof(e);

  fitness.begin_stage(stage_id, "adam", "grad_step");
  auto t0 = std::chrono::high_resolution_clock::now();
  float best_loss = 1e30f;
  std::vector<float> best_x = x;          // keep the best parameters seen

  for (int g = 0; g < gen; g++) {
    int batch_id = g % ds.num_batches;
    int global_gen = gen_offset + g;

    fitness.compute_gradient(batch_id, global_gen, grad);

    // Cosine learning-rate schedule.
    float lr = 0.5f * lr0 * (1.0f + cosf(3.14159265f * (float)g / (float)gen));
    float bc1 = 1.0f - powf(beta1, g + 1);   // bias-correction terms: O(1) per step
    float bc2 = 1.0f - powf(beta2, g + 1);
    for (int i = 0; i < nparam; i++) {
      m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
      v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
      float mh = m[i] / bc1;
      float vh = v[i] / bc2;
      x[i] -= lr * mh / (sqrtf(vh) + eps);
    }

    fitness.model()->set_parameters(x.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    float loss = fitness.compute_loss(batch_id, global_gen, stage_id, g + 1, (float)dt);
    if (loss < best_loss) {
      best_loss = loss;
      best_x = x;
    }

    if (g % 5 == 0 || g == gen - 1) {
      printf("  Adam step %5d: loss=%.4f eV, best=%.4f eV lr=%.2e (%.1fs)\n",
             g + 1, loss, best_loss, lr, dt);
    }
  }

  // Restore the best parameters found (a high lr can overshoot late in the run).
  fitness.model()->set_parameters(best_x.data());
}
