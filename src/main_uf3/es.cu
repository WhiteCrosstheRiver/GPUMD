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

#include "es.cuh"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

void run_es(UF3_Parameters& para, Uf3Fitness& fitness)
{
  int nparam = fitness.num_parameters();
  int pop = para.population;
  int gen = para.generation;
  int batch = para.batch;

  const auto& train_set = fitness.train_set();
  int nframes = (int)train_set.size();

  // Current best
  std::vector<float> best(nparam);
  std::vector<float> trial(nparam);
  fitness.model()->get_parameters(best.data());

  float best_rmse = 1e30f;
  srand(12345);

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int g = 0; g < gen; g++) {
    float total_rmse = 0.0f;
    float gen_best = 1e30f;

    for (int p = 0; p < pop; p++) {
      // Perturb
      for (int i = 0; i < nparam; i++)
        trial[i] = best[i] + (rand() / (float)RAND_MAX - 0.5f) * 0.02f;

      // Batch
      std::vector<int> bidx(batch);
      for (int b = 0; b < batch; b++) bidx[b] = rand() % nframes;

      float rmse = fitness.compute_loss_for_params(trial.data(), bidx);
      total_rmse += rmse;

      if (rmse < best_rmse) {
        best_rmse = rmse;
        for (int i = 0; i < nparam; i++) best[i] = trial[i];
        fitness.model()->set_parameters(best.data());
      }
      if (rmse < gen_best) gen_best = rmse;
    }

    if (g % 5 == 0 || g == gen - 1) {
      auto t1 = std::chrono::high_resolution_clock::now();
      double dt = std::chrono::duration<double>(t1 - t0).count();
      printf("  ES gen %5d: best=%.3f eV, avg=%.3f eV (%.1fs)\n",
             g, best_rmse, total_rmse / pop, dt);
    }
  }
}
