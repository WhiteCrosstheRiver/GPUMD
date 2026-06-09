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
#include <cstdio>
#include <cstdlib>
#include <vector>

void run_es(
  const UF3_Parameters& para,
  const UF3_OptimizerStage& stage,
  int stage_id,
  int gen_offset,
  Uf3Fitness& fitness)
{
  int nparam = fitness.num_parameters();
  int pop = stage.population;
  int gen = stage.generation;
  const auto& ds = fitness.dataset();

  std::vector<float> best(nparam), trial(nparam);
  fitness.model()->get_parameters(best.data());

  float best_rmse = 1e30f;
  srand(12345);

  fitness.begin_stage(stage_id, "es", "es_generation");
  auto t0 = std::chrono::high_resolution_clock::now();

  for (int g = 0; g < gen; g++) {
    int batch_id = g % ds.num_batches;
    int global_gen = gen_offset + g;
    float total_rmse = 0.0f;
    float gen_best = 1e30f;

    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();

    for (int p = 0; p < pop; p++) {
      for (int i = 0; i < nparam; i++) {
        trial[i] = best[i] + (rand() / (float)RAND_MAX - 0.5f) * 0.02f;
      }

      // All individuals in this generation share the same local_iter (g+1).
      // The duplicate-logging guard in compute_loss_for_params ensures only
      // the first individual at a checkpoint actually writes to the log files.
      float rmse = fitness.compute_loss_for_params(
        trial.data(), batch_id, global_gen, stage_id, g + 1, (float)dt);
      total_rmse += rmse;

      if (rmse < best_rmse) {
        best_rmse = rmse;
        for (int i = 0; i < nparam; i++) {
          best[i] = trial[i];
        }
        fitness.model()->set_parameters(best.data());
      }
      if (rmse < gen_best) {
        gen_best = rmse;
      }
    }

    if (g % 5 == 0 || g == gen - 1) {
      printf("  ES gen %5d: best=%.3f eV, avg=%.3f eV (%.1fs)\n",
             g + 1, best_rmse, total_rmse / pop, dt);
    }
  }
}
