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

#include "snes.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void compute_utilities(int pop, std::vector<float>& u)
{
  u.resize(pop);
  for (int i = 0; i < pop; i++) {
    float x = (i + 0.5f) / pop;
    u[i] = std::max(0.0f, logf(pop / 2.0f + 1.0f) - logf((float)(i + 1)));
  }
  float sum = 0;
  for (int i = 0; i < pop; i++) {
    sum += u[i];
  }
  float mean = sum / pop;
  for (int i = 0; i < pop; i++) {
    u[i] -= mean;
  }
}

void run_snes(
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

  std::vector<float> mu(nparam);
  fitness.model()->get_parameters(mu.data());
  std::vector<float> sigma(nparam, 0.05f);

  std::vector<float> utility;
  compute_utilities(pop, utility);

  std::vector<float> population(nparam * pop);
  std::vector<float> trial_pop(nparam * pop);   // P trial parameter sets
  std::vector<float> fitness_vals(pop);
  std::vector<int> indices(pop);

  std::mt19937 rng(12345);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  float eta_sigma = 0.1f;
  float eta_mu = 1.0f;

  fitness.begin_stage(stage_id, "snes", "snes_generation");
  auto t0 = std::chrono::high_resolution_clock::now();
  float best_rmse = 1e30f;

  for (int g = 0; g < gen; g++) {
    int batch_id = g % ds.num_batches;
    int global_gen = gen_offset + g;

    // Build all P trial parameter sets at once so the GPU can fuse-evaluate
    // them in a single launch chain.
    for (int p = 0; p < pop; p++) {
      for (int i = 0; i < nparam; i++) {
        float s = normal(rng);
        population[p * nparam + i] = s;
        trial_pop[p * nparam + i] = mu[i] + sigma[i] * s;
      }
      indices[p] = p;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    fitness.compute_loss_population(
      trial_pop.data(), pop, batch_id, global_gen, stage_id,
      g + 1, (float)dt, fitness_vals.data());

    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return fitness_vals[a] < fitness_vals[b]; });

    float best_in_gen = fitness_vals[indices[0]];
    if (best_in_gen < best_rmse) {
      best_rmse = best_in_gen;
    }

    for (int i = 0; i < nparam; i++) {
      float grad_mu = 0.0f, grad_sig = 0.0f;
      for (int r = 0; r < pop; r++) {
        int p = indices[r];
        float s = population[p * nparam + i];
        float u = utility[r];
        grad_mu += u * s;
        grad_sig += u * (s * s - 1.0f);
      }
      mu[i] += eta_mu * sigma[i] * grad_mu / pop;
      sigma[i] *= expf(0.5f * eta_sigma * grad_sig / pop);
      sigma[i] = std::max(1e-5f, sigma[i]);
    }

    fitness.model()->set_parameters(mu.data());

    if (g % 5 == 0 || g == gen - 1) {
      float avg_fit = 0;
      for (int p = 0; p < pop; p++) {
        avg_fit += fitness_vals[p];
      }
      avg_fit /= pop;
      printf("  SNES gen %5d: best=%.3f eV, avg=%.3f eV (%.1fs)\n",
             g + 1, best_in_gen, avg_fit, dt);
    }
  }
}
