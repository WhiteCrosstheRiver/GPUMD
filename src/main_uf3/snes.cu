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

// Compute utility weights from ranking (SNES paper, Table 1)
static void compute_utilities(int pop, std::vector<float>& u)
{
  u.resize(pop);
  for (int i = 0; i < pop; i++) {
    // Standard normal order statistic approximation
    float x = (i + 0.5f) / pop;
    // Inverse CDF approximation (Blom)
    u[i] = std::max(0.0f, logf(pop / 2.0f + 1.0f) - logf((float)(i + 1)));
  }
  // Normalize to zero mean
  float sum = 0;
  for (int i = 0; i < pop; i++) sum += u[i];
  float mean = sum / pop;
  for (int i = 0; i < pop; i++) u[i] -= mean;
}

void run_snes(UF3_Parameters& para, Uf3Fitness& fitness)
{
  int nparam = fitness.num_parameters();
  int pop = para.population;
  int gen = para.generation;
  int batch = para.batch;

  const auto& train_set = fitness.train_set();
  int nframes = (int)train_set.size();

  // Initialize mu (center of search distribution) and sigma
  std::vector<float> mu(nparam);
  fitness.model()->get_parameters(mu.data());

  std::vector<float> sigma(nparam, 0.05f); // initial step size

  // Compute utility weights
  std::vector<float> utility;
  compute_utilities(pop, utility);

  // Working arrays
  std::vector<float> population(nparam * pop);
  std::vector<float> fitness_vals(pop);
  std::vector<float> trial(nparam);
  std::vector<int> indices(pop);

  std::mt19937 rng(12345);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  // SNES hyperparameter
  float eta_sigma = 0.1f;
  float eta_mu = 1.0f;

  auto t0 = std::chrono::high_resolution_clock::now();
  float best_rmse = 1e30f;

  for (int g = 0; g < gen; g++) {
    // Random batch (same for all population members this generation)
    std::vector<int> bidx(batch);
    for (int b = 0; b < batch; b++) bidx[b] = rng() % nframes;

    // Sample population and evaluate fitness
    for (int p = 0; p < pop; p++) {
      for (int i = 0; i < nparam; i++) {
        float s = normal(rng);
        population[p * nparam + i] = s;
        trial[i] = mu[i] + sigma[i] * s;
      }
      fitness_vals[p] = fitness.compute_loss_for_params(trial.data(), bidx, g);
      indices[p] = p;
    }

    // Sort by fitness (ascending = better)
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return fitness_vals[a] < fitness_vals[b]; });

    float best_in_gen = fitness_vals[indices[0]];
    if (best_in_gen < best_rmse) best_rmse = best_in_gen;

    // Update mu and sigma using natural gradient
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
      sigma[i] = std::max(1e-5f, sigma[i]); // prevent collapse
    }

    fitness.model()->set_parameters(mu.data());

    if (g % 5 == 0 || g == gen - 1) {
      auto t1 = std::chrono::high_resolution_clock::now();
      double dt = std::chrono::duration<double>(t1 - t0).count();
      float avg_fit = 0;
      for (int p = 0; p < pop; p++) avg_fit += fitness_vals[p];
      avg_fit /= pop;
      printf("  SNES gen %5d: best=%.3f eV, avg=%.3f eV (%.1fs)\n",
             g, best_in_gen, avg_fit, dt);
    }
  }
}
