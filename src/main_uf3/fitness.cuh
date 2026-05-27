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

/*----------------------------------------------------------------------------80
UF3 fitness: computes full loss (energy + force RMSE + L1/L2 regularization).
Matches NEP loss.out format for comparison.
------------------------------------------------------------------------------*/

#pragma once
#include "dataset.cuh"
#include "uf3.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class Uf3Fitness
{
public:
  Uf3Fitness(UF3_Parameters& para, Uf3Model* model,
             const std::vector<Uf3Frame>& train_set);

  void set_test_set(const std::vector<Uf3Frame>& test_set) { test_set_ = &test_set; }

  // Evaluate loss for given parameters. Returns energy RMSE (eV/atom).
  // Populates loss struct for optimizer use.
  float compute_loss(const std::vector<int>& batch_indices, int generation);

  float compute_loss_for_params(
    const float* params, const std::vector<int>& batch_indices, int generation);

  int num_parameters() const { return model_->num_parameters(); }
  Uf3Model* model() { return model_; }
  const std::vector<Uf3Frame>& train_set() const { return train_set_; }

  // Loss components (filled by compute_loss)
  float loss_e = 0, loss_f = 0, loss_l1 = 0, loss_l2 = 0, loss_total = 0;

private:
  Uf3Model* model_;
  const std::vector<Uf3Frame>& train_set_;
  const std::vector<Uf3Frame>* test_set_ = nullptr;
  GPU_Vector<float> d_energy_, d_fx_, d_fy_, d_fz_;
  std::vector<float> h_energy_, h_fx_, h_fy_, h_fz_;
  float lambda_e_ = 1, lambda_f_ = 1, lambda_1_ = 0, lambda_2_ = 0;
  FILE* floss_ = nullptr;
  int max_batch_atoms_ = 0;
};
