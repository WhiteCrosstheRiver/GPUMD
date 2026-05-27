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
UF3 fitness: computes RMSE loss for energy on GPU-evaluated batches.
Connects Uf3Model to optimizers.
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

  // Evaluate loss for current model parameters. Returns energy RMSE.
  // batch_indices: random subset of training frames
  float compute_loss(const std::vector<int>& batch_indices);

  float compute_loss_for_params(
    const float* params, const std::vector<int>& batch_indices);

  int num_parameters() const { return model_->num_parameters(); }
  float current_rmse() const { return current_rmse_; }

  Uf3Model* model() { return model_; }
  const std::vector<Uf3Frame>& train_set() const { return train_set_; }

private:
  Uf3Model* model_;
  const std::vector<Uf3Frame>& train_set_;
  GPU_Vector<float> d_energy_;
  std::vector<float> h_energy_;
  float current_rmse_ = 0.0f;
};
