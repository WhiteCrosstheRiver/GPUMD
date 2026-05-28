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

#pragma once
#include "dataset.cuh"
#include "dataset_gpu.cuh"
#include "parameters.cuh"
#include "uf3.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class Uf3Fitness
{
public:
  Uf3Fitness(UF3_Parameters& para, Uf3Model* model,
             const Uf3DatasetGPU& dataset,
             const std::vector<Uf3Frame>& train_set);

  float compute_loss(int batch_id, int generation, int stage_id = 0);
  float compute_loss_for_params(const float* params, int batch_id, int generation, int stage_id = 0);
  void compute_gradient(int batch_id, int generation, std::vector<float>& grad);

  int num_parameters() const { return model_->num_parameters(); }
  Uf3Model* model() { return model_; }
  const Uf3DatasetGPU& dataset() const { return dataset_; }

  float loss_e = 0, loss_f = 0, loss_l1 = 0, loss_l2 = 0, loss_total = 0;
  float test_e = 0, test_f = 0;

  ~Uf3Fitness()
  {
    if (floss_) {
      fclose(floss_);
      floss_ = nullptr;
    }
  }

private:
  void accumulate_regularization_gradient(std::vector<float>& grad);

  Uf3Model* model_;
  const Uf3DatasetGPU& dataset_;
  const std::vector<Uf3Frame>& train_set_;
  std::vector<Uf3Frame> test_set_;
  int test_set_size_ = 0;
  GPU_Vector<float> d_energy_;
  std::vector<float> h_energy_, h_fx_, h_fy_, h_fz_;
  float lambda_e_ = 1, lambda_f_ = 1, lambda_1_ = 0, lambda_2_ = 0;
  FILE* floss_ = nullptr;
};
