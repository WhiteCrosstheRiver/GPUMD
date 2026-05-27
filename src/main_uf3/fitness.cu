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

#include "fitness.cuh"
#include <cmath>

Uf3Fitness::Uf3Fitness(
  UF3_Parameters& para, Uf3Model* model,
  const std::vector<Uf3Frame>& train_set)
  : model_(model), train_set_(train_set)
{
  d_energy_.resize(para.batch);
  h_energy_.resize(para.batch);
}

float Uf3Fitness::compute_loss(const std::vector<int>& batch_indices)
{
  int batch_size = (int)batch_indices.size();
  model_->evaluate(train_set_, batch_indices, d_energy_);
  d_energy_.copy_to_host(h_energy_.data());

  double sum = 0.0;
  for (int b = 0; b < batch_size; b++) {
    int fidx = batch_indices[b];
    sum += fabs((double)h_energy_[b] - (double)train_set_[fidx].energy);
  }
  current_rmse_ = (float)(sum / batch_size);
  return current_rmse_;
}

float Uf3Fitness::compute_loss_for_params(
  const float* params, const std::vector<int>& batch_indices)
{
  model_->set_parameters(params);
  return compute_loss(batch_indices);
}
