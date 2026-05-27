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
#include <cstdio>

Uf3Fitness::Uf3Fitness(
  UF3_Parameters& para, Uf3Model* model,
  const std::vector<Uf3Frame>& train_set)
  : model_(model), train_set_(train_set)
{
  lambda_e_ = (float)para.lambda_e;
  lambda_f_ = (float)para.lambda_f;
  lambda_1_ = (float)para.lambda_v;  // reused for L1
  lambda_2_ = 0.0f;                  // L2 not configured yet
  max_batch_atoms_ = para.batch * 200;

  d_energy_.resize(para.batch);
  d_fx_.resize(max_batch_atoms_);
  d_fy_.resize(max_batch_atoms_);
  d_fz_.resize(max_batch_atoms_);
  h_energy_.resize(para.batch);
  h_fx_.resize(max_batch_atoms_);
  h_fy_.resize(max_batch_atoms_);
  h_fz_.resize(max_batch_atoms_);

  floss_ = fopen("loss.out", "w");
  if (floss_) {
    fprintf(floss_, "# gen L_t L_1 L_2 L_e_train L_f_train L_e_test L_f_test\n");
    fflush(floss_);
  }
}

float Uf3Fitness::compute_loss(const std::vector<int>& batch_indices, int generation)
{
  int B = (int)batch_indices.size();
  int total = 0;
  for (int b = 0; b < B; b++) total += train_set_[batch_indices[b]].num_atoms;

  // Ensure force buffers are large enough
  if (total > max_batch_atoms_) {
    max_batch_atoms_ = total;
    d_fx_.resize(total); d_fy_.resize(total); d_fz_.resize(total);
    h_fx_.resize(total);  h_fy_.resize(total);  h_fz_.resize(total);
  }

  // GPU evaluation (energy + forces)
  model_->evaluate(train_set_, batch_indices, d_energy_);
  d_energy_.copy_to_host(h_energy_.data());

  // Download forces (from model's GPU buffers, written by force kernel)
  cudaMemcpy(h_fx_.data(), model_->d_fx.data(), total * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_fy_.data(), model_->d_fy.data(), total * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_fz_.data(), model_->d_fz.data(), total * sizeof(float), cudaMemcpyDeviceToHost);

  // Compute energy RMSE
  double e_sum2 = 0;
  for (int b = 0; b < B; b++) {
    int fidx = batch_indices[b];
    double diff = (double)h_energy_[b] - (double)train_set_[fidx].energy;
    e_sum2 += diff * diff;
  }
  float e_rmse = (float)sqrt(e_sum2 / B); // per-frame RMSE (eV)

  // Compute force RMSE
  double f_sum2 = 0; int f_count = 0;
  { int off = 0;
    for (int b = 0; b < B; b++) {
      const Uf3Frame& f = train_set_[batch_indices[b]];
      for (int i = 0; i < f.num_atoms; i++) {
        double dx = (double)h_fx_[off+i] - (double)f.fx[i];
        double dy = (double)h_fy_[off+i] - (double)f.fy[i];
        double dz = (double)h_fz_[off+i] - (double)f.fz[i];
        f_sum2 += dx*dx + dy*dy + dz*dz;
        f_count += 3;
      }
      off += f.num_atoms;
    }
  }
  float f_rmse = (float)sqrt(f_sum2 / f_count); // per-component force RMSE (eV/A)

  // L1/L2 regularization
  float l1 = 0, l2 = 0;
  std::vector<float> params(model_->num_parameters());
  model_->get_parameters(params.data());
  for (size_t i = 0; i < params.size(); i++) { l1 += fabsf(params[i]); l2 += params[i]*params[i]; }
  l1 /= params.size(); l2 = sqrtf(l2 / params.size());

  // Total loss (NEP convention: per-component RMSE weighted by lambda)
  loss_e = e_rmse / train_set_[batch_indices[0]].num_atoms; // eV/atom
  loss_f = f_rmse;
  loss_l1 = 0; loss_l2 = 0;
  if (lambda_1_ > 0) { loss_l1 = lambda_1_ * l1; loss_l2 = lambda_2_ * l2; }
  loss_total = lambda_e_ * loss_e + lambda_f_ * loss_f + loss_l1 + loss_l2;

  // Write loss.out every 100 generations
  if (floss_ && generation % 100 == 0) {
    float e_test = 0, f_test = 0;
    if (test_set_ && !test_set_->empty()) {
      // Quick test set evaluation (single frame for speed)
      std::vector<int> tidx(1, 0);
      GPU_Vector<float> de;
      model_->evaluate(*test_set_, tidx, de);
      de.copy_to_host(h_energy_.data());
      e_test = fabsf(h_energy_[0] - (*test_set_)[0].energy) / (*test_set_)[0].num_atoms;
    }
    fprintf(floss_, "%d %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
            generation, loss_total, loss_l1, loss_l2, loss_e, loss_f, e_test, f_test);
    fflush(floss_);
  }

  return loss_total;
}

float Uf3Fitness::compute_loss_for_params(
  const float* params, const std::vector<int>& batch_indices, int generation)
{
  model_->set_parameters(params);
  return compute_loss(batch_indices, generation);
}
