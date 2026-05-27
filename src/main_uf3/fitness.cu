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
#include "utilities/error.cuh"
#include <cmath>
#include <cstdio>

Uf3Fitness::Uf3Fitness(
  UF3_Parameters& para, Uf3Model* model,
  const std::vector<Uf3Frame>& train_set)
  : model_(model), train_set_(train_set)
{
  lambda_e_ = (float)para.lambda_e;
  lambda_f_ = (float)para.lambda_f;
  lambda_1_ = (float)para.lambda_1;
  lambda_2_ = (float)para.lambda_2;
  d_energy_.resize(para.batch);
  h_energy_.resize(para.batch);

  // Load test set internally (like NEP)
  if (!para.test_data.empty()) {
    test_set_ = load_uf3_frames(para.test_data.c_str(), 0.0f);
    test_set_size_ = (int)test_set_.size();
    printf("Loaded %d test frames.\n", test_set_size_);
  } else {
    printf("Warning: no test data specified. Test RMSE will be 0.\n");
  }

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

  if (total > (int)h_fx_.size()) {
    h_fx_.resize(total); h_fy_.resize(total); h_fz_.resize(total);
  }

  model_->evaluate(train_set_, batch_indices, d_energy_);
  d_energy_.copy_to_host(h_energy_.data());
  cudaMemcpy(h_fx_.data(), model_->d_fx.data(), total * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_fy_.data(), model_->d_fy.data(), total * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_fz_.data(), model_->d_fz.data(), total * sizeof(float), cudaMemcpyDeviceToHost);

  // Energy RMSE (per atom)
  double e_sum2 = 0;
  for (int b = 0; b < B; b++) {
    int fidx = batch_indices[b];
    int na = train_set_[fidx].num_atoms;
    double diff = (double)h_energy_[b] / na - (double)train_set_[fidx].energy / na;
    e_sum2 += diff * diff;
  }
  loss_e = (float)sqrt(e_sum2 / B);

  // Force RMSE (per component)
  double f_sum2 = 0; int total_atoms = 0;
  { int off = 0;
    for (int b = 0; b < B; b++) {
      const Uf3Frame& f = train_set_[batch_indices[b]];
      total_atoms += f.num_atoms;
      for (int i = 0; i < f.num_atoms; i++) {
        double dx = (double)h_fx_[off+i] - (double)f.fx[i];
        double dy = (double)h_fy_[off+i] - (double)f.fy[i];
        double dz = (double)h_fz_[off+i] - (double)f.fz[i];
        f_sum2 += dx*dx + dy*dy + dz*dz;
      }
      off += f.num_atoms;
    }
  }
  loss_f = (float)sqrt(f_sum2 / (3.0 * total_atoms));

  // L1/L2 regularization
  float l1 = 0, l2 = 0;
  { std::vector<float> params(model_->num_parameters());
    model_->get_parameters(params.data());
    for (size_t i = 0; i < params.size(); i++) { l1 += fabsf(params[i]); l2 += params[i]*params[i]; }
    l1 /= params.size(); l2 = sqrtf(l2 / params.size()); }
  loss_l1 = l1; loss_l2 = l2;
  loss_total = lambda_e_ * loss_e + lambda_f_ * loss_f + lambda_1_ * loss_l1 + lambda_2_ * loss_l2;

  // Test set evaluation (internal, every 100 generations)
  if (test_set_size_ > 0 && generation % 100 == 0) {
    int ntest = std::min(100, test_set_size_);
    std::vector<int> tidx(ntest);
    for (int i = 0; i < ntest; i++) tidx[i] = i;
    int t_total = 0; for (int i=0;i<ntest;i++) t_total += test_set_[i].num_atoms;
    if (t_total > (int)h_fx_.size()) { h_fx_.resize(t_total); h_fy_.resize(t_total); h_fz_.resize(t_total); }

    GPU_Vector<float> d_te(ntest);
    model_->evaluate(test_set_, tidx, d_te);
    d_te.copy_to_host(h_energy_.data());
    cudaMemcpy(h_fx_.data(), model_->d_fx.data(), t_total*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy_.data(), model_->d_fy.data(), t_total*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz_.data(), model_->d_fz.data(), t_total*sizeof(float), cudaMemcpyDeviceToHost);

    double te_sum2 = 0;
    for (int i = 0; i < ntest; i++) {
      int na = test_set_[i].num_atoms;
      double diff = (double)h_energy_[i]/na - (double)test_set_[i].energy/na;
      te_sum2 += diff * diff;
    }
    test_e = (float)sqrt(te_sum2 / ntest);

    double tf_sum2 = 0; int tf_atoms = 0;
    { int off = 0;
      for (int i = 0; i < ntest; i++) {
        const Uf3Frame& f = test_set_[i];
        tf_atoms += f.num_atoms;
        for (int j = 0; j < f.num_atoms; j++) {
          double dx = (double)h_fx_[off+j] - (double)f.fx[j];
          double dy = (double)h_fy_[off+j] - (double)f.fy[j];
          double dz = (double)h_fz_[off+j] - (double)f.fz[j];
          tf_sum2 += dx*dx + dy*dy + dz*dz;
        }
        off += f.num_atoms;
      }
    }
    test_f = (float)sqrt(tf_sum2 / (3.0 * tf_atoms));

    fprintf(floss_, "%d %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
            generation, loss_total, loss_l1, loss_l2, loss_e, loss_f, test_e, test_f);
    fflush(floss_);
  } else if (floss_ && generation % 100 == 0) {
    fprintf(floss_, "%d %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
            generation, loss_total, loss_l1, loss_l2, loss_e, loss_f, test_e, test_f);
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
