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
#include <algorithm>
#include <cmath>
#include <cstdio>

Uf3Fitness::Uf3Fitness(
  UF3_Parameters& para, Uf3Model* model,
  const Uf3DatasetGPU& dataset,
  const std::vector<Uf3Frame>& train_set)
  : model_(model), dataset_(dataset), train_set_(train_set)
{
  lambda_e_ = (float)para.lambda_e;
  lambda_f_ = (float)para.lambda_f;
  lambda_1_ = (float)para.lambda_1;
  lambda_2_ = (float)para.lambda_2;
  d_energy_.resize(para.batch);
  h_energy_.resize(para.batch);
  int max_batch_atoms = para.batch * 200;
  h_fx_.reserve(max_batch_atoms);
  h_fy_.reserve(max_batch_atoms);
  h_fz_.reserve(max_batch_atoms);

  if (!para.test_data.empty() && para.test_data != "none") {
    test_set_ = load_uf3_frames(para.test_data.c_str(), 0.0f);
    test_set_size_ = (int)test_set_.size();
    printf("Loaded %d test frames.\n", test_set_size_);
  } else {
    printf("Warning: no test data specified. Test RMSE will be 0.\n");
  }

  floss_ = fopen("loss.out", "w");
  if (floss_) {
    fprintf(floss_, "# stage gen L_t L_1 L_2 L_e_train L_f_train L_e_test L_f_test\n");
    fflush(floss_);
  }
}

void Uf3Fitness::accumulate_regularization_gradient(std::vector<float>& grad)
{
  int nparam = model_->num_parameters();
  if ((int)grad.size() != nparam) {
    grad.resize(nparam, 0.0f);
  }

  std::vector<float> params(nparam);
  model_->get_parameters(params.data());
  if (lambda_1_ > 0.0f) {
    float scale = lambda_1_ / nparam;
    for (int i = 0; i < nparam; i++) {
      float s = params[i] >= 0.0f ? 1.0f : -1.0f;
      if (params[i] == 0.0f) {
        s = 0.0f;
      }
      grad[i] += scale * s;
    }
  }
  if (lambda_2_ > 0.0f && loss_l2 > 1e-12f) {
    float scale = lambda_2_ / ((float)nparam * loss_l2);
    for (int i = 0; i < nparam; i++) {
      grad[i] += scale * params[i];
    }
  }
}

float Uf3Fitness::compute_loss(int batch_id, int generation, int stage_id)
{
  const auto& bidx = dataset_.batches[batch_id % dataset_.num_batches];
  int B = (int)bidx.size();
  int total = 0;
  for (int b = 0; b < B; b++) {
    total += dataset_.h_natoms[bidx[b]];
  }

  if (total > (int)h_fx_.size()) {
    h_fx_.resize(total);
    h_fy_.resize(total);
    h_fz_.resize(total);
  }

  model_->evaluate(dataset_, bidx, d_energy_);

  d_energy_.copy_to_host(h_energy_.data());
  cudaMemcpy(h_fx_.data(), model_->d_fx.data(), total * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_fy_.data(), model_->d_fy.data(), total * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_fz_.data(), model_->d_fz.data(), total * sizeof(float), cudaMemcpyDeviceToHost);

  double e_sum2 = 0;
  for (int b = 0; b < B; b++) {
    int fidx = bidx[b];
    int na = train_set_[fidx].num_atoms;
    double diff = (double)h_energy_[b] / na - (double)train_set_[fidx].energy / na;
    e_sum2 += diff * diff;
  }
  loss_e = (float)sqrt(e_sum2 / B);

  double f_sum2 = 0;
  int total_atoms = 0;
  int off = 0;
  for (int b = 0; b < B; b++) {
    const Uf3Frame& f = train_set_[bidx[b]];
    total_atoms += f.num_atoms;
    for (int i = 0; i < f.num_atoms; i++) {
      double dx = (double)h_fx_[off + i] - (double)f.fx[i];
      double dy = (double)h_fy_[off + i] - (double)f.fy[i];
      double dz = (double)h_fz_[off + i] - (double)f.fz[i];
      f_sum2 += dx * dx + dy * dy + dz * dz;
    }
    off += f.num_atoms;
  }
  loss_f = (float)sqrt(f_sum2 / (3.0 * total_atoms));

  float l1 = 0, l2 = 0;
  {
    std::vector<float> params(model_->num_parameters());
    model_->get_parameters(params.data());
    for (size_t i = 0; i < params.size(); i++) {
      l1 += fabsf(params[i]);
      l2 += params[i] * params[i];
    }
    l1 /= params.size();
    l2 = sqrtf(l2 / params.size());
  }
  loss_l1 = l1;
  loss_l2 = l2;
  loss_total = lambda_e_ * loss_e + lambda_f_ * loss_f + lambda_1_ * loss_l1 + lambda_2_ * loss_l2;

  if (test_set_size_ > 0 && generation > 0 && generation % 100 == 0) {
    int ntest = std::min(100, test_set_size_);
    std::vector<int> tidx(ntest);
    for (int i = 0; i < ntest; i++) {
      tidx[i] = i;
    }
    int t_total = 0;
    for (int i = 0; i < ntest; i++) {
      t_total += test_set_[i].num_atoms;
    }
    if (t_total > (int)h_fx_.size()) {
      h_fx_.resize(t_total);
      h_fy_.resize(t_total);
      h_fz_.resize(t_total);
    }

    GPU_Vector<float> d_te(ntest);
    model_->evaluate(test_set_, tidx, d_te);
    d_te.copy_to_host(h_energy_.data());
    cudaMemcpy(h_fx_.data(), model_->d_fx.data(), t_total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy_.data(), model_->d_fy.data(), t_total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz_.data(), model_->d_fz.data(), t_total * sizeof(float), cudaMemcpyDeviceToHost);

    double te_sum2 = 0;
    for (int i = 0; i < ntest; i++) {
      int na = test_set_[i].num_atoms;
      double diff = (double)h_energy_[i] / na - (double)test_set_[i].energy / na;
      te_sum2 += diff * diff;
    }
    test_e = (float)sqrt(te_sum2 / ntest);

    double tf_sum2 = 0;
    int tf_atoms = 0, toff = 0;
    for (int i = 0; i < ntest; i++) {
      const Uf3Frame& f = test_set_[i];
      tf_atoms += f.num_atoms;
      for (int j = 0; j < f.num_atoms; j++) {
        double dx = (double)h_fx_[toff + j] - (double)f.fx[j];
        double dy = (double)h_fy_[toff + j] - (double)f.fy[j];
        double dz = (double)h_fz_[toff + j] - (double)f.fz[j];
        tf_sum2 += dx * dx + dy * dy + dz * dz;
      }
      toff += f.num_atoms;
    }
    test_f = (float)sqrt(tf_sum2 / (3.0 * tf_atoms));

    fprintf(floss_, "%d %d %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
            stage_id, generation, loss_total, loss_l1, loss_l2, loss_e, loss_f, test_e, test_f);
    fflush(floss_);
  } else if (floss_ && generation > 0 && generation % 100 == 0) {
    fprintf(floss_, "%d %d %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
            stage_id, generation, loss_total, loss_l1, loss_l2, loss_e, loss_f, test_e, test_f);
    fflush(floss_);
  }

  return loss_total;
}

void Uf3Fitness::compute_gradient(int batch_id, int generation, std::vector<float>& grad)
{
  const auto& bidx = dataset_.batches[batch_id % dataset_.num_batches];
  compute_loss(batch_id, generation, 0);
  model_->compute_loss_gradient(
    dataset_, bidx, loss_e, loss_f, lambda_e_, lambda_f_, grad);
  accumulate_regularization_gradient(grad);
}

float Uf3Fitness::compute_loss_for_params(
  const float* params, int batch_id, int generation, int stage_id)
{
  model_->set_parameters(params);
  return compute_loss(batch_id, generation, stage_id);
}
