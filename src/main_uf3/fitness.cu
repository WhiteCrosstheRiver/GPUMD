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
Uf3Fitness — fast training-loss evaluator.  Hot-path goal: zero per-step
heap allocation, zero force-vector D2H.  The training RMSE numerators are
reduced on the GPU and only 2 floats are copied back per step.  Test-set
RMSE goes through a slow CPU path that only triggers every 100 generations.
------------------------------------------------------------------------------*/

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

  // Pre-allocate every hot-path GPU buffer (sized to dataset max-batch).
  int B_max = dataset.max_batch_size > 0 ? dataset.max_batch_size : para.batch;
  int nparam = model_->num_parameters();
  d_energy_.resize(B_max);
  d_loss_sum_.resize(2);
  d_grad_.resize(nparam);
  d_ediff_.resize(B_max);
  // Pinned host buffer enables cudaMemcpyAsync to overlap the scalar D2H with
  // the host-side L1/L2 computation that follows.
  CHECK(cudaHostAlloc((void**)&h_loss_sum_pinned_, 2 * sizeof(float),
                      cudaHostAllocDefault));
  h_loss_sum_pinned_[0] = 0.0f;
  h_loss_sum_pinned_[1] = 0.0f;

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

void Uf3Fitness::compute_l1_l2_host(float& l1, float& l2)
{
  int nparam = model_->num_parameters();
  std::vector<float> params(nparam);
  model_->get_parameters(params.data());
  float s1 = 0, s2 = 0;
  for (int i = 0; i < nparam; i++) {
    s1 += fabsf(params[i]);
    s2 += params[i] * params[i];
  }
  l1 = s1 / nparam;
  l2 = sqrtf(s2 / nparam);
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
      float s = params[i] > 0.0f ? 1.0f : (params[i] < 0.0f ? -1.0f : 0.0f);
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

float Uf3Fitness::compute_loss_train_only(int batch_id)
{
  int bid = batch_id % dataset_.num_batches;
  int B = dataset_.batch_size(bid);
  int total_atoms = dataset_.batch_total_atoms[bid];
  cudaStream_t s = model_->stream();

  // 1. Forward pass on the GPU (energies in d_energy_buf, forces in d_fx/fy/fz).
  model_->evaluate_batch(dataset_, bid, d_energy_);

  // 2. GPU reduction: only 2 floats need to come back to host.  Issue Async
  //    D2H into pinned memory so the host-side L1/L2 work can run in parallel.
  model_->compute_loss_reduction(dataset_, bid, d_loss_sum_);
  CHECK(cudaMemcpyAsync(h_loss_sum_pinned_, d_loss_sum_.data(),
                        2 * sizeof(float), cudaMemcpyDeviceToHost, s));

  // 3. L1/L2 on host coefficient mirror — overlaps the scalar D2H above.
  compute_l1_l2_host(loss_l1, loss_l2);

  // 4. Now we need loss_e / loss_f, so synchronize the compute stream.
  CHECK(cudaStreamSynchronize(s));
  loss_e = (float)sqrt(h_loss_sum_pinned_[0] / std::max(1, B));
  loss_f = (float)sqrt(h_loss_sum_pinned_[1] / std::max(1, 3 * total_atoms));

  loss_total = lambda_e_ * loss_e + lambda_f_ * loss_f
             + lambda_1_ * loss_l1 + lambda_2_ * loss_l2;
  return loss_total;
}

float Uf3Fitness::compute_loss(int batch_id, int generation, int stage_id)
{
  float total = compute_loss_train_only(batch_id);

  // Optional test-set evaluation (slow path; runs at most every 100 gens).
  if (test_set_size_ > 0 && generation > 0 && generation % 100 == 0) {
    evaluate_test_set(generation, stage_id);
  }
  if (floss_ && generation > 0 && generation % 100 == 0) {
    fprintf(floss_, "%d %d %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
            stage_id, generation, loss_total, loss_l1, loss_l2,
            loss_e, loss_f, test_e, test_f);
    fflush(floss_);
  }
  return total;
}

void Uf3Fitness::evaluate_test_set(int generation, int /*stage_id*/)
{
  int ntest = std::min(100, test_set_size_);
  std::vector<int> tidx(ntest);
  int t_total = 0;
  for (int i = 0; i < ntest; i++) {
    tidx[i] = i;
    t_total += test_set_[i].num_atoms;
  }
  if ((int)h_fx_test_.size() < t_total) {
    h_fx_test_.resize(t_total);
    h_fy_test_.resize(t_total);
    h_fz_test_.resize(t_total);
  }
  if ((int)h_energy_test_.size() < ntest) {
    h_energy_test_.resize(ntest);
  }

  // Slow path — uses the legacy host-frame evaluator.
  model_->evaluate(test_set_, tidx, d_energy_test_);
  d_energy_test_.copy_to_host(h_energy_test_.data());

  // For the test-set we re-use d_fx because Uf3Model::evaluate(frames,...)
  // writes into batch-local positions for the host-frame path; we read back
  // exactly t_total floats from each axis.
  cudaMemcpy(h_fx_test_.data(), model_->d_fx.data(), t_total * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_fy_test_.data(), model_->d_fy.data(), t_total * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_fz_test_.data(), model_->d_fz.data(), t_total * sizeof(float),
             cudaMemcpyDeviceToHost);

  double te_sum2 = 0;
  for (int i = 0; i < ntest; i++) {
    int na = test_set_[i].num_atoms;
    double diff = (double)h_energy_test_[i] / na
                - (double)test_set_[i].energy / na;
    te_sum2 += diff * diff;
  }
  test_e = (float)sqrt(te_sum2 / ntest);

  double tf_sum2 = 0;
  int tf_atoms = 0, toff = 0;
  for (int i = 0; i < ntest; i++) {
    const Uf3Frame& f = test_set_[i];
    tf_atoms += f.num_atoms;
    for (int j = 0; j < f.num_atoms; j++) {
      double dx = (double)h_fx_test_[toff + j] - (double)f.fx[j];
      double dy = (double)h_fy_test_[toff + j] - (double)f.fy[j];
      double dz = (double)h_fz_test_[toff + j] - (double)f.fz[j];
      tf_sum2 += dx * dx + dy * dy + dz * dz;
    }
    toff += f.num_atoms;
  }
  test_f = (float)sqrt(tf_sum2 / (3.0 * std::max(1, tf_atoms)));

  // Force buffer was just clobbered by the test-set evaluator (which uses
  // batch-local offsets) — caller must not re-use d_fx for the train batch
  // after this without first re-running evaluate_batch(...).  No-op here:
  // compute_loss returns immediately after.
  (void)generation;
}

void Uf3Fitness::compute_gradient(int batch_id, int generation,
                                  std::vector<float>& grad)
{
  (void)generation;
  int bid = batch_id % dataset_.num_batches;
  // Train-only loss leaves d_energy_buf and d_fx/fy/fz populated and untouched
  // by any test-set side-effect, so the gradient kernel can read them directly.
  compute_loss_train_only(bid);
  model_->compute_loss_gradient(
    dataset_, bid, loss_e, loss_f, lambda_e_, lambda_f_,
    d_grad_, d_ediff_, grad);
  accumulate_regularization_gradient(grad);
}

float Uf3Fitness::compute_loss_for_params(
  const float* params, int batch_id, int generation, int stage_id)
{
  // Use the async path so the host-side coeff packing + raw H2D + GPU spline
  // build chain naturally onto the model's compute stream.  evaluate_batch /
  // compute_loss_reduction launch on the same stream, so no extra sync needed
  // until we want loss_e/loss_f.
  model_->set_parameters_async(params, model_->stream());
  return compute_loss(batch_id, generation, stage_id);
}

Uf3Fitness::~Uf3Fitness()
{
  if (floss_) {
    fclose(floss_);
    floss_ = nullptr;
  }
  if (h_loss_sum_pinned_) {
    cudaFreeHost(h_loss_sum_pinned_);
    h_loss_sum_pinned_ = nullptr;
  }
  if (h_loss_sum_pop_pinned_) {
    cudaFreeHost(h_loss_sum_pop_pinned_);
    h_loss_sum_pop_pinned_ = nullptr;
  }
}

void Uf3Fitness::compute_loss_population(
  const float* host_pop_params, int pop,
  int batch_id, int generation, int stage_id,
  float* out_loss_total)
{
  if (pop <= 0) return;

  // 3B fallback: pop kernels currently only cover 2B.  Use the existing
  // single-individual path one at a time.
  if (model_->has_3b()) {
    int nparam = model_->num_parameters();
    for (int p = 0; p < pop; p++) {
      out_loss_total[p] = compute_loss_for_params(
        host_pop_params + (size_t)p * nparam, batch_id, generation, stage_id);
    }
    return;
  }

  int bid = batch_id % dataset_.num_batches;
  int B = dataset_.batch_size(bid);
  int total_atoms = dataset_.batch_total_atoms[bid];
  cudaStream_t s = model_->stream();

  // Grow pop-side buffers as needed.
  if (pop > pop_capacity_h_) {
    if (h_loss_sum_pop_pinned_) cudaFreeHost(h_loss_sum_pop_pinned_);
    CHECK(cudaHostAlloc((void**)&h_loss_sum_pop_pinned_,
                        (size_t)2 * pop * sizeof(float), cudaHostAllocDefault));
    pop_capacity_h_ = pop;
  }
  if ((int)d_loss_sum_pop_.size() < 2 * pop) d_loss_sum_pop_.resize(2 * pop);
  if ((int)d_energy_pop_.size() < (size_t)pop * B) d_energy_pop_.resize((size_t)pop * B);

  // 1. Async build P spline tables on the GPU.
  model_->set_population_parameters_async(host_pop_params, pop, s);
  // 2. P x forward passes (one launch per kernel, grid=(B,P)).
  model_->evaluate_batch_population(dataset_, bid, pop, d_energy_pop_);
  // 3. P x GPU reductions; output 2*P floats.
  model_->compute_loss_reduction_population(dataset_, bid, pop, d_loss_sum_pop_);
  // 4. Async D2H of the 2*P scalars.
  CHECK(cudaMemcpyAsync(h_loss_sum_pop_pinned_, d_loss_sum_pop_.data(),
                        (size_t)2 * pop * sizeof(float),
                        cudaMemcpyDeviceToHost, s));

  // 5. Overlap host work: per-individual L1/L2 from the raw 2B coeffs the
  //    caller just handed us.  3B is excluded (we early-returned above).
  int nparam = model_->num_parameters();
  std::vector<float> reg_per_ind(2 * pop, 0.0f);  // [p*2+0]=L1, +1=L2
  for (int p = 0; p < pop; p++) {
    const float* params = host_pop_params + (size_t)p * nparam;
    float s1 = 0.0f, s2 = 0.0f;
    for (int i = 0; i < nparam; i++) {
      float v = params[i];
      s1 += fabsf(v);
      s2 += v * v;
    }
    reg_per_ind[p * 2 + 0] = s1 / nparam;
    reg_per_ind[p * 2 + 1] = sqrtf(s2 / nparam);
  }

  // 6. Sync the compute stream so the loss numerators are visible.
  CHECK(cudaStreamSynchronize(s));

  float e_norm = 1.0f / std::max(1, B);
  float f_norm = 1.0f / std::max(1, 3 * total_atoms);
  for (int p = 0; p < pop; p++) {
    float le = sqrtf(h_loss_sum_pop_pinned_[p * 2 + 0] * e_norm);
    float lf = sqrtf(h_loss_sum_pop_pinned_[p * 2 + 1] * f_norm);
    float l1 = reg_per_ind[p * 2 + 0];
    float l2 = reg_per_ind[p * 2 + 1];
    out_loss_total[p] = lambda_e_ * le + lambda_f_ * lf
                      + lambda_1_ * l1 + lambda_2_ * l2;
  }

  // Mirror compute_loss's floss_ logging behavior using individual 0 as the
  // "representative" sample.  Matches the cadence (every 100 gens, skip gen 0).
  if (floss_ && generation > 0 && generation % 100 == 0) {
    float le0 = sqrtf(h_loss_sum_pop_pinned_[0] * e_norm);
    float lf0 = sqrtf(h_loss_sum_pop_pinned_[1] * f_norm);
    fprintf(floss_, "%d %d %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
            stage_id, generation, out_loss_total[0],
            reg_per_ind[0], reg_per_ind[1], le0, lf0, test_e, test_f);
    fflush(floss_);
  }
}
