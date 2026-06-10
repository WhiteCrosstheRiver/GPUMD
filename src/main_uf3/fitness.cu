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
RMSE goes through a slow CPU path that only triggers at logging checkpoints.

Logging design
--------------
Every optimizer stage writes to two files:
  loss.out                  — unified log across all stages
  loss_stage_N_OPT.out      — per-stage log for debugging / plotting

Both files share the same column layout.  Column semantics:
  stage      integer stage index (0-based)
  optimizer  adam | lbfgs | snes | es | lstsq
  iter_unit  what one local_iter increment means:
               grad_step        — one gradient update (Adam, LBFGS)
               lbfgs_step       — one LBFGS outer step
               snes_generation  — one SNES population evaluation
               es_generation    — one ES generation
               solve            — lstsq one-shot solve
  local_iter within-stage iteration counter (1-based)
  wall_time_s elapsed seconds since stage start

Rows are written at local_iter == 1 and every 100 thereafter.  A
duplicate-logging guard (last_logged_local_iter_) prevents multiple writes
when compute_loss is called several times per iteration (LBFGS line-search,
ES inner population loop).
------------------------------------------------------------------------------*/

#include "fitness.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Log format shared by loss.out and per-stage files
// ---------------------------------------------------------------------------
static const char LOG_HEADER[] =
  "# stage  optimizer     iter_unit         local_iter  wall_time_s  "
  "L_t          L_1          L_2          L_e_train    L_f_train    "
  "L_e_test     L_f_test\n";

static const char LOG_FMT[] =
  "%-8d%-14s%-18s%-12d%-13.2f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f\n";

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
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
    float nn_cut = para.n_max_3b[0] > 0
                     ? (float)std::max(para.rc_3b[0], para.rc_3b[1]) : 0.0f;
    float ghost_cut = std::max((float)para.rc_2b, nn_cut);
    test_set_ = load_uf3_frames(para.test_data.c_str(), para.elements, nn_cut, ghost_cut);
    if (para.min_atoms > 1) {
      test_set_.erase(
        std::remove_if(test_set_.begin(), test_set_.end(),
                       [&](const Uf3Frame& f) { return f.num_atoms < para.min_atoms; }),
        test_set_.end());
    }
    test_set_size_ = (int)test_set_.size();
    printf("Loaded %d test frames.\n", test_set_size_);
  } else {
    printf("Warning: no test data specified. Test RMSE will be 0.\n");
  }

  floss_ = fopen("loss.out", "w");
  if (floss_) {
    fputs(LOG_HEADER, floss_);
    fflush(floss_);
  }
}

// ---------------------------------------------------------------------------
// Stage context — call once per optimizer stage before the training loop
// ---------------------------------------------------------------------------
void Uf3Fitness::begin_stage(int stage_id, const char* opt_name, const char* iter_unit)
{
  cur_stage_id_ = stage_id;
  cur_opt_name_ = opt_name;
  cur_iter_unit_ = iter_unit;
  last_logged_local_iter_ = -1;

  if (fstage_) {
    fclose(fstage_);
    fstage_ = nullptr;
  }

  char fname[128];
  snprintf(fname, sizeof(fname), "loss_stage_%d_%s.out", stage_id, opt_name);
  fstage_ = fopen(fname, "w");
  if (fstage_) {
    fputs(LOG_HEADER, fstage_);
    fflush(fstage_);
  }
  printf("  Stage %d [%s, iter_unit=%s] — log: %s\n",
         stage_id, opt_name, iter_unit, fname);
}

// ---------------------------------------------------------------------------
// Shared log-line writer: stdout + loss.out + per-stage file
// ---------------------------------------------------------------------------
void Uf3Fitness::write_log_line(
  int stage_id, const char* opt, const char* iunit,
  int local_iter, float wall_time_s,
  float lt, float l1, float l2, float le, float lf, float te, float tf)
{
  printf(LOG_FMT, stage_id, opt, iunit, local_iter, wall_time_s,
         lt, l1, l2, le, lf, te, tf);
  fflush(stdout);
  if (floss_) {
    fprintf(floss_, LOG_FMT, stage_id, opt, iunit, local_iter, wall_time_s,
            lt, l1, l2, le, lf, te, tf);
    fflush(floss_);
  }
  if (fstage_) {
    fprintf(fstage_, LOG_FMT, stage_id, opt, iunit, local_iter, wall_time_s,
            lt, l1, l2, le, lf, te, tf);
    fflush(fstage_);
  }
}

// ---------------------------------------------------------------------------
// L1 / L2 regularization — computed on the host coefficient mirror
// ---------------------------------------------------------------------------
void Uf3Fitness::compute_l1_l2_host(float& l1, float& l2)
{
  int nparam = model_->num_parameters();
  if ((int)h_params_cache_.size() < nparam) h_params_cache_.resize(nparam);
  model_->get_parameters(h_params_cache_.data());
  float s1 = 0, s2 = 0;
  for (int i = 0; i < nparam; i++) {
    s1 += fabsf(h_params_cache_[i]);
    s2 += h_params_cache_[i] * h_params_cache_[i];
  }
  l1 = s1 / nparam;
  l2 = sqrtf(s2 / nparam);
}

void Uf3Fitness::accumulate_regularization_gradient(std::vector<float>& grad)
{
  int nparam = model_->num_parameters();
  if ((int)grad.size() != nparam) grad.resize(nparam, 0.0f);
  if ((int)h_params_cache_.size() < nparam) h_params_cache_.resize(nparam);
  model_->get_parameters(h_params_cache_.data());
  const std::vector<float>& params = h_params_cache_;
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

// ---------------------------------------------------------------------------
// Hot-path train loss — GPU forward + 2-float D2H, no test-set side-effects
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Test-set evaluation (slow path — CPU D2H loops, called at checkpoints only)
// ---------------------------------------------------------------------------
void Uf3Fitness::evaluate_test_set(int /*generation*/, int /*stage_id*/)
{
  int ntest = test_set_size_;  // evaluate all test frames
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

  // Forces are laid out at EXPANDED (real+ghost) offsets in model_->d_fx; gather
  // the real-atom forces (first n_real of each frame's expanded block) into the
  // contiguous per-frame layout the RMSE loop below expects.
  int t_exp = 0;
  for (int i = 0; i < ntest; i++) t_exp += test_set_[i].num_total;
  std::vector<float> exp_fx(t_exp), exp_fy(t_exp), exp_fz(t_exp);
  cudaMemcpy(exp_fx.data(), model_->d_fx.data(), t_exp * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(exp_fy.data(), model_->d_fy.data(), t_exp * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(exp_fz.data(), model_->d_fz.data(), t_exp * sizeof(float), cudaMemcpyDeviceToHost);
  {
    int eoff = 0, roff = 0;
    for (int i = 0; i < ntest; i++) {
      int nr = test_set_[i].num_atoms;
      for (int j = 0; j < nr; j++) {
        h_fx_test_[roff + j] = exp_fx[eoff + j];
        h_fy_test_[roff + j] = exp_fy[eoff + j];
        h_fz_test_[roff + j] = exp_fz[eoff + j];
      }
      eoff += test_set_[i].num_total;
      roff += nr;
    }
  }

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
}

// ---------------------------------------------------------------------------
// Main loss entry-point (single individual)
// ---------------------------------------------------------------------------
float Uf3Fitness::compute_loss(int batch_id, int generation, int stage_id,
                                int local_iter, float wall_time_s)
{
  float total = compute_loss_train_only(batch_id);

  // Log at local_iter==1 (stage start) and every 100 steps thereafter.
  // The duplicate guard prevents double-logging when the same local_iter is
  // passed multiple times (LBFGS line-search, ES inner population loop).
  bool is_checkpoint = (local_iter % 100 == 0 || local_iter == 1)
                        && local_iter != last_logged_local_iter_;

  if (test_set_size_ > 0 && is_checkpoint) {
    evaluate_test_set(generation, stage_id);
  }
  if (is_checkpoint) {
    last_logged_local_iter_ = local_iter;
    write_log_line(stage_id, cur_opt_name_.c_str(), cur_iter_unit_.c_str(),
                   local_iter, wall_time_s,
                   loss_total, loss_l1, loss_l2, loss_e, loss_f, test_e, test_f);
  }
  return total;
}

float Uf3Fitness::compute_loss_for_params(
  const float* params, int batch_id, int generation, int stage_id,
  int local_iter, float wall_time_s)
{
  model_->set_parameters_async(params, model_->stream());
  return compute_loss(batch_id, generation, stage_id, local_iter, wall_time_s);
}

void Uf3Fitness::compute_gradient(int batch_id, int generation,
                                  std::vector<float>& grad)
{
  (void)generation;
  int bid = batch_id % dataset_.num_batches;
  compute_loss_train_only(bid);
  model_->compute_loss_gradient(
    dataset_, bid, loss_e, loss_f, lambda_e_, lambda_f_,
    d_grad_, d_ediff_, grad);
  accumulate_regularization_gradient(grad);
  // Project the 3B gradient onto the neighbour-swap-symmetric subspace so the
  // optimizer stays consistent with the symmetrized model (set_parameters keeps
  // the iterate symmetric; this keeps the search direction symmetric too).
  model_->symmetrize_3b_gradient(grad);
  // Keep frozen (edge) coefficients fixed at 0 for smooth cutoffs.
  const std::vector<char>& fr = model_->frozen();
  for (int i = 0; i < (int)grad.size() && i < (int)fr.size(); i++)
    if (fr[i]) grad[i] = 0.0f;
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
Uf3Fitness::~Uf3Fitness()
{
  if (floss_) {
    fclose(floss_);
    floss_ = nullptr;
  }
  if (fstage_) {
    fclose(fstage_);
    fstage_ = nullptr;
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

// ---------------------------------------------------------------------------
// Population-mode loss (SNES / ES)
// ---------------------------------------------------------------------------
void Uf3Fitness::compute_loss_population(
  const float* host_pop_params, int pop,
  int batch_id, int generation, int stage_id,
  int local_iter, float wall_time_s,
  float* out_loss_total)
{
  if (pop <= 0) return;

  // Correctness-first: the 2B population fast-path predates the PBC ghost-atom
  // and 1-body (e0) changes and would evaluate the wrong model, so route every
  // individual through the fully-correct single-individual path for now.  The
  // performance phase will restore a correct *parallel* population kernel
  // (ghost-aware 2B+3B+e0) — the priority path for SNES.
  if (true || model_->has_3b()) {
    int nparam = model_->num_parameters();
    for (int p = 0; p < pop; p++) {
      out_loss_total[p] = compute_loss_for_params(
        host_pop_params + (size_t)p * nparam, batch_id, generation, stage_id,
        local_iter, wall_time_s);
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
  // 2. Fused forward + reduction: single kernel per (frame, individual),
  //    no intermediate force/energy buffers written to global memory.
  model_->evaluate_and_reduce_population(dataset_, bid, pop, d_loss_sum_pop_);
  // 3. Async D2H of the 2*P scalars.
  CHECK(cudaMemcpyAsync(h_loss_sum_pop_pinned_, d_loss_sum_pop_.data(),
                        (size_t)2 * pop * sizeof(float),
                        cudaMemcpyDeviceToHost, s));

  // 5. Overlap host work: per-individual L1/L2 from the raw 2B coeffs.
  int nparam = model_->num_parameters();
  if ((int)reg_per_ind_.size() < 2 * pop) reg_per_ind_.assign(2 * pop, 0.0f);
  else std::fill(reg_per_ind_.begin(), reg_per_ind_.begin() + 2 * pop, 0.0f);
  for (int p = 0; p < pop; p++) {
    const float* params = host_pop_params + (size_t)p * nparam;
    float s1 = 0.0f, s2 = 0.0f;
    for (int i = 0; i < nparam; i++) {
      float v = params[i];
      s1 += fabsf(v);
      s2 += v * v;
    }
    reg_per_ind_[p * 2 + 0] = s1 / nparam;
    reg_per_ind_[p * 2 + 1] = sqrtf(s2 / nparam);
  }

  // 6. Sync the compute stream so the loss numerators are visible.
  CHECK(cudaStreamSynchronize(s));

  float e_norm = 1.0f / std::max(1, B);
  float f_norm = 1.0f / std::max(1, 3 * total_atoms);
  for (int p = 0; p < pop; p++) {
    float le = sqrtf(h_loss_sum_pop_pinned_[p * 2 + 0] * e_norm);
    float lf = sqrtf(h_loss_sum_pop_pinned_[p * 2 + 1] * f_norm);
    float l1 = reg_per_ind_[p * 2 + 0];
    float l2 = reg_per_ind_[p * 2 + 1];
    out_loss_total[p] = lambda_e_ * le + lambda_f_ * lf
                      + lambda_1_ * l1 + lambda_2_ * l2;
  }

  // 7. At checkpoints: find best individual, evaluate test set on it, log.
  bool is_checkpoint = (local_iter % 100 == 0 || local_iter == 1)
                        && local_iter != last_logged_local_iter_;
  if (is_checkpoint) {
    // Find best individual by total loss.
    int best_p = 0;
    float best_val = out_loss_total[0];
    for (int p = 1; p < pop; p++) {
      if (out_loss_total[p] < best_val) {
        best_val = out_loss_total[p];
        best_p = p;
      }
    }

    // Evaluate test set on best individual's parameters.
    // (Previously, test_e/test_f were never updated on the population path,
    //  so they always showed 0 in loss.out — fixed here.)
    if (test_set_size_ > 0) {
      model_->set_parameters_async(
        host_pop_params + (size_t)best_p * nparam, s);
      CHECK(cudaStreamSynchronize(s));
      evaluate_test_set(generation, stage_id);
    }

    last_logged_local_iter_ = local_iter;
    float le_best = sqrtf(h_loss_sum_pop_pinned_[best_p * 2 + 0] * e_norm);
    float lf_best = sqrtf(h_loss_sum_pop_pinned_[best_p * 2 + 1] * f_norm);
    float l1_best = reg_per_ind_[best_p * 2 + 0];
    float l2_best = reg_per_ind_[best_p * 2 + 1];
    write_log_line(stage_id, cur_opt_name_.c_str(), cur_iter_unit_.c_str(),
                   local_iter, wall_time_s,
                   best_val, l1_best, l2_best, le_best, lf_best, test_e, test_f);
  }
}
