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
#include <string>
#include <vector>

class Uf3Fitness
{
public:
  Uf3Fitness(UF3_Parameters& para, Uf3Model* model,
             const Uf3DatasetGPU& dataset,
             const std::vector<Uf3Frame>& train_set);

  // Call at the start of each optimizer stage to set context for logging.
  // Opens loss_stage_N_OPT.out and resets the per-stage iteration counter.
  void begin_stage(int stage_id, const char* opt_name, const char* iter_unit);

  // local_iter: within-stage iteration number (1-based).
  //   Adam/LBFGS: gradient step count.
  //   SNES/ES:    generation count (one population evaluation each).
  //   lstsq:      always 1.
  // wall_time_s: elapsed seconds since stage start.
  float compute_loss(int batch_id, int generation, int stage_id,
                     int local_iter, float wall_time_s);
  float compute_loss_for_params(const float* params, int batch_id, int generation,
                                int stage_id, int local_iter, float wall_time_s);
  void compute_gradient(int batch_id, int generation, std::vector<float>& grad);

  // Whole-generation loss evaluation: builds and reduces `pop` parameter sets
  // in one launch chain.  out_loss_total[p] receives the unified loss for
  // individual p (energy + force + L1/L2 regularization).
  //
  // Falls back to a per-individual loop when has_3b is true (no pop kernel for
  // 3B yet).  Logs best-individual RMSE at checkpoints and evaluates the test
  // set on that best individual (fixes the stale-zero test RMSE bug on the
  // population path).
  void compute_loss_population(
    const float* host_pop_params,
    int pop,
    int batch_id,
    int generation,
    int stage_id,
    int local_iter,
    float wall_time_s,
    float* out_loss_total);

  // Train-only loss (no test-set side-effects).  Leaves d_fx/fy/fz and
  // d_energy_buf populated for the batch, so the caller may follow it with
  // compute_loss_gradient(...) without re-running the forward pass.
  float compute_loss_train_only(int batch_id);

  int num_parameters() const { return model_->num_parameters(); }
  Uf3Model* model() { return model_; }
  const Uf3DatasetGPU& dataset() const { return dataset_; }

  float loss_e = 0, loss_f = 0, loss_l1 = 0, loss_l2 = 0, loss_total = 0;
  float test_e = 0, test_f = 0;

  ~Uf3Fitness();

private:
  void accumulate_regularization_gradient(std::vector<float>& grad);
  void compute_l1_l2_host(float& l1, float& l2);
  void evaluate_test_set(int generation, int stage_id);
  void write_log_line(int stage_id, const char* opt, const char* iunit,
                      int local_iter, float wall_time_s,
                      float lt, float l1, float l2,
                      float le, float lf, float te, float tf);

  Uf3Model* model_;
  const Uf3DatasetGPU& dataset_;
  const std::vector<Uf3Frame>& train_set_;
  std::vector<Uf3Frame> test_set_;
  int test_set_size_ = 0;

  // Train-path GPU buffers (pre-allocated; never resized on the hot path).
  GPU_Vector<float> d_energy_;     // per-batch predicted energies
  GPU_Vector<float> d_loss_sum_;   // [2]: e_sum2, f_sum2 (GPU-side RMSE numerators)
  GPU_Vector<float> d_grad_;       // [nparam] workspace for loss-gradient kernel
  GPU_Vector<float> d_ediff_;      // [batch] workspace for per-frame energy residuals
  // Pinned host mirror so the D2H of the loss reduction can be Async on the
  // model's compute stream — lets host-side L1/L2 work overlap the scalar copy.
  float* h_loss_sum_pinned_ = nullptr;  // size 2

  // Population-mode buffers (lazy-grown in compute_loss_population).
  GPU_Vector<float> d_loss_sum_pop_;       // [2 * pop]
  GPU_Vector<float> d_energy_pop_;         // [pop * batch]
  float* h_loss_sum_pop_pinned_ = nullptr; // [2 * pop_capacity_h_]
  int pop_capacity_h_ = 0;

  // Test-path host buffers (used only at logging checkpoints).
  std::vector<float> h_energy_test_, h_fx_test_, h_fy_test_, h_fz_test_;
  GPU_Vector<float> d_energy_test_;

  float lambda_e_ = 1, lambda_f_ = 1, lambda_1_ = 0, lambda_2_ = 0;

  // Logging state.
  FILE* floss_ = nullptr;    // main loss.out (all stages)
  FILE* fstage_ = nullptr;   // per-stage loss file (reset by begin_stage)
  std::string cur_opt_name_;
  std::string cur_iter_unit_;
  int cur_stage_id_ = -1;
  // Prevents duplicate log lines when compute_loss is called multiple times
  // with the same local_iter (LBFGS line-search, ES inner pop loop).
  int last_logged_local_iter_ = -1;
};
