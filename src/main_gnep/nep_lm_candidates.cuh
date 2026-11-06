// SPDX-License-Identifier: GPL-3.0-or-later

/*----------------------------------------------------------------------------80
Candidate training losses for the native LM loop.
------------------------------------------------------------------------------*/

#pragma once
#include "dataset.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

namespace lm_training
{

struct Candidate_Loss {
  int index;
  double loss;
  Batch_MSE mse;
};

template <typename Evaluate_Train>
static std::vector<Candidate_Loss> evaluate_candidate_losses(
  const float* parameters,
  const float* deltas,
  int count,
  const std::vector<bool>& valid,
  double before,
  std::vector<float>& candidate,
  Evaluate_Train evaluate_train)
{
  std::vector<Candidate_Loss> losses;
  candidate.resize(count);
  bool calculate_neighbor = true;
  for (int j = 0; j < int(valid.size()); ++j) {
    if (!valid[j]) {
      continue;
    }
    for (int i = 0; i < count; ++i) {
      candidate[i] = parameters[i] + deltas[size_t(j) * count + i];
    }
    Batch_MSE mse;
    const double value = evaluate_train(candidate.data(), calculate_neighbor, mse);
    calculate_neighbor = false;
    if (std::isfinite(value) && value < before) {
      losses.push_back({j, value, mse});
    }
  }
  // The original loop keeps the earliest damping index when losses are equal.
  std::stable_sort(
    losses.begin(), losses.end(), [](const Candidate_Loss& a, const Candidate_Loss& b) {
      return a.loss < b.loss;
    });
  return losses;
}

} // namespace lm_training
