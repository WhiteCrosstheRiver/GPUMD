/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

// Host-only, double precision MM solver. This header deliberately has no CUDA
// dependencies, so the exact solver used by training can be tested on a CPU.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fcvp
{
struct Group {
  // Already weighted and divided by the observation count: H=A'WA/N,
  // g=A'Wy/N, yy=y'Wy/N. H is row-major and symmetric.
  std::vector<double> h, g;
  double yy = 0.0, lambda = 0.0;
};
struct Options {
  int iterations = 40, coordinate_sweeps = 20000, total_parameters = 1;
  double epsilon = 1e-8, tolerance = 1e-8;
  double lambda1 = 0.0, lambda2 = 0.0;
  double fixed_l1 = 0.0, fixed_l2_squared = 0.0;
  double proximal_relative = 1e-12;
};
struct Result {
  std::vector<double> beta, objectives;
  bool converged = false;
  int iterations = 0;
  double proximal_shift = 0.0;
};

inline double squared_error(const Group& group, const std::vector<double>& x)
{
  // Extended precision reduces cancellation near an exact fit. The final
  // candidate must also be checked by the native predictor, outside this solver.
  long double value = group.yy;
  const int p = x.size();
  for (int i = 0; i < p; ++i) {
    value -= 2.0L * x[i] * group.g[i];
    for (int j = 0; j < p; ++j)
      value += (long double)x[i] * group.h[i * p + j] * x[j];
  }
  return std::max(0.0, static_cast<double>(value));
}

inline double
objective(const std::vector<Group>& groups, const std::vector<double>& x, const Options& o)
{
  double value = 0.0, l1 = o.fixed_l1, l2 = o.fixed_l2_squared;
  for (const auto& group : groups)
    if (group.lambda > 0)
      value += group.lambda * std::sqrt(squared_error(group, x) + o.epsilon * o.epsilon);
  for (double v : x) {
    l1 += std::abs(v);
    l2 += v * v;
  }
  return value + o.lambda1 * l1 / o.total_parameters +
         o.lambda2 * std::sqrt(l2 / o.total_parameters + o.epsilon * o.epsilon);
}

inline bool cholesky_solve(std::vector<double> h, std::vector<double> rhs, std::vector<double>& x)
{
  const int p = rhs.size();
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j <= i; ++j) {
      double v = h[i * p + j];
      for (int k = 0; k < j; ++k)
        v -= h[i * p + k] * h[j * p + k];
      if (i == j) {
        if (!(v > 0.0) || !std::isfinite(v))
          return false;
        h[i * p + i] = std::sqrt(v);
      } else
        h[i * p + j] = v / h[j * p + j];
    }
    for (int j = 0; j < i; ++j)
      rhs[i] -= h[i * p + j] * rhs[j];
    rhs[i] /= h[i * p + i];
  }
  x = rhs;
  for (int i = p - 1; i >= 0; --i) {
    for (int j = i + 1; j < p; ++j)
      x[i] -= h[j * p + i] * x[j];
    x[i] /= h[i * p + i];
    if (!std::isfinite(x[i]))
      return false;
  }
  return true;
}

inline Result solve(const std::vector<Group>& groups, std::vector<double> x, const Options& o)
{
  const int p = x.size();
  if (
    !p || o.total_parameters < p || o.iterations < 1 || o.coordinate_sweeps < 1 ||
    !(o.epsilon * o.epsilon > 0) || !std::isfinite(o.epsilon * o.epsilon) || !(o.epsilon > 0) ||
    !(o.tolerance > 0) || !std::isfinite(o.tolerance) || o.lambda1 < 0 || o.lambda2 < 0 ||
    !std::isfinite(o.lambda1 + o.lambda2 + o.fixed_l1 + o.fixed_l2_squared) ||
    !(o.proximal_relative > 0) || !std::isfinite(o.proximal_relative))
    throw std::invalid_argument("Invalid FCVP options");
  for (const auto& group : groups) {
    if (
      group.h.size() != size_t(p * p) || group.g.size() != size_t(p) || group.lambda < 0 ||
      !std::isfinite(group.lambda + group.yy))
      throw std::invalid_argument("Invalid FCVP Gram group");
    for (double v : group.h)
      if (!std::isfinite(v))
        throw std::invalid_argument("Nonfinite Gram matrix");
    for (double v : group.g)
      if (!std::isfinite(v))
        throw std::invalid_argument("Nonfinite RHS");
  }
  for (double v : x)
    if (!std::isfinite(v))
      throw std::invalid_argument("Nonfinite beta");
  Result result;
  result.objectives.push_back(objective(groups, x, o));
  for (int iteration = 0; iteration < o.iterations; ++iteration) {
    std::vector<double> h(p * p, 0.0), rhs(p, 0.0), candidate;
    for (const auto& group : groups) {
      if (group.lambda == 0)
        continue;
      const double w = group.lambda / std::sqrt(squared_error(group, x) + o.epsilon * o.epsilon);
      for (int i = 0; i < p * p; ++i)
        h[i] += w * group.h[i];
      for (int i = 0; i < p; ++i)
        rhs[i] += w * group.g[i];
    }
    double norm2 = o.fixed_l2_squared;
    for (double v : x)
      norm2 += v * v;
    const double l2 = o.lambda2 / (o.total_parameters *
                                   std::sqrt(norm2 / o.total_parameters + o.epsilon * o.epsilon));
    double scale = 1.0;
    for (int i = 0; i < p; ++i) {
      h[i * p + i] += l2;
      scale = std::max(scale, std::abs(h[i * p + i]));
    }
    // A proximal term about x (not a ridge penalty about zero) stabilizes
    // composition/bias null spaces without silently changing the objective.
    double shift = scale * o.proximal_relative;
    bool solved = false;
    std::vector<double> stable_h, stable_rhs;
    for (int attempt = 0; attempt < 8 && !solved; ++attempt) {
      stable_h = h;
      stable_rhs = rhs;
      for (int i = 0; i < p; ++i) {
        stable_h[i * p + i] += shift;
        stable_rhs[i] += shift * x[i];
      }
      solved = cholesky_solve(stable_h, stable_rhs, candidate);
      if (!solved)
        shift *= 10;
    }
    if (!solved)
      throw std::runtime_error("FCVP quadratic solve failed");
    result.proximal_shift = std::max(result.proximal_shift, shift);
    const double l1 = o.lambda1 / o.total_parameters;
    if (l1 > 0.0) {
      // Exact nonsmooth L1 coordinate minimization of the convex majorizer.
      candidate = x;
      for (int sweep = 0; sweep < o.coordinate_sweeps; ++sweep) {
        double change = 0.0, magnitude = 1.0;
        for (int i = 0; i < p; ++i) {
          double r = stable_rhs[i];
          for (int j = 0; j < p; ++j)
            if (j != i)
              r -= stable_h[i * p + j] * candidate[j];
          const double next =
            std::copysign(std::max(0.0, std::abs(r) - l1), r) / stable_h[i * p + i];
          change = std::max(change, std::abs(next - candidate[i]));
          magnitude = std::max(magnitude, std::abs(next));
          candidate[i] = next;
        }
        if (change <= o.tolerance * magnitude)
          break;
      }
    }
    // Guard finite precision and inexact coordinate solves with the actual
    // smoothed objective, not just the quadratic surrogate.
    double value = objective(groups, candidate, o);
    for (int backtrack = 0;
         backtrack < 40 && (!std::isfinite(value) || value > result.objectives.back());
         ++backtrack) {
      for (int i = 0; i < p; ++i)
        candidate[i] = 0.5 * (candidate[i] + x[i]);
      value = objective(groups, candidate, o);
    }
    if (!std::isfinite(value) || value > result.objectives.back())
      break;
    double change = 0.0, magnitude = 1.0;
    for (int i = 0; i < p; ++i) {
      change = std::max(change, std::abs(candidate[i] - x[i]));
      magnitude = std::max(magnitude, std::abs(x[i]));
    }
    x.swap(candidate);
    result.objectives.push_back(value);
    ++result.iterations;
    if (change <= o.tolerance * magnitude) {
      result.converged = true;
      break;
    }
  }
  result.beta = x;
  return result;
}
} // namespace fcvp
