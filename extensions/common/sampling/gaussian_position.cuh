/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#pragma once

#include "model/box.cuh"
#include <cmath>
#include <random>

// x = x0 + N(0, sigma), y = y0 + N(0, sigma). Does not wrap.
inline void sample_gaussian_xy(
  double x0, double y0, double sigma, std::mt19937& gen, double& x, double& y)
{
  std::normal_distribution<double> dist(0.0, sigma);
  x = x0 + dist(gen);
  y = y0 + dist(gen);
}

inline void wrap_xy_into_box(const Box& box, double& x, double& y)
{
  const double z = 0.0;
  double sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
  double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
  if (box.pbc_x == 1) {
    sx -= floor(sx);
  }
  if (box.pbc_y == 1) {
    sy -= floor(sy);
  }
  x = box.cpu_h[0] * sx + box.cpu_h[1] * sy;
  y = box.cpu_h[3] * sx + box.cpu_h[4] * sy;
}
