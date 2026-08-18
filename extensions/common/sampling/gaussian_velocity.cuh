/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#pragma once

#include <cmath>
#include <random>

// |v| fixed; theta ~ |N(0, sigma_deg)|; phi ~ U(0, 2 pi); beam along -z.
inline void sample_gaussian_beam_velocity(
  double v_mag, double theta_sigma_deg, std::mt19937& gen, double v[3])
{
  const double pi = 3.14159265358979;
  std::normal_distribution<double> unit_normal(0.0, 1.0);
  std::uniform_real_distribution<double> uniform_phi(0.0, 2.0 * pi);
  const double theta = fabs(unit_normal(gen) * theta_sigma_deg * pi / 180.0);
  const double phi = uniform_phi(gen);
  v[0] = v_mag * sin(theta) * cos(phi);
  v[1] = v_mag * sin(theta) * sin(phi);
  v[2] = -v_mag * cos(theta);
}
