/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#pragma once

#include "model/atom.cuh"
#include "model/box.cuh"

// Anchor is (px, py, zmax). zmax is the highest atom within XY radius (PBC);
// if none, the global zmax is used.
inline void query_local_zmax(
  const Atom& atom,
  const Box& box,
  double px,
  double py,
  double radius,
  double& ax,
  double& ay,
  double& az)
{
  const int N = atom.number_of_atoms;
  int best = -1;
  double best_z = 0.0;
  const double r2 = radius * radius;
  for (int i = 0; i < N; ++i) {
    double dx = atom.cpu_position_per_atom[i] - px;
    double dy = atom.cpu_position_per_atom[i + N] - py;
    double dz = 0.0;
    apply_mic(box, dx, dy, dz);
    if (dx * dx + dy * dy <= r2) {
      const double z = atom.cpu_position_per_atom[i + 2 * N];
      if (best < 0 || z > best_z) {
        best_z = z;
        best = i;
      }
    }
  }
  if (best < 0) {
    for (int i = 0; i < N; ++i) {
      const double z = atom.cpu_position_per_atom[i + 2 * N];
      if (best < 0 || z > best_z) {
        best_z = z;
        best = i;
      }
    }
  }
  ax = px;
  ay = py;
  az = atom.cpu_position_per_atom[best + 2 * N];
}
