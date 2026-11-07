/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#pragma once

#include "model/atom.cuh"
#include "model/box.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

// XY max-z height map (PBC in x/y). Build once per deposit command; each atom
// samples a few bins instead of scanning all N.
struct LocalSurfaceMap
{
  int nx = 0;
  int ny = 0;
  double cell = 1.0;
  double global_zmax = 0.0;
  std::vector<double> zmax;

  void build(const Atom& atom, const Box& box, double cell_size)
  {
    const int N = atom.number_of_atoms;
    if (N <= 0) {
      return;
    }
    cell = cell_size;
    const double Lx = box.cpu_h[0];
    const double Ly = box.cpu_h[4];
    nx = std::max(1, (int)std::ceil(Lx / cell));
    ny = std::max(1, (int)std::ceil(Ly / cell));
    const double unset = -1.0e300;
    zmax.assign(static_cast<size_t>(nx) * ny, unset);

    const double* r0 = atom.cpu_position_per_atom.data();
    const double* rx = r0;
    const double* ry = r0 + N;
    const double* rz = r0 + 2 * N;
    auto wrap = [](int i, int ncell) {
      int rem = i % ncell;
      return rem < 0 ? rem + ncell : rem;
    };
    global_zmax = rz[0];
    for (int i = 0; i < N; ++i) {
      if (rz[i] > global_zmax) {
        global_zmax = rz[i];
      }
      const int ix = wrap((int)std::floor(rx[i] / cell), nx);
      const int iy = wrap((int)std::floor(ry[i] / cell), ny);
      double& cur = zmax[ix + nx * iy];
      if (rz[i] > cur) {
        cur = rz[i];
      }
    }
  }

  // Highest atom z within XY radius of (px,py) (PBC); global max if none.
  double query(double px, double py, double radius) const
  {
    auto wrap = [](int i, int ncell) {
      int rem = i % ncell;
      return rem < 0 ? rem + ncell : rem;
    };
    const double Lx_cells = nx;
    const double Ly_cells = ny;
    const double cx0 = px / cell;
    const double cy0 = py / cell;
    const int span = (int)std::ceil(radius / cell);
    const int icx = (int)std::floor(cx0);
    const int icy = (int)std::floor(cy0);
    bool found = false;
    double best = 0.0;
    for (int ix = icx - span; ix <= icx + span; ++ix) {
      for (int iy = icy - span; iy <= icy + span; ++iy) {
        const int wx = wrap(ix, nx);
        const int wy = wrap(iy, ny);
        const double v = zmax[wx + nx * wy];
        if (v <= -1.0e299) {
          continue;
        }
        // bin center must lie within the radius (slightly conservative on edges)
        double bx = (ix + 0.5) * cell - px;
        double by = (iy + 0.5) * cell - py;
        // minimum-image in bin units around the query cell
        double dxc = (double)(ix - icx);
        double dyc = (double)(iy - icy);
        if (std::abs(dxc) > Lx_cells / 2) {
          dxc -= (dxc > 0 ? Lx_cells : -Lx_cells);
        }
        if (std::abs(dyc) > Ly_cells / 2) {
          dyc -= (dyc > 0 ? Ly_cells : -Ly_cells);
        }
        bx = dxc * cell;
        by = dyc * cell;
        if (bx * bx + by * by <= (radius + 0.7071 * cell) * (radius + 0.7071 * cell)) {
          if (!found || v > best) {
            best = v;
            found = true;
          }
        }
      }
    }
    return found ? best : global_zmax;
  }
};

// Anchor is (px, py, zmax). zmax is the highest atom within XY radius (PBC);
// if none, the global zmax is used. Keeps the original per-atom scan behavior.
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
