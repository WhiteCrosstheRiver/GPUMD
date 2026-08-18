/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#pragma once

#include "model/atom.cuh"
#include "model/box.cuh"
#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>

// keep[i] != 0 if atom i is connected to the lowest-z atom.
// Edges: distance <= cutoff. XY uses PBC; z does not.
inline void find_main_component_from_min_z(
  const Atom& atom, const Box& box, double cutoff, std::vector<char>& keep)
{
  const int N = atom.number_of_atoms;
  keep.assign(N, 0);
  if (N <= 0) {
    return;
  }
  if (N == 1) {
    keep[0] = 1;
    return;
  }

  const double* x = atom.cpu_position_per_atom.data();
  const double* y = x + N;
  const double* z = x + 2 * N;

  int seed = 0;
  double zmin = z[0];
  double zmax = z[0];
  for (int i = 1; i < N; ++i) {
    if (z[i] < zmin) {
      zmin = z[i];
      seed = i;
    }
    if (z[i] > zmax) {
      zmax = z[i];
    }
  }

  const double cutoff_sq = cutoff * cutoff;
  const double cell_size = cutoff;
  const double lx = box.cpu_h[0];
  const double ly = box.cpu_h[4];
  const int nx = std::max(1, (int)std::ceil(lx / cell_size));
  const int ny = std::max(1, (int)std::ceil(ly / cell_size));
  const int nz = std::max(1, (int)std::ceil((zmax - zmin + 1.0e-8) / cell_size));
  auto wrap = [](int i, int n) {
    int r = i % n;
    return r < 0 ? r + n : r;
  };
  auto cell_id = [&](int ix, int iy, int iz) { return ix + nx * (iy + ny * iz); };

  std::vector<std::vector<int>> cells(static_cast<size_t>(nx) * ny * nz);
  std::vector<int> cx(N), cy(N), cz(N);
  for (int i = 0; i < N; ++i) {
    cx[i] = wrap((int)std::floor(x[i] / cell_size), nx);
    cy[i] = wrap((int)std::floor(y[i] / cell_size), ny);
    int iz = (int)std::floor((z[i] - zmin) / cell_size);
    if (iz < 0) {
      iz = 0;
    }
    if (iz >= nz) {
      iz = nz - 1;
    }
    cz[i] = iz;
    cells[cell_id(cx[i], cy[i], cz[i])].push_back(i);
  }

  keep[seed] = 1;
  std::queue<int> q;
  q.push(seed);
  while (!q.empty()) {
    const int i = q.front();
    q.pop();
    for (int ox = -1; ox <= 1; ++ox) {
      for (int oy = -1; oy <= 1; ++oy) {
        for (int oz = -1; oz <= 1; ++oz) {
          const int niz = cz[i] + oz;
          if (niz < 0 || niz >= nz) {
            continue;
          }
          const int nix = wrap(cx[i] + ox, nx);
          const int niy = wrap(cy[i] + oy, ny);
          const std::vector<int>& bucket = cells[cell_id(nix, niy, niz)];
          for (int j : bucket) {
            if (keep[j] || j == i) {
              continue;
            }
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double mic_z = 0.0;
            apply_mic(box, dx, dy, mic_z);
            const double dz = z[j] - z[i];
            if (dx * dx + dy * dy + dz * dz <= cutoff_sq) {
              keep[j] = 1;
              q.push(j);
            }
          }
        }
      }
    }
  }
}
