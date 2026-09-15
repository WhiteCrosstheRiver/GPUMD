/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#pragma once

#include "../topology/connectivity_bfs.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

// label 0 = unfixed, 1 = fixed.
// Substrate = BFS component from the lowest atom along `axis` (optional type mask).
// Unfix the shear of thickness `offset` below the local surface of that component.
// Atoms not in the substrate stay unfixed. Optional cubic region: substrate
// outside the box is always fixed.
// The height map and labeling passes are threaded (per-thread maps, reduce; the
// label loop is an embarrassingly parallel read-only map).
inline __host__ bool classify_substrate_shell(
  const Atom& atom,
  const Box& box,
  int axis,
  double cutoff,
  double offset,
  const std::vector<char>* eligible,
  const double* region_cubic,
  std::vector<int>& label)
{
  const int N = atom.number_of_atoms;
  label.assign(N, 0);
  if (N <= 0) {
    return false;
  }

  std::vector<char> keep;
  find_main_component_from_min_axis(atom, box, cutoff, axis, eligible, keep);

  int n_sub = 0;
  for (int i = 0; i < N; ++i) {
    n_sub += keep[i] ? 1 : 0;
  }
  if (n_sub == 0) {
    return false;
  }

  const double* r0 = atom.cpu_position_per_atom.data();
  const double* r[3] = {r0, r0 + N, r0 + 2 * N};
  const int t0 = (axis + 1) % 3;
  const int t1 = (axis + 2) % 3;
  const double cell_size = 1.0; // height map; independent of BFS cutoff so grooves stay local
  const double L[3] = {box.cpu_h[0], box.cpu_h[4], box.cpu_h[8]};
  const int nt0 = std::max(1, (int)std::ceil(L[t0] / cell_size));
  const int nt1 = std::max(1, (int)std::ceil(L[t1] / cell_size));
  auto wrap = [](int i, int n) {
    int rem = i % n;
    if (rem < 0) {
      rem += n;
    }
    return rem;
  };
  const double unset = -1.0e300;
  std::vector<int> b0(N), b1(N);
  const int n_hcell = nt0 * nt1;
  std::vector<double> hmap(n_hcell, unset);
  {
    // bin index threaded; height map via per-thread maps reduced into hmap
    const int hw = (int)std::thread::hardware_concurrency();
    const int n_thread = std::max(1, std::min(hw > 0 ? std::max(1, hw / 2) : 1, std::max(1, N / 65536)));
    std::vector<std::thread> workers;
    std::vector<std::vector<double>> t_hmap(n_thread, std::vector<double>(n_hcell, unset));
    const int chunk = (N + n_thread - 1) / n_thread;
    for (int t = 0; t < n_thread; ++t) {
      const int lo = t * chunk;
      const int hi = std::min(N, lo + chunk);
      if (lo >= hi) {
        break;
      }
      workers.emplace_back([&, lo, hi, t]() {
        auto& hm = t_hmap[t];
        for (int i = lo; i < hi; ++i) {
          b0[i] = wrap((int)std::floor(r[t0][i] / cell_size), nt0);
          b1[i] = wrap((int)std::floor(r[t1][i] / cell_size), nt1);
          if (!keep[i]) {
            continue;
          }
          const int id = b0[i] + nt0 * b1[i];
          if (r[axis][i] > hm[id]) {
            hm[id] = r[axis][i];
          }
        }
      });
    }
    for (auto& w : workers) {
      w.join();
    }
    for (int t = 0; t < n_thread; ++t) {
      for (int c = 0; c < n_hcell; ++c) {
        if (t_hmap[t][c] > hmap[c]) {
          hmap[c] = t_hmap[t][c];
        }
      }
    }
  }

  auto in_region = [&](int i) {
    if (region_cubic == nullptr) {
      return true;
    }
    const double x = r[0][i];
    const double y = r[1][i];
    const double z = r[2][i];
    return x >= region_cubic[0] && x <= region_cubic[1] && y >= region_cubic[2] &&
           y <= region_cubic[3] && z >= region_cubic[4] && z <= region_cubic[5];
  };

  {
    const int hw = (int)std::thread::hardware_concurrency();
    const int n_thread = std::max(1, std::min(hw > 0 ? std::max(1, hw / 2) : 1, std::max(1, N / 65536)));
    std::vector<std::thread> workers;
    const int chunk = (N + n_thread - 1) / n_thread;
    for (int t = 0; t < n_thread; ++t) {
      const int lo = t * chunk;
      const int hi = std::min(N, lo + chunk);
      if (lo >= hi) {
        break;
      }
      workers.emplace_back([&, lo, hi]() {
        for (int i = lo; i < hi; ++i) {
          if (!keep[i]) {
            label[i] = 0;
            continue;
          }
          if (!in_region(i)) {
            label[i] = 1;
            continue;
          }
          const double local = hmap[b0[i] + nt0 * b1[i]];
          if (r[axis][i] + offset >= local) {
            label[i] = 0;
          } else {
            label[i] = 1;
          }
        }
      });
    }
    for (auto& w : workers) {
      w.join();
    }
  }
  return true;
}
