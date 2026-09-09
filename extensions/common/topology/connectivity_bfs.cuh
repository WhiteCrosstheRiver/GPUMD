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

// keep[i] != 0 if atom i is connected to the lowest-coord atom along `axis`.
// Edges: distance <= cutoff. The two transverse directions use PBC; `axis` does not.
// If eligible is non-null, only those atoms are in the graph.
inline void find_main_component_from_min_axis(
  const Atom& atom,
  const Box& box,
  double cutoff,
  int axis,
  const std::vector<char>* eligible,
  std::vector<char>& keep)
{
  const int N = atom.number_of_atoms;
  keep.assign(N, 0);
  if (N <= 0) {
    return;
  }

  const double* r0 = atom.cpu_position_per_atom.data();
  const double* r[3] = {r0, r0 + N, r0 + 2 * N};
  const int t0 = (axis + 1) % 3;
  const int t1 = (axis + 2) % 3;
  auto is_ok = [&](int i) { return eligible == nullptr || (*eligible)[i]; };

  int seed = -1;
  double amin = 0.0;
  double amax = 0.0;
  for (int i = 0; i < N; ++i) {
    if (!is_ok(i)) {
      continue;
    }
    const double a = r[axis][i];
    if (seed < 0) {
      seed = i;
      amin = a;
      amax = a;
      continue;
    }
    if (a < amin) {
      amin = a;
      seed = i;
    }
    if (a > amax) {
      amax = a;
    }
  }
  if (seed < 0) {
    return;
  }

  const double cutoff_sq = cutoff * cutoff;
  const double cell_size = cutoff;
  const double L[3] = {box.cpu_h[0], box.cpu_h[4], box.cpu_h[8]};
  const int nt0 = std::max(1, (int)std::ceil(L[t0] / cell_size));
  const int nt1 = std::max(1, (int)std::ceil(L[t1] / cell_size));
  const int na = std::max(1, (int)std::ceil((amax - amin + 1.0e-8) / cell_size));
  auto wrap = [](int i, int n) {
    int rem = i % n;
    return rem < 0 ? rem + n : rem;
  };
  auto cell_id = [&](int i0, int i1, int ia) { return i0 + nt0 * (i1 + nt1 * ia); };

  const int ncells = nt0 * nt1 * na;
  std::vector<int> c0(N), c1(N), ca(N), cell_of(N);
  std::vector<int> cell_count(ncells, 0);
  for (int i = 0; i < N; ++i) {
    c0[i] = wrap((int)std::floor(r[t0][i] / cell_size), nt0);
    c1[i] = wrap((int)std::floor(r[t1][i] / cell_size), nt1);
    int ia = (int)std::floor((r[axis][i] - amin) / cell_size);
    if (ia < 0) {
      ia = 0;
    }
    if (ia >= na) {
      ia = na - 1;
    }
    ca[i] = ia;
    cell_of[i] = cell_id(c0[i], c1[i], ia);
    cell_count[cell_of[i]] += 1;
  }
  std::vector<int> cell_offset(ncells + 1, 0);
  for (int c = 0; c < ncells; ++c) {
    cell_offset[c + 1] = cell_offset[c] + cell_count[c];
  }
  std::vector<int> cell_contents(N);
  std::fill(cell_count.begin(), cell_count.end(), 0);
  for (int i = 0; i < N; ++i) {
    const int id = cell_of[i];
    cell_contents[cell_offset[id] + cell_count[id]] = i;
    cell_count[id] += 1;
  }

  keep[seed] = 1;
  std::queue<int> q;
  q.push(seed);
  while (!q.empty()) {
    const int i = q.front();
    q.pop();
    for (int o0 = -1; o0 <= 1; ++o0) {
      for (int o1 = -1; o1 <= 1; ++o1) {
        for (int oa = -1; oa <= 1; ++oa) {
          const int nia = ca[i] + oa;
          if (nia < 0 || nia >= na) {
            continue;
          }
          const int ni0 = wrap(c0[i] + o0, nt0);
          const int ni1 = wrap(c1[i] + o1, nt1);
          const int nid = cell_id(ni0, ni1, nia);
          for (int k = cell_offset[nid]; k < cell_offset[nid + 1]; ++k) {
            const int j = cell_contents[k];
            if (keep[j] || j == i || !is_ok(j)) {
              continue;
            }
            double dx = r[0][j] - r[0][i];
            double dy = r[1][j] - r[1][i];
            double dz = r[2][j] - r[2][i];
            apply_mic(box, dx, dy, dz);
            if (axis == 0) {
              dx = r[0][j] - r[0][i];
            } else if (axis == 1) {
              dy = r[1][j] - r[1][i];
            } else {
              dz = r[2][j] - r[2][i];
            }
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

inline void find_main_component_from_min_z(
  const Atom& atom, const Box& box, double cutoff, std::vector<char>& keep)
{
  find_main_component_from_min_axis(atom, box, cutoff, 2, nullptr, keep);
}
