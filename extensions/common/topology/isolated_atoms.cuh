/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#pragma once

#include "coord_expr.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Count + prefix + packed contents, same layout as the GPU neighbor cell list.
struct IsolatedCells
{
  int N = 0;
  double cell_size = 0.0;
  double zmin = 0.0;
  int nx = 1;
  int ny = 1;
  int nz = 1;
  std::vector<int> cx, cy, cz;
  std::vector<int> cell_offset;
  std::vector<int> cell_contents;

  void build(const Atom& atom, const Box& box, double cell_size_in)
  {
    N = atom.number_of_atoms;
    cell_size = cell_size_in;
    cx.clear();
    cy.clear();
    cz.clear();
    cell_offset.clear();
    cell_contents.clear();
    if (N <= 0 || cell_size <= 0.0) {
      return;
    }

    const double* r0 = atom.cpu_position_per_atom.data();
    const double* r[3] = {r0, r0 + N, r0 + 2 * N};
    zmin = r[2][0];
    double zmax = r[2][0];
    for (int i = 1; i < N; ++i) {
      if (r[2][i] < zmin) {
        zmin = r[2][i];
      }
      if (r[2][i] > zmax) {
        zmax = r[2][i];
      }
    }

    const double Lx = box.cpu_h[0];
    const double Ly = box.cpu_h[4];
    nx = std::max(1, (int)std::ceil(Lx / cell_size));
    ny = std::max(1, (int)std::ceil(Ly / cell_size));
    nz = std::max(1, (int)std::ceil((zmax - zmin + 1.0e-8) / cell_size));
    auto wrap = [](int i, int ncell) {
      int rem = i % ncell;
      return rem < 0 ? rem + ncell : rem;
    };
    auto cell_id = [&](int ix, int iy, int iz) { return ix + nx * (iy + ny * iz); };

    const int ncells = nx * ny * nz;
    cx.resize(N);
    cy.resize(N);
    cz.resize(N);
    std::vector<int> cell_of(N);
    std::vector<int> cell_count(ncells, 0);
    for (int i = 0; i < N; ++i) {
      cx[i] = wrap((int)std::floor(r[0][i] / cell_size), nx);
      cy[i] = wrap((int)std::floor(r[1][i] / cell_size), ny);
      int iz = (int)std::floor((r[2][i] - zmin) / cell_size);
      if (iz < 0) {
        iz = 0;
      }
      if (iz >= nz) {
        iz = nz - 1;
      }
      cz[i] = iz;
      cell_of[i] = cell_id(cx[i], cy[i], iz);
      cell_count[cell_of[i]] += 1;
    }
    cell_offset.assign(ncells + 1, 0);
    for (int c = 0; c < ncells; ++c) {
      cell_offset[c + 1] = cell_offset[c] + cell_count[c];
    }
    cell_contents.resize(N);
    std::fill(cell_count.begin(), cell_count.end(), 0);
    for (int i = 0; i < N; ++i) {
      const int id = cell_of[i];
      cell_contents[cell_offset[id] + cell_count[id]] = i;
      cell_count[id] += 1;
    }
  }

  bool covers(int num_atoms, double cutoff) const
  {
    return N == num_atoms && cell_size + 1.0e-15 >= cutoff;
  }
};

// Mark candidates whose neighbor counts within cutoff satisfy `expr`.
// XY uses PBC; z does not (same convention as delete disconnected).
// only: the matching atom.
// full: the matching atom and every neighbor in the cutoff.
// selected: the matching atom and neighbors whose species is in selected_types
//           (X in that set means all neighbors, i.e. full).
// cells: if non-null and covers N plus cutoff, reuse; otherwise build locally.
// skip: if non-null, those atoms are already gone (not neighbors, not candidates).
inline void mark_isolated(
  const Atom& atom,
  const Box& box,
  double cutoff,
  const CoordExpr& expr,
  const std::vector<char>* candidate,
  IsolatedMode mode,
  const std::unordered_set<std::string>* selected_types,
  std::vector<char>& isolated,
  const IsolatedCells* cells = nullptr,
  const std::vector<char>* skip = nullptr)
{
  const int N = atom.number_of_atoms;
  isolated.assign(N, 0);
  if (N <= 0 || cutoff <= 0.0) {
    return;
  }

  const bool selected_all =
    mode == IsolatedMode::Full ||
    (mode == IsolatedMode::Selected && selected_types != nullptr &&
     (selected_types->count("X") || selected_types->count("x")));

  IsolatedCells local;
  const IsolatedCells* use = cells;
  if (use == nullptr || !use->covers(N, cutoff)) {
    local.build(atom, box, cutoff);
    use = &local;
  }

  // species id for each atom (index into unique_symbols); avoids string hashing per neighbor
  std::vector<std::string> unique_symbols(atom.cpu_type_size.size());
  {
    std::vector<char> seen(atom.cpu_type_size.size(), 0);
    for (int i = 0; i < N; ++i) {
      const int t = atom.cpu_type[i];
      if (t >= 0 && t < (int)unique_symbols.size() && !seen[t]) {
        unique_symbols[t] = atom.cpu_atom_symbol[i];
        seen[t] = 1;
      }
    }
  }
  std::vector<int> sym_id(N);
  for (int i = 0; i < N; ++i) {
    sym_id[i] = atom.cpu_type[i];
  }
  const auto needs_by_sym = coord_expr_needs_by_sym(expr);

  const double* r0 = atom.cpu_position_per_atom.data();
  const double* r[3] = {r0, r0 + N, r0 + 2 * N};
  auto is_skipped = [&](int i) { return skip != nullptr && (*skip)[i]; };
  auto is_candidate = [&](int i) {
    if (is_skipped(i)) {
      return false;
    }
    return candidate == nullptr || (*candidate)[i];
  };

  const double cutoff_sq = cutoff * cutoff;
  const int nx = use->nx;
  const int ny = use->ny;
  const int nz = use->nz;
  auto wrap = [](int i, int ncell) {
    int rem = i % ncell;
    return rem < 0 ? rem + ncell : rem;
  };
  auto cell_id = [&](int ix, int iy, int iz) { return ix + nx * (iy + ny * iz); };

  std::unordered_map<std::string, int> by_sym;
  std::vector<int> by_id(unique_symbols.size(), 0);
  std::vector<int> touched;
  std::vector<int> neigh;
  neigh.reserve(32);

  for (int i = 0; i < N; ++i) {
    if (!is_candidate(i)) {
      continue;
    }
    by_sym.clear();
    for (int id : touched) {
      by_id[id] = 0;
    }
    touched.clear();
    neigh.clear();
    int total = 0;
    for (int ox = -1; ox <= 1; ++ox) {
      for (int oy = -1; oy <= 1; ++oy) {
        for (int oz = -1; oz <= 1; ++oz) {
          const int niz = use->cz[i] + oz;
          if (niz < 0 || niz >= nz) {
            continue;
          }
          const int nix = wrap(use->cx[i] + ox, nx);
          const int niy = wrap(use->cy[i] + oy, ny);
          const int nid = cell_id(nix, niy, niz);
          for (int k = use->cell_offset[nid]; k < use->cell_offset[nid + 1]; ++k) {
            const int j = use->cell_contents[k];
            if (j == i || is_skipped(j)) {
              continue;
            }
            // x,y periodic via minimum image; z is free, use raw difference
            double dx = r[0][j] - r[0][i];
            double dy = r[1][j] - r[1][i];
            const double Lx = box.cpu_h[0];
            const double Ly = box.cpu_h[4];
            dx -= Lx * std::round(dx / Lx);
            dy -= Ly * std::round(dy / Ly);
            const double dz = r[2][j] - r[2][i];
            if (dx * dx + dy * dy + dz * dz <= cutoff_sq) {
              ++total;
              const int sid = sym_id[j];
              if (by_id[sid] == 0) {
                touched.push_back(sid);
              }
              by_id[sid] += 1;
              if (mode != IsolatedMode::Only) {
                neigh.push_back(j);
              }
            }
          }
        }
      }
    }
    if (needs_by_sym) {
      for (int id : touched) {
        by_sym[unique_symbols[id]] = by_id[id];
      }
    }
    if (!eval_coord_expr(expr, total, by_sym)) {
      continue;
    }
    isolated[i] = 1;
    if (mode == IsolatedMode::Only) {
      continue;
    }
    for (int j : neigh) {
      if (is_skipped(j)) {
        continue;
      }
      if (selected_all) {
        isolated[j] = 1;
      } else if (
        mode == IsolatedMode::Selected && selected_types != nullptr &&
        selected_types->count(atom.cpu_atom_symbol[j])) {
        isolated[j] = 1;
      }
    }
  }
}
