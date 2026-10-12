/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#pragma once

/*------------------------------------------------------------------------------
fix shell v2: growth-aware substrate shell (host code only).

v1 (substrate_shell.cuh, classify_substrate_shell) is not touched. `fix shell`
still runs v1 when none of the v2 keywords below is given, so existing etch
loops keep their exact behaviour.

v2 keywords (all opt-in):
  surface radius R   local surface = highest substrate atom whose height-map
                     cell lies within R of the atom's cell (disc max filter).
                     R = 0 is the v1 rule (own 1 A column only).
  cell c             height-map cell size, default 1.0 A (v1 value).
  mode recede|grow   recede (default): classify from scratch on every call
                     (v1 semantics; atoms can be released when the surface
                     recedes). grow: ratchet. Atoms frozen by an earlier grow
                     call stay frozen and only mobile atoms are re-examined.
  connect full|frozen|none
                     substrate test. full: flood fill from the lowest atom
                     over all eligible atoms. frozen (grow default): flood
                     fill from the frozen band through mobile atoms only,
                     O(N_mobile). none: every eligible atom is substrate
                     (use right after `delete disconnected`).
  every K            grow only: classify on every K-th call, re-apply the
                     cached frozen set on the others.
  reset              grow only: drop the ratchet and classify from scratch.

Grow-mode invariants (the caller enforces them):
  * shell-frozen atoms do not move between calls: they are in the fixed group
    of every run while the state is valid. Integrate::finalize() invalidates
    the state after any run in which a grow `fix shell` was not committed;
  * shell-frozen atoms are never deleted (delete protects atom.cpu_fixed);
  * atom.cpu_shell_frozen is appended with 0 for new atoms and compacted on
    delete (AtomMutation), so indices stay aligned.
Under these invariants an incremental grow call freezes exactly the atoms a
full classification would newly freeze (tested in tests/extensions/shell_v2).

The flood fill uses lateral cells at least `cutoff` wide (floor(L/cutoff)
cells). The v1 BFS uses ceil(L/cutoff) cells, which leaves a narrow last cell
at the periodic seam and misses pairs that straddle it; see
tests/extensions/shell_v2 (seam test).
------------------------------------------------------------------------------*/

#include "model/atom.cuh"
#include "model/box.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace shell2
{

constexpr double kUnset = -1.0e300;

enum class Mode { Recede, Grow };
enum class Connect { Full, Frozen, None };

struct Params {
  int axis = 2;
  double cutoff = 0.0;
  double offset = 0.0;
  double surface_radius = 0.0;
  double cell_size = 1.0;
  Mode mode = Mode::Recede;
  Connect connect = Connect::Full;
  int every = 1;
  bool reset = false;
  bool has_region = false;
  double region[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<std::string> types; // sorted and unique; empty = every species

  // Everything that decides which atoms a classification freezes.
  // `every` and `reset` only decide when to classify.
  bool same_classifier(const Params& o) const
  {
    if (axis != o.axis || cutoff != o.cutoff || offset != o.offset ||
        surface_radius != o.surface_radius || cell_size != o.cell_size ||
        connect != o.connect || has_region != o.has_region || types != o.types) {
      return false;
    }
    if (has_region) {
      for (int k = 0; k < 6; ++k) {
        if (region[k] != o.region[k]) {
          return false;
        }
      }
    }
    return true;
  }
};

// Lateral height-map grid; the binning formula is the v1 one.
struct LateralGrid {
  int axis = 2;
  int t0 = 0;
  int t1 = 1;
  int n0 = 1;
  int n1 = 1;
  int pbc0 = 1;
  int pbc1 = 1;
  double cell = 1.0;
  double L0 = 0.0;
  double L1 = 0.0;

  void init(const Box& box, int axis_in, double cell_in)
  {
    axis = axis_in;
    t0 = (axis + 1) % 3;
    t1 = (axis + 2) % 3;
    cell = cell_in;
    const double L[3] = {box.cpu_h[0], box.cpu_h[4], box.cpu_h[8]};
    const int pbc[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
    L0 = L[t0];
    L1 = L[t1];
    pbc0 = pbc[t0];
    pbc1 = pbc[t1];
    n0 = std::max(1, (int)std::ceil(L0 / cell));
    n1 = std::max(1, (int)std::ceil(L1 / cell));
  }
  bool same(const LateralGrid& o) const
  {
    return axis == o.axis && n0 == o.n0 && n1 == o.n1 && cell == o.cell && L0 == o.L0 &&
           L1 == o.L1 && pbc0 == o.pbc0 && pbc1 == o.pbc1;
  }
  int size() const { return n0 * n1; }
  static int wrap(int i, int n)
  {
    int rem = i % n;
    return rem < 0 ? rem + n : rem;
  }
  int cell_of(double a0, double a1) const
  {
    return wrap((int)std::floor(a0 / cell), n0) + n0 * wrap((int)std::floor(a1 / cell), n1);
  }
};

// Reusable buffers; kept inside GrowState so repeated calls do not reallocate.
struct Workspace {
  // flood fill
  std::vector<int> nodes;
  std::vector<int> node_cell;
  std::vector<int> node_c0;
  std::vector<int> node_c1;
  std::vector<int> node_ca;
  std::vector<int> cell_offset;
  std::vector<int> cell_nodes;
  std::vector<int> frontier;
  std::vector<int> next;
  std::vector<char> visited;
  // classification
  std::vector<char> keep;
  std::vector<int> atom_cell;
  std::vector<int> cand;
  std::vector<int> cand_cell;
  std::vector<int> seeds;
  std::vector<char> cand_sub;
  std::vector<double> hmap;
  std::vector<double> hsmooth;
  std::vector<int> disc0;
  std::vector<int> disc1;
};

struct GrowState {
  bool valid = false;
  Params params;
  LateralGrid grid;
  int calls_since_classify = 0;
  std::vector<double> frozen_top; // per lateral cell: highest shell-frozen atom
  Workspace ws;
  long long n_full = 0;
  long long n_incremental = 0;
  long long n_reused = 0;
};

struct Report {
  enum Kind { Full, Incremental, Reused } kind = Full;
  bool ok = true;
  bool fallback_full_bfs = false;
  int n_atoms = 0;
  int n_substrate = 0;  // substrate atoms examined this call
  int n_candidates = 0; // grow: mobile eligible atoms examined
  int n_new_frozen = 0;
  int n_frozen = 0; // shell-frozen after the call
  double h_min = 0.0;
  double h_mean = 0.0;
  double h_max = 0.0;
  double ms = 0.0;
};

inline bool in_region(const Params& p, const double* const r[3], int i)
{
  if (!p.has_region) {
    return true;
  }
  const double x = r[0][i];
  const double y = r[1][i];
  const double z = r[2][i];
  return x >= p.region[0] && x <= p.region[1] && y >= p.region[2] && y <= p.region[3] &&
         z >= p.region[4] && z <= p.region[5];
}

// Distance test with the v1 convention: minimum image in the box, raw
// difference along the substrate axis.
inline bool within_cutoff(
  const Box& box, const double* const r[3], int axis, int i, int j, double cutoff_sq)
{
  double d[3] = {r[0][j] - r[0][i], r[1][j] - r[1][i], r[2][j] - r[2][i]};
  const double raw_axis = d[axis];
  apply_mic(box, d[0], d[1], d[2]);
  d[axis] = raw_axis;
  return d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= cutoff_sq;
}

/*
Multi-source flood fill over a node set.
  nodes[k]   atom index of node k
  is_seed[k] 1: node starts in the component; 0: node may be reached
On return visited[k] = 1 for every node connected to a seed through nodes.
Lateral cells are at least `cutoff` wide, so a 3x3x3 stencil sees every pair
within cutoff, including pairs across the periodic seam.
*/
inline void flood_fill(
  const Box& box,
  const double* const r[3],
  int axis,
  double cutoff,
  const std::vector<int>& nodes,
  const std::vector<char>& is_seed,
  Workspace& ws,
  std::vector<char>& visited)
{
  const int nn = (int)nodes.size();
  visited.assign(nn, 0);
  if (nn == 0) {
    return;
  }
  const int t0 = (axis + 1) % 3;
  const int t1 = (axis + 2) % 3;
  const double L[3] = {box.cpu_h[0], box.cpu_h[4], box.cpu_h[8]};
  const int nt0 = std::max(1, (int)std::floor(L[t0] / cutoff));
  const int nt1 = std::max(1, (int)std::floor(L[t1] / cutoff));
  const double w0 = L[t0] / nt0;
  const double w1 = L[t1] / nt1;
  double amin = r[axis][nodes[0]];
  double amax = amin;
  for (int k = 1; k < nn; ++k) {
    const double a = r[axis][nodes[k]];
    amin = std::min(amin, a);
    amax = std::max(amax, a);
  }
  const int na = std::max(1, (int)std::floor((amax - amin) / cutoff) + 1);
  const int ncell = nt0 * nt1 * na;

  ws.node_c0.resize(nn);
  ws.node_c1.resize(nn);
  ws.node_ca.resize(nn);
  ws.node_cell.resize(nn);
  ws.cell_offset.assign(ncell + 1, 0);
  for (int k = 0; k < nn; ++k) {
    const int i = nodes[k];
    const int c0 = LateralGrid::wrap((int)std::floor(r[t0][i] / w0), nt0);
    const int c1 = LateralGrid::wrap((int)std::floor(r[t1][i] / w1), nt1);
    int ca = (int)std::floor((r[axis][i] - amin) / cutoff);
    ca = std::min(std::max(ca, 0), na - 1);
    ws.node_c0[k] = c0;
    ws.node_c1[k] = c1;
    ws.node_ca[k] = ca;
    const int id = c0 + nt0 * (c1 + nt1 * ca);
    ws.node_cell[k] = id;
    ws.cell_offset[id + 1] += 1;
  }
  for (int c = 0; c < ncell; ++c) {
    ws.cell_offset[c + 1] += ws.cell_offset[c];
  }
  ws.cell_nodes.resize(nn);
  {
    std::vector<int>& cursor = ws.next; // borrowed as scratch
    cursor.assign(ws.cell_offset.begin(), ws.cell_offset.end() - 1);
    for (int k = 0; k < nn; ++k) {
      ws.cell_nodes[cursor[ws.node_cell[k]]++] = k;
    }
  }

  const double cutoff_sq = cutoff * cutoff;
  ws.frontier.clear();
  for (int k = 0; k < nn; ++k) {
    if (is_seed[k]) {
      visited[k] = 1;
      ws.frontier.push_back(k);
    }
  }
  while (!ws.frontier.empty()) {
    ws.next.clear();
    for (const int u : ws.frontier) {
      const int i = nodes[u];
      // unique lateral neighbours (fewer than 3 distinct when nt < 3)
      int l0[3];
      int n0 = 0;
      for (int o = -1; o <= 1; ++o) {
        const int c = LateralGrid::wrap(ws.node_c0[u] + o, nt0);
        bool dup = false;
        for (int q = 0; q < n0; ++q) {
          dup = dup || (l0[q] == c);
        }
        if (!dup) {
          l0[n0++] = c;
        }
      }
      int l1[3];
      int n1 = 0;
      for (int o = -1; o <= 1; ++o) {
        const int c = LateralGrid::wrap(ws.node_c1[u] + o, nt1);
        bool dup = false;
        for (int q = 0; q < n1; ++q) {
          dup = dup || (l1[q] == c);
        }
        if (!dup) {
          l1[n1++] = c;
        }
      }
      for (int oa = -1; oa <= 1; ++oa) {
        const int ca = ws.node_ca[u] + oa;
        if (ca < 0 || ca >= na) {
          continue;
        }
        for (int q1 = 0; q1 < n1; ++q1) {
          for (int q0 = 0; q0 < n0; ++q0) {
            const int id = l0[q0] + nt0 * (l1[q1] + nt1 * ca);
            for (int s = ws.cell_offset[id]; s < ws.cell_offset[id + 1]; ++s) {
              const int v = ws.cell_nodes[s];
              if (visited[v]) {
                continue;
              }
              if (within_cutoff(box, r, axis, i, nodes[v], cutoff_sq)) {
                visited[v] = 1;
                ws.next.push_back(v);
              }
            }
          }
        }
      }
    }
    ws.frontier.swap(ws.next);
  }
}

// keep[i] = 1 if atom i is connected to the lowest eligible atom along axis
// (same definition as v1, with the seam-safe cell grid).
inline void main_component(
  const Atom& atom,
  const Box& box,
  int axis,
  double cutoff,
  const std::vector<char>* eligible,
  Workspace& ws,
  std::vector<char>& keep)
{
  const int N = atom.number_of_atoms;
  keep.assign(N, 0);
  const double* r0 = atom.cpu_position_per_atom.data();
  const double* const r[3] = {r0, r0 + N, r0 + 2 * N};
  ws.nodes.clear();
  int seed_node = -1;
  double amin = 0.0;
  for (int i = 0; i < N; ++i) {
    if (eligible != nullptr && !(*eligible)[i]) {
      continue;
    }
    if (seed_node < 0 || r[axis][i] < amin) {
      amin = r[axis][i];
      seed_node = (int)ws.nodes.size();
    }
    ws.nodes.push_back(i);
  }
  if (seed_node < 0) {
    return;
  }
  std::vector<char> is_seed(ws.nodes.size(), 0);
  is_seed[seed_node] = 1;
  std::vector<char> visited;
  flood_fill(box, r, axis, cutoff, ws.nodes, is_seed, ws, visited);
  for (size_t k = 0; k < ws.nodes.size(); ++k) {
    if (visited[k]) {
      keep[ws.nodes[k]] = 1;
    }
  }
}

// Disc max filter of the height map. radius <= 0 copies (v1 rule).
inline void smooth_heights(
  const LateralGrid& g,
  const std::vector<double>& h,
  double radius,
  Workspace& ws,
  std::vector<double>& out)
{
  if (radius <= 0.0) {
    out = h;
    return;
  }
  ws.disc0.clear();
  ws.disc1.clear();
  const int m = (int)std::floor(radius / g.cell + 1.0e-9);
  const double r2 = radius * radius * (1.0 + 1.0e-12);
  for (int d1 = -m; d1 <= m; ++d1) {
    for (int d0 = -m; d0 <= m; ++d0) {
      const double x = d0 * g.cell;
      const double y = d1 * g.cell;
      if (x * x + y * y <= r2) {
        ws.disc0.push_back(d0);
        ws.disc1.push_back(d1);
      }
    }
  }
  out.assign(g.size(), kUnset);
  const int nd = (int)ws.disc0.size();
  for (int c1 = 0; c1 < g.n1; ++c1) {
    for (int c0 = 0; c0 < g.n0; ++c0) {
      double best = kUnset;
      for (int k = 0; k < nd; ++k) {
        int a0 = c0 + ws.disc0[k];
        int a1 = c1 + ws.disc1[k];
        if (g.pbc0) {
          a0 = LateralGrid::wrap(a0, g.n0);
        } else if (a0 < 0 || a0 >= g.n0) {
          continue;
        }
        if (g.pbc1) {
          a1 = LateralGrid::wrap(a1, g.n1);
        } else if (a1 < 0 || a1 >= g.n1) {
          continue;
        }
        const double v = h[a0 + g.n0 * a1];
        if (v > best) {
          best = v;
        }
      }
      out[c0 + g.n0 * c1] = best;
    }
  }
}

inline void height_stats(const std::vector<double>& h, Report& rep)
{
  int n = 0;
  double sum = 0.0;
  rep.h_min = 0.0;
  rep.h_max = 0.0;
  for (const double v : h) {
    if (v <= kUnset * 0.5) {
      continue;
    }
    if (n == 0) {
      rep.h_min = v;
      rep.h_max = v;
    }
    rep.h_min = std::min(rep.h_min, v);
    rep.h_max = std::max(rep.h_max, v);
    sum += v;
    ++n;
  }
  rep.h_mean = n > 0 ? sum / n : 0.0;
}

/*
Full classification (v1 rule plus optional disc filter).
frozen[i] = 1 for substrate atoms deeper than `offset` below their local
surface, and for substrate atoms outside `region cubic`. Atoms that are not in
the substrate stay 0. Returns false if no substrate was found.
use_v1_bfs = true routes connectivity through the original v1 BFS; it exists
only so the tests can check every other step against v1 bit by bit.
*/
template <typename V1Bfs>
inline bool classify_full_impl(
  const Atom& atom,
  const Box& box,
  const Params& p,
  const std::vector<char>* eligible,
  Workspace& ws,
  std::vector<char>& frozen,
  Report& rep,
  V1Bfs v1_bfs)
{
  const int N = atom.number_of_atoms;
  frozen.assign(N, 0);
  rep.n_atoms = N;
  if (N <= 0) {
    return false;
  }
  const double* r0 = atom.cpu_position_per_atom.data();
  const double* const r[3] = {r0, r0 + N, r0 + 2 * N};

  if (p.connect == Connect::None) {
    ws.keep.assign(N, 0);
    for (int i = 0; i < N; ++i) {
      ws.keep[i] = (eligible == nullptr || (*eligible)[i]) ? 1 : 0;
    }
  } else if (!v1_bfs(ws.keep)) {
    main_component(atom, box, p.axis, p.cutoff, eligible, ws, ws.keep);
  }

  LateralGrid g;
  g.init(box, p.axis, p.cell_size);
  ws.atom_cell.resize(N);
  ws.hmap.assign(g.size(), kUnset);
  int n_sub = 0;
  for (int i = 0; i < N; ++i) {
    const int c = g.cell_of(r[g.t0][i], r[g.t1][i]);
    ws.atom_cell[i] = c;
    if (ws.keep[i]) {
      ++n_sub;
      if (r[p.axis][i] > ws.hmap[c]) {
        ws.hmap[c] = r[p.axis][i];
      }
    }
  }
  rep.n_substrate = n_sub;
  if (n_sub == 0) {
    return false;
  }
  smooth_heights(g, ws.hmap, p.surface_radius, ws, ws.hsmooth);
  height_stats(ws.hsmooth, rep);

  for (int i = 0; i < N; ++i) {
    if (!ws.keep[i]) {
      continue;
    }
    if (!in_region(p, r, i)) {
      frozen[i] = 1;
      continue;
    }
    frozen[i] = (r[p.axis][i] + p.offset >= ws.hsmooth[ws.atom_cell[i]]) ? 0 : 1;
  }
  return true;
}

inline bool classify_full(
  const Atom& atom,
  const Box& box,
  const Params& p,
  const std::vector<char>* eligible,
  Workspace& ws,
  std::vector<char>& frozen,
  Report& rep)
{
  return classify_full_impl(
    atom, box, p, eligible, ws, frozen, rep, [](std::vector<char>&) { return false; });
}

inline void rebuild_frozen_top(
  const Atom& atom, const LateralGrid& g, int axis, std::vector<double>& top)
{
  const int N = atom.number_of_atoms;
  const double* r0 = atom.cpu_position_per_atom.data();
  const double* const r[3] = {r0, r0 + N, r0 + 2 * N};
  top.assign(g.size(), kUnset);
  for (int i = 0; i < N; ++i) {
    if (!atom.cpu_shell_frozen[i]) {
      continue;
    }
    const int c = g.cell_of(r[g.t0][i], r[g.t1][i]);
    if (r[axis][i] > top[c]) {
      top[c] = r[axis][i];
    }
  }
}

/*
Grow-mode update of atom.cpu_shell_frozen (the ratchet).
  first call / invalid state / changed parameters / reset / nothing frozen yet:
      full classification; cpu_shell_frozen = result
  otherwise, on every K-th call: incremental
      candidates = eligible atoms that are not shell-frozen
      substrate  = candidates reached from the frozen band (connect frozen),
                   or the full component (connect full), or all (connect none)
      surface    = max(frozen_top, substrate candidates) per cell, disc filtered
      freeze     = substrate candidates deeper than offset (or outside region)
  other calls: reuse (nothing recomputed)
*/
inline void update_grow(
  Atom& atom,
  const Box& box,
  const Params& p,
  const std::vector<char>* eligible,
  GrowState& st,
  Report& rep)
{
  const auto t_begin = std::chrono::steady_clock::now();
  const int N = atom.number_of_atoms;
  rep = Report();
  rep.n_atoms = N;
  std::vector<char>& sf = atom.cpu_shell_frozen;

  LateralGrid g;
  g.init(box, p.axis, p.cell_size);
  bool need_full = p.reset || !st.valid || !p.same_classifier(st.params) ||
                   (int)sf.size() != N || !g.same(st.grid);
  int n_frozen = 0;
  if (!need_full) {
    for (int i = 0; i < N; ++i) {
      n_frozen += sf[i] ? 1 : 0;
    }
    need_full = (n_frozen == 0);
  }

  auto finish = [&]() {
    rep.ms = std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t_begin)
               .count();
  };

  if (!need_full && st.calls_since_classify + 1 < p.every) {
    ++st.calls_since_classify;
    ++st.n_reused;
    rep.kind = Report::Reused;
    rep.n_frozen = n_frozen;
    finish();
    return;
  }

  const double* r0 = atom.cpu_position_per_atom.data();
  const double* const r[3] = {r0, r0 + N, r0 + 2 * N};
  Workspace& ws = st.ws;

  if (need_full) {
    Params pf = p;
    if (pf.connect == Connect::Frozen) {
      pf.connect = Connect::Full; // nothing to seed from yet
    }
    std::vector<char> frozen;
    rep.ok = classify_full(atom, box, pf, eligible, ws, frozen, rep);
    sf.assign(N, 0);
    if (rep.ok) {
      sf.swap(frozen);
    }
    rep.kind = Report::Full;
    st.valid = true;
    st.params = p;
    st.params.reset = false;
    st.grid = g;
    st.calls_since_classify = 0;
    rebuild_frozen_top(atom, g, p.axis, st.frozen_top);
    rep.n_frozen = 0;
    for (int i = 0; i < N; ++i) {
      rep.n_frozen += sf[i] ? 1 : 0;
    }
    rep.n_new_frozen = rep.n_frozen;
    ++st.n_full;
    finish();
    return;
  }

  // ---- incremental ----
  rep.kind = Report::Incremental;
  ws.cand.clear();
  double zmin = 0.0;
  for (int i = 0; i < N; ++i) {
    if (sf[i] || (eligible != nullptr && !(*eligible)[i])) {
      continue;
    }
    if (ws.cand.empty() || r[p.axis][i] < zmin) {
      zmin = r[p.axis][i];
    }
    ws.cand.push_back(i);
  }
  const int nc = (int)ws.cand.size();
  rep.n_candidates = nc;
  ws.cand_sub.assign(nc, 0);

  if (nc > 0) {
    bool use_full_bfs = (p.connect == Connect::Full);
    if (p.connect == Connect::Frozen) {
      ws.seeds.clear();
      const double band = zmin - p.cutoff;
      for (int j = 0; j < N; ++j) {
        if (sf[j] && r[p.axis][j] >= band) {
          ws.seeds.push_back(j);
        }
      }
      if (ws.seeds.empty()) {
        use_full_bfs = true;
        rep.fallback_full_bfs = true;
      } else {
        // nodes: candidates (0..nc-1) then seeds
        std::vector<int> nodes;
        nodes.reserve(nc + ws.seeds.size());
        nodes.insert(nodes.end(), ws.cand.begin(), ws.cand.end());
        nodes.insert(nodes.end(), ws.seeds.begin(), ws.seeds.end());
        std::vector<char> is_seed(nodes.size(), 0);
        std::fill(is_seed.begin() + nc, is_seed.end(), 1);
        std::vector<char> visited;
        flood_fill(box, r, p.axis, p.cutoff, nodes, is_seed, ws, visited);
        for (int k = 0; k < nc; ++k) {
          ws.cand_sub[k] = visited[k];
        }
      }
    }
    if (use_full_bfs) {
      main_component(atom, box, p.axis, p.cutoff, eligible, ws, ws.keep);
      for (int k = 0; k < nc; ++k) {
        ws.cand_sub[k] = ws.keep[ws.cand[k]];
      }
    }
    if (p.connect == Connect::None) {
      std::fill(ws.cand_sub.begin(), ws.cand_sub.end(), 1);
    }
  }

  ws.hmap = st.frozen_top;
  ws.cand_cell.resize(nc);
  int n_sub = 0;
  for (int k = 0; k < nc; ++k) {
    const int i = ws.cand[k];
    const int c = g.cell_of(r[g.t0][i], r[g.t1][i]);
    ws.cand_cell[k] = c;
    if (ws.cand_sub[k]) {
      ++n_sub;
      if (r[p.axis][i] > ws.hmap[c]) {
        ws.hmap[c] = r[p.axis][i];
      }
    }
  }
  rep.n_substrate = n_sub;
  smooth_heights(g, ws.hmap, p.surface_radius, ws, ws.hsmooth);
  height_stats(ws.hsmooth, rep);

  int n_new = 0;
  for (int k = 0; k < nc; ++k) {
    if (!ws.cand_sub[k]) {
      continue;
    }
    const int i = ws.cand[k];
    bool freeze;
    if (!in_region(p, r, i)) {
      freeze = true;
    } else {
      freeze = !(r[p.axis][i] + p.offset >= ws.hsmooth[ws.cand_cell[k]]);
    }
    if (freeze) {
      sf[i] = 1;
      ++n_new;
      if (r[p.axis][i] > st.frozen_top[ws.cand_cell[k]]) {
        st.frozen_top[ws.cand_cell[k]] = r[p.axis][i];
      }
    }
  }
  rep.n_new_frozen = n_new;
  rep.n_frozen = n_frozen + n_new;
  st.calls_since_classify = 0;
  ++st.n_incremental;
  finish();
}

inline void print_report(const Params& p, const Report& rep)
{
  const char* kind = rep.kind == Report::Full          ? "full"
                     : rep.kind == Report::Incremental ? "incremental"
                                                       : "reused";
  const char* conn = p.connect == Connect::Full     ? "full"
                     : p.connect == Connect::Frozen ? "frozen"
                                                    : "none";
  if (p.mode == Mode::Grow) {
    printf(
      "Fix shell v2 (grow, %s, connect %s%s): frozen %d of %d (+%d), examined %d, "
      "surface R = %g A, every %d, %.3f ms.\n",
      kind,
      conn,
      rep.fallback_full_bfs ? ", full-BFS fallback" : "",
      rep.n_frozen,
      rep.n_atoms,
      rep.n_new_frozen,
      rep.kind == Report::Full ? rep.n_substrate : rep.n_candidates,
      p.surface_radius,
      p.every,
      rep.ms);
  } else {
    printf(
      "Fix shell v2 (recede, connect %s): substrate %d of %d, surface R = %g A, %.3f ms.\n",
      conn,
      rep.n_substrate,
      rep.n_atoms,
      p.surface_radius,
      rep.ms);
  }
  if (rep.kind != Report::Reused) {
    printf(
      "    local surface along axis: min %.4f, mean %.4f, max %.4f A; offset %g A.\n",
      rep.h_min,
      rep.h_mean,
      rep.h_max,
      p.offset);
  }
}

} // namespace shell2
