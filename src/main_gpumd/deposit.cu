/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#include "deposit.cuh"
#include "atom_mutation.cuh"
#include "variable.cuh"
#include "force/force.cuh"
#include "utilities/common.cuh"
#include "model/group.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

extern double get_mass_from_symbol(const std::string& symbol);
extern void read_xyz_in_line_3(
  std::ifstream& input,
  const int N,
  const int has_velocity_in_xyz,
  const bool has_mass,
  const bool has_charge,
  const int num_columns,
  const int* property_offset,
  int& number_of_types,
  std::vector<std::string>& atom_symbols,
  std::vector<std::string>& cpu_atom_symbol,
  std::vector<int>& cpu_type,
  std::vector<double>& cpu_mass,
  std::vector<float>& cpu_charge,
  std::vector<double>& cpu_position_per_atom,
  std::vector<double>& cpu_velocity_per_atom,
  std::vector<Group>& group);

struct DepositConfig
{
  enum Style
  {
    Grid,
    Random,
    Gaussian,
    Point
  };
  enum Entity
  {
    Atom,
    Molecule
  };
  enum Direction
  {
    Axis,
    Vector,
    Target
  };
  enum Surface
  {
    Fixed,
    Global,
    Local
  };
  enum Velocity
  {
    Constant,
    Uniform,
    VectorVelocity,
    BeamVelocity,
    ThermalVelocity
  };
  enum Select
  {
    Sequential,
    RandomSelect
  };

  Style style = Point;
  Entity entity = Atom;
  std::string source;

  bool number_all = false;
  bool has_number = false;
  int number = 1;

  bool has_direction = false;
  Direction direction_kind = Axis;
  double flight[3] = {0.0, 0.0, -1.0};
  double target[3] = {0.0, 0.0, 0.0};

  bool has_normal = false;
  double normal[3] = {0.0, 0.0, 1.0};

  bool has_basis = false;
  double basis[3] = {0.0, 0.0, 0.0};

  bool has_surface = false;
  Surface surface_kind = Fixed;
  double surface_H = 0.0;
  double surface_gap = 0.0;
  double surface_radius = 0.0;
  bool surface_mobile = false; // surface local: measure only non-fixed atoms

  bool has_region = false;
  double region[4] = {0.0, 0.0, 0.0, 0.0};

  bool has_spacing = false;
  double spacing[2] = {0.0, 0.0};

  bool has_origin = false;
  double origin[2] = {0.0, 0.0};

  bool has_sigma = false;
  double sigma = 0.0;

  bool has_spread = false;
  double spread_sigma_deg = 0.0;

  bool has_velocity = false;
  Velocity velocity_kind = Constant;
  double speed = 0.0;
  double speed_min = 0.0;
  double speed_max = 0.0;
  double vector_velocity[3] = {0.0, 0.0, 0.0};
  double drift[3] = {0.0, 0.0, 0.0};
  double temperature = 0.0;
  double entity_mass = 0.0;

  bool mixture = false;
  std::vector<std::string> mixture_symbols;
  std::vector<double> mixture_weights;
  bool box_region = false;
  double xyz_bounds[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  bool skip_flight = false;

  bool has_near = false;
  double near_r = 0.0;

  bool has_select = false;
  Select select = RandomSelect;

  bool has_attempt = false;
  int attempt = 10;

  bool has_seed = false;
  unsigned seed = 0;
};

struct DepositFrame
{
  double u[3] = {1.0, 0.0, 0.0};
  double v[3] = {0.0, 1.0, 0.0};
  double n[3] = {0.0, 0.0, 1.0};
  double d[3] = {0.0, 0.0, -1.0};
  int cartesian_n = -1;
};

struct DepositUV
{
  double u = 0.0;
  double v = 0.0;
};

struct EntityTemplate
{
  int n_atoms = 0;
  std::vector<std::string> symbol;
  std::vector<int> type;
  std::vector<double> mass;
  std::vector<double> rel_pos;
  double radius = 0.0;
};

static void normalize3(double v[3], const char* what)
{
  const double n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  if (!std::isfinite(n2) || n2 <= 1.0e-30) {
    PRINT_INPUT_ERROR(what);
  }
  const double inv = 1.0 / sqrt(n2);
  v[0] *= inv;
  v[1] *= inv;
  v[2] *= inv;
}

static double norm3(const double v[3])
{
  return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static void normalize_or_die(double v[3], const char* what)
{
  const double n = norm3(v);
  if (!std::isfinite(n) || n <= 1.0e-30) {
    PRINT_INPUT_ERROR(what);
  }
  v[0] /= n;
  v[1] /= n;
  v[2] /= n;
}

static double dot3(const double a[3], const double b[3])
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void cross3(const double a[3], const double b[3], double c[3])
{
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

static int cartesian_axis(const double n[3])
{
  for (int a = 0; a < 3; ++a) {
    if (fabs(fabs(n[a]) - 1.0) < 1.0e-12 && fabs(n[(a + 1) % 3]) < 1.0e-12 &&
        fabs(n[(a + 2) % 3]) < 1.0e-12) {
      return a;
    }
  }
  return -1;
}

static void build_deposit_frame(const DepositConfig& cfg, DepositFrame& frame)
{
  if (cfg.has_normal) {
    frame.n[0] = cfg.normal[0];
    frame.n[1] = cfg.normal[1];
    frame.n[2] = cfg.normal[2];
  } else {
    frame.n[0] = -cfg.flight[0];
    frame.n[1] = -cfg.flight[1];
    frame.n[2] = -cfg.flight[2];
  }
  normalize_or_die(frame.n, "deposit normal must be non-zero.");
  frame.cartesian_n = cartesian_axis(frame.n);

  if (cfg.direction_kind == DepositConfig::Target) {
    frame.d[0] = -frame.n[0];
    frame.d[1] = -frame.n[1];
    frame.d[2] = -frame.n[2];
  } else {
    frame.d[0] = cfg.flight[0];
    frame.d[1] = cfg.flight[1];
    frame.d[2] = cfg.flight[2];
    normalize_or_die(frame.d, "deposit direction must be non-zero.");
  }

  double a[3];
  if (cfg.has_basis) {
    a[0] = cfg.basis[0];
    a[1] = cfg.basis[1];
    a[2] = cfg.basis[2];
  } else {
    int k = 0;
    if (fabs(frame.n[1]) < fabs(frame.n[k])) {
      k = 1;
    }
    if (fabs(frame.n[2]) < fabs(frame.n[k])) {
      k = 2;
    }
    a[0] = a[1] = a[2] = 0.0;
    a[k] = 1.0;
  }

  const double adotn = dot3(a, frame.n);
  frame.u[0] = a[0] - adotn * frame.n[0];
  frame.u[1] = a[1] - adotn * frame.n[1];
  frame.u[2] = a[2] - adotn * frame.n[2];
  normalize_or_die(frame.u, "basis is parallel to the surface normal.");
  cross3(frame.n, frame.u, frame.v);
}

static void uvH_to_xyz(const DepositFrame& frame, double u, double v, double H, double r[3])
{
  for (int d = 0; d < 3; ++d) {
    r[d] = u * frame.u[d] + v * frame.v[d] + H * frame.n[d];
  }
}

static double wrap01(double s)
{
  return s - floor(s);
}

static void wrap_lateral(const Box& box, const DepositFrame& frame, double r[3])
{
  if (frame.cartesian_n < 0) {
    return;
  }
  double s[3];
  s[0] = box.cpu_h[9] * r[0] + box.cpu_h[10] * r[1] + box.cpu_h[11] * r[2];
  s[1] = box.cpu_h[12] * r[0] + box.cpu_h[13] * r[1] + box.cpu_h[14] * r[2];
  s[2] = box.cpu_h[15] * r[0] + box.cpu_h[16] * r[1] + box.cpu_h[17] * r[2];
  const int pbc[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
  for (int a = 0; a < 3; ++a) {
    if (a == frame.cartesian_n || pbc[a] != 1) {
      continue;
    }
    s[a] = wrap01(s[a]);
  }
  r[0] = box.cpu_h[0] * s[0] + box.cpu_h[1] * s[1] + box.cpu_h[2] * s[2];
  r[1] = box.cpu_h[3] * s[0] + box.cpu_h[4] * s[1] + box.cpu_h[5] * s[2];
  r[2] = box.cpu_h[6] * s[0] + box.cpu_h[7] * s[1] + box.cpu_h[8] * s[2];
}

static int wrap_cell(int i, int n)
{
  int rem = i % n;
  return rem < 0 ? rem + n : rem;
}

static double min_image_1d(double d, double L, int pbc)
{
  if (pbc != 1 || L <= 0.0) {
    return d;
  }
  d -= L * nearbyint(d / L);
  return d;
}

struct SurfaceIndex
{
  double cell = 1.0;
  int nu = 1;
  int nv = 1;
  double Lu = 1.0;
  double Lv = 1.0;
  double u0 = 0.0;
  double v0 = 0.0;
  int pbc_u = 0;
  int pbc_v = 0;
  double global_H = 0.0;
  double build_radius = -1.0; // radius the index was built with (batch reuse)
  bool has_local_index = false;
  std::vector<double> cell_maxH; // per-cell highest H, for query pruning
  std::vector<int> head;
  std::vector<int> next;
  std::vector<double> u;
  std::vector<double> v;
  std::vector<double> H;

  void reduce_global(const Atom& atom, const DepositFrame& frame)
  {
    has_local_index = false;
    build_radius = -1.0;
    head.clear();
    next.clear();
    cell_maxH.clear();
    u.clear();
    v.clear();
    H.clear();
    const int N = atom.number_of_atoms;
    if (N < 1) {
      PRINT_INPUT_ERROR("surface global/local needs existing atoms.");
    }
    const double* r0 = atom.cpu_position_per_atom.data();
    global_H = r0[0] * frame.n[0] + r0[N] * frame.n[1] + r0[2 * N] * frame.n[2];
    for (int i = 1; i < N; ++i) {
      const double h = r0[i] * frame.n[0] + r0[i + N] * frame.n[1] + r0[i + 2 * N] * frame.n[2];
      if (h > global_H) {
        global_H = h;
      }
    }
  }

  void build(
    const Atom& atom,
    const Box& box,
    const DepositFrame& frame,
    double radius,
    double filter_radius = 0.0,
    const std::vector<char>* mobile_mask = nullptr)
  {
    has_local_index = false;
    const int N = atom.number_of_atoms;
    if (N < 1) {
      PRINT_INPUT_ERROR("surface global/local needs existing atoms.");
    }
    const double* r0 = atom.cpu_position_per_atom.data();
    u.resize(N);
    v.resize(N);
    H.resize(N);
    // when a mobile mask is given, fixed atoms are skipped: the surface (and the
    // global-H fallback) is measured over mobile atoms only, so a fixed wall
    // above the reactive region cannot hijack the launch height
    int n_kept = 0;
    global_H = -1.0e300;
    for (int i = 0; i < N; ++i) {
      const double xyz[3] = {r0[i], r0[N + i], r0[2 * N + i]};
      u[i] = dot3(xyz, frame.u);
      v[i] = dot3(xyz, frame.v);
      H[i] = dot3(xyz, frame.n);
      if (mobile_mask == nullptr || (i < (int)mobile_mask->size() && (*mobile_mask)[i])) {
        ++n_kept;
        if (H[i] > global_H) {
          global_H = H[i];
        }
      }
    }
    if (n_kept == 0) {
      PRINT_INPUT_ERROR("surface local mobile: no mobile atoms to measure the surface.");
    }

    if (frame.cartesian_n == 2) {
      Lu = box.cpu_h[0];
      Lv = box.cpu_h[4];
      pbc_u = box.pbc_x;
      pbc_v = box.pbc_y;
    } else if (frame.cartesian_n == 0) {
      Lu = box.cpu_h[4];
      Lv = box.cpu_h[8];
      pbc_u = box.pbc_y;
      pbc_v = box.pbc_z;
    } else if (frame.cartesian_n == 1) {
      Lu = box.cpu_h[0];
      Lv = box.cpu_h[8];
      pbc_u = box.pbc_x;
      pbc_v = box.pbc_z;
    } else {
      Lu = Lv = 0.0;
      pbc_u = pbc_v = 0;
    }

    cell = radius > 0.0 ? radius : 1.0;
    if (pbc_u && Lu > 0.0) {
      nu = std::max(1, (int)ceil(Lu / cell));
      cell = Lu / nu;
    } else {
      nu = 1;
    }
    if (pbc_v && Lv > 0.0) {
      nv = std::max(1, (int)ceil(Lv / cell));
      } else {
      nv = 1;
    }
    if (pbc_u) {
      for (int i = 0; i < N; ++i) {
        u[i] = wrap01(u[i] / Lu) * Lu;
      }
    }
    if (pbc_v) {
      for (int i = 0; i < N; ++i) {
        v[i] = wrap01(v[i] / Lv) * Lv;
      }
    }
    if (!pbc_u || !pbc_v) {
      double umin = u[0], umax = u[0], vmin = v[0], vmax = v[0];
      for (int i = 1; i < N; ++i) {
        umin = std::min(umin, u[i]);
        umax = std::max(umax, u[i]);
        vmin = std::min(vmin, v[i]);
        vmax = std::max(vmax, v[i]);
      }
      if (!pbc_u) {
        u0 = umin - radius;
        Lu = std::max(umax - umin, cell);
        nu = std::max(1, (int)ceil((umax - umin + 2.0 * radius) / cell));
        for (int i = 0; i < N; ++i) {
          u[i] -= u0;
        }
      }
      if (!pbc_v) {
        v0 = vmin - radius;
        Lv = std::max(vmax - vmin, cell);
        nv = std::max(1, (int)ceil((vmax - vmin + 2.0 * radius) / cell));
        for (int i = 0; i < N; ++i) {
          v[i] -= v0;
        }
      }
    }

    head.assign(static_cast<size_t>(nu) * nv, -1);
    next.assign(N, -1);
    for (int i = 0; i < N; ++i) {
      if (mobile_mask != nullptr && (i >= (int)mobile_mask->size() || !(*mobile_mask)[i])) {
        continue;
      }
      int iu = (int)floor(u[i] / cell);
      int iv = (int)floor(v[i] / cell);
      if (pbc_u) {
        iu = wrap_cell(iu, nu);
    } else {
        iu = std::min(nu - 1, std::max(0, iu));
      }
      if (pbc_v) {
        iv = wrap_cell(iv, nv);
      } else {
        iv = std::min(nv - 1, std::max(0, iv));
      }
      const int c = iu + nu * iv;
      next[i] = head[c];
      head[c] = i;
    }

    // per-cell highest H: lets query skip whole (deep) cell chains once a
    // candidate best is known — columns hold ~2000 atoms but only their top
    // few can answer a max-H query. Chains are additionally relinked in
    // descending-H order so the skip test fires after the first few atoms.
    cell_maxH.assign(head.size(), -1.0e300);
    for (int i = 0; i < N; ++i) {
      if (mobile_mask != nullptr && (i >= (int)mobile_mask->size() || !(*mobile_mask)[i])) {
        continue;
      }
      int iu = (int)floor(u[i] / cell);
      int iv = (int)floor(v[i] / cell);
      if (pbc_u) {
        iu = wrap_cell(iu, nu);
      } else {
        iu = std::min(nu - 1, std::max(0, iu));
      }
      if (pbc_v) {
        iv = wrap_cell(iv, nv);
      } else {
        iv = std::min(nv - 1, std::max(0, iv));
      }
      const int c = iu + nu * iv;
      if (H[i] > cell_maxH[c]) {
        cell_maxH[c] = H[i];
      }
    }
    {
      // relink each cell chain in descending-H order
      std::vector<std::vector<int>> by_cell(head.size());
      for (int i = 0; i < N; ++i) {
        if (mobile_mask != nullptr && (i >= (int)mobile_mask->size() || !(*mobile_mask)[i])) {
          continue;
        }
        int iu = (int)floor(u[i] / cell);
        int iv = (int)floor(v[i] / cell);
        if (pbc_u) {
          iu = wrap_cell(iu, nu);
        } else {
          iu = std::min(nu - 1, std::max(0, iu));
        }
        if (pbc_v) {
          iv = wrap_cell(iv, nv);
        } else {
          iv = std::min(nv - 1, std::max(0, iv));
        }
        by_cell[iu + nu * iv].push_back(i);
      }
      std::fill(head.begin(), head.end(), -1);
      std::fill(next.begin(), next.end(), -1);
      for (size_t c = 0; c < by_cell.size(); ++c) {
        auto& v = by_cell[c];
        std::sort(v.begin(), v.end(), [&](int a, int b) { return H[a] > H[b]; });
        for (int k = (int)v.size() - 1; k >= 0; --k) {
          const int i = v[k];
          next[i] = head[c];
          head[c] = i;
        }
      }
    }
    if (filter_radius > 0.0) {
      // no-op placeholder retained for interface stability
      (void)filter_radius;
    }
    has_local_index = true;
  }

  double query(double qu, double qv, double radius) const
  {
    const double r2 = radius * radius;
    if (pbc_u && Lu > 0.0) {
      qu = wrap01(qu / Lu) * Lu;
    } else {
      qu -= u0;
    }
    if (pbc_v && Lv > 0.0) {
      qv = wrap01(qv / Lv) * Lv;
    } else {
      qv -= v0;
    }
    const int span = (int)ceil(radius / cell);
    const int iu0 = (int)floor(qu / cell);
    const int iv0 = (int)floor(qv / cell);
    // Collect the candidate cells, then walk them in descending cell_maxH so
    // an early high `best` prunes every remaining deep chain (exact for a
    // max-H query: a cell whose highest atom cannot beat `best` is skipped).
    const int n_cell = (2 * span + 1) * (2 * span + 1);
    std::vector<int> cells(n_cell);
    std::vector<double> cellH(n_cell);
    int n_used = 0;
    for (int du = -span; du <= span; ++du) {
      for (int dv = -span; dv <= span; ++dv) {
        int iu = iu0 + du;
        int iv = iv0 + dv;
        if (pbc_u) {
          iu = wrap_cell(iu, nu);
        } else if (iu < 0 || iu >= nu) {
          continue;
        }
        if (pbc_v) {
          iv = wrap_cell(iv, nv);
        } else if (iv < 0 || iv >= nv) {
          continue;
        }
        cells[n_used] = iu + nu * iv;
        cellH[n_used] = cell_maxH.empty() ? 0.0e300 : cell_maxH[cells[n_used]];
        ++n_used;
      }
    }
    // insertion sort by descending cellH (n_used <= 512, usually ~25)
    for (int a = 1; a < n_used; ++a) {
      const int cc = cells[a];
      const double ch = cellH[a];
      int b = a - 1;
      while (b >= 0 && cellH[b] < ch) {
        cells[b + 1] = cells[b];
        cellH[b + 1] = cellH[b];
        --b;
      }
      cells[b + 1] = cc;
      cellH[b + 1] = ch;
    }
    bool found = false;
    double best = 0.0;
    for (int ci = 0; ci < n_used; ++ci) {
      if (found && cellH[ci] <= best) {
        break; // all remaining cells are even lower
      }
      const int c = cells[ci];
      for (int a = head[c]; a >= 0; a = next[a]) {
        if (found && H[a] <= best) {
          // chains are H-descending, so the rest of this chain cannot win
          break;
        }
        const double wu = min_image_1d(u[a] - qu, Lu, pbc_u);
        const double wv = min_image_1d(v[a] - qv, Lv, pbc_v);
        if (wu * wu + wv * wv <= r2) {
          if (!found || H[a] > best) {
            best = H[a];
            found = true;
          }
        }
      }
    }
    return found ? best : global_H;
  }

  // Insert one already-placed atom so later queries in the same batch see it.
  void insert_one(const DepositFrame& frame, const double r[3])
  {
    const double height = dot3(r, frame.n);
    if (height > global_H) {
      global_H = height;
    }
    if (!has_local_index) {
      return;
    }
    u.push_back(dot3(r, frame.u));
    v.push_back(dot3(r, frame.v));
    H.push_back(height);
    next.push_back(-1);
    double qu = u.back();
    double qv = v.back();
    if (pbc_u && Lu > 0.0) {
      qu = wrap01(qu / Lu) * Lu;
      u.back() = qu;
    } else {
      qu -= u0;
    }
    if (pbc_v && Lv > 0.0) {
      qv = wrap01(qv / Lv) * Lv;
      v.back() = qv;
    } else {
      qv -= v0;
    }
    int iu = (int)floor(qu / cell);
    int iv = (int)floor(qv / cell);
    if (pbc_u) {
      iu = wrap_cell(iu, nu);
    } else {
      iu = std::min(nu - 1, std::max(0, iu));
    }
    if (pbc_v) {
      iv = wrap_cell(iv, nv);
    } else {
      iv = std::min(nv - 1, std::max(0, iv));
    }
    const int a = (int)u.size() - 1;
    int* link = &head[iu + nu * iv];
    while (*link >= 0 && H[*link] >= height) {
      link = &next[*link];
    }
    next[a] = *link;
    *link = a;
    if (!cell_maxH.empty() && H[a] > cell_maxH[iu + nu * iv]) {
      cell_maxH[iu + nu * iv] = H[a];
    }
  }
};

struct NearIndex
{
  double cell = 1.0;
  bool brute_force = false;
  int nx = 1, ny = 1, nz = 1;
  int pbc[3] = {0, 0, 0};
  double L[3] = {1.0, 1.0, 1.0};
  const Box* box = nullptr;
  int overflow = -1;
  std::vector<int> head;
  std::vector<int> next;
  std::vector<double> x, y, z;

  int id(int ix, int iy, int iz) const { return ix + nx * (iy + ny * iz); }

  bool cell_of(double px, double py, double pz, int& ix, int& iy, int& iz) const
  {
    ix = (int)floor(px / cell);
    iy = (int)floor(py / cell);
    iz = (int)floor(pz / cell);
    if (pbc[0]) {
      ix = wrap_cell(ix, nx);
    } else if (ix < 0 || ix >= nx) {
      return false;
    }
    if (pbc[1]) {
      iy = wrap_cell(iy, ny);
    } else if (iy < 0 || iy >= ny) {
      return false;
    }
    if (pbc[2]) {
      iz = wrap_cell(iz, nz);
    } else if (iz < 0 || iz >= nz) {
      return false;
    }
    return true;
  }

  void link_atom(int i)
  {
    int ix, iy, iz;
    if (!cell_of(x[i], y[i], z[i], ix, iy, iz)) {
      next[i] = overflow;
      overflow = i;
      return;
    }
    const int c = id(ix, iy, iz);
    next[i] = head[c];
    head[c] = i;
  }

  void rebuild_cells()
  {
    overflow = -1;
    head.assign(static_cast<size_t>(nx) * ny * nz, -1);
    next.assign(x.size(), -1);
    for (int i = 0; i < (int)x.size(); ++i) {
      link_atom(i);
    }
  }

  void build(const Atom& atom, const Box& box_, double cutoff)
  {
    box = &box_;
    cell = cutoff > 0.0 ? cutoff : 1.0;
    L[0] = box_.cpu_h[0];
    L[1] = box_.cpu_h[4];
    L[2] = box_.cpu_h[8];
    pbc[0] = box_.pbc_x;
    pbc[1] = box_.pbc_y;
    pbc[2] = box_.pbc_z;
    brute_force = !std::isfinite(cell) || !(cell > 0.0);
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        if (!std::isfinite(box_.cpu_h[3 * row + col]))
          PRINT_INPUT_ERROR("Exclusion distance requires a finite box matrix.");
        if (row != col && fabs(box_.cpu_h[3 * row + col]) > 1.0e-10)
          brute_force = true;
      }
      if (!(L[row] > 0.0) || !std::isfinite(L[row] / cell) ||
          L[row] / cell > std::numeric_limits<int>::max() - 1 ||
          (pbc[row] && L[row] < cutoff))
        brute_force = true;
    }
    const int N = atom.number_of_atoms;
    const double* r0 = atom.cpu_position_per_atom.data();
    x.resize(N);
    y.resize(N);
    z.resize(N);
    for (int i = 0; i < N; ++i) {
      x[i] = r0[i];
      y[i] = r0[i + N];
      z[i] = r0[i + 2 * N];
    }
    if (brute_force) {
      head.clear();
      next.clear();
      overflow = -1;
      return;
    }
    nx = pbc[0] && L[0] > 0.0 ? std::max(1, (int)floor(L[0] / cell)) : std::max(1, (int)ceil(L[0] / cell));
    ny = pbc[1] && L[1] > 0.0 ? std::max(1, (int)floor(L[1] / cell)) : std::max(1, (int)ceil(L[1] / cell));
    nz = pbc[2] && L[2] > 0.0 ? std::max(1, (int)floor(L[2] / cell)) : std::max(1, (int)ceil(L[2] / cell));
    if (pbc[0] && L[0] > 0.0) {
      cell = std::min(cell, L[0] / nx);
    }
    for (int i = 0; i < N; ++i) {
      if (pbc[0]) {
        x[i] = wrap01(x[i] / L[0]) * L[0];
      }
      if (pbc[1]) {
        y[i] = wrap01(y[i] / L[1]) * L[1];
      }
      if (pbc[2]) {
        z[i] = wrap01(z[i] / L[2]) * L[2];
      }
    }
    rebuild_cells();
  }

  void insert_one(double px, double py, double pz)
  {
    x.push_back(px);
    y.push_back(py);
    z.push_back(pz);
    if (brute_force) return;
    next.push_back(-1);
    link_atom((int)x.size() - 1);
  }

  void insert(const double* xyz, int n_atoms)
  {
    for (int a = 0; a < n_atoms; ++a) {
      insert_one(xyz[a * 3], xyz[a * 3 + 1], xyz[a * 3 + 2]);
    }
  }

  bool overlaps(const double* xyz, int n_atoms, double cutoff) const
  {
    const double cut2 = cutoff * cutoff;
    if (brute_force) {
      for (int a = 0; a < n_atoms; ++a) {
        for (size_t j = 0; j < x.size(); ++j) {
          double rx = xyz[a * 3] - x[j];
          double ry = xyz[a * 3 + 1] - y[j];
          double rz = xyz[a * 3 + 2] - z[j];
          apply_mic(*box, rx, ry, rz);
          if (rx * rx + ry * ry + rz * rz < cut2) return true;
        }
      }
      return false;
    }
    const int span = 1;
    for (int a = 0; a < n_atoms; ++a) {
      const double px = xyz[a * 3];
      const double py = xyz[a * 3 + 1];
      const double pz = xyz[a * 3 + 2];
      int ix0 = (int)floor(px / cell);
      int iy0 = (int)floor(py / cell);
      int iz0 = (int)floor(pz / cell);
      for (int dx = -span; dx <= span; ++dx) {
        for (int dy = -span; dy <= span; ++dy) {
          for (int dz = -span; dz <= span; ++dz) {
            int ix = ix0 + dx;
            int iy = iy0 + dy;
            int iz = iz0 + dz;
            if (pbc[0]) {
              ix = wrap_cell(ix, nx);
            } else if (ix < 0 || ix >= nx) {
              continue;
            }
            if (pbc[1]) {
              iy = wrap_cell(iy, ny);
            } else if (iy < 0 || iy >= ny) {
              continue;
            }
            if (pbc[2]) {
              iz = wrap_cell(iz, nz);
            } else if (iz < 0 || iz >= nz) {
              continue;
            }
            for (int j = head[id(ix, iy, iz)]; j >= 0; j = next[j]) {
              double rx = px - x[j];
              double ry = py - y[j];
              double rz = pz - z[j];
              apply_mic(*box, rx, ry, rz);
              if (rx * rx + ry * ry + rz * rz < cut2) {
                return true;
              }
            }
          }
        }
      }
      for (int j = overflow; j >= 0; j = next[j]) {
        double rx = px - x[j];
        double ry = py - y[j];
        double rz = pz - z[j];
        apply_mic(*box, rx, ry, rz);
        if (rx * rx + ry * ry + rz * rz < cut2) {
          return true;
        }
      }
    }
    return false;
  }
};

struct GridView
{
  int nu = 0;
  int nv = 0;
  double umin = 0.0;
  double vmin = 0.0;
  double du = 1.0;
  double dv = 1.0;
  int size() const { return nu * nv; }
  DepositUV at(int s) const
  {
    return {umin + (s / nv) * du, vmin + (s % nv) * dv};
  }
};

static int count_grid_axis(double lo, double hi, double d)
{
  int n = 0;
  const double eps = 1.0e-10;
  for (;;) {
    if (lo + n * d > hi + eps) {
      break;
    }
    ++n;
    if (n > 100000000) {
      PRINT_INPUT_ERROR("grid spacing produced too many sites.");
    }
  }
  return n;
}

static GridView make_grid_view(const DepositConfig& cfg)
{
  GridView g;
  g.umin = cfg.region[0];
  g.vmin = cfg.region[2];
  g.du = cfg.spacing[0];
  g.dv = cfg.spacing[1];
  g.nu = count_grid_axis(cfg.region[0], cfg.region[1], cfg.spacing[0]);
  g.nv = count_grid_axis(cfg.region[2], cfg.region[3], cfg.spacing[1]);
  if (g.nu > std::numeric_limits<int>::max() / g.nv) {
    PRINT_INPUT_ERROR("grid spacing produced too many sites.");
  }
  if (g.size() < 1) {
    PRINT_INPUT_ERROR("grid region/spacing produced no sites.");
  }
  return g;
}

// Only Cartesian slab geometry has a simple lower bound on the distance
// between wrapped grid sites. Reject tilted cells/bases and a periodic normal.
static bool safe_cartesian_lateral_axes(
  const DepositFrame& frame, const Box& box, int axes[2])
{
  if (frame.cartesian_n < 0) {
    return false;
  }
  const int pbc[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
  if (pbc[frame.cartesian_n]) {
    return false;
  }
  const double eps = 1.0e-10;
  for (int row = 0; row < 3; ++row) {
    if (!(box.cpu_h[row * 3 + row] > 0.0)) {
      return false;
    }
    for (int col = 0; col < 3; ++col) {
      if (row != col && fabs(box.cpu_h[row * 3 + col]) > eps) {
        return false;
      }
    }
  }
  const double* basis[2] = {frame.u, frame.v};
  for (int k = 0; k < 2; ++k) {
    axes[k] = -1;
    for (int axis = 0; axis < 3; ++axis) {
      if (axis != frame.cartesian_n && fabs(fabs(basis[k][axis]) - 1.0) < eps) {
        axes[k] = axis;
      } else if (fabs(basis[k][axis]) > eps) {
        return false;
      }
    }
  }
  return axes[0] >= 0 && axes[1] >= 0 && axes[0] != axes[1];
}

static bool grid_sites_separated(
  const DepositConfig& cfg, const DepositFrame& frame, const Box& box, double min_distance)
{
  if (cfg.style != DepositConfig::Grid) {
    return false;
  }
  int axes[2];
  if (!safe_cartesian_lateral_axes(frame, box, axes)) {
    return false;
  }
  const GridView grid = make_grid_view(cfg);
  const int counts[2] = {grid.nu, grid.nv};
  for (int k = 0; k < 2; ++k) {
    if (counts[k] < 2) {
      continue;
    }
    const double spacing = cfg.spacing[k];
    if (spacing < min_distance) {
      return false;
    }
    const int pbc[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
    if (pbc[axes[k]]) {
      const double length = box.cpu_h[axes[k] * 3 + axes[k]];
      const double span = (counts[k] - 1) * spacing;
      // A span >= L can wrap onto any earlier site; the seam is the
      // additional nearest-neighbor gap when the span is inside one cell.
      if (span >= length || length - span < min_distance) {
        return false;
      }
    }
  }
  return true;
}

struct DepositSampler
{
  std::mt19937 rng;
  std::normal_distribution<double> normal{0.0, 1.0};
  std::uniform_real_distribution<double> phi{0.0, 2.0 * 3.14159265358979};
  std::uniform_real_distribution<double> speed{0.0, 1.0};
  std::uniform_real_distribution<double> uu{0.0, 1.0};
  std::uniform_real_distribution<double> vv{0.0, 1.0};
  bool has_speed = false;
  bool has_uv = false;
};

static DepositUV sample_style_uv(const DepositConfig& cfg, DepositSampler& s)
{
  DepositUV uv;
  if (cfg.style == DepositConfig::Point) {
    uv.u = cfg.origin[0];
    uv.v = cfg.origin[1];
    return uv;
  }
  if (cfg.style == DepositConfig::Gaussian) {
    uv.u = cfg.origin[0] + s.normal(s.rng) * cfg.sigma;
    uv.v = cfg.origin[1] + s.normal(s.rng) * cfg.sigma;
    return uv;
  }
  uv.u = (cfg.region[0] < cfg.region[1]) ? s.uu(s.rng) : cfg.region[0];
  uv.v = (cfg.region[2] < cfg.region[3]) ? s.vv(s.rng) : cfg.region[2];
  return uv;
}

static void sample_flight_direction(
  const DepositConfig& cfg,
  const DepositFrame& frame,
  const double r[3],
  DepositSampler& s,
  double dir[3])
{
  double d[3];
  if (cfg.direction_kind == DepositConfig::Target) {
    d[0] = cfg.target[0] - r[0];
    d[1] = cfg.target[1] - r[1];
    d[2] = cfg.target[2] - r[2];
    normalize_or_die(d, "direction target coincides with a launch position.");
  } else {
    d[0] = frame.d[0];
    d[1] = frame.d[1];
    d[2] = frame.d[2];
  }
  if (!cfg.has_spread) {
    dir[0] = d[0];
    dir[1] = d[1];
    dir[2] = d[2];
    return;
  }
  double e1[3];
  const double d_dot_u = dot3(d, frame.u);
  e1[0] = frame.u[0] - d_dot_u * d[0];
  e1[1] = frame.u[1] - d_dot_u * d[1];
  e1[2] = frame.u[2] - d_dot_u * d[2];
  if (norm3(e1) <= 1.0e-12) {
    e1[0] = frame.v[0];
    e1[1] = frame.v[1];
    e1[2] = frame.v[2];
  }
  normalize_or_die(e1, "cannot build angular spread frame.");
  double e2[3];
  cross3(d, e1, e2);
  const double theta = fabs(s.normal(s.rng) * cfg.spread_sigma_deg * 3.14159265358979 / 180.0);
  const double phi = s.phi(s.rng);
  const double st = sin(theta);
  const double ct = cos(theta);
  for (int k = 0; k < 3; ++k) {
    dir[k] = st * cos(phi) * e1[k] + st * sin(phi) * e2[k] + ct * d[k];
  }
}

static double sample_speed(const DepositConfig& cfg, DepositSampler& s)
{
  if (cfg.velocity_kind == DepositConfig::Constant ||
      cfg.velocity_kind == DepositConfig::BeamVelocity) {
    return cfg.speed;
  }
  if (!s.has_speed) {
    return cfg.speed_min;
  }
  return s.speed(s.rng);
}

static void sample_velocity(
  const DepositConfig& cfg,
  const DepositFrame& frame,
  const double r[3],
  DepositSampler& s,
  double vel[3])
{
  if (cfg.velocity_kind == DepositConfig::VectorVelocity) {
    for (int k = 0; k < 3; ++k) {
      vel[k] = cfg.vector_velocity[k] * TIME_UNIT_CONVERSION;
    }
    return;
  }
  if (cfg.velocity_kind == DepositConfig::ThermalVelocity) {
    // GPUMD masses are amu; kB*T/m converts to (angstrom/fs)^2 here.
    const double sigma = sqrt(8.617333262145e-5 * cfg.temperature /
                              (103.6426965268 * cfg.entity_mass));
    for (int k = 0; k < 3; ++k) {
      vel[k] = (cfg.drift[k] + sigma * s.normal(s.rng)) * TIME_UNIT_CONVERSION;
    }
    return;
  }
  double dir[3];
  sample_flight_direction(cfg, frame, r, s, dir);
  const double speed = sample_speed(cfg, s) *
    (cfg.velocity_kind == DepositConfig::BeamVelocity ? TIME_UNIT_CONVERSION : 1.0);
  vel[0] = speed * dir[0];
  vel[1] = speed * dir[1];
  vel[2] = speed * dir[2];
}

static double resolve_launch_H(
  const DepositConfig& cfg,
  const SurfaceIndex& surface,
  double u,
  double v)
{
  if (cfg.surface_kind == DepositConfig::Fixed) {
    return cfg.surface_H;
  }
  if (cfg.surface_kind == DepositConfig::Global) {
    return surface.global_H + cfg.surface_gap;
  }
  return surface.query(u, v, cfg.surface_radius) + cfg.surface_gap;
}

static void fill_entity_xyz(const EntityTemplate& tmpl, const double r[3], double* xyz)
{
  for (int n = 0; n < tmpl.n_atoms; ++n) {
    for (int d = 0; d < 3; ++d) {
      xyz[n * 3 + d] = r[d] + tmpl.rel_pos[n + tmpl.n_atoms * d];
    }
  }
}

static void place_entity(
  NewAtoms& added,
  int n_total,
  int first,
  const EntityTemplate& tmpl,
  const double r[3],
  const double vel[3])
{
  for (int n = 0; n < tmpl.n_atoms; ++n) {
    const int i = first + n;
    added.symbol[i] = tmpl.symbol[n];
    added.type[i] = tmpl.type[n];
    added.mass[i] = tmpl.mass[n];
    added.charge[i] = 0.0f;
    for (int d = 0; d < 3; ++d) {
      added.position[i + n_total * d] = r[d] + tmpl.rel_pos[n + tmpl.n_atoms * d];
      added.velocity[i + n_total * d] = vel[d];
    }
  }
}

static NewAtoms make_batch(int n_atoms)
{
  NewAtoms added;
  added.n = n_atoms;
  added.type.resize(n_atoms);
  added.mass.resize(n_atoms);
  added.charge.resize(n_atoms, 0.0f);
  added.symbol.resize(n_atoms);
  added.position.resize(n_atoms * 3);
  added.velocity.resize(n_atoms * 3);
  return added;
}

static std::vector<std::string> symbols_from_atom(const Atom& atoms)
{
  std::vector<std::string> by_type(atoms.cpu_type_size.size());
  for (int i = 0; i < atoms.number_of_atoms; ++i) {
    const int t = atoms.cpu_type[i];
    if (t >= 0 && t < (int)by_type.size() && by_type[t].empty()) {
      by_type[t] = atoms.cpu_atom_symbol[i];
    }
  }
  return by_type;
}

static std::vector<std::string> symbols_from_potential()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }
  std::string filename_potential;
  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() >= 2 && tokens[0] == "potential") {
      filename_potential = tokens[1];
      break;
    }
  }
  if (filename_potential.empty()) {
    PRINT_INPUT_ERROR("There is no 'potential' keyword in run.in.");
  }
  std::ifstream input_potential(filename_potential);
  if (!input_potential.is_open()) {
    PRINT_INPUT_ERROR("Cannot open potential file.");
  }
  std::vector<std::string> tokens = get_tokens(input_potential);
  if (tokens.size() < 3) {
    PRINT_INPUT_ERROR("The first line of the potential file should have at least 3 items.");
  }
  const int number_of_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if ((int)tokens.size() != 2 + number_of_types) {
    PRINT_INPUT_ERROR("The first line of the potential file should have the correct number of atom symbols.");
  }
  std::vector<std::string> atom_symbols(number_of_types);
  for (int n = 0; n < number_of_types; ++n) {
    atom_symbols[n] = tokens[2 + n];
  }
  return atom_symbols;
}

static int type_of_symbol(const std::string& symbol, const std::vector<std::string>& allowed)
{
  for (size_t t = 0; t < allowed.size(); ++t) {
    if (symbol == allowed[t]) {
      return (int)t;
    }
  }
  PRINT_INPUT_ERROR("Atom symbol is not allowed by the used potential.");
  return -1;
}

static std::vector<std::string> resolve_allowed_symbols(const Atom& atoms, const std::string& needed)
{
  std::vector<std::string> allowed = symbols_from_atom(atoms);
  for (const auto& s : allowed) {
    if (s == needed) {
      return allowed;
    }
  }
  return symbols_from_potential();
}

static void read_molecule_xyz(
  const std::string& filename,
  const std::vector<std::string>& allowed_symbols,
  std::vector<std::string>& molecule_symbols,
  std::vector<double>& molecule_positions,
  std::vector<double>& molecule_masses)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Failed to open molecule xyz file.");
  }
  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() != 1) {
    PRINT_INPUT_ERROR("The first line for the xyz file should have one value.");
  }
  const int N = get_int_from_token(tokens[0], __FILE__, __LINE__);
  if (N < 1) {
    PRINT_INPUT_ERROR("Number of atoms should >= 1.");
  }
  std::string line;
  std::getline(input, line);
  int has_velocity_in_xyz = 0;
  bool has_mass = false;
  bool has_charge = false;
  int num_columns = 4;
  int property_offset[5] = {0, 1, -1, -1, -1};
  int number_of_types = (int)allowed_symbols.size();
  std::vector<Group> empty_group;
  molecule_symbols.resize(N);
  molecule_positions.resize(N * 3);
  molecule_masses.resize(N);
  std::vector<int> cpu_type(N);
  std::vector<float> cpu_charge(N);
  std::vector<double> cpu_velocity_per_atom(N * 3);
  std::vector<std::string> atom_symbols_copy = allowed_symbols;
  read_xyz_in_line_3(
    input,
    N,
    has_velocity_in_xyz,
    has_mass,
    has_charge,
    num_columns,
    property_offset,
    number_of_types,
    atom_symbols_copy,
    molecule_symbols,
    cpu_type,
    molecule_masses,
    cpu_charge,
    molecule_positions,
    cpu_velocity_per_atom,
    empty_group);
}

static void finish_entity_radius(EntityTemplate& tmpl)
{
  tmpl.radius = 0.0;
  for (int n = 0; n < tmpl.n_atoms; ++n) {
    double r2 = 0.0;
    for (int d = 0; d < 3; ++d) {
      const double x = tmpl.rel_pos[n + tmpl.n_atoms * d];
      r2 += x * x;
    }
    tmpl.radius = std::max(tmpl.radius, sqrt(r2));
  }
}

static void resolve_entity_template(const DepositConfig& cfg, const Atom& atoms, EntityTemplate& tmpl)
{
  if (cfg.entity == DepositConfig::Atom) {
    const std::vector<std::string> allowed = resolve_allowed_symbols(atoms, cfg.source);
    tmpl.n_atoms = 1;
    tmpl.symbol = {cfg.source};
    tmpl.type = {type_of_symbol(cfg.source, allowed)};
    tmpl.mass = {get_mass_from_symbol(cfg.source)};
    tmpl.rel_pos = {0.0, 0.0, 0.0};
    tmpl.radius = 0.0;
    return;
  }

  const std::vector<std::string> allowed = symbols_from_potential();
  std::vector<std::string> symbols;
  std::vector<double> positions;
  std::vector<double> masses;
  read_molecule_xyz(cfg.source, allowed, symbols, positions, masses);
  tmpl.n_atoms = (int)symbols.size();
  tmpl.symbol = symbols;
  tmpl.mass = masses;
  tmpl.type.resize(tmpl.n_atoms);
  for (int n = 0; n < tmpl.n_atoms; ++n) {
    tmpl.type[n] = type_of_symbol(symbols[n], allowed);
  }
  double com[3] = {0.0, 0.0, 0.0};
  double total_mass = 0.0;
  for (int n = 0; n < tmpl.n_atoms; ++n) {
    total_mass += masses[n];
    for (int d = 0; d < 3; ++d) {
      com[d] += masses[n] * positions[n + tmpl.n_atoms * d];
    }
  }
  if (total_mass <= 0.0) {
    PRINT_INPUT_ERROR("molecule mass must be > 0.");
  }
  for (int d = 0; d < 3; ++d) {
    com[d] /= total_mass;
  }
  tmpl.rel_pos.resize(tmpl.n_atoms * 3);
  for (int n = 0; n < tmpl.n_atoms; ++n) {
    for (int d = 0; d < 3; ++d) {
      tmpl.rel_pos[n + tmpl.n_atoms * d] = positions[n + tmpl.n_atoms * d] - com[d];
    }
  }
  finish_entity_radius(tmpl);
  printf("Read molecule from %s (%d atoms, mass %.6f).\n", cfg.source.c_str(), tmpl.n_atoms, total_mass);
}

struct Launch
{
  double r[3];
  double vel[3];
};

// The near test can never reject a single-atom entity launched `gap` above a
// local or global surface when gap >= near_r, the query radius >= near_r, and
// the atoms of one command are at least near_r apart: every existing atom
// within the query radius lies at least `gap` below the launch point (the
// local surface is the highest of them; later commands of a batch see earlier
// ones through surface.insert_one), and every other atom is farther than the
// radius laterally. Then the NearIndex over all atoms is not built.
// Not used with `surface ... mobile`: fixed atoms are then not in the surface
// index and may lie above the measured surface.
static bool near_test_is_redundant(
  const DepositConfig& cfg, const DepositFrame& frame, const Box& box)
{
  if (!cfg.has_near) {
    return true;
  }
  if (cfg.entity != DepositConfig::Atom || cfg.surface_mobile) {
    return false;
  }
  int axes[2];
  if (!safe_cartesian_lateral_axes(frame, box, axes)) {
    return false;
  }
  const bool one_atom = !cfg.number_all && cfg.number == 1;
  const bool spaced_grid = grid_sites_separated(cfg, frame, box, cfg.near_r);
  if (!one_atom && !spaced_grid) {
    return false;
  }
  if (cfg.surface_kind == DepositConfig::Global) {
    return cfg.surface_gap >= cfg.near_r;
  }
  if (cfg.surface_kind == DepositConfig::Local) {
    return cfg.surface_gap >= cfg.near_r && cfg.surface_radius >= cfg.near_r;
  }
  return false;
}

namespace {

double named_real(std::vector<std::string>& t, size_t i, const Box& box, VariableScope* vars)
{
  if (i >= t.size()) {
    PRINT_INPUT_ERROR("Missing numeric deposit argument.");
  }
  double v = 0.0;
  if (vars != nullptr) {
    v = vars->resolve_real(t[i], box, "Invalid deposit numeric argument.");
  } else if (!is_valid_real(t[i].c_str(), &v)) {
    PRINT_INPUT_ERROR("Invalid deposit numeric argument.");
  }
  if (!std::isfinite(v)) {
    PRINT_INPUT_ERROR("Deposit arguments must be finite.");
  }
  t[i] = format_variable_number(v);
  return v;
}

int named_int(std::vector<std::string>& t, size_t i, const Box& box, VariableScope* vars)
{
  const double v = named_real(t, i, box, vars);
  if (v < 0.0 || v > static_cast<double>(std::numeric_limits<int>::max()) ||
      floor(v) != v) {
    PRINT_INPUT_ERROR("Deposit argument must be a nonnegative integer.");
  }
  return static_cast<int>(v);
}

void named_size(const std::vector<std::string>& t, size_t n, const char* syntax)
{
  if (t.size() != n) {
    PRINT_INPUT_ERROR(syntax);
  }
}

void named_positive(double v, const char* msg)
{
  if (!(v > 0.0)) {
    PRINT_INPUT_ERROR(msg);
  }
}

void parse_named_facet(
  std::vector<std::string>& t, DepositConfig& cfg, const Box& box, VariableScope* vars)
{
  if (t.empty()) {
    PRINT_INPUT_ERROR("Empty deposit facet.");
  }
  if (t[0] == "particle") {
    if (t.size() < 3) {
      PRINT_INPUT_ERROR("Usage: particle atom SYMBOL | mixture SYMBOL WEIGHT ... | molecule PATH");
    }
    if (t[1] == "atom" || t[1] == "molecule") {
      named_size(t, 3, "Usage: particle atom SYMBOL | molecule PATH");
      cfg.entity = t[1] == "atom" ? DepositConfig::Atom : DepositConfig::Molecule;
      cfg.source = vars ? vars->expand_text(t[2], box) : t[2];
      t[2] = cfg.source;
      if (cfg.source.empty()) {
        PRINT_INPUT_ERROR("Particle source cannot be empty.");
      }
    } else if (t[1] == "mixture") {
      if (t.size() < 6 || t.size() % 2 != 0) {
        PRINT_INPUT_ERROR("Usage: particle mixture SYMBOL WEIGHT SYMBOL WEIGHT ...");
      }
      cfg.entity = DepositConfig::Atom;
      cfg.mixture = true;
      double sum = 0.0;
      for (size_t i = 2; i < t.size(); i += 2) {
        const std::string symbol = vars ? vars->expand_text(t[i], box) : t[i];
        if (symbol.empty() || std::find(cfg.mixture_symbols.begin(), cfg.mixture_symbols.end(), symbol) !=
                                cfg.mixture_symbols.end()) {
          PRINT_INPUT_ERROR("Mixture symbols must be distinct and nonempty.");
        }
        t[i] = symbol;
        cfg.mixture_symbols.push_back(symbol);
        const double w = named_real(t, i + 1, box, vars);
        if (w < 0.0) PRINT_INPUT_ERROR("Mixture probabilities must be nonnegative.");
        cfg.mixture_weights.push_back(w);
        sum += w;
      }
      if (!std::isfinite(sum) || fabs(sum - 1.0) > 1.0e-8) {
        PRINT_INPUT_ERROR("Mixture probabilities must sum to one.");
      }
    } else {
      PRINT_INPUT_ERROR("Particle style must be atom, mixture, or molecule.");
    }
  } else if (t[0] == "placement") {
    if (t.size() < 2) {
      PRINT_INPUT_ERROR("Missing placement style.");
    }
    if (t[1] == "point") {
      named_size(t, 5, "Usage: placement point at u v");
      if (t[2] != "at") PRINT_INPUT_ERROR("Usage: placement point at u v");
      cfg.style = DepositConfig::Point;
      cfg.origin[0] = named_real(t, 3, box, vars);
      cfg.origin[1] = named_real(t, 4, box, vars);
    } else if (t[1] == "grid") {
      if (t.size() != 5 && t.size() != 7) PRINT_INPUT_ERROR("Usage: placement grid spacing du dv [select random|sequential]");
      if (t[2] != "spacing") PRINT_INPUT_ERROR("Grid requires spacing.");
      cfg.style = DepositConfig::Grid;
      cfg.spacing[0] = named_real(t, 3, box, vars);
      cfg.spacing[1] = named_real(t, 4, box, vars);
      named_positive(cfg.spacing[0], "Grid spacing must be positive.");
      named_positive(cfg.spacing[1], "Grid spacing must be positive.");
      cfg.select = DepositConfig::RandomSelect;
      if (t.size() == 7) {
        if (t[5] != "select" || (t[6] != "random" && t[6] != "sequential"))
          PRINT_INPUT_ERROR("Grid select must be random or sequential.");
        cfg.select = t[6] == "random" ? DepositConfig::RandomSelect : DepositConfig::Sequential;
      }
    } else if (t[1] == "random") {
      if (t.size() != 2 && t.size() != 4) PRINT_INPUT_ERROR("Usage: placement random [attempt N]");
      cfg.style = DepositConfig::Random;
      if (t.size() == 4) {
        if (t[2] != "attempt") PRINT_INPUT_ERROR("Usage: placement random [attempt N]");
        cfg.attempt = named_int(t, 3, box, vars);
        named_positive(cfg.attempt, "Attempt must be positive.");
      }
    } else if (t[1] == "gaussian") {
      if (t.size() != 7 && t.size() != 9)
        PRINT_INPUT_ERROR("Usage: placement gaussian center u v sigma S [attempt N]");
      if (t[2] != "center" || t[5] != "sigma")
        PRINT_INPUT_ERROR("Usage: placement gaussian center u v sigma S [attempt N]");
      cfg.style = DepositConfig::Gaussian;
      cfg.origin[0] = named_real(t, 3, box, vars);
      cfg.origin[1] = named_real(t, 4, box, vars);
      cfg.sigma = named_real(t, 6, box, vars);
      if (cfg.sigma < 0.0) PRINT_INPUT_ERROR("Gaussian sigma must be nonnegative.");
      if (t.size() == 9) {
        if (t[7] != "attempt") PRINT_INPUT_ERROR("Gaussian requires attempt N.");
        cfg.attempt = named_int(t, 8, box, vars);
        named_positive(cfg.attempt, "Attempt must be positive.");
      }
    } else {
      PRINT_INPUT_ERROR("Placement style must be point, grid, random, or gaussian.");
    }
  } else if (t[0] == "region") {
    if (t.size() < 2) PRINT_INPUT_ERROR("Missing region style.");
    if (t[1] == "box") {
      named_size(t, 9, "Usage: region box bounds xmin xmax ymin ymax zmin zmax");
      if (t[2] != "bounds") PRINT_INPUT_ERROR("Box region requires bounds.");
      cfg.box_region = true;
      for (int k = 0; k < 6; ++k) cfg.xyz_bounds[k] = named_real(t, 3 + k, box, vars);
      for (int k = 0; k < 3; ++k) {
        if (cfg.xyz_bounds[2*k] > cfg.xyz_bounds[2*k+1]) PRINT_INPUT_ERROR("Box bounds are reversed.");
      }
    } else if (t[1] == "plane" || t[1] == "surface") {
      if (t.size() < 12 || t[2] != "normal" || t[6] != "bounds")
        PRINT_INPUT_ERROR("Usage: region plane|surface normal nx ny nz bounds umin umax vmin vmax height ...");
      for (int k = 0; k < 3; ++k) cfg.normal[k] = named_real(t, 3 + k, box, vars);
      normalize3(cfg.normal, "Region normal must be nonzero.");
      cfg.has_normal = true;
      cfg.has_region = true;
      for (int k = 0; k < 4; ++k) cfg.region[k] = named_real(t, 7 + k, box, vars);
      if (cfg.region[0] > cfg.region[1] || cfg.region[2] > cfg.region[3])
        PRINT_INPUT_ERROR("Region bounds are reversed.");
      if (t[11] != "height") PRINT_INPUT_ERROR("Region requires height.");
      if (t[1] == "plane") {
        if (t.size() != 13 && t.size() != 17)
          PRINT_INPUT_ERROR("Usage: region plane normal ... bounds ... height H [basis bx by bz]");
        cfg.surface_kind = DepositConfig::Fixed;
        cfg.surface_H = named_real(t, 12, box, vars);
        if (t.size() == 17) {
          if (t[13] != "basis") PRINT_INPUT_ERROR("Usage: basis bx by bz");
          for (int k = 0; k < 3; ++k) cfg.basis[k] = named_real(t, 14 + k, box, vars);
          cfg.has_basis = true;
        }
      } else if (t.size() == 17 && t[12] == "global" && t[13] == "gap" && t[15] == "scope") {
        cfg.surface_kind = DepositConfig::Global;
        cfg.surface_gap = named_real(t, 14, box, vars);
        if (t[16] != "all" && t[16] != "mobile") PRINT_INPUT_ERROR("Surface scope must be all or mobile.");
        cfg.surface_mobile = t[16] == "mobile";
      } else if (t.size() == 19 && t[12] == "local" && t[13] == "radius" &&
                 t[15] == "gap" && t[17] == "scope") {
        cfg.surface_kind = DepositConfig::Local;
        cfg.surface_radius = named_real(t, 14, box, vars);
        cfg.surface_gap = named_real(t, 16, box, vars);
        named_positive(cfg.surface_radius, "Local surface radius must be positive.");
        if (t[18] != "all" && t[18] != "mobile") PRINT_INPUT_ERROR("Surface scope must be all or mobile.");
        cfg.surface_mobile = t[18] == "mobile";
      } else {
        PRINT_INPUT_ERROR("Usage: region surface normal ... bounds ... height global gap D scope all|mobile, or height local radius R gap D scope all|mobile");
      }
      if (cfg.surface_gap < 0.0) PRINT_INPUT_ERROR("Surface gap must be nonnegative.");
    } else {
      PRINT_INPUT_ERROR("Region style must be plane, surface, or box.");
    }
  } else if (t[0] == "velocity") {
    if (t.size() < 2) PRINT_INPUT_ERROR("Missing velocity style.");
    if (t[1] == "vector") {
      named_size(t, 7, "Usage: velocity vector vx vy vz units angstrom/fs");
      for (int k = 0; k < 3; ++k) cfg.vector_velocity[k] = named_real(t, 2 + k, box, vars);
      if (t[5] != "units" || t[6] != "angstrom/fs") PRINT_INPUT_ERROR("Velocity units must be angstrom/fs.");
      cfg.velocity_kind = DepositConfig::VectorVelocity;
    } else if (t[1] == "beam") {
      if (t.size() != 10 && t.size() != 12)
        PRINT_INPUT_ERROR("Usage: velocity beam speed V units angstrom/fs direction dx dy dz [spread degrees]");
      if (t[2] != "speed" || t[4] != "units" || t[5] != "angstrom/fs" || t[6] != "direction")
        PRINT_INPUT_ERROR("Invalid beam syntax or units.");
      cfg.speed = named_real(t, 3, box, vars);
      named_positive(cfg.speed, "Beam speed must be positive.");
      for (int k = 0; k < 3; ++k) cfg.flight[k] = named_real(t, 7 + k, box, vars);
      normalize3(cfg.flight, "Beam direction must be nonzero.");
      cfg.has_direction = true;
      cfg.velocity_kind = DepositConfig::BeamVelocity;
      if (t.size() == 12) {
        if (t[10] != "spread") PRINT_INPUT_ERROR("Beam requires spread degrees.");
        cfg.spread_sigma_deg = named_real(t, 11, box, vars);
        if (cfg.spread_sigma_deg < 0.0 || cfg.spread_sigma_deg >= 90.0)
          PRINT_INPUT_ERROR("Beam spread must be in [0,90) degrees.");
        cfg.has_spread = cfg.spread_sigma_deg > 0.0;
      }
    } else if (t[1] == "thermal") {
      named_size(t, 10, "Usage: velocity thermal temperature T drift vx vy vz units angstrom/fs");
      if (t[2] != "temperature" || t[4] != "drift" || t[8] != "units" || t[9] != "angstrom/fs")
        PRINT_INPUT_ERROR("Invalid thermal syntax or units.");
      cfg.temperature = named_real(t, 3, box, vars);
      if (cfg.temperature < 0.0) PRINT_INPUT_ERROR("Thermal temperature must be nonnegative.");
      for (int k = 0; k < 3; ++k) cfg.drift[k] = named_real(t, 5 + k, box, vars);
      cfg.velocity_kind = DepositConfig::ThermalVelocity;
    } else {
      PRINT_INPUT_ERROR("Velocity style must be vector, beam, or thermal.");
    }
    cfg.has_velocity = true;
  } else if (t[0] == "flight") {
    if (t.size() == 3 && t[1] == "skip" && t[2] == "off") {
      cfg.skip_flight = false;
    } else if (t.size() == 5 && t[1] == "skip" && t[2] == "on" &&
               t[3] == "clearance" && t[4] == "auto") {
      cfg.skip_flight = true;
    } else {
      PRINT_INPUT_ERROR("Usage: flight skip off | flight skip on clearance auto");
    }
  } else if (t[0] == "exclusion") {
    named_size(t, 3, "Usage: exclusion distance R");
    if (t[1] != "distance") PRINT_INPUT_ERROR("Usage: exclusion distance R");
    cfg.near_r = named_real(t, 2, box, vars);
    named_positive(cfg.near_r, "Exclusion distance must be positive.");
    cfg.has_near = true;
  } else {
    PRINT_INPUT_ERROR("Unknown deposit facet.");
  }
}

} // namespace

bool DepositRegistry::skips_flight(const std::string& name) const
{
  const auto it = recipes_.find(name);
  if (it == recipes_.end()) return false;
  const auto& flight = it->second.flight;
  return flight.size() == 5 && flight[1] == "skip" && flight[2] == "on";
}

DepositEvent DepositRegistry::execute(
  const std::vector<std::string>& tokens,
  Box& box,
  Atom& atoms,
  std::vector<Group>& groups,
  GPU_Vector<double>& thermo,
  Force& force,
  VariableScope* variables)
{
  if (tokens.size() < 3 || tokens[0] != "deposit") {
    PRINT_INPUT_ERROR("Usage: deposit NAME FACET ... | deposit NAME emit ... | deposit NAME pulse ...");
  }
  const std::string name = variables ? variables->expand_text(tokens[1], box) : tokens[1];
  if (name.empty()) PRINT_INPUT_ERROR("Deposit recipe name cannot be empty.");
  const std::string action = tokens[2];

  if (action != "emit" && action != "pulse") {
    if (action != "particle" && action != "placement" && action != "region" &&
        action != "velocity" && action != "flight" && action != "exclusion" &&
        action != "seed") {
      PRINT_INPUT_ERROR("Unknown deposit facet.");
    }
    std::vector<std::string> facet(tokens.begin() + 2, tokens.end());
    auto& recipe = recipes_[name];
    if (action == "seed") {
      named_size(facet, 2, "Usage: deposit NAME seed INTEGER");
      const int seed = named_int(facet, 1, box, variables);
      recipe.rng.seed(static_cast<unsigned>(seed));
      recipe.rng_initialized = true;
      printf("Deposit recipe %s: seed set to %d.\n", name.c_str(), seed);
      return {};
    }
    DepositConfig check;
    parse_named_facet(facet, check, box, variables);
    if (action == "particle") recipe.particle = facet;
    else if (action == "placement") recipe.placement = facet;
    else if (action == "region") recipe.region = facet;
    else if (action == "velocity") recipe.velocity = facet;
    else if (action == "flight") recipe.flight = facet;
    else recipe.exclusion = facet;
    printf("Deposit recipe %s: %s configured.\n", name.c_str(), action.c_str());
    return {};
  }

  auto found = recipes_.find(name);
  if (found == recipes_.end()) PRINT_INPUT_ERROR("Unknown deposit recipe.");
  Recipe& recipe = found->second;
  if (recipe.particle.empty() || recipe.placement.empty() || recipe.region.empty() ||
      recipe.velocity.empty()) {
    PRINT_INPUT_ERROR("Deposit recipe requires particle, placement, region, and velocity facets.");
  }
  const bool pulse = action == "pulse";
  if ((!pulse && tokens.size() != 5) || (pulse && tokens.size() != 11) || tokens[3] != "number") {
    PRINT_INPUT_ERROR("Usage: deposit NAME emit number N|all, or pulse number N|all wait entry relax N max_wait M");
  }
  if (pulse && (tokens[5] != "wait" || tokens[6] != "entry" ||
                tokens[7] != "relax" || tokens[9] != "max_wait")) {
    PRINT_INPUT_ERROR("Usage: pulse number N|all wait entry relax N max_wait M");
  }
  std::vector<std::string> count_token = {"number", tokens[4]};
  const bool number_all = tokens[4] == "all";
  const int number = number_all ? 0 : named_int(count_token, 1, box, variables);
  if (!number_all && number < 1) PRINT_INPUT_ERROR("Emit number must be positive.");
  int relax_steps = 0, max_wait = 0;
  if (pulse) {
    std::vector<std::string> v = {"relax", tokens[8], "max_wait", tokens[10]};
    relax_steps = named_int(v, 1, box, variables);
    max_wait = named_int(v, 3, box, variables);
    if (max_wait < 1 || relax_steps > std::numeric_limits<int>::max() - max_wait)
      PRINT_INPUT_ERROR("Pulse requires max_wait > 0 and a representable total step count.");
  }

  DepositConfig cfg;
  auto read = [&](const std::vector<std::string>& facet) {
    if (!facet.empty()) {
      std::vector<std::string> copy = facet;
      parse_named_facet(copy, cfg, box, nullptr);
    }
  };
  read(recipe.particle);
  read(recipe.placement);
  read(recipe.region);
  read(recipe.velocity);
  read(recipe.flight);
  read(recipe.exclusion);
  cfg.number = number;
  cfg.number_all = number_all;
  cfg.has_number = true;
  if (number_all && cfg.style != DepositConfig::Grid)
    PRINT_INPUT_ERROR("Emit number all requires grid placement.");
  if (cfg.box_region && cfg.style != DepositConfig::Random)
    PRINT_INPUT_ERROR("Box region currently requires random placement.");
  if (cfg.box_region && (cfg.xyz_bounds[0] == cfg.xyz_bounds[1] ||
                         cfg.xyz_bounds[2] == cfg.xyz_bounds[3] ||
                         cfg.xyz_bounds[4] == cfg.xyz_bounds[5])) {
    PRINT_INPUT_ERROR("Random box region must have nonzero volume.");
  }
  if (!cfg.box_region && cfg.style == DepositConfig::Point) {
    if (cfg.origin[0] < cfg.region[0] || cfg.origin[0] > cfg.region[1] ||
        cfg.origin[1] < cfg.region[2] || cfg.origin[1] > cfg.region[3])
      PRINT_INPUT_ERROR("Placement center lies outside region bounds.");
  }
  if (!cfg.box_region && cfg.style == DepositConfig::Random &&
      (cfg.region[0] == cfg.region[1] || cfg.region[2] == cfg.region[3])) {
    PRINT_INPUT_ERROR("Random placement requires nonzero region area.");
  }
  if (!cfg.has_direction) {
    cfg.flight[0] = cfg.vector_velocity[0];
    cfg.flight[1] = cfg.vector_velocity[1];
    cfg.flight[2] = cfg.vector_velocity[2];
    if (cfg.velocity_kind == DepositConfig::ThermalVelocity) {
      for (int k = 0; k < 3; ++k) cfg.flight[k] = cfg.drift[k];
    }
    if (norm3(cfg.flight) <= 1.0e-30) {
      cfg.flight[0] = -cfg.normal[0];
      cfg.flight[1] = -cfg.normal[1];
      cfg.flight[2] = -cfg.normal[2];
    }
  }
  DepositFrame frame;
  build_deposit_frame(cfg, frame);

  double clearance = 0.0;
  if (cfg.skip_flight || pulse) {
    if (cfg.box_region) PRINT_INPUT_ERROR("Flight skip and pulse require a plane or surface region.");
    if (frame.cartesian_n < 0) PRINT_INPUT_ERROR("Flight skip and pulse require a Cartesian normal.");
    const int pbc[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
    if (pbc[frame.cartesian_n]) PRINT_INPUT_ERROR("Flight skip and pulse require a nonperiodic injection axis.");
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        if (a != b && fabs(box.cpu_h[3*a+b]) > 1.0e-10)
          PRINT_INPUT_ERROR("Flight skip and pulse require an orthogonal box.");
      }
    }
    clearance = force.deposition_clearance();
    if (!(clearance > 0.0) || !std::isfinite(clearance))
      PRINT_INPUT_ERROR("Potential has no finite-range deposition clearance; flight skip/pulse unsupported.");
  }

  AtomMutation::sync_cpu_from_gpu(atoms);
  const int first_atom = atoms.number_of_atoms;
  double substrate_max = -std::numeric_limits<double>::infinity();
  if (cfg.skip_flight || pulse) {
    if (first_atom == 0) PRINT_INPUT_ERROR("Flight skip and pulse require existing atoms.");
    for (int i = 0; i < first_atom; ++i) {
      double h = 0.0;
      for (int d = 0; d < 3; ++d)
        h += atoms.cpu_position_per_atom[i + d * first_atom] * frame.n[d];
      if (!std::isfinite(h))
        PRINT_INPUT_ERROR("Flight skip/pulse requires finite substrate coordinates.");
      substrate_max = std::max(substrate_max, h);
    }
  }

  std::vector<EntityTemplate> templates;
  if (cfg.mixture) {
    for (const auto& symbol : cfg.mixture_symbols) {
      DepositConfig one = cfg;
      one.mixture = false;
      one.source = symbol;
      EntityTemplate t;
      resolve_entity_template(one, atoms, t);
      templates.push_back(t);
    }
  } else {
    EntityTemplate t;
    resolve_entity_template(cfg, atoms, t);
    templates.push_back(t);
  }
  for (const auto& tmpl : templates) {
    if (tmpl.n_atoms < 1) PRINT_INPUT_ERROR("Deposit particle template is empty.");
    for (double m : tmpl.mass)
      if (!std::isfinite(m) || !(m > 0.0)) PRINT_INPUT_ERROR("Deposit particle mass must be finite and positive.");
    for (double x : tmpl.rel_pos)
      if (!std::isfinite(x)) PRINT_INPUT_ERROR("Deposit particle coordinates must be finite.");
  }
  SurfaceIndex surface;
  if (!cfg.box_region && cfg.surface_kind == DepositConfig::Global) {
    if (cfg.surface_mobile) {
      bool any = false;
      surface.global_H = -std::numeric_limits<double>::infinity();
      for (int i = 0; i < first_atom; ++i) {
        if (i < (int)atoms.cpu_fixed.size() && atoms.cpu_fixed[i]) continue;
        any = true;
        double h = 0.0;
        for (int d = 0; d < 3; ++d) h += atoms.cpu_position_per_atom[i + d * first_atom] * frame.n[d];
        surface.global_H = std::max(surface.global_H, h);
      }
      if (!any) PRINT_INPUT_ERROR("Surface mobile scope has no mobile atoms.");
    } else {
      surface.reduce_global(atoms, frame);
    }
  } else if (!cfg.box_region && cfg.surface_kind == DepositConfig::Local) {
    if (box.pbc_x || box.pbc_y || box.pbc_z) {
      if (frame.cartesian_n < 0)
        PRINT_INPUT_ERROR("Periodic local surfaces require a Cartesian normal.");
      for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
          if (row != col && fabs(box.cpu_h[3 * row + col]) > 1.0e-10)
            PRINT_INPUT_ERROR("Periodic local surfaces require an orthogonal box.");
    }
    std::vector<char> keep;
    if (cfg.surface_mobile) {
      keep.assign(first_atom, 1);
      for (int i = 0; i < first_atom; ++i)
        if (i < (int)atoms.cpu_fixed.size() && atoms.cpu_fixed[i]) keep[i] = 0;
    }
    surface.build(atoms, box, frame, cfg.surface_radius, 0.0, cfg.surface_mobile ? &keep : nullptr);
  }
  NearIndex near;
  const bool near_redundant = cfg.has_near && !cfg.skip_flight &&
                              near_test_is_redundant(cfg, frame, box);
  double max_template_radius = 0.0;
  for (const auto& tmpl : templates)
    max_template_radius = std::max(max_template_radius, tmpl.radius);
  const bool skip_new_new = cfg.has_near && !cfg.skip_flight && cfg.style == DepositConfig::Grid &&
    grid_sites_separated(cfg, frame, box, 2.0 * max_template_radius + cfg.near_r);
  if (cfg.has_near && !near_redundant) near.build(atoms, box, cfg.near_r);

  if (!recipe.rng_initialized) {
    std::random_device rd;
    recipe.rng.seed(rd());
    recipe.rng_initialized = true;
  }
  DepositSampler sampler;
  sampler.rng = recipe.rng;
  if (cfg.style == DepositConfig::Random && !cfg.box_region) {
    sampler.uu = std::uniform_real_distribution<double>(cfg.region[0], cfg.region[1]);
    sampler.vv = std::uniform_real_distribution<double>(cfg.region[2], cfg.region[3]);
  }
  const GridView grid = cfg.style == DepositConfig::Grid ? make_grid_view(cfg) : GridView{};
  const int entities = number_all ? grid.size() : number;
  if (entities > 100000000 || entities < 1) PRINT_INPUT_ERROR("Emit count is invalid or too large.");
  if (cfg.style == DepositConfig::Grid && entities > grid.size())
    PRINT_INPUT_ERROR("Emit count exceeds grid sites.");
  std::vector<int> grid_order;
  if (cfg.style == DepositConfig::Grid) {
    grid_order.resize(grid.size());
    std::iota(grid_order.begin(), grid_order.end(), 0);
    if (cfg.select == DepositConfig::RandomSelect)
      std::shuffle(grid_order.begin(), grid_order.end(), sampler.rng);
  }
  int grid_cursor = 0;
  std::discrete_distribution<int> mixture_pick(cfg.mixture_weights.begin(), cfg.mixture_weights.end());
  std::vector<Launch> launches;
  std::vector<int> template_ids;
  launches.reserve(entities);
  template_ids.reserve(entities);
  double omitted_min = std::numeric_limits<double>::infinity();
  double omitted_max = 0.0;
  const double entry_height = (cfg.skip_flight || pulse) ? substrate_max + clearance : 0.0;
  double running_envelope = substrate_max;

  for (int k = 0; k < entities; ++k) {
    const int t_id = cfg.mixture ? mixture_pick(sampler.rng) : 0;
    const EntityTemplate& tmpl = templates[t_id];
    bool accepted = false;
    const int attempts = cfg.style == DepositConfig::Grid ? (number_all ? 1 : grid.size()) :
                         cfg.style == DepositConfig::Point ? 1 : cfg.attempt;
    for (int attempt = 0; attempt < attempts; ++attempt) {
      if (cfg.style == DepositConfig::Grid && grid_cursor >= grid.size()) break;
      double r[3];
      DepositUV uv;
      if (cfg.box_region) {
        for (int d = 0; d < 3; ++d) {
          std::uniform_real_distribution<double> dist(cfg.xyz_bounds[2*d], cfg.xyz_bounds[2*d+1]);
          r[d] = dist(sampler.rng);
        }
      } else {
        if (cfg.style == DepositConfig::Grid) {
          const int site = grid_order[grid_cursor++];
          uv = grid.at(site);
        } else {
          uv = sample_style_uv(cfg, sampler);
          if (!std::isfinite(uv.u) || !std::isfinite(uv.v))
            PRINT_INPUT_ERROR("Sampled nonfinite deposition position.");
          if (cfg.style == DepositConfig::Gaussian &&
              (uv.u < cfg.region[0] || uv.u > cfg.region[1] ||
               uv.v < cfg.region[2] || uv.v > cfg.region[3])) continue;
        }
        const double H = resolve_launch_H(cfg, surface, uv.u, uv.v);
        if (!std::isfinite(H)) PRINT_INPUT_ERROR("Sampled nonfinite deposition height.");
        uvH_to_xyz(frame, uv.u, uv.v, H, r);
        wrap_lateral(box, frame, r);
      }
      for (double x : r)
        if (!std::isfinite(x)) PRINT_INPUT_ERROR("Sampled nonfinite deposition position.");
      DepositConfig velocity_cfg = cfg;
      velocity_cfg.entity_mass = 0.0;
      for (double m : tmpl.mass) velocity_cfg.entity_mass += m;
      Launch L;
      for (int d = 0; d < 3; ++d) L.r[d] = r[d];
      sample_velocity(velocity_cfg, frame, r, sampler, L.vel);
      for (int d = 0; d < 3; ++d)
        if (!std::isfinite(L.vel[d])) PRINT_INPUT_ERROR("Sampled nonfinite deposition velocity.");
      double omitted = 0.0;
      if (cfg.skip_flight) {
        const double vn = dot3(L.vel, frame.n);
        if (!(vn < -1.0e-14)) PRINT_INPUT_ERROR("Flight skip requires velocity toward the surface.");
        double low = std::numeric_limits<double>::infinity();
        for (int a = 0; a < tmpl.n_atoms; ++a) {
          double h = dot3(L.r, frame.n);
          for (int d = 0; d < 3; ++d) h += tmpl.rel_pos[a + tmpl.n_atoms*d] * frame.n[d];
          low = std::min(low, h);
        }
        const double target = running_envelope + clearance + 1.0e-8;
        if (low > target) {
          omitted = (low - target) / -vn;
          for (int d = 0; d < 3; ++d) L.r[d] += L.vel[d] * omitted;
        }
        wrap_lateral(box, frame, L.r);
      }
      std::vector<double> cand((size_t)tmpl.n_atoms * 3);
      fill_entity_xyz(tmpl, L.r, cand.data());
      for (double x : cand)
        if (!std::isfinite(x)) PRINT_INPUT_ERROR("Sampled nonfinite deposition position.");
      if (cfg.has_near && !near_redundant &&
          near.overlaps(cand.data(), tmpl.n_atoms, cfg.near_r)) continue;
      if (cfg.has_near && !near_redundant && !skip_new_new)
        near.insert(cand.data(), tmpl.n_atoms);
      if (cfg.skip_flight) {
        const double omitted_fs = omitted * TIME_UNIT_CONVERSION;
        omitted_min = std::min(omitted_min, omitted_fs);
        omitted_max = std::max(omitted_max, omitted_fs);
        for (int a = 0; a < tmpl.n_atoms; ++a) {
          const double xyz[3] = {cand[3*a], cand[3*a+1], cand[3*a+2]};
          running_envelope = std::max(running_envelope, dot3(xyz, frame.n));
        }
      }
      launches.push_back(L);
      template_ids.push_back(t_id);
      accepted = true;
      break;
    }
    if (!accepted && !number_all)
      PRINT_INPUT_ERROR("Deposit failed: no valid insertion after placement attempts.");
  }
  if (launches.empty()) PRINT_INPUT_ERROR("Deposit found no valid insertion sites.");
  recipe.rng = sampler.rng;

  int added_count = 0;
  for (int id : template_ids) added_count += templates[id].n_atoms;
  NewAtoms added = make_batch(added_count);
  int offset = 0;
  int max_type = -1;
  for (size_t k = 0; k < launches.size(); ++k) {
    const EntityTemplate& tmpl = templates[template_ids[k]];
    place_entity(added, added_count, offset, tmpl, launches[k].r, launches[k].vel);
    offset += tmpl.n_atoms;
    for (int t : tmpl.type) max_type = std::max(max_type, t);
  }
  if ((int)atoms.cpu_type_size.size() < max_type + 1)
    atoms.cpu_type_size.resize(max_type + 1, 0);
  AtomMutation::append_atoms(atoms, groups, thermo, force, added);
  printf("Deposit recipe %s: emitted %d entities (%d atoms); total atoms %d.\n",
         name.c_str(), (int)launches.size(), added_count, atoms.number_of_atoms);
  if (cfg.skip_flight)
    printf("Flight skip omitted %.9g to %.9g fs of free flight; MD time unchanged.\n",
           omitted_min, omitted_max);

  DepositEvent event;
  event.active = true;
  event.pulse = pulse;
  event.first_atom = first_atom;
  event.count = added_count;
  for (int d = 0; d < 3; ++d) event.normal[d] = frame.n[d];
  event.entry_height = entry_height;
  event.relax_steps = relax_steps;
  event.max_wait = max_wait;
  event.omitted_time_min_fs = cfg.skip_flight ? omitted_min : 0.0;
  event.omitted_time_max_fs = cfg.skip_flight ? omitted_max : 0.0;
  return event;
}
