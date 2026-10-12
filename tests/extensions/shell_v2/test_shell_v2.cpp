// CPU tests for fix shell v2 (extensions/common/surface/substrate_shell_v2.cuh).
//
//   T1  v2 full classification reproduces v1 labels bit for bit (R = 0, v1 BFS)
//   T2  v2 flood fill equals brute-force connectivity; v1 BFS is checked too
//   T3  seam case: a bond across the periodic x boundary
//   T4  grow mode: incremental result == ratchet of full classifications
//   T5  timing at production size
//   T6  deep atoms left unfrozen by the v1 1 A column rule
//
// Build and run: bash run.sh

#include "../../../extensions/common/surface/substrate_shell.cuh"
#include "../../../extensions/common/surface/substrate_shell_v2.cuh"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>

namespace
{

int g_fail = 0;
#define EXPECT(cond, ...)                                                                          \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      ++g_fail;                                                                                    \
      printf("  FAIL %s:%d: ", __FILE__, __LINE__);                                               \
      printf(__VA_ARGS__);                                                                         \
      printf("\n");                                                                                \
    }                                                                                              \
  } while (0)

const double A_SI = 65.3244 / 12.0; // NEP Si lattice implied by the production box

Box make_box(double lx, double ly, double lz)
{
  Box box;
  for (double& v : box.cpu_h) {
    v = 0.0;
  }
  box.cpu_h[0] = lx;
  box.cpu_h[4] = ly;
  box.cpu_h[8] = lz;
  box.cpu_h[9] = 1.0 / lx;
  box.cpu_h[13] = 1.0 / ly;
  box.cpu_h[17] = 1.0 / lz;
  box.pbc_x = 1;
  box.pbc_y = 1;
  box.pbc_z = 0;
  box.thickness_x = lx;
  box.thickness_y = ly;
  box.thickness_z = lz;
  return box;
}

struct Builder {
  std::vector<double> x, y, z;
  std::vector<std::string> sym;
  void add(double xi, double yi, double zi, const char* s = "Si")
  {
    x.push_back(xi);
    y.push_back(yi);
    z.push_back(zi);
    sym.push_back(s);
  }
  Atom atom() const
  {
    Atom a;
    const int N = (int)x.size();
    a.number_of_atoms = N;
    a.cpu_position_per_atom.resize(3 * N);
    for (int i = 0; i < N; ++i) {
      a.cpu_position_per_atom[i] = x[i];
      a.cpu_position_per_atom[i + N] = y[i];
      a.cpu_position_per_atom[i + 2 * N] = z[i];
    }
    a.cpu_atom_symbol = sym;
    a.cpu_type.assign(N, 0);
    a.cpu_fixed.assign(N, 0);
    a.cpu_shell_frozen.assign(N, 0);
    return a;
  }
};

double wrap_l(double v, double L)
{
  v = std::fmod(v, L);
  return v < 0.0 ? v + L : v;
}

// diamond slab (001), nx x ny x nz conventional cells, thermal jitter, rough top,
// a trench along y, adatoms, a floating cluster, some Ge
Builder rough_slab(int nx, int ny, int nz, unsigned seed, double ge_fraction = 0.2)
{
  static const double basis[8][3] = {
    {0, 0, 0},
    {0, 0.5, 0.5},
    {0.5, 0, 0.5},
    {0.5, 0.5, 0},
    {0.25, 0.25, 0.25},
    {0.25, 0.75, 0.75},
    {0.75, 0.25, 0.75},
    {0.75, 0.75, 0.25}};
  std::mt19937 rng(seed);
  std::normal_distribution<double> jit(0.0, 0.1);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  const double Lx = nx * A_SI;
  const double Ly = ny * A_SI;
  const double ztop = nz * A_SI;
  Builder b;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        for (const auto& s : basis) {
          const double xi = (i + s[0]) * A_SI + jit(rng);
          const double yi = (j + s[1]) * A_SI + jit(rng);
          const double zi = (k + s[2]) * A_SI + jit(rng);
          if (zi > ztop - 3.0 && u01(rng) < 0.35) {
            continue; // rough top
          }
          if (std::fabs(xi - 0.3 * Lx) < 4.0 && zi > ztop - 9.0) {
            continue; // trench along y
          }
          b.add(wrap_l(xi, Lx), wrap_l(yi, Ly), zi, u01(rng) < ge_fraction ? "Ge" : "Si");
        }
      }
    }
  }
  // adatoms bonded on top of random surface atoms
  const int n0 = (int)b.x.size();
  for (int t = 0; t < nx * ny / 2; ++t) {
    int best = -1;
    for (int q = 0; q < 40; ++q) {
      const int c = (int)(u01(rng) * n0);
      if (best < 0 || b.z[c] > b.z[best]) {
        best = c;
      }
    }
    const double th = 0.6 * u01(rng);
    const double ph = 6.283185307 * u01(rng);
    b.add(
      wrap_l(b.x[best] + 2.35 * std::sin(th) * std::cos(ph), Lx),
      wrap_l(b.y[best] + 2.35 * std::sin(th) * std::sin(ph), Ly),
      b.z[best] + 2.35 * std::cos(th));
  }
  // floating cluster
  for (int t = 0; t < 4; ++t) {
    b.add(0.5 * Lx + 2.3 * t, 0.5 * Ly, ztop + 12.0);
  }
  return b;
}

std::vector<char> types_mask(const Atom& a, std::initializer_list<const char*> types)
{
  std::vector<char> m(a.number_of_atoms, 0);
  for (int i = 0; i < a.number_of_atoms; ++i) {
    for (const char* t : types) {
      if (a.cpu_atom_symbol[i] == t) {
        m[i] = 1;
      }
    }
  }
  return m;
}

// O(N^2) reference: component of the lowest eligible atom
std::vector<char> brute_component(
  const Atom& a, const Box& box, int axis, double cutoff, const std::vector<char>* elig)
{
  const int N = a.number_of_atoms;
  const double* r0 = a.cpu_position_per_atom.data();
  const double* const r[3] = {r0, r0 + N, r0 + 2 * N};
  std::vector<int> parent(N);
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](int i) {
    while (parent[i] != i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  const double c2 = cutoff * cutoff;
  for (int i = 0; i < N; ++i) {
    if (elig && !(*elig)[i]) {
      continue;
    }
    for (int j = i + 1; j < N; ++j) {
      if (elig && !(*elig)[j]) {
        continue;
      }
      if (shell2::within_cutoff(box, r, axis, i, j, c2)) {
        parent[find(i)] = find(j);
      }
    }
  }
  int seed = -1;
  for (int i = 0; i < N; ++i) {
    if (elig && !(*elig)[i]) {
      continue;
    }
    if (seed < 0 || r[axis][i] < r[axis][seed]) {
      seed = i;
    }
  }
  std::vector<char> keep(N, 0);
  if (seed < 0) {
    return keep;
  }
  const int root = find(seed);
  for (int i = 0; i < N; ++i) {
    if ((!elig || (*elig)[i]) && find(i) == root) {
      keep[i] = 1;
    }
  }
  return keep;
}

int count(const std::vector<char>& v)
{
  int n = 0;
  for (char c : v) {
    n += c ? 1 : 0;
  }
  return n;
}

// ---------------------------------------------------------------- T1
void test_v1_equivalence()
{
  printf("T1: v2 full classification == v1 (R = 0, v1 BFS)\n");
  const Box box = make_box(12 * A_SI, 12 * A_SI, 1000.0);
  int cases = 0;
  for (unsigned seed = 1; seed <= 6; ++seed) {
    const Builder b = rough_slab(12, 12, 6, seed);
    const Atom a = b.atom();
    const std::vector<char> ge_si = types_mask(a, {"Si", "Ge"});
    const std::vector<char> si_only = types_mask(a, {"Si"});
    const double region[6] = {-10.0, 40.0, -10.0, 70.0, -10.0, 1000.0};
    for (int variant = 0; variant < 4; ++variant) {
      const std::vector<char>* elig = (variant == 1) ? &si_only : (variant == 2 ? &ge_si : nullptr);
      const double* reg = (variant == 3) ? region : nullptr;
      for (double offset : {5.0, 12.0, 20.0}) {
        std::vector<int> v1;
        const bool ok1 = classify_substrate_shell(a, box, 2, 2.9, offset, elig, reg, v1);

        shell2::Params p;
        p.axis = 2;
        p.cutoff = 2.9;
        p.offset = offset;
        p.has_region = reg != nullptr;
        if (reg) {
          std::copy(reg, reg + 6, p.region);
        }
        shell2::Workspace ws;
        shell2::Report rep;
        std::vector<char> v2;
        const bool ok2 = shell2::classify_full_impl(
          a, box, p, elig, ws, v2, rep, [&](std::vector<char>& keep) {
            find_main_component_from_min_axis(a, box, 2.9, 2, elig, keep);
            return true;
          });
        EXPECT(ok1 == ok2, "ok flags differ (seed %u variant %d)", seed, variant);
        int diff = 0;
        for (int i = 0; i < a.number_of_atoms; ++i) {
          diff += (v1[i] != 0) != (v2[i] != 0);
        }
        EXPECT(diff == 0, "%d labels differ (seed %u variant %d offset %g)", diff, seed, variant, offset);
        ++cases;
      }
    }
  }
  printf("  %d cases compared\n", cases);
}

// ---------------------------------------------------------------- T2 / T3
void test_connectivity()
{
  printf("T2: flood fill vs brute force\n");
  const Box box = make_box(6 * A_SI, 6 * A_SI, 1000.0); // 32.66 A: last 2.9 A cell is 0.76 A wide
  int v1_mismatch_cases = 0;
  for (unsigned seed = 11; seed <= 20; ++seed) {
    Builder b = rough_slab(6, 6, 3, seed);
    // chains on top-layer atoms with x < 1 A: S -> P -> R straight up, then Q
    // across the x seam from R (2.0 A, PBC). Q's only neighbour is R.
    // v1 cells (2.9 A, ceil -> 12 cells, last one 0.76 A wide): R in cell 0,
    // Q in cell 10, which are not stencil neighbours.
    const double Lx = box.cpu_h[0];
    const int n_slab = (int)b.x.size();
    double ztop = 0.0;
    for (int i = 0; i < n_slab; ++i) {
      ztop = std::max(ztop, b.z[i]);
    }
    int added = 0;
    for (int i = 0; i < n_slab && added < 4; ++i) {
      if (b.x[i] < 1.0 && b.z[i] > ztop - 6.0 && b.z[i] < ztop - 1.0) {
        const double zr = b.z[i] + 2 * 2.3;
        b.add(b.x[i], b.y[i], b.z[i] + 2.3);
        b.add(b.x[i], b.y[i], zr);
        b.add(b.x[i] - 2.0 + Lx, b.y[i], zr);
        ++added;
      }
    }
    const Atom a = b.atom();
    const std::vector<char> ref = brute_component(a, box, 2, 2.9, nullptr);
    shell2::Workspace ws;
    std::vector<char> v2;
    shell2::main_component(a, box, 2, 2.9, nullptr, ws, v2);
    std::vector<char> v1;
    find_main_component_from_min_axis(a, box, 2.9, 2, nullptr, v1);
    int d2 = 0;
    int d1 = 0;
    for (int i = 0; i < a.number_of_atoms; ++i) {
      d2 += (ref[i] != 0) != (v2[i] != 0);
      d1 += (ref[i] != 0) != (v1[i] != 0);
    }
    EXPECT(d2 == 0, "v2 flood fill differs from brute force in %d atoms (seed %u)", d2, seed);
    v1_mismatch_cases += d1 > 0;
  }
  printf("  v2: identical to brute force in all cases\n");
  printf("  v1 BFS differs from brute force in %d of 10 cases (seam cells)\n", v1_mismatch_cases);

  printf("T3: one bond across the periodic x seam\n");
  {
    const Box sbox = make_box(65.3244, 65.3244, 200.0);
    Builder b;
    b.add(0.5, 10.0, 0.0);  // lowest atom (seed), cell 0
    b.add(63.7, 10.0, 0.1); // PBC distance 2.12 A, v1 cell 21 of 23 -> not a v1 neighbour of cell 0
    const Atom a = b.atom();
    std::vector<char> v1;
    find_main_component_from_min_axis(a, sbox, 2.9, 2, nullptr, v1);
    shell2::Workspace ws;
    std::vector<char> v2;
    shell2::main_component(a, sbox, 2, 2.9, nullptr, ws, v2);
    printf("  v1 keeps the second atom: %s;  v2 keeps it: %s\n", v1[1] ? "yes" : "NO", v2[1] ? "yes" : "NO");
    EXPECT(v2[1] == 1, "v2 must connect across the seam");
  }
}

// ---------------------------------------------------------------- T4
// Growth: deposit bonded atoms on top, some flyers, jitter mobile atoms,
// periodic cleanup of flyers and random mobile deletions (order-preserving,
// like AtomMutation::remove_atoms). Shell-frozen atoms never move or vanish.
struct GrowSim {
  Builder b;
  std::vector<char> sf_ref; // ratchet of full classifications
  std::mt19937 rng;
  Box box;
  explicit GrowSim(unsigned seed) : rng(seed), box(make_box(12 * A_SI, 12 * A_SI, 1000.0))
  {
    b = rough_slab(12, 12, 4, seed, 0.0);
    // remove the floating cluster of the generator: it would never be cleaned in this sim
    b.x.resize(b.x.size() - 4);
    b.y.resize(b.y.size() - 4);
    b.z.resize(b.z.size() - 4);
    b.sym.resize(b.sym.size() - 4);
  }
  void step(const Atom& a, int n_dep, bool cleanup, std::vector<char>& sf_live)
  {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::normal_distribution<double> jit(0.0, 0.04);
    const double Lx = box.cpu_h[0];
    const double Ly = box.cpu_h[4];
    const int N = a.number_of_atoms;
    // jitter mobile atoms only
    for (int i = 0; i < N; ++i) {
      if (sf_live[i]) {
        continue;
      }
      b.x[i] = wrap_l(b.x[i] + jit(rng), Lx);
      b.y[i] = wrap_l(b.y[i] + jit(rng), Ly);
      b.z[i] += jit(rng);
    }
    // deposit
    for (int t = 0; t < n_dep; ++t) {
      int best = -1;
      for (int q = 0; q < 60; ++q) {
        const int c = (int)(u01(rng) * N);
        if (best < 0 || b.z[c] > b.z[best]) {
          best = c;
        }
      }
      const bool flyer = u01(rng) < 0.1;
      const double d = flyer ? 7.0 : 2.35;
      const double th = 0.7 * u01(rng);
      const double ph = 6.283185307 * u01(rng);
      b.add(
        wrap_l(b.x[best] + d * std::sin(th) * std::cos(ph), Lx),
        wrap_l(b.y[best] + d * std::sin(th) * std::sin(ph), Ly),
        b.z[best] + d * std::cos(th));
      sf_live.push_back(0);
      sf_ref.push_back(0);
    }
    if (cleanup) {
      // delete disconnected mobile atoms and ~0.2% random mobile atoms
      const Atom cur = b.atom();
      shell2::Workspace ws;
      std::vector<char> keep;
      shell2::main_component(cur, box, 2, 2.9, nullptr, ws, keep);
      std::vector<char> del(cur.number_of_atoms, 0);
      for (int i = 0; i < cur.number_of_atoms; ++i) {
        if (sf_live[i]) {
          continue;
        }
        del[i] = !keep[i] || u01(rng) < 0.002;
      }
      Builder nb;
      std::vector<char> nsf;
      std::vector<char> nref;
      for (int i = 0; i < cur.number_of_atoms; ++i) {
        if (del[i]) {
          continue;
        }
        nb.add(b.x[i], b.y[i], b.z[i], b.sym[i].c_str());
        nsf.push_back(sf_live[i]);
        nref.push_back(sf_ref[i]);
      }
      b = nb;
      sf_live.swap(nsf);
      sf_ref.swap(nref);
    }
  }
};

void test_grow(shell2::Connect connect, int every, double radius)
{
  const char* cname = connect == shell2::Connect::Frozen ? "frozen"
                      : connect == shell2::Connect::Full ? "full"
                                                         : "none";
  printf("T4: grow incremental == ratchet of full (connect %s, every %d, R = %g)\n", cname, every, radius);
  GrowSim sim(7 + every);
  shell2::Params p;
  p.axis = 2;
  p.cutoff = 2.9;
  p.offset = 11.0;
  p.surface_radius = radius;
  p.mode = shell2::Mode::Grow;
  p.connect = connect;
  p.every = every;
  shell2::GrowState st;
  std::vector<char> sf_live(sim.b.x.size(), 0);
  sim.sf_ref.assign(sim.b.x.size(), 0);
  int n_incr = 0;
  int n_mismatch_steps = 0;
  int calls_since = 0;
  bool ref_started = false;
  for (int k = 0; k < 90; ++k) {
    Atom a = sim.b.atom();
    a.cpu_shell_frozen = sf_live;
    shell2::Report rep;
    shell2::update_grow(a, sim.box, p, nullptr, st, rep);
    sf_live = a.cpu_shell_frozen;
    // reference on the same schedule: classify when the live state classified
    const bool classified = rep.kind != shell2::Report::Reused;
    if (classified) {
      shell2::Params pf = p;
      if (pf.connect == shell2::Connect::Frozen) {
        pf.connect = shell2::Connect::Full;
      }
      shell2::Workspace ws;
      shell2::Report r2;
      std::vector<char> full;
      shell2::classify_full(a, sim.box, pf, nullptr, ws, full, r2);
      for (int i = 0; i < a.number_of_atoms; ++i) {
        sim.sf_ref[i] = (ref_started ? sim.sf_ref[i] : 0) || full[i];
      }
      ref_started = ref_started || count(sim.sf_ref) > 0;
      calls_since = 0;
    } else {
      ++calls_since;
    }
    n_incr += rep.kind == shell2::Report::Incremental;
    int diff = 0;
    for (int i = 0; i < a.number_of_atoms; ++i) {
      diff += (sf_live[i] != 0) != (sim.sf_ref[i] != 0);
    }
    if (diff) {
      ++n_mismatch_steps;
      if (n_mismatch_steps <= 3) {
        printf("  step %d (%s): %d atoms differ\n", k,
               rep.kind == shell2::Report::Full ? "full" : (rep.kind == shell2::Report::Incremental ? "incremental" : "reused"),
               diff);
      }
    }
    sim.step(a, 60, (k % 5) == 4, sf_live);
  }
  Atom a = sim.b.atom();
  printf("  final N = %d, shell-frozen = %d, incremental calls = %d, mismatching steps = %d\n",
         a.number_of_atoms, count(sf_live), n_incr, n_mismatch_steps);
  EXPECT(n_incr > 0, "no incremental call happened");
  EXPECT(n_mismatch_steps == 0, "incremental and reference diverged");
}

// ---------------------------------------------------------------- T6
// With thermal displacement, a 1 A column can hold a deep atom and nothing
// above it; v1 then treats that atom as surface and leaves it unfrozen deep
// in the bulk. Each such atom also pulls its NEP influence halo into the
// force pass. A disc of radius >= 2 A removes them.
void test_column_artifact()
{
  printf("T6: deep atoms left unfrozen by the 1 A column rule\n");
  const Box box = make_box(12 * A_SI, 12 * A_SI, 1000.0);
  static const double basis[8][3] = {
    {0, 0, 0}, {0, .5, .5}, {.5, 0, .5}, {.5, .5, 0},
    {.25, .25, .25}, {.25, .75, .75}, {.75, .25, .75}, {.75, .75, .25}};
  for (double sigma : {0.05, 0.10}) {
    std::mt19937 rng(3);
    std::normal_distribution<double> jit(0.0, sigma);
    Builder b;
    for (int k = 0; k < 40; ++k) {
      for (int j = 0; j < 12; ++j) {
        for (int i = 0; i < 12; ++i) {
          for (const auto& s : basis) {
            b.add(wrap_l((i + s[0]) * A_SI + jit(rng), box.cpu_h[0]),
                  wrap_l((j + s[1]) * A_SI + jit(rng), box.cpu_h[4]),
                  (k + s[2]) * A_SI + jit(rng));
          }
        }
      }
    }
    const Atom a = b.atom();
    double ztop = 0.0;
    for (double v : b.z) {
      ztop = std::max(ztop, v);
    }
    const double offset = 20.0;
    std::vector<int> v1;
    classify_substrate_shell(a, box, 2, 2.9, offset, nullptr, nullptr, v1);
    int deep_v1 = 0;
    for (int i = 0; i < a.number_of_atoms; ++i) {
      deep_v1 += (!v1[i] && b.z[i] < ztop - offset - 5.0);
    }
    shell2::Params p;
    p.axis = 2;
    p.cutoff = 2.9;
    p.offset = offset;
    p.surface_radius = 3.0;
    shell2::Workspace ws;
    shell2::Report rep;
    std::vector<char> f;
    shell2::classify_full(a, box, p, nullptr, ws, f, rep);
    int deep_v2 = 0;
    for (int i = 0; i < a.number_of_atoms; ++i) {
      deep_v2 += (!f[i] && b.z[i] < ztop - offset - 5.0);
    }
    printf("  jitter %.2f A, N = %d: unfrozen atoms > 5 A below the shell: v1 %d, v2 (R = 3) %d\n",
           sigma, a.number_of_atoms, deep_v1, deep_v2);
    EXPECT(deep_v2 == 0, "surface radius 3 should leave no deep unfrozen atom");
  }
}

// ---------------------------------------------------------------- T5
void test_timing()
{
  printf("T5: timing at production size (single thread)\n");
  const Box box = make_box(12 * A_SI, 12 * A_SI, 1000.0);
  for (int nz : {20, 60, 100}) {
    Builder b = rough_slab(12, 12, nz, 99, 0.2);
    Atom a = b.atom();
    const int N = a.number_of_atoms;
    auto ms_since = [](std::chrono::steady_clock::time_point t) {
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    };
    std::vector<int> v1;
    auto t = std::chrono::steady_clock::now();
    const int reps = 5;
    for (int r = 0; r < reps; ++r) {
      classify_substrate_shell(a, box, 2, 2.9, 20.0, nullptr, nullptr, v1);
    }
    const double t_v1 = ms_since(t) / reps;

    shell2::Params p;
    p.axis = 2;
    p.cutoff = 2.9;
    p.offset = 20.0;
    p.surface_radius = 3.0;
    p.mode = shell2::Mode::Grow;
    p.connect = shell2::Connect::Frozen;
    shell2::GrowState st;
    shell2::Report rep;
    t = std::chrono::steady_clock::now();
    shell2::update_grow(a, box, p, nullptr, st, rep); // full
    const double t_full = ms_since(t);
    const int frozen_full = rep.n_frozen;
    // next pulse: 5 atoms appended on top
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    double t_inc = 0.0;
    const int pulses = 20;
    for (int q = 0; q < pulses; ++q) {
      for (int k = 0; k < 5; ++k) {
        int best = 0;
        for (int s = 0; s < 60; ++s) {
          const int c = (int)(u01(rng) * a.number_of_atoms);
          if (a.cpu_position_per_atom[c + 2 * a.number_of_atoms] > a.cpu_position_per_atom[best + 2 * a.number_of_atoms]) {
            best = c;
          }
        }
        const int n = a.number_of_atoms;
        b.add(a.cpu_position_per_atom[best], a.cpu_position_per_atom[best + n], a.cpu_position_per_atom[best + 2 * n] + 2.35);
        std::vector<char> sf = a.cpu_shell_frozen;
        sf.push_back(0);
        a = b.atom();
        a.cpu_shell_frozen = sf;
      }
      t = std::chrono::steady_clock::now();
      shell2::update_grow(a, box, p, nullptr, st, rep);
      t_inc += ms_since(t);
    }
    printf("  N = %6d: v1 %7.2f ms | v2 grow full %7.2f ms | v2 grow incremental %6.2f ms "
           "(frozen %d, examined %d mobile)\n",
           N, t_v1, t_full, t_inc / pulses, frozen_full, rep.n_candidates);
  }
}

} // namespace

int main()
{
  test_v1_equivalence();
  test_connectivity();
  test_grow(shell2::Connect::Frozen, 1, 6.0);
  test_grow(shell2::Connect::Frozen, 3, 6.0);
  test_grow(shell2::Connect::Full, 1, 0.0);
  test_grow(shell2::Connect::Frozen, 1, 0.0);
  test_column_artifact();
  test_timing();
  printf(g_fail == 0 ? "ALL TESTS PASSED\n" : "%d CHECKS FAILED\n", g_fail);
  return g_fail == 0 ? 0 : 1;
}
