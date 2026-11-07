// Cell-list checks for the CPU topology helpers used by `delete disconnected`,
// `delete isolated` and `fix shell` (extensions/common/topology).
//   N1  BFS component of the lowest atom == brute force, several box widths
//   N2  delete-isolated coordination counts == brute force, several box widths
// Box widths that are not multiples of the 2.9 A cutoff exercise the periodic
// seam; widths below 3 cutoffs exercise duplicated stencil cells.
// Build: bash run.sh (compiled against this tree), or point ROOT elsewhere.

#include "extensions/common/topology/connectivity_bfs.cuh"
#include "extensions/common/topology/isolated_atoms.cuh"

#include <cstdio>
#include <numeric>
#include <random>

namespace
{
int g_fail = 0;

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
  return box;
}

Atom random_atoms(double lx, double ly, double h, double density, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const int N = std::max(4, (int)(density * lx * ly * h));
  Atom a;
  a.number_of_atoms = N;
  a.cpu_position_per_atom.resize(3 * N);
  for (int i = 0; i < N; ++i) {
    a.cpu_position_per_atom[i] = u(rng) * lx;
    a.cpu_position_per_atom[i + N] = u(rng) * ly;
    a.cpu_position_per_atom[i + 2 * N] = u(rng) * h;
  }
  a.cpu_atom_symbol.assign(N, "Si");
  a.cpu_type.assign(N, 0);
  a.cpu_type_size.assign(1, N);
  a.cpu_fixed.assign(N, 0);
  a.cpu_shell_frozen.assign(N, 0);
  return a;
}

double d2(const Atom& a, const Box& box, int i, int j)
{
  const int N = a.number_of_atoms;
  const double* r = a.cpu_position_per_atom.data();
  double dx = r[j] - r[i];
  double dy = r[j + N] - r[i + N];
  const double dz = r[j + 2 * N] - r[i + 2 * N];
  dx -= box.cpu_h[0] * std::round(dx / box.cpu_h[0]);
  dy -= box.cpu_h[4] * std::round(dy / box.cpu_h[4]);
  return dx * dx + dy * dy + dz * dz;
}
} // namespace

int main()
{
  const double cutoff = 2.9;
  const double widths[] = {5.0, 7.0, 10.0, 32.6622, 65.3244};
  printf("N1: BFS component vs brute force\n");
  for (double L : widths) {
    int bad = 0;
    for (unsigned seed = 1; seed <= 20; ++seed) {
      const Atom a = random_atoms(L, L, 12.0, 0.035, seed);
      const int N = a.number_of_atoms;
      std::vector<int> parent(N);
      std::iota(parent.begin(), parent.end(), 0);
      auto find = [&](int i) {
        while (parent[i] != i) {
          parent[i] = parent[parent[i]];
          i = parent[i];
        }
        return i;
      };
      const Box box = make_box(L, L, 100.0);
      for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
          if (d2(a, box, i, j) <= cutoff * cutoff) {
            parent[find(i)] = find(j);
          }
        }
      }
      int seed_atom = 0;
      for (int i = 1; i < N; ++i) {
        if (a.cpu_position_per_atom[i + 2 * N] < a.cpu_position_per_atom[seed_atom + 2 * N]) {
          seed_atom = i;
        }
      }
      std::vector<char> keep;
      find_main_component_from_min_z(a, box, cutoff, keep);
      for (int i = 0; i < N; ++i) {
        bad += (find(i) == find(seed_atom)) != (keep[i] != 0);
      }
    }
    printf("  L = %8.4f A: %d mismatching atoms over 20 structures\n", L, bad);
    g_fail += bad != 0;
  }
  printf("N2: delete-isolated coordination vs brute force\n");
  for (double L : widths) {
    int bad = 0;
    for (unsigned seed = 1; seed <= 20; ++seed) {
      const Atom a = random_atoms(L, L, 12.0, 0.035, 100 + seed);
      const int N = a.number_of_atoms;
      const Box box = make_box(L, L, 100.0);
      std::vector<int> coord(N, 0);
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          if (i != j && d2(a, box, i, j) <= cutoff * cutoff) {
            ++coord[i];
          }
        }
      }
      for (int k = 1; k <= 4; ++k) {
        CoordExpr e;
        e.kind = CoordExpr::CMP;
        e.species = {"X"};
        e.cmp = CoordCmp::LT;
        e.value = k;
        std::vector<char> iso;
        mark_isolated(a, box, cutoff, e, nullptr, IsolatedMode::Only, nullptr, iso);
        for (int i = 0; i < N; ++i) {
          bad += (coord[i] < k) != (iso[i] != 0);
        }
      }
    }
    printf("  L = %8.4f A: %d mismatching flags over 20 structures x 4 thresholds\n", L, bad);
    g_fail += bad != 0;
  }
  printf(g_fail == 0 ? "ALL CELL TESTS PASSED\n" : "%d CELL TEST GROUPS FAILED\n", g_fail);
  return g_fail == 0 ? 0 : 1;
}
