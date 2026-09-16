/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#pragma once

#include "model/atom.cuh"
#include "model/box.cuh"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace connectivity_bfs_detail {

// Worker threads for the parallel passes. Runs alongside one GPU job; only
// CPU arrays are touched.
inline int bfs_thread_count(int N)
{
  const int hw = (int)std::thread::hardware_concurrency();
  const int half = hw > 0 ? std::max(1, hw / 2) : 1;
  return std::max(1, std::min(half, std::max(1, N / 65536)));
}

} // namespace connectivity_bfs_detail

// keep[i] != 0 if atom i is connected to the lowest-coord atom along `axis`.
// Edges: distance <= cutoff. The two transverse directions use PBC; `axis` does not.
// If eligible is non-null, only those atoms are in the graph.
// Visited set is identical to the serial queue version: atoms are claimed by
// compare-and-swap during a level-synchronous flood fill from the single
// lowest-axis seed; the cell build and seed scan are threaded.
inline __host__ void find_main_component_from_min_axis(
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
  {
    const int n_thread = connectivity_bfs_detail::bfs_thread_count(N);
    std::vector<double> t_min(n_thread, 0.0), t_max(n_thread, 0.0);
    std::vector<int> t_arg(n_thread, -1);
    std::vector<std::thread> workers;
    const int chunk = (N + n_thread - 1) / n_thread;
    for (int t = 0; t < n_thread; ++t) {
      const int lo = t * chunk;
      const int hi = std::min(N, lo + chunk);
      if (lo >= hi) {
        break;
      }
      workers.emplace_back([&, lo, hi, t]() {
        for (int i = lo; i < hi; ++i) {
          if (!is_ok(i)) {
            continue;
          }
          const double a = r[axis][i];
          if (t_arg[t] < 0) {
            t_arg[t] = i;
            t_min[t] = a;
            t_max[t] = a;
            continue;
          }
          if (a < t_min[t]) {
            t_min[t] = a;
            t_arg[t] = i;
          }
          if (a > t_max[t]) {
            t_max[t] = a;
          }
        }
      });
    }
    for (auto& w : workers) {
      w.join();
    }
    for (int t = 0; t < n_thread; ++t) {
      if (t_arg[t] < 0) {
        continue;
      }
      if (seed < 0 || t_min[t] < amin) {
        amin = t_min[t];
        seed = t_arg[t];
      }
      if (t_max[t] > amax) {
        amax = t_max[t];
      }
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
    if (rem < 0) {
      rem += n;
    }
    return rem;
  };
  auto cell_id = [&](int i0, int i1, int ia) { return i0 + nt0 * (i1 + nt1 * ia); };

  const int ncells = nt0 * nt1 * na;
  std::vector<int> c0(N), c1(N), ca(N), cell_of(N);
  {
    const int n_thread = connectivity_bfs_detail::bfs_thread_count(N);
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
        }
      });
    }
    for (auto& w : workers) {
      w.join();
    }
  }
  std::vector<int> cell_offset(ncells + 1, 0);
  for (int i = 0; i < N; ++i) {
    cell_offset[cell_of[i] + 1] += 1;
  }
  for (int c = 0; c < ncells; ++c) {
    cell_offset[c + 1] += cell_offset[c];
  }
  std::vector<int> cell_contents(N);
  {
    // in-cell order does not affect the resulting component
    std::vector<int> cursor(cell_offset.begin(), cell_offset.end() - 1);
    for (int i = 0; i < N; ++i) {
      const int id = cell_of[i];
      cell_contents[cursor[id]] = i;
      cursor[id] += 1;
    }
  }

  // persistent spin-wait workers: level expansion is sequential-dependent, so
  // amortizing thread creation across ALL levels (not per level) is the only
  // way parallelism pays for slab-shaped components (~700 levels of us-scale
  // work each; per-level spawn costs ~0.5 s alone)
  std::atomic<char>* keep_atomic = reinterpret_cast<std::atomic<char>*>(keep.data());
  keep_atomic[seed].store(1);
  std::vector<int> frontier{seed};

  const int n_pool = connectivity_bfs_detail::bfs_thread_count(N);
  std::atomic<int> gen{0};     // owner bumps per level
  std::atomic<int> workers_done{0};
  std::atomic<int> stop_flag{0};
  std::vector<int> shared_frontier; // owner writes, workers read (published via gen)
  int shared_size = 0;
  std::vector<std::vector<int>> next_private(n_pool + 1);

  auto expand_range = [&](int lo, int hi, int tid) {
    std::vector<int>& next = next_private[tid];
    next.clear();
    for (int fi = lo; fi < hi; ++fi) {
      const int i = shared_frontier[fi];
      for (int o0 = -1; o0 <= 1; ++o0) {
        for (int o1 = -1; o1 <= 1; ++o1) {
          for (int oa = -1; oa <= 1; ++oa) {
            const int nia = ca[i] + oa;
            if (nia < 0 || nia >= na) {
              continue;
            }
            const int nid = cell_id(wrap(c0[i] + o0, nt0), wrap(c1[i] + o1, nt1), nia);
            for (int k = cell_offset[nid]; k < cell_offset[nid + 1]; ++k) {
              const int j = cell_contents[k];
              if (j == i || keep[j] || !is_ok(j)) {
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
                char expected = 0;
                if (keep_atomic[j].compare_exchange_strong(expected, 1)) {
                  next.push_back(j);
                }
              }
            }
          }
        }
      }
    }
  };

  std::vector<std::thread> pool;
  std::mutex idle_mu;
  std::condition_variable idle_cv;
  std::atomic<int> wake_epoch{0};
  for (int t = 1; t <= n_pool; ++t) {
    pool.emplace_back([&, t]() {
      int seen = 0;
      while (true) {
        {
          std::unique_lock<std::mutex> lk(idle_mu);
          idle_cv.wait(lk, [&]() {
            return wake_epoch.load(std::memory_order_acquire) != seen ||
                   stop_flag.load(std::memory_order_acquire);
          });
        }
        if (stop_flag.load(std::memory_order_acquire)) {
          return;
        }
        seen = wake_epoch.load(std::memory_order_acquire);
        if (stop_flag.load(std::memory_order_acquire)) {
          return;
        }
        const int total = shared_size;
        const int active = total >= 64 ? n_pool + 1 : 1;
        if (active > 1) {
          const int chunk = (total + active - 1) / active;
          const int lo = t * chunk;
          const int hi = std::min(total, lo + chunk);
          if (lo < hi) {
            expand_range(lo, hi, t);
          }
        }
        workers_done.fetch_add(1, std::memory_order_release);
      }
    });
  }

  int my_gen = 0;
  while (!frontier.empty()) {
    shared_frontier = frontier; // copy: workers read the stable copy
    shared_size = (int)frontier.size();
    ++my_gen;
    workers_done.store(0, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lk(idle_mu);
      wake_epoch.store(my_gen, std::memory_order_release);
    }
    idle_cv.notify_all();
    const int total = shared_size;
    if (total >= 64) {
      const int active = n_pool + 1;
      const int chunk = (total + active - 1) / active;
      expand_range(0, std::min(total, chunk), 0);
      while (workers_done.load(std::memory_order_acquire) < n_pool) {
      }
      std::vector<int> next;
      size_t sum = 0;
      for (int t = 0; t <= n_pool; ++t) {
        sum += next_private[t].size();
      }
      next.reserve(sum);
      for (int t = 0; t <= n_pool; ++t) {
        next.insert(next.end(), next_private[t].begin(), next_private[t].end());
      }
      frontier.swap(next);
    } else {
      // small level: expand serially, workers see active==1 and just ack
      const int tid = 0;
      next_private[tid].clear();
      const int saved_lo = 0;
      const int saved_hi = total;
      // reuse the same lambda serially
      {
        std::vector<int>& next = next_private[tid];
        for (int fi = saved_lo; fi < saved_hi; ++fi) {
          const int i = shared_frontier[fi];
          for (int o0 = -1; o0 <= 1; ++o0) {
            for (int o1 = -1; o1 <= 1; ++o1) {
              for (int oa = -1; oa <= 1; ++oa) {
                const int nia = ca[i] + oa;
                if (nia < 0 || nia >= na) {
                  continue;
                }
                const int nid = cell_id(wrap(c0[i] + o0, nt0), wrap(c1[i] + o1, nt1), nia);
                for (int k = cell_offset[nid]; k < cell_offset[nid + 1]; ++k) {
                  const int j = cell_contents[k];
                  if (j == i || keep[j] || !is_ok(j)) {
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
                    char expected = 0;
                    if (keep_atomic[j].compare_exchange_strong(expected, 1)) {
                      next.push_back(j);
                    }
                  }
                }
              }
            }
          }
        }
      }
      while (workers_done.load(std::memory_order_acquire) < n_pool) {
      }
      frontier.swap(next_private[tid]);
    }
  }
  stop_flag.store(1, std::memory_order_release);
  ++my_gen;
  {
    std::lock_guard<std::mutex> lk(idle_mu);
    wake_epoch.store(my_gen, std::memory_order_release);
  }
  idle_cv.notify_all();
  for (auto& w : pool) {
    w.join();
  }
}
inline __host__ void find_main_component_from_min_z(
  const Atom& atom, const Box& box, double cutoff, std::vector<char>& keep)
{
  find_main_component_from_min_axis(atom, box, cutoff, 2, nullptr, keep);
}
