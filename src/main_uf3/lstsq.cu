/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "lstsq.cuh"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---- Cholesky decomposition (in-place) ----
static bool cholesky(std::vector<float>& A, int n)
{
  for (int j = 0; j < n; j++) {
    float s = 0;
    for (int k = 0; k < j; k++) s += A[j*n+k] * A[j*n+k];
    float diag = A[j*n+j] - s;
    if (diag <= 0) { A[j*n+j] = 0; return false; }
    A[j*n+j] = sqrtf(diag);
    float invLjj = 1.0f / A[j*n+j];
    for (int i = j+1; i < n; i++) {
      s = 0;
      for (int k = 0; k < j; k++) s += A[i*n+k] * A[j*n+k];
      A[i*n+j] = (A[i*n+j] - s) * invLjj;
    }
  }
  return true;
}

// ---- Solve L L^T x = b (L is lower-triangular) ----
static void solve_cholesky(const std::vector<float>& L, int n,
                            const std::vector<float>& b, std::vector<float>& x)
{
  // Forward: L y = b
  x.resize(n);
  for (int i = 0; i < n; i++) {
    float s = b[i];
    for (int j = 0; j < i; j++) s -= L[i*n+j] * x[j];
    x[i] = s / L[i*n+i];
  }
  // Backward: L^T x = y (overwrite x in-place)
  for (int i = n-1; i >= 0; i--) {
    float s = x[i];
    for (int j = i+1; j < n; j++) s -= L[j*n+i] * x[j];
    x[i] = s / L[i*n+i];
  }
}

// ---- Evaluate 2B basis value for coefficient k at distance r ----
static float eval_basis(int coeff_idx, int nint, float r, float kmin, float kd, float rc)
{
  if (r >= rc) return 0;
  int m = (int)((r - kmin) / kd);
  if (m < 0) m = 0; if (m >= nint) m = nint - 1;
  float u = (r - (kmin + m*kd)) / kd;
  int crel = coeff_idx - m + 3;
  if (crel < 0 || crel > 3) return 0;
  if (crel == 0) return (1-u)*(1-u)*(1-u)/6;
  if (crel == 1) return (3*u*u*u - 6*u*u + 4)/6;
  if (crel == 2) return (-3*u*u*u + 3*u*u + 3*u + 1)/6;
  return u*u*u/6;
}

void run_lstsq(UF3_Parameters& para, Uf3Fitness& fitness)
{
  int nparam = fitness.num_parameters();
  int ncoeff = fitness.model()->ncoeff_2b();
  int npairs = fitness.model()->num_pairs();
  int nint = fitness.model()->nknots_2b() - 1;
  float rc = fitness.model()->rc_2b();
  float kmin = fitness.model()->knots_2b()[0];
  float kd = (fitness.model()->knots_2b().back() - kmin) / nint;

  const auto& train_set = fitness.train_set();
  int nframes = (int)train_set.size();
  int use_frames = std::min(nframes, para.batch); // use up to batch frames

  auto t0 = std::chrono::high_resolution_clock::now();

  // Build normal equation: ATA (nparam×nparam) and ATb (nparam)
  std::vector<double> ATA(nparam * nparam, 0.0);
  std::vector<double> ATb(nparam, 0.0);
  std::vector<float> basis_sum(nparam); // per-frame basis accumulator

  srand(12345);
  for (int f = 0; f < use_frames; f++) {
    int fidx = rand() % nframes;
    const Uf3Frame& fr = train_set[fidx];
    int n = fr.num_atoms;

    // Accumulate basis sum for each coefficient
    for (int k = 0; k < nparam; k++) basis_sum[k] = 0;
    for (int i = 0; i < n; i++) {
      int ti = fr.types[i];
      for (int j = i+1; j < n; j++) {
        int tj = fr.types[j];
        float dx = fr.x[i]-fr.x[j], dy = fr.y[i]-fr.y[j], dz = fr.z[i]-fr.z[j];
        float r = sqrtf(dx*dx+dy*dy+dz*dz);
        int pair_idx = ti * fitness.model()->num_types() + tj;
        for (int c = 0; c < ncoeff; c++) {
          float bv = eval_basis(c, nint, r, kmin, kd, rc);
          basis_sum[pair_idx * ncoeff + c] += bv;
        }
      }
    }

    double target = (double)fr.energy;
    for (int k = 0; k < nparam; k++) {
      double ak = (double)basis_sum[k];
      ATb[k] += ak * target;
      for (int m = 0; m < nparam; m++)
        ATA[k*nparam + m] += ak * (double)basis_sum[m];
    }
  }

  // Regularization (small diagonal to ensure positive definiteness)
  for (int k = 0; k < nparam; k++) ATA[k*nparam + k] += 1e-6;

  // Convert to float for Cholesky
  std::vector<float> ATAf(nparam * nparam);
  std::vector<float> ATbf(nparam);
  for (int i = 0; i < nparam; i++) { ATbf[i] = (float)ATb[i];
    for (int j = 0; j < nparam; j++) ATAf[i*nparam + j] = (float)ATA[i*nparam + j]; }

  // Cholesky + solve
  bool ok = cholesky(ATAf, nparam);
  if (!ok) { printf("  lstsq: ATA not positive definite, using diagonal\n"); }

  std::vector<float> x;
  solve_cholesky(ATAf, nparam, ATbf, x);

  // Apply solution
  fitness.model()->set_parameters(x.data());

  auto t1 = std::chrono::high_resolution_clock::now();
  double dt = std::chrono::duration<double>(t1 - t0).count();

  // Evaluate RMS
  std::vector<int> bidx(1);
  float rmse = 0;
  for (int f = 0; f < std::min(100, nframes); f++) {
    bidx[0] = f;
    rmse += fitness.compute_loss(bidx, 0);
  }
  rmse /= std::min(100, nframes);

  printf("  lstsq solution: %d params, %d frames, %.2f s\n", nparam, use_frames, dt);
  printf("  Energy RMSE = %.3f eV/frame\n", rmse);
}
