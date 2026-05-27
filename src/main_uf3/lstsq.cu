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

// ---- Evaluate 2B basis VALUE for coefficient k at distance r ----
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

// ---- Evaluate 2B basis DERIVATIVE d/dr for coefficient k at distance r ----
static float eval_basis_deriv(int coeff_idx, int nint, float r,
                               float kmin, float kd, float rc)
{
  if (r >= rc) return 0;
  int m = (int)((r - kmin) / kd);
  if (m < 0) m = 0; if (m >= nint) m = nint - 1;
  float u = (r - (kmin + m*kd)) / kd;
  int crel = coeff_idx - m + 3;
  if (crel < 0 || crel > 3) return 0;
  // dB/du / kd
  float ddu = 0;
  if (crel == 0) ddu = -3*(1-u)*(1-u)/6;         // d/du (1-u)^3
  else if (crel == 1) ddu = (9*u*u - 12*u)/6;     // d/du (3u^3-6u^2+4)
  else if (crel == 2) ddu = (-9*u*u + 6*u + 3)/6; // d/du (-3u^3+3u^2+3u+1)
  else ddu = 3*u*u/6;
  return ddu / kd; // chain rule: dB/dr = dB/du * du/dr = dB/du / kd
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

    // Energy contribution
    double target = (double)fr.energy;
    for (int k = 0; k < nparam; k++) {
      double ak = (double)basis_sum[k];
      ATb[k] += ak * target;
      for (int m = 0; m < nparam; m++)
        ATA[k*nparam + m] += ak * (double)basis_sum[m];
    }

    // Force contribution (weighted by lambda_f, normalized by force components)
    float lf = (float)para.lambda_f;
    if (lf > 0 && n > 0) {
      float force_norm = 1.0f / (3.0f * n); // normalize by number of force components per frame
      for (int i = 0; i < n; i++) {
        int ti = fr.types[i];
        // Force basis vector for atom i: G_k,iα = -Σ_j B'_k(r_ij) * (r_iα - r_jα) / r_ij
        std::vector<double> force_basis_x(nparam, 0), force_basis_y(nparam, 0), force_basis_z(nparam, 0);
        for (int j = 0; j < n; j++) {
          if (i == j) continue;
          int tj = fr.types[j];
          float dx = fr.x[i]-fr.x[j], dy = fr.y[i]-fr.y[j], dz = fr.z[i]-fr.z[j];
          float r = sqrtf(dx*dx+dy*dy+dz*dz);
          if (r >= rc) continue;
          int pair_idx = ti * fitness.model()->num_types() + tj;
          float inv_r = 1.0f / r;
          for (int c = 0; c < ncoeff; c++) {
            float dbdr = eval_basis_deriv(c, nint, r, kmin, kd, rc);
            float factor = -dbdr * inv_r;
            int kk = pair_idx * ncoeff + c;
            force_basis_x[kk] += factor * dx;
            force_basis_y[kk] += factor * dy;
            force_basis_z[kk] += factor * dz;
          }
        }
        // Add force equations: ATA += lf * force_norm * G·G^T, ATb += lf * force_norm * G·F_ref
        float wf = lf * force_norm;
        for (int k = 0; k < nparam; k++) {
          double gkx = force_basis_x[k], gky = force_basis_y[k], gkz = force_basis_z[k];
          if (fabs(gkx) < 1e-10 && fabs(gky) < 1e-10 && fabs(gkz) < 1e-10) continue;
          ATb[k] += wf * (gkx * (double)fr.fx[i] + gky * (double)fr.fy[i] + gkz * (double)fr.fz[i]);
          for (int m = 0; m < nparam; m++) {
            double gk_gm = gkx*force_basis_x[m] + gky*force_basis_y[m] + gkz*force_basis_z[m];
            if (gk_gm != 0) ATA[k*nparam + m] += wf * gk_gm;
          }
        }
      }
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

  // Evaluate with full loss (energy + force) to match other optimizers
  std::vector<int> bidx(1);
  float total_loss = 0;
  int eval_frames = std::min(50, nframes);
  for (int f = 0; f < eval_frames; f++) {
    bidx[0] = f;
    total_loss += fitness.compute_loss(bidx, 0);
  }
  total_loss /= eval_frames;

  printf("  lstsq solution: %d params, %d frames, %.2f s\n", nparam, use_frames, dt);
  printf("  Loss (E+F) = %.3f eV [E=%.3f F=%.3f eV/A]\n",
         total_loss, fitness.loss_e, fitness.loss_f);
}
