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

static void solve_cholesky(const std::vector<float>& L, int n,
                            const std::vector<float>& b, std::vector<float>& x)
{
  x.resize(n);
  for (int i = 0; i < n; i++) {
    float s = b[i];
    for (int j = 0; j < i; j++) s -= L[i*n+j] * x[j];
    x[i] = s / L[i*n+i];
  }
  for (int i = n-1; i >= 0; i--) {
    float s = x[i];
    for (int j = i+1; j < n; j++) s -= L[j*n+i] * x[j];
    x[i] = s / L[i*n+i];
  }
}

// ---- Evaluate 2B basis VALUE ----
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

// ---- Evaluate 2B basis DERIVATIVE ----
static float eval_basis_deriv(int coeff_idx, int nint, float r, float kmin, float kd, float rc)
{
  if (r >= rc) return 0;
  int m = (int)((r - kmin) / kd);
  if (m < 0) m = 0; if (m >= nint) m = nint - 1;
  float u = (r - (kmin + m*kd)) / kd;
  int crel = coeff_idx - m + 3;
  if (crel < 0 || crel > 3) return 0;
  float ddu = 0;
  if (crel == 0) ddu = -3*(1-u)*(1-u)/6;
  else if (crel == 1) ddu = (9*u*u - 12*u)/6;
  else if (crel == 2) ddu = (-9*u*u + 6*u + 3)/6;
  else ddu = 3*u*u/6;
  return ddu / kd;
}

void run_lstsq(UF3_Parameters& para, Uf3Fitness& fitness)
{
  int ncoeff = fitness.model()->ncoeff_2b();
  int npairs = fitness.model()->num_pairs();
  int nt = fitness.model()->num_types();
  int nint = fitness.model()->nknots_2b() - 1;
  float rc = fitness.model()->rc_2b();
  float kmin = fitness.model()->knots_2b()[0];
  float kd = (fitness.model()->knots_2b().back() - kmin) / nint;

  bool has_3b = fitness.model()->has_3b();
  int num_params_2b = npairs * ncoeff;
  int nparam = fitness.model()->num_parameters();

  int nc3[3] = {0}, nint3[3] = {0};
  float kmin3[3] = {0}, kd3[3] = {0}, rc3[3] = {0};
  if (has_3b) {
    for (int d=0;d<3;d++) {
      nc3[d] = fitness.model()->ncoeff_3b(d);
      nint3[d] = fitness.model()->nknots_3b(d) - 1;
      rc3[d] = fitness.model()->rc_3b(d);
      kmin3[d] = fitness.model()->knots_3b(d)[0];
      kd3[d] = (fitness.model()->knots_3b(d).back() - kmin3[d]) / nint3[d];
    }
  }

  const auto& train_set = fitness.train_set();
  int nframes = (int)train_set.size();
  int use_frames = std::min(nframes, para.batch);

  auto t0 = std::chrono::high_resolution_clock::now();

  // Normal equations: ATA (nparam×nparam), ATb (nparam)
  std::vector<double> ATA(nparam * nparam, 0.0);
  std::vector<double> ATb(nparam, 0.0);
  std::vector<float> basis_sum(nparam);

  srand(12345);
  for (int f = 0; f < use_frames; f++) {
    int fidx = rand() % nframes;
    const Uf3Frame& fr = train_set[fidx];
    int n = fr.num_atoms;

    for (int k = 0; k < nparam; k++) basis_sum[k] = 0;

    // ---- 2B basis ----
    for (int i = 0; i < n; i++) {
      int ti = fr.types[i];
      for (int j = i+1; j < n; j++) {
        int tj = fr.types[j];
        float dx = fr.x[i]-fr.x[j], dy = fr.y[i]-fr.y[j], dz = fr.z[i]-fr.z[j];
        float r = sqrtf(dx*dx+dy*dy+dz*dz);
        if (r >= rc) continue;
        int pair_idx = ti * nt + tj;
        for (int c = 0; c < ncoeff; c++)
          basis_sum[pair_idx * ncoeff + c] += eval_basis(c, nint, r, kmin, kd, rc);
      }
    }

    // ---- 2B force contribution to normal equations ----
    float lf = (float)para.lambda_f;
    if (lf > 0) {
      float fnorm = 1.0f / (3.0f * n); // normalize by force components per frame
      for (int i = 0; i < n; i++) {
        int ti = fr.types[i];
        std::vector<double> fbx(num_params_2b, 0), fby(num_params_2b, 0), fbz(num_params_2b, 0);
        for (int j = 0; j < n; j++) {
          if (i == j) continue;
          int tj = fr.types[j];
          float dx = fr.x[i]-fr.x[j], dy = fr.y[i]-fr.y[j], dz = fr.z[i]-fr.z[j];
          float r = sqrtf(dx*dx+dy*dy+dz*dz);
          if (r >= rc) continue;
          float inv_r = 1.0f / r;
          int pair_idx = ti * nt + tj;
          for (int c = 0; c < ncoeff; c++) {
            float dbdr = eval_basis_deriv(c, nint, r, kmin, kd, rc);
            float factor = -dbdr * inv_r;
            int kk = pair_idx * ncoeff + c;
            fbx[kk] += factor * dx; fby[kk] += factor * dy; fbz[kk] += factor * dz;
          }
        }
        // Add force equations: ATA += lf * fnorm * G·G^T, ATb += lf * fnorm * G·F_ref
        float wf = lf * fnorm;
        for (int k = 0; k < num_params_2b; k++) {
          if (fbx[k]==0 && fby[k]==0 && fbz[k]==0) continue;
          ATb[k] += wf * (fbx[k]*fr.fx[i] + fby[k]*fr.fy[i] + fbz[k]*fr.fz[i]);
          for (int m = 0; m < num_params_2b; m++) {
            double gk_gm = fbx[k]*fbx[m] + fby[k]*fby[m] + fbz[k]*fbz[m];
            if (gk_gm != 0) ATA[k*nparam + m] += wf * gk_gm;
          }
        }
      }
    }

    // ---- 3B basis (B_p(rij)*B_q(rik)*B_r(rjk)) ----
    if (has_3b) {
      for (int i = 0; i < n; i++) {
        int ti = fr.types[i];
        for (int j = i+1; j < n; j++) {
          float dx12 = fr.x[j]-fr.x[i], dy12 = fr.y[j]-fr.y[i], dz12 = fr.z[j]-fr.z[i];
          float r12 = sqrtf(dx12*dx12+dy12*dy12+dz12*dz12);
          if (r12 >= rc3[0]) continue;
          int tj = fr.types[j];
          for (int k = j+1; k < n; k++) {
            float dx13 = fr.x[k]-fr.x[i], dy13 = fr.y[k]-fr.y[i], dz13 = fr.z[k]-fr.z[i];
            float r13 = sqrtf(dx13*dx13+dy13*dy13+dz13*dz13);
            if (r13 >= rc3[1]) continue;
            float dx23 = fr.x[k]-fr.x[j], dy23 = fr.y[k]-fr.y[j], dz23 = fr.z[k]-fr.z[j];
            float r23 = sqrtf(dx23*dx23+dy23*dy23+dz23*dz23);
            if (r23 >= rc3[2]) continue;
            int tk = fr.types[k];
            int trip_idx = (ti*nt + tj)*nt + tk;
            int off3 = num_params_2b + trip_idx * nc3[0] * nc3[1] * nc3[2];
            for (int p = 0; p < nc3[0]; p++) {
              float bp = eval_basis(p, nint3[0], r12, kmin3[0], kd3[0], rc3[0]);
              if (bp == 0) continue;
              for (int q = 0; q < nc3[1]; q++) {
                float bq = eval_basis(q, nint3[1], r13, kmin3[1], kd3[1], rc3[1]);
                if (bq == 0) continue;
                float bpbq = bp * bq;
                for (int r = 0; r < nc3[2]; r++) {
                  float br = eval_basis(r, nint3[2], r23, kmin3[2], kd3[2], rc3[2]);
                  if (br == 0) continue;
                  basis_sum[off3 + p + q * nc3[0] + r * nc3[0] * nc3[1]] += bpbq * br;
                }
              }
            }
          }
        }
      }
    }

    double target = (double)fr.energy;
    for (int k = 0; k < nparam; k++) {
      double ak = (double)basis_sum[k];
      if (ak == 0) continue;
      ATb[k] += ak * target;
      for (int m = 0; m < nparam; m++) {
        if (basis_sum[m] == 0) continue;
        ATA[k*nparam + m] += ak * (double)basis_sum[m];
      }
    }
  }

  // Regularization
  for (int k = 0; k < nparam; k++) ATA[k*nparam + k] += 1e-6;

  // Convert to float for Cholesky
  std::vector<float> ATAf(nparam * nparam);
  std::vector<float> ATbf(nparam);
  for (int i = 0; i < nparam; i++) { ATbf[i] = (float)ATb[i];
    for (int j = 0; j < nparam; j++) ATAf[i*nparam + j] = (float)ATA[i*nparam + j]; }

  bool ok = cholesky(ATAf, nparam);
  if (!ok) printf("  lstsq: ATA not PD, using diagonal\n");

  std::vector<float> x;
  solve_cholesky(ATAf, nparam, ATbf, x);
  fitness.model()->set_parameters(x.data());

  auto t1 = std::chrono::high_resolution_clock::now();
  printf("  lstsq: %d params%s, %d frames, %.2f s\n",
         nparam, has_3b ? " (2B+3B)" : " (2B)", use_frames,
         std::chrono::duration<double>(t1-t0).count());

  std::vector<int> bidx(1); float tl = 0; int en = std::min(20, nframes);
  for (int f = 0; f < en; f++) { bidx[0] = f; tl += fitness.compute_loss(bidx, 0); }
  tl /= en;
  printf("  Loss = %.3f [E=%.3f F=%.3f eV/A]\n", tl, fitness.loss_e, fitness.loss_f);
}
