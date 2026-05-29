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
#include "utilities/gpu_macro.cuh"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---- Cholesky decomposition (in-place) ----
static bool cholesky(std::vector<float>& A, int n) {
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
                            const std::vector<float>& b, std::vector<float>& x) {
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

// ---- GPU helpers ----
__device__ inline float _bval(int ci, int ni, float r, float km, float kd) {
  int m = (int)((r - km) / kd); if (m < 0) m = 0; if (m >= ni) m = ni - 1;
  float u = (r - (km + m*kd)) / kd;
  int cr = ci - m + 3; if (cr < 0 || cr > 3) return 0;
  if (cr == 0) return (1-u)*(1-u)*(1-u)/6;
  if (cr == 1) return (3*u*u*u - 6*u*u + 4)/6;
  if (cr == 2) return (-3*u*u*u + 3*u*u + 3*u + 1)/6;
  return u*u*u/6;
}

// Basis derivative dB/dr for force equations
__device__ inline float _dbval(int ci, int ni, float r, float km, float kd) {
  int m = (int)((r - km) / kd); if (m < 0) m = 0; if (m >= ni) m = ni - 1;
  float u = (r - (km + m*kd)) / kd;
  int cr = ci - m + 3; if (cr < 0 || cr > 3) return 0;
  float ddu = 0;
  if (cr == 0) ddu = -3*(1-u)*(1-u)/6;
  else if (cr == 1) ddu = (9*u*u - 12*u)/6;
  else if (cr == 2) ddu = (-9*u*u + 6*u + 3)/6;
  else ddu = 3*u*u/6;
  return ddu / kd;
}

// GPU kernel: multi-threaded per frame with shared memory atomicAdd.
// blockDim threads cooperate: each handles a subset of atoms, atomically
// accumulates basis contributions into per-block shared memory.
static __global__ void lstsq_accumulate(
  int nf, const int* __restrict__ fidx, const int* __restrict__ nat, const int* __restrict__ off,
  const int* __restrict__ typ, const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
  int ncoeff, int npairs, int nt, int nint, float kmin, float kd, float rc, int num_params_2b,
  int has_3b, int nc0, int nc1, int nc2, int ni0, int ni1, int ni2,
  float k0, float kd0, float r0, float k1, float kd1, float r1, float k2, float kd2, float r2,
  int nparam, float* __restrict__ basis_out,
  double* __restrict__ d_ATA, double* __restrict__ d_ATb,
  const float* __restrict__ d_target,
  const float* __restrict__ d_fxref, const float* __restrict__ d_fyref, const float* __restrict__ d_fzref,
  float lambda_f, int num_params_2b_only)
{
  extern __shared__ float s_basis[]; // dynamically sized: nparam elements
  int b = blockIdx.x; if (b >= nf) return;
  int tid = threadIdx.x, stride = blockDim.x;
  int fid = fidx[b], n = nat[fid], o = off[fid];

  // Zero shared memory (parallel across threads)
  for (int k = tid; k < nparam; k += stride) s_basis[k] = 0;
  __syncthreads();

  // 2B: each thread handles atoms i = tid, tid+stride, ...
  for (int i = tid; i < n; i += stride) {
    int ti = typ[o+i];
    for (int j = i+1; j < n; j++) {
      int tj = typ[o+j];
      float dx = x[o+i]-x[o+j], dy = y[o+i]-y[o+j], dz = z[o+i]-z[o+j];
      float r = sqrtf(dx*dx+dy*dy+dz*dz); if (r >= rc) continue;
      int pi = ti * nt + tj;
      for (int c = 0; c < ncoeff; c++) {
        float bv = _bval(c, nint, r, kmin, kd);
        if (bv != 0) atomicAdd(&s_basis[pi * ncoeff + c], bv);
      }
    }
  }

  // 2B force equations (add to ATA/ATb with lambda_f weight, matching SNES loss)
  if (d_fxref && lambda_f > 0) {
    float fnorm = 1.0f / (3.0f * n); // per-component normalization (matching NEP)
    float wf = lambda_f * fnorm;
    for (int i = tid; i < n; i += stride) {
      int ti = typ[o+i];
      float fxi = d_fxref[o+i], fyi = d_fyref[o+i], fzi = d_fzref[o+i];
      if (fxi == 0 && fyi == 0 && fzi == 0) continue;
      for (int j = 0; j < n; j++) {
        if (i == j) continue;
        int tj = typ[o+j];
        float dx = x[o+i]-x[o+j], dy = y[o+i]-y[o+j], dz = z[o+i]-z[o+j];
        float r = sqrtf(dx*dx+dy*dy+dz*dz); if (r >= rc) continue;
        float inv_r = 1.0f / r;
        int pi = ti * nt + tj;
        for (int c = 0; c < ncoeff; c++) {
          float dbdr = _dbval(c, nint, r, kmin, kd);
          float factor = -dbdr * inv_r;
          int kk = pi * ncoeff + c;
          float gx = factor * dx, gy = factor * dy, gz = factor * dz;
          // Accumulate force contribution to ATb
          float fcontrib = gx * (double)fxi + gy * (double)fyi + gz * (double)fzi;
          if (fcontrib != 0) atomicAdd(&d_ATb[kk], wf * (double)fcontrib);
          // Accumulate to ATA: ATA[kk][mm] += wf * (gx*gx' + gy*gy' + gz*gz')
          for (int cc = 0; cc < ncoeff; cc++) {
            int mm = pi * ncoeff + cc;
            if (mm < kk) continue; // only upper triangle for efficiency
            float dbdr2 = _dbval(cc, nint, r, kmin, kd);
            float factor2 = -dbdr2 * inv_r;
            float gx2 = factor2 * dx, gy2 = factor2 * dy, gz2 = factor2 * dz;
            double gdot = (double)gx*gx2 + (double)gy*gy2 + (double)gz*gz2;
            if (gdot != 0) {
              atomicAdd(&d_ATA[kk * nparam + mm], wf * gdot);
              if (kk != mm) atomicAdd(&d_ATA[mm * nparam + kk], wf * gdot);
            }
          }
        }
      }
    }
  }

  // 3B: each thread handles center atom i = tid, tid+stride, ...
  if (has_3b) {
    for (int i = tid; i < n; i += stride) {
      int ti = typ[o+i];
      for (int j = i+1; j < n; j++) {
        float dx12=x[o+j]-x[o+i], dy12=y[o+j]-y[o+i], dz12=z[o+j]-z[o+i];
        float r12=sqrtf(dx12*dx12+dy12*dy12+dz12*dz12); if (r12>=r0) continue;
        int tj=typ[o+j];
        for (int k=j+1;k<n;k++){
          float dx13=x[o+k]-x[o+i], dy13=y[o+k]-y[o+i], dz13=z[o+k]-z[o+i];
          float r13=sqrtf(dx13*dx13+dy13*dy13+dz13*dz13); if (r13>=r1) continue;
          float dx23=x[o+k]-x[o+j], dy23=y[o+k]-y[o+j], dz23=z[o+k]-z[o+j];
          float r23=sqrtf(dx23*dx23+dy23*dy23+dz23*dz23); if (r23>=r2) continue;
          int tk=typ[o+k], trip=(ti*nt+tj)*nt+tk;
          int off3 = num_params_2b + trip * nc0 * nc1 * nc2;
          for(int p=0;p<nc0;p++){ float bp=_bval(p,ni0,r12,k0,kd0); if(bp==0)continue;
          for(int q=0;q<nc1;q++){ float bq=_bval(q,ni1,r13,k1,kd1); if(bq==0)continue;
          float bpbq=bp*bq;
          for(int r=0;r<nc2;r++){ float br=_bval(r,ni2,r23,k2,kd2); if(br==0)continue;
          atomicAdd(&s_basis[off3 + p + q*nc0 + r*nc0*nc1], bpbq * br);
          }}}
        }
      }
    }
  }
  __syncthreads();

  // All threads: atomically accumulate basis into ATA/ATb (distributed)
  // Thread k handles rows k, k+stride, ... of the ATA matrix
  if (d_ATA && d_ATb) {
    float targ = d_target[fid];
    for (int k = tid; k < nparam; k += stride) {
      float bk = s_basis[k]; if (bk == 0) continue;
      atomicAdd(&d_ATb[k], (double)bk * (double)targ);
      for (int m = 0; m < nparam; m++) {
        float bm = s_basis[m]; if (bm == 0) continue;
        atomicAdd(&d_ATA[k * nparam + m], (double)bk * (double)bm);
      }
    }
  }

  // Also write basis to global output (for debug/download)
  if (tid == 0) {
    float* g_basis = basis_out + b * nparam;
    for (int kk = 0; kk < nparam; kk++) g_basis[kk] = s_basis[kk];
  }
}

// GPU reduction: accumulate basis vectors into ATA matrix and ATb vector
static __global__ void lstsq_reduce_ata(
  int nf, int nparam, const float* __restrict__ basis, const float* __restrict__ target,
  double* __restrict__ d_ATA, double* __restrict__ d_ATb)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nparam) return;
  int k = idx; // This thread handles row k of ATA

  // For row k, accumulate contributions from all frames
  // This is O(nparam * nf) per block — let's use a different decomposition.
  // Actually, let each block handle one frame: atomicAdd d_ATA[basis_k][basis_m]
  // and d_ATb[basis_k] * target.
  // We'll use a simple approach: blocks process frames, threads process coefficients.
  // For now: CPU path below is the fallback.
}

void run_lstsq(
  const UF3_Parameters& para,
  const UF3_OptimizerStage& stage,
  int stage_id,
  int gen_offset,
  Uf3Fitness& fitness)
{
  if (stage.generation > 1) {
    printf("  Warning: lstsq runs once; generation=%d ignored.\n", stage.generation);
  }
  fitness.begin_stage(stage_id, "lstsq", "solve");

  int ncoeff = fitness.model()->ncoeff_2b(), npairs = fitness.model()->num_pairs();
  int nt = fitness.model()->num_types(), nint = fitness.model()->nknots_2b() - 1;
  float rc = fitness.model()->rc_2b(), kmin = fitness.model()->knots_2b()[0];
  float kd = (fitness.model()->knots_2b().back() - kmin) / nint;
  bool has_3b = fitness.model()->has_3b();
  int num_params_2b = npairs * ncoeff, nparam = fitness.model()->num_parameters();
  int nc3[3]={0},ni3[3]={0}; float k3[3]={0},kd3[3]={0},r3[3]={0};
  if(has_3b) for(int d=0;d<3;d++){ nc3[d]=fitness.model()->ncoeff_3b(d); ni3[d]=fitness.model()->nknots_3b(d)-1;
    r3[d]=fitness.model()->rc_3b(d); k3[d]=fitness.model()->knots_3b(d)[0];
    kd3[d]=(fitness.model()->knots_3b(d).back()-k3[d])/ni3[d]; }

  const auto& ds = fitness.dataset();
  int use_frames = ds.num_frames;
  if (!stage.full_batch) {
    int cap = stage.batch >= 0 ? stage.batch : para.batch;
    use_frames = std::min(ds.num_frames, cap);
  }

  auto t0 = std::chrono::high_resolution_clock::now();

  // Use pre-loaded GPU data directly — NO H2D upload

  std::vector<int> h_bidx(use_frames);
  for (int i = 0; i < use_frames; i++) h_bidx[i] = i;
  GPU_Vector<int> d_bidx(use_frames);
  d_bidx.copy_from_host(h_bidx.data());

  GPU_Vector<float> d_basis(use_frames * nparam);
  GPU_Vector<double> d_ATA(nparam * nparam), d_ATb(nparam);
  // Zero ATA/ATb on GPU
  cudaMemset(d_ATA.data(), 0, nparam * nparam * sizeof(double));
  cudaMemset(d_ATb.data(), 0, nparam * sizeof(double));

  // Launch GPU kernel: 1 frame per block, 64 threads per block, shared memory
  const int BLK = 64;
  size_t smem = nparam * sizeof(float);
  lstsq_accumulate<<<use_frames, BLK, smem>>>(
    use_frames, d_bidx.data(), ds.d_natoms.data(), ds.d_offsets.data(),
    ds.d_types.data(), ds.d_x.data(), ds.d_y.data(), ds.d_z.data(),
    ncoeff, npairs, nt, nint, kmin, kd, rc, num_params_2b,
    has_3b ? 1 : 0, nc3[0], nc3[1], nc3[2], ni3[0], ni3[1], ni3[2],
    k3[0], kd3[0], r3[0], k3[1], kd3[1], r3[1], k3[2], kd3[2], r3[2],
    nparam, d_basis.data(), d_ATA.data(), d_ATb.data(), ds.d_energy_ref.data(),
    ds.d_fx_ref.data(), ds.d_fy_ref.data(), ds.d_fz_ref.data(), (float)para.lambda_f, num_params_2b);
  GPU_CHECK_KERNEL

  // Download ATA and ATb from GPU (already accumulated atomically)
  std::vector<double> ATA(nparam * nparam), ATb(nparam);
  cudaMemcpy(ATA.data(), d_ATA.data(), nparam*nparam*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(ATb.data(), d_ATb.data(), nparam*sizeof(double), cudaMemcpyDeviceToHost);

  for (int k = 0; k < nparam; k++) ATA[k*nparam + k] += 1e-6;
  std::vector<float> ATAf(nparam*nparam), ATbf(nparam);
  for (int i = 0; i < nparam; i++) { ATbf[i] = (float)ATb[i];
    for (int j = 0; j < nparam; j++) ATAf[i*nparam+j] = (float)ATA[i*nparam+j]; }
  bool ok = cholesky(ATAf, nparam);
  if (!ok) printf("  lstsq: ATA not PD\n");
  std::vector<float> x; solve_cholesky(ATAf, nparam, ATbf, x);
  fitness.model()->set_parameters(x.data());

  auto t1 = std::chrono::high_resolution_clock::now();
  float dt = (float)std::chrono::duration<double>(t1 - t0).count();
  printf("  lstsq GPU: %d params%s, %d frames, %.2f s\n",
         nparam, has_3b?" (2B+3B)":" (2B)", use_frames, dt);
  // local_iter=1: lstsq runs once; triggers the local_iter==1 logging checkpoint.
  float tl = fitness.compute_loss(0, gen_offset, stage_id, 1, dt);
  printf("  Loss=%.3f [E=%.3f F=%.3f eV/A]\n", tl, fitness.loss_e, fitness.loss_f);
}
