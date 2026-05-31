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
#include <algorithm>
#include <cublas_v2.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---- Cholesky decomposition (in-place, double precision) ----
// Double precision is important: the 2B/3B normal matrix mixes well-constrained
// columns with nearly-unconstrained 3B columns spanning many orders of
// magnitude, where float Cholesky loses positive-definiteness.
static bool cholesky(std::vector<double>& A, int n) {
  for (int j = 0; j < n; j++) {
    double s = 0;
    for (int k = 0; k < j; k++) s += A[j*n+k] * A[j*n+k];
    double diag = A[j*n+j] - s;
    if (diag <= 0) { A[j*n+j] = 0; return false; }
    A[j*n+j] = sqrt(diag);
    double invLjj = 1.0 / A[j*n+j];
    for (int i = j+1; i < n; i++) {
      s = 0;
      for (int k = 0; k < j; k++) s += A[i*n+k] * A[j*n+k];
      A[i*n+j] = (A[i*n+j] - s) * invLjj;
    }
  }
  return true;
}
static void solve_cholesky(const std::vector<double>& L, int n,
                            const std::vector<double>& b, std::vector<double>& x) {
  x.resize(n);
  for (int i = 0; i < n; i++) {
    double s = b[i];
    for (int j = 0; j < i; j++) s -= L[i*n+j] * x[j];
    x[i] = s / L[i*n+i];
  }
  for (int i = n-1; i >= 0; i--) {
    double s = x[i];
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

// ===========================================================================
// Explicit weighted design-matrix construction (column-major) for cuBLAS.
//
// UF3 is linear in its coefficients, so least-squares is the right trainer.
// We build A (M rows x nparam cols) and b (M), where rows are:
//   - one ENERGY row per frame            (1-body counts + 0.5*2B + per-centre 3B)
//   - three FORCE rows per real atom       (2B + 3B force features)
// weighted to match the loss lambda_e*MSE_e_peratom + lambda_f*MSE_f, then form
// the normal equations AtA / Atb via cuBLAS.  Force features include the 3B
// term (the previous version omitted them, leaving 3B forces unconstrained and
// huge).  Column-major layout: A[(size_t)col*M + row].
// ===========================================================================

// Energy rows: one block per frame.  Builds the feature vector in shared memory
// then writes it (scaled by w_e) into the frame's row of A.
static __global__ void lstsq_energy_rows(
  int nf, int M, const int* __restrict__ fidx, const int* __restrict__ nat,
  const int* __restrict__ nat_tot, const int* __restrict__ off,
  const int* __restrict__ typ, const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
  int ncoeff, int nt, int nint, float kmin, float kd, float rc, int num_params_2b,
  int has_3b, int nc0, int nc1, int nc2, int ni0, int ni1, int ni2,
  float k0, float kd0, float r0, float k1, float kd1, float r1, float k2, float kd2, float r2,
  int nparam, int e0_off, int num_frames_total,
  const float* __restrict__ d_target, float lambda_e,
  const int* __restrict__ nn_off, const int* __restrict__ nn_lst, const int* __restrict__ nn_frame_off,
  double* __restrict__ A, double* __restrict__ bvec)
{
  extern __shared__ float s_basis[];
  int row = blockIdx.x; if (row >= nf) return;
  int tid = threadIdx.x, stride = blockDim.x;
  int fid = fidx[row], n = nat[fid], n_tot = nat_tot[fid], o = off[fid];
  for (int k = tid; k < nparam; k += stride) s_basis[k] = 0.0f;
  __syncthreads();

  for (int i = tid; i < n; i += stride) atomicAdd(&s_basis[e0_off + typ[o+i]], 1.0f);

  for (int i = tid; i < n; i += stride) {
    int ti = typ[o+i];
    for (int j = 0; j < n_tot; j++) {
      if (j == i) continue;
      int tj = typ[o+j];
      float dx = x[o+i]-x[o+j], dy = y[o+i]-y[o+j], dz = z[o+i]-z[o+j];
      float r = sqrtf(dx*dx+dy*dy+dz*dz); if (r >= rc) continue;
      int pi = ti*nt + tj;
      for (int c = 0; c < ncoeff; c++) { float bv=_bval(c,nint,r,kmin,kd); if(bv!=0) atomicAdd(&s_basis[pi*ncoeff+c], 0.5f*bv); }
    }
  }

  if (has_3b) {
    int nn_base = nn_frame_off[fid];
    for (int i = tid; i < n; i += stride) {
      int ti = typ[o+i]; int nn_start = nn_off[nn_base+i]; int nni = nn_off[nn_base+i+1]-nn_start;
      for (int jj = 0; jj < nni; jj++) {
        int j = nn_lst[nn_start+jj];
        float dx12=x[o+j]-x[o+i],dy12=y[o+j]-y[o+i],dz12=z[o+j]-z[o+i];
        float r12=sqrtf(dx12*dx12+dy12*dy12+dz12*dz12); if(r12>=r0)continue; int tj=typ[o+j];
        for (int kk=jj+1; kk<nni; kk++) {
          int k=nn_lst[nn_start+kk];
          float dx13=x[o+k]-x[o+i],dy13=y[o+k]-y[o+i],dz13=z[o+k]-z[o+i];
          float r13=sqrtf(dx13*dx13+dy13*dy13+dz13*dz13); if(r13>=r1)continue;
          float dx23=x[o+k]-x[o+j],dy23=y[o+k]-y[o+j],dz23=z[o+k]-z[o+j];
          float r23=sqrtf(dx23*dx23+dy23*dy23+dz23*dz23); if(r23>=r2)continue;
          int tk=typ[o+k], trip=(ti*nt+tj)*nt+tk; int off3=num_params_2b+trip*nc0*nc1*nc2;
          for(int p=0;p<nc0;p++){float bp=_bval(p,ni0,r12,k0,kd0);if(bp==0)continue;
          for(int q=0;q<nc1;q++){float bq=_bval(q,ni1,r13,k1,kd1);if(bq==0)continue;float bpbq=bp*bq;
          for(int r=0;r<nc2;r++){float br=_bval(r,ni2,r23,k2,kd2);if(br==0)continue;
          atomicAdd(&s_basis[off3+p+q*nc0+r*nc0*nc1], bpbq*br);}}}
        }
      }
    }
  }
  __syncthreads();

  float na=(float)n; if(na<1.0f) na=1.0f;
  double we = sqrt((double)lambda_e/(double)num_frames_total)/(double)na;
  for (int k = tid; k < nparam; k += stride) A[(size_t)k*M + row] = we * (double)s_basis[k];
  if (tid == 0) bvec[row] = we * (double)d_target[fid];
}

// Force rows: set the RHS (b) and the 2B + 3B force features.  One block per
// frame; threads stride over real centre atoms.  Force row index for real atom
// (frame f, local i, component d) = nf + 3*(realbase[f]+i) + d.
static __global__ void lstsq_force_rows(
  int nf, int M, const int* __restrict__ fidx, const int* __restrict__ nat,
  const int* __restrict__ nat_tot, const int* __restrict__ off, const int* __restrict__ realbase,
  const int* __restrict__ parent,
  const int* __restrict__ typ, const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
  int ncoeff, int nt, int nint, float kmin, float kd, float rc, int num_params_2b,
  int has_3b, int nc0, int nc1, int nc2, int ni0, int ni1, int ni2,
  float k0, float kd0, float r0, float k1, float kd1, float r1, float k2, float kd2, float r2,
  int nparam, float wf,
  const float* __restrict__ fxref, const float* __restrict__ fyref, const float* __restrict__ fzref,
  const int* __restrict__ nn_off, const int* __restrict__ nn_lst, const int* __restrict__ nn_frame_off,
  double* __restrict__ A, double* __restrict__ bvec)
{
  int blk = blockIdx.x; if (blk >= nf) return;
  int tid = threadIdx.x, stride = blockDim.x;
  int fid = fidx[blk], n = nat[fid], n_tot = nat_tot[fid], o = off[fid];
  int rb = realbase[blk];   // chunk-local real-atom base (block position, not global fid)

  // RHS + 2B force features
  for (int i = tid; i < n; i += stride) {
    int rowx = nf + 3*(rb+i) + 0, rowy = rowx+1, rowz = rowx+2;
    bvec[rowx] = (double)wf*(double)fxref[o+i];
    bvec[rowy] = (double)wf*(double)fyref[o+i];
    bvec[rowz] = (double)wf*(double)fzref[o+i];
    int ti = typ[o+i];
    for (int j = 0; j < n_tot; j++) {
      if (j == i) continue;
      int tj = typ[o+j];
      float dx=x[o+i]-x[o+j],dy=y[o+i]-y[o+j],dz=z[o+i]-z[o+j];
      float d2=dx*dx+dy*dy+dz*dz,invr=rsqrtf(d2),r=d2*invr; if(r>=rc)continue;
      int pi=ti*nt+tj;
      for (int c=0;c<ncoeff;c++){ float db=_dbval(c,nint,r,kmin,kd); if(db==0)continue;
        // F_i = -phi'(r)*rinv*dx ;  per-coeff feature = -db*rinv*dx
        double f = -(double)db*(double)invr*(double)wf; int col=pi*ncoeff+c;
        atomicAdd(&A[(size_t)col*M+rowx], f*(double)dx);
        atomicAdd(&A[(size_t)col*M+rowy], f*(double)dy);
        atomicAdd(&A[(size_t)col*M+rowz], f*(double)dz);
      }
    }
  }

  // 3B force features (per-centre triplets; scatter to i, parent_j, parent_k).
  if (has_3b) {
    int nn_base = nn_frame_off[fid];
    for (int i = tid; i < n; i += stride) {
      int ti=typ[o+i]; int nn_start=nn_off[nn_base+i]; int nni=nn_off[nn_base+i+1]-nn_start;
      int rowi = nf + 3*(rb+i);
      for (int jj=0; jj<nni; jj++) {
        int j=nn_lst[nn_start+jj];
        float dx12=x[o+j]-x[o+i],dy12=y[o+j]-y[o+i],dz12=z[o+j]-z[o+i];
        float d12=dx12*dx12+dy12*dy12+dz12*dz12,inv12=rsqrtf(d12),r12=d12*inv12; if(r12>=r0)continue;
        int tj=typ[o+j]; int pj=parent[o+j]; int rowj=nf+3*(rb+pj);
        int m0=(int)((r12-k0)/kd0); if(m0<0)m0=0; if(m0>=ni0)m0=ni0-1; float u0=(r12-(k0+m0*kd0))/kd0;
        float b0[4],db0[4]; { float u=u0,u2=u*u,u3=u2*u; b0[0]=(1-3*u+3*u2-u3)/6;b0[1]=(4-6*u2+3*u3)/6;b0[2]=(1+3*u+3*u2-3*u3)/6;b0[3]=u3/6;
          float om=1-u; db0[0]=-om*om*0.5f/kd0; db0[1]=u*(3*u-4)*0.5f/kd0; db0[2]=(-3*u2+2*u+1)*0.5f/kd0; db0[3]=u2*0.5f/kd0; }
        int p0=m0-3; if(p0<0)p0=0;
        for (int kk=jj+1; kk<nni; kk++) {
          int k=nn_lst[nn_start+kk];
          float dx13=x[o+k]-x[o+i],dy13=y[o+k]-y[o+i],dz13=z[o+k]-z[o+i];
          float d13=dx13*dx13+dy13*dy13+dz13*dz13,inv13=rsqrtf(d13),r13=d13*inv13; if(r13>=r1)continue;
          float dx23=x[o+k]-x[o+j],dy23=y[o+k]-y[o+j],dz23=z[o+k]-z[o+j];
          float d23=dx23*dx23+dy23*dy23+dz23*dz23,inv23=rsqrtf(d23),r23=d23*inv23; if(r23>=r2)continue;
          int tk=typ[o+k]; int pk=parent[o+k]; int rowk=nf+3*(rb+pk);
          int trip=(ti*nt+tj)*nt+tk; int off3=num_params_2b+trip*nc0*nc1*nc2;
          int m1=(int)((r13-k1)/kd1); if(m1<0)m1=0; if(m1>=ni1)m1=ni1-1; float u1=(r13-(k1+m1*kd1))/kd1;
          int m2=(int)((r23-k2)/kd2); if(m2<0)m2=0; if(m2>=ni2)m2=ni2-1; float u2=(r23-(k2+m2*kd2))/kd2;
          float b1[4],db1[4],b2[4],db2[4];
          { float u=u1,uu=u*u,uuu=uu*u; b1[0]=(1-3*u+3*uu-uuu)/6;b1[1]=(4-6*uu+3*uuu)/6;b1[2]=(1+3*u+3*uu-3*uuu)/6;b1[3]=uuu/6;
            float om=1-u; db1[0]=-om*om*0.5f/kd1; db1[1]=u*(3*u-4)*0.5f/kd1; db1[2]=(-3*uu+2*u+1)*0.5f/kd1; db1[3]=uu*0.5f/kd1; }
          { float u=u2,uu=u*u,uuu=uu*u; b2[0]=(1-3*u+3*uu-uuu)/6;b2[1]=(4-6*uu+3*uuu)/6;b2[2]=(1+3*u+3*uu-3*uuu)/6;b2[3]=uuu/6;
            float om=1-u; db2[0]=-om*om*0.5f/kd2; db2[1]=u*(3*u-4)*0.5f/kd2; db2[2]=(-3*uu+2*u+1)*0.5f/kd2; db2[3]=uu*0.5f/kd2; }
          int p1=m1-3; if(p1<0)p1=0; int p2=m2-3; if(p2<0)p2=0;
          for(int dp=0;dp<4;dp++){int p=p0+dp; if(p>=nc0)break; float Bp=b0[dp],dBp=db0[dp];
          for(int dq=0;dq<4;dq++){int q=p1+dq; if(q>=nc1)break; float Bq=b1[dq],dBq=db1[dq];
          for(int dr=0;dr<4;dr++){int rr=p2+dr; if(rr>=nc2)break;
            int col=off3+p+q*nc0+rr*nc0*nc1;
            float G12=dBp*Bq*b2[dr];          // d(dE/dr12)/dC
            float G13=Bp*dBq*b2[dr];
            float G23=Bp*Bq*db2[dr];
            // F_i = G12*inv12*dx12 + G13*inv13*dx13 (per coeff), scaled by wf.
            double fix=(double)wf*((double)G12*inv12*dx12+(double)G13*inv13*dx13);
            double fiy=(double)wf*((double)G12*inv12*dy12+(double)G13*inv13*dy13);
            double fiz=(double)wf*((double)G12*inv12*dz12+(double)G13*inv13*dz13);
            atomicAdd(&A[(size_t)col*M+rowi+0],fix); atomicAdd(&A[(size_t)col*M+rowi+1],fiy); atomicAdd(&A[(size_t)col*M+rowi+2],fiz);
            double fjx=(double)wf*(-(double)G12*inv12*dx12+(double)G23*inv23*dx23);
            double fjy=(double)wf*(-(double)G12*inv12*dy12+(double)G23*inv23*dy23);
            double fjz=(double)wf*(-(double)G12*inv12*dz12+(double)G23*inv23*dz23);
            atomicAdd(&A[(size_t)col*M+rowj+0],fjx); atomicAdd(&A[(size_t)col*M+rowj+1],fjy); atomicAdd(&A[(size_t)col*M+rowj+2],fjz);
            double fkx=(double)wf*(-(double)G13*inv13*dx13-(double)G23*inv23*dx23);
            double fky=(double)wf*(-(double)G13*inv13*dy13-(double)G23*inv23*dy23);
            double fkz=(double)wf*(-(double)G13*inv13*dz13-(double)G23*inv23*dz23);
            atomicAdd(&A[(size_t)col*M+rowk+0],fkx); atomicAdd(&A[(size_t)col*M+rowk+1],fky); atomicAdd(&A[(size_t)col*M+rowk+2],fkz);
          }}}
        }
      }
    }
  }
}

// ---- Curvature (second-difference) regularization --------------------------
// Adds lambda * (D2 c)^T (D2 c) penalties to the normal matrix, where D2 is the
// discrete second-difference operator along a coefficient sequence.  This is
// what keeps the under-constrained 3B spline grid smooth and physical — without
// it the energy-only 3B fit produces wildly oscillating coefficients and absurd
// forces.  Indices id[0..2] are the three stencil positions in the global
// parameter vector; the stencil weights are (1, -2, 1).
static inline void add_curvature_stencil(std::vector<double>& ATA, int nparam,
                                         int i0, int i1, int i2, double lam)
{
  const int idx[3] = {i0, i1, i2};
  const double w[3] = {1.0, -2.0, 1.0};
  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 3; b++)
      ATA[(size_t)idx[a] * nparam + idx[b]] += lam * w[a] * w[b];
}

// 2B: penalize curvature along each pair's 1D coefficient sequence.
// 3B: penalize curvature along each of the three grid axes independently.
static void add_curvature_regularization(
  std::vector<double>& ATA, int nparam,
  int npairs, int ncoeff,
  bool has_3b, int num_params_2b, int num_trips, int nc0, int nc1, int nc2,
  double lam2b, double lam3b)
{
  if (lam2b > 0.0) {
    for (int p = 0; p < npairs; p++) {
      int base = p * ncoeff;
      for (int c = 1; c < ncoeff - 1; c++)
        add_curvature_stencil(ATA, nparam, base + c - 1, base + c, base + c + 1, lam2b);
    }
  }
  if (has_3b && lam3b > 0.0) {
    int gsz = nc0 * nc1 * nc2;
    for (int t = 0; t < num_trips; t++) {
      int base = num_params_2b + t * gsz;
      auto gidx = [&](int a, int b, int c) { return base + a + b * nc0 + c * nc0 * nc1; };
      for (int c = 0; c < nc2; c++)
        for (int b = 0; b < nc1; b++)
          for (int a = 1; a < nc0 - 1; a++)
            add_curvature_stencil(ATA, nparam, gidx(a-1,b,c), gidx(a,b,c), gidx(a+1,b,c), lam3b);
      for (int c = 0; c < nc2; c++)
        for (int a = 0; a < nc0; a++)
          for (int b = 1; b < nc1 - 1; b++)
            add_curvature_stencil(ATA, nparam, gidx(a,b-1,c), gidx(a,b,c), gidx(a,b+1,c), lam3b);
      for (int b = 0; b < nc1; b++)
        for (int a = 0; a < nc0; a++)
          for (int c = 1; c < nc2 - 1; c++)
            add_curvature_stencil(ATA, nparam, gidx(a,b,c-1), gidx(a,b,c), gidx(a,b,c+1), lam3b);
    }
  }
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
    int cap = stage.batch;
    use_frames = std::min(ds.num_frames, cap);
  }

  auto t0 = std::chrono::high_resolution_clock::now();

  // e0 (1-body) columns live at the end of the parameter vector.
  int e0_off = nparam - nt;

  // Global totals (for consistent loss-matching weights across chunks).
  int total_real = 0;
  for (int i = 0; i < use_frames; i++) total_real += ds.h_natoms[i];
  if (total_real < 1) total_real = 1;
  float lambda_e = (float)para.lambda_e, lambda_f = (float)para.lambda_f;
  float wf = (lambda_f > 0.0f) ? sqrtf(lambda_f / (3.0f * (float)total_real)) : 0.0f;

  // The full design matrix A (M x nparam) can exceed GPU memory for big datasets
  // with 3B.  Build it in row-chunks of frames and accumulate the normal
  // equations AtA / Atb with cuBLAS (beta=1) — mathematically identical to a
  // single solve, but bounded memory (O(maxM x nparam) instead of O(M x nparam)).
  const int BLK = 64;
  size_t smem = nparam * sizeof(float);
  GPU_Vector<double> d_ATA((size_t)nparam * nparam), d_ATb(nparam);
  cudaMemset(d_ATA.data(), 0, (size_t)nparam * nparam * sizeof(double));
  cudaMemset(d_ATb.data(), 0, nparam * sizeof(double));
  cublasHandle_t cb; cublasCreate(&cb);
  double one = 1.0;

  // Row budget per chunk (cap the A buffer ~3 GB).
  size_t budget = (size_t)3ull << 30;
  long long maxM = (long long)(budget / ((size_t)nparam * sizeof(double)));
  if (maxM < 4096) maxM = 4096;
  if (const char* e = getenv("UF3_MAXM")) maxM = atoll(e);  // testing: force chunking

  GPU_Vector<double> d_A, d_b;       // reused across chunks (grown as needed)
  GPU_Vector<int> d_cbidx, d_crealbase;
  int n_chunks = 0;
  for (int f0 = 0; f0 < use_frames; ) {
    // Greedily grow a chunk until its row count would exceed maxM.
    std::vector<int> cb_idx, cb_realbase;
    long long Mc = 0; int creal = 0; int f1 = f0;
    while (f1 < use_frames) {
      int na = ds.h_natoms[f1];
      long long add = 1 + 3LL * na;
      if (Mc + add > maxM && f1 > f0) break;
      cb_idx.push_back(f1); cb_realbase.push_back(creal);
      creal += na; Mc += add; f1++;
    }
    int ncf = f1 - f0;
    long long Mchunk = (long long)ncf + 3LL * creal;
    n_chunks++;

    if ((int)d_cbidx.size() < ncf) { d_cbidx.resize(ncf); d_crealbase.resize(ncf); }
    d_cbidx.copy_from_host(cb_idx.data());
    d_crealbase.copy_from_host(cb_realbase.data());
    if ((long long)d_A.size() < Mchunk * nparam) d_A.resize((size_t)Mchunk * nparam);
    if ((long long)d_b.size() < Mchunk) d_b.resize((size_t)Mchunk);
    cudaMemset(d_A.data(), 0, (size_t)Mchunk * nparam * sizeof(double));
    cudaMemset(d_b.data(), 0, (size_t)Mchunk * sizeof(double));

    lstsq_energy_rows<<<ncf, BLK, smem>>>(
      ncf, (int)Mchunk, d_cbidx.data(), ds.d_natoms.data(), ds.d_natoms_tot.data(), ds.d_offsets.data(),
      ds.d_types.data(), ds.d_x.data(), ds.d_y.data(), ds.d_z.data(),
      ncoeff, nt, nint, kmin, kd, rc, num_params_2b,
      has_3b ? 1 : 0, nc3[0], nc3[1], nc3[2], ni3[0], ni3[1], ni3[2],
      k3[0], kd3[0], r3[0], k3[1], kd3[1], r3[1], k3[2], kd3[2], r3[2],
      nparam, e0_off, use_frames, ds.d_energy_ref.data(), lambda_e,
      ds.d_nn_off.data(), ds.d_nn_lst.data(), ds.d_nn_frame_off.data(),
      d_A.data(), d_b.data());
    GPU_CHECK_KERNEL

    if (lambda_f > 0.0f) {
      lstsq_force_rows<<<ncf, BLK>>>(
        ncf, (int)Mchunk, d_cbidx.data(), ds.d_natoms.data(), ds.d_natoms_tot.data(),
        ds.d_offsets.data(), d_crealbase.data(), ds.d_parent.data(),
        ds.d_types.data(), ds.d_x.data(), ds.d_y.data(), ds.d_z.data(),
        ncoeff, nt, nint, kmin, kd, rc, num_params_2b,
        has_3b ? 1 : 0, nc3[0], nc3[1], nc3[2], ni3[0], ni3[1], ni3[2],
        k3[0], kd3[0], r3[0], k3[1], kd3[1], r3[1], k3[2], kd3[2], r3[2],
        nparam, wf, ds.d_fx_ref.data(), ds.d_fy_ref.data(), ds.d_fz_ref.data(),
        ds.d_nn_off.data(), ds.d_nn_lst.data(), ds.d_nn_frame_off.data(),
        d_A.data(), d_b.data());
      GPU_CHECK_KERNEL
    }

    // Accumulate normal equations: AtA += A_chunk^T A_chunk, Atb += A_chunk^T b.
    cublasDsyrk(cb, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, nparam, (int)Mchunk,
                &one, d_A.data(), (int)Mchunk, &one, d_ATA.data(), nparam);
    cublasDgemv(cb, CUBLAS_OP_T, (int)Mchunk, nparam, &one, d_A.data(), (int)Mchunk,
                d_b.data(), 1, &one, d_ATb.data(), 1);
    f0 = f1;
  }
  cublasDestroy(cb);

  std::vector<double> ATA((size_t)nparam * nparam), ATb(nparam);
  cudaMemcpy(ATA.data(), d_ATA.data(), (size_t)nparam*nparam*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(ATb.data(), d_ATb.data(), nparam*sizeof(double), cudaMemcpyDeviceToHost);
  // Valid data is in the host lower triangle — mirror it to the upper triangle.
  for (int i = 0; i < nparam; i++)
    for (int j = i + 1; j < nparam; j++)
      ATA[(size_t)i*nparam + j] = ATA[(size_t)j*nparam + i];

  // Tikhonov ridge.  Two components: (1) a relative term so well-constrained
  // columns are barely perturbed, and (2) a per-column floor proportional to
  // the mean diagonal so nearly-unconstrained 3B columns (diagonal ~0, e.g.
  // triplets/basis that never activate) stay positive-definite and solve to ~0.
  double diag_mean = 0.0;
  for (int k = 0; k < nparam; k++) diag_mean += ATA[k*nparam + k];
  diag_mean /= std::max(1, nparam);
  double ridge_rel = 1e-8;
  double ridge_floor = 1e-6 * (diag_mean > 0.0 ? diag_mean : 1.0);
  for (int k = 0; k < nparam; k++) {
    double dk = ATA[k*nparam + k];
    ATA[k*nparam + k] = dk + ridge_rel * dk + ridge_floor;
  }

  // Curvature regularization (scaled relative to the mean diagonal so it is
  // invariant to the absolute weighting scale).  3B needs much stronger
  // smoothing than 2B since the energy-only 3B fit is heavily under-constrained.
  double dm = (diag_mean > 0.0 ? diag_mean : 1.0);
  double lam2b = 1e-4 * dm;
  double lam3b = 1e-3 * dm;
  if (const char* e = getenv("UF3_C2")) lam2b = atof(e) * dm;
  if (const char* e = getenv("UF3_C3")) lam3b = atof(e) * dm;
  add_curvature_regularization(ATA, nparam, npairs, ncoeff,
                               has_3b, num_params_2b, fitness.model()->num_triplets(),
                               nc3[0], nc3[1], nc3[2], lam2b, lam3b);

  // Constrain frozen (edge) coefficients to 0: decouple their row/column so the
  // free coefficients are solved as if the frozen ones do not contribute, and
  // the frozen ones solve to exactly 0.  Gives smooth spline cutoffs.
  {
    const std::vector<char>& fr = fitness.model()->frozen();
    for (int k = 0; k < nparam; k++) {
      if (!fr[k]) continue;
      for (int j = 0; j < nparam; j++) { ATA[(size_t)k*nparam + j] = 0.0; ATA[(size_t)j*nparam + k] = 0.0; }
      ATA[(size_t)k*nparam + k] = (ridge_floor > 0.0 ? ridge_floor : 1.0);
      ATb[k] = 0.0;
    }
  }

  bool ok = cholesky(ATA, nparam);   // in-place Cholesky in double
  if (!ok) printf("  lstsq: ATA not PD (after ridge)\n");
  std::vector<double> xd; solve_cholesky(ATA, nparam, ATb, xd);
  std::vector<float> x(nparam);
  for (int i = 0; i < nparam; i++) x[i] = (float)xd[i];
  fitness.model()->set_parameters(x.data());

  auto t1 = std::chrono::high_resolution_clock::now();
  float dt = (float)std::chrono::duration<double>(t1 - t0).count();
  printf("  lstsq GPU: %d params%s, %d frames, %.2f s\n",
         nparam, has_3b?" (2B+3B)":" (2B)", use_frames, dt);
  // local_iter=1: lstsq runs once; triggers the local_iter==1 logging checkpoint.
  float tl = fitness.compute_loss(0, gen_offset, stage_id, 1, dt);
  printf("  Loss=%.3f [E=%.3f F=%.3f eV/A]\n", tl, fitness.loss_e, fitness.loss_f);
}
