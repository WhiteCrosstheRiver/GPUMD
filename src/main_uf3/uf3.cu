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

#include "uf3.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>

// ---- GPU helpers ----------------------------------------------------------
__device__ inline float uf3_eval_cubic(float4 c, float u) {
  return c.x + u * (c.y + u * (c.z + u * c.w));
}
__device__ inline int uf3_find_interval(float r, float kmin, float kdelta, int nint) {
  int i = (int)((r - kmin) / kdelta);
  if (i < 0) i = 0; if (i >= nint) i = nint - 1;
  return i;
}

// ---- 2B kernel (multi-threaded: each thread handles some atoms) -----------
static __global__ void uf3_eval_2b(
  int nf, const int* __restrict__ fidx, const int* __restrict__ nat,
  const int* __restrict__ off, const int* __restrict__ typ,
  const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
  const float4* __restrict__ coeff, int np, int nint, float kmin, float kd, float rc,
  const int* __restrict__ tmap, int nt, float* __restrict__ ene)
{
  int b = blockIdx.x; if (b >= nf) return;
  int tid = threadIdx.x, stride = blockDim.x;
  int fid = fidx[b], n = nat[fid], o = off[fid];
  float pe = 0;
  for (int i = tid; i < n; i += stride) {
    for (int j = i+1; j < n; j++) {
      float dx = x[o+i]-x[o+j], dy = y[o+i]-y[o+j], dz = z[o+i]-z[o+j];
      float r = sqrtf(dx*dx+dy*dy+dz*dz); if (r >= rc) continue;
      int m = uf3_find_interval(r, kmin, kd, nint);
      float u = (r - (kmin + m*kd)) / kd;
      float4 c = __ldg(&coeff[tmap[typ[o+i]*nt+typ[o+j]] * nint + m]);
      pe += uf3_eval_cubic(c, u);
    }
  }
  // Warp/block reduction
  __shared__ float s_pe[64];
  s_pe[tid] = pe; __syncthreads();
  for (int s = stride/2; s > 0; s >>= 1) {
    if (tid < s) s_pe[tid] += s_pe[tid + s];
    __syncthreads();
  }
  if (tid == 0) ene[b] = s_pe[0];
}

// ---- 2B force kernel (multi-threaded) -----------------------------------
static __global__ void uf3_eval_2b_force(
  int nf, const int* __restrict__ fidx, const int* __restrict__ nat,
  const int* __restrict__ off, const int* __restrict__ typ,
  const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
  const float4* __restrict__ coeff, int np, int nint, float kmin, float kd, float rc,
  const int* __restrict__ tmap, int nt,
  float* __restrict__ fx, float* __restrict__ fy, float* __restrict__ fz)
{
  int b = blockIdx.x; if (b >= nf) return;
  int tid = threadIdx.x, stride = blockDim.x;
  int fid = fidx[b], n = nat[fid], o = off[fid];

  for (int i = tid; i < n; i += stride) {
    float fxi = 0, fyi = 0, fzi = 0;
    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      float dx = x[o+i]-x[o+j], dy = y[o+i]-y[o+j], dz = z[o+i]-z[o+j];
      float r = sqrtf(dx*dx+dy*dy+dz*dz); if (r >= rc) continue;
      int m = uf3_find_interval(r, kmin, kd, nint);
      float u = (r - (kmin + m*kd)) / kd;
      float4 c = __ldg(&coeff[tmap[typ[o+i]*nt+typ[o+j]] * nint + m]);
      float deriv = (c.y + u * (2.0f*c.z + u * 3.0f*c.w)) / kd;
      float f = deriv / r;
      fxi += f * dx; fyi += f * dy; fzi += f * dz;
    }
    fx[o+i] = fxi; fy[o+i] = fyi; fz[o+i] = fzi;
  }
}

// ---- 3B kernel (neighbor-list-based, multi-threaded) ---------------------
static __global__ void uf3_eval_3b(
  int nf, const int* __restrict__ fidx, const int* __restrict__ nat,
  const int* __restrict__ off, const int* __restrict__ typ,
  const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
  const float* __restrict__ tensor, int nc0,int nc1,int nc2,
  const float4* __restrict__ b0, int ni0, const float4* __restrict__ b1, int ni1,
  const float4* __restrict__ b2, int ni2,
  float km0,float kd0,float rc0, float km1,float kd1,float rc1,
  float km2,float kd2,float rc2, const int* __restrict__ tmap, int ntr,int nt,
  const int* __restrict__ nn_off, const int* __restrict__ nn_lst,
  const int* __restrict__ nn_frame_off,
  float* __restrict__ ene)
{
  int b = blockIdx.x; if (b >= nf) return;
  int tid = threadIdx.x, stride = blockDim.x;
  int fid = fidx[b], n = nat[fid], o = off[fid];
  float pe = 0;

  // nn_frame_off[fid] = start index in nn_off for this frame's atoms
  int nn_base = nn_frame_off[fid];

  for (int i = tid; i < n; i += stride) {
    int ti = typ[o+i];
    int nni = nn_off[nn_base + i + 1] - nn_off[nn_base + i];
    int nn_start = nn_off[nn_base + i];

    // Iterate over all pairs (j,k) from i's neighbor list
    for (int jj = 0; jj < nni; jj++) {
      int j = nn_lst[nn_start + jj];
      if (j <= i) continue; // ensure unique triplets: i < j < k
      float dx12=x[o+j]-x[o+i], dy12=y[o+j]-y[o+i], dz12=z[o+j]-z[o+i];
      float r12=sqrtf(dx12*dx12+dy12*dy12+dz12*dz12); if (r12>=rc0) continue;
      int tj=typ[o+j];

      for (int kk = jj+1; kk < nni; kk++) {
        int k = nn_lst[nn_start + kk];
        if (k <= j) continue;
        float dx13=x[o+k]-x[o+i], dy13=y[o+k]-y[o+i], dz13=z[o+k]-z[o+i];
        float r13=sqrtf(dx13*dx13+dy13*dy13+dz13*dz13); if (r13>=rc1) continue;
        float dx23=x[o+k]-x[o+j], dy23=y[o+k]-y[o+j], dz23=z[o+k]-z[o+j];
        float r23=sqrtf(dx23*dx23+dy23*dy23+dz23*dz23); if (r23>=rc2) continue;
        int tk=typ[o+k];
        int m0=uf3_find_interval(r12,km0,kd0,ni0), m1=uf3_find_interval(r13,km1,kd1,ni1), m2=uf3_find_interval(r23,km2,kd2,ni2);
        float u0=(r12-(km0+m0*kd0))/kd0, u1=(r13-(km1+m1*kd1))/kd1, u2=(r23-(km2+m2*kd2))/kd2;
        float vb0[4],vb1[4],vb2[4];
        for(int p=0;p<4;p++){vb0[p]=uf3_eval_cubic(__ldg(&b0[m0*4+p]),u0);}
        for(int p=0;p<4;p++){vb1[p]=uf3_eval_cubic(__ldg(&b1[m1*4+p]),u1);}
        for(int p=0;p<4;p++){vb2[p]=uf3_eval_cubic(__ldg(&b2[m2*4+p]),u2);}
        int p0=m0-3;if(p0<0)p0=0;int p1=m1-3;if(p1<0)p1=0;int p2=m2-3;if(p2<0)p2=0;
        const float* C=&tensor[tmap[(ti*nt+tj)*nt+tk]*nc0*nc1*nc2];
        for(int dp=0;dp<4;dp++){int q=p0+dp;if(q>=nc0)continue;float bp=vb0[dp];
        for(int dq=0;dq<4;dq++){int r=p1+dq;if(r>=nc1)continue;float bq=vb1[dq];
        for(int dr=0;dr<4;dr++){int s=p2+dr;if(s>=nc2)continue;
        pe+=C[q+r*nc0+s*nc0*nc1]*bp*bq*vb2[dr];}}}
      }
    }
  }
  // Block reduction
  __shared__ float s_pe[64];
  s_pe[tid] = pe; __syncthreads();
  for (int s = stride/2; s > 0; s >>= 1) {
    if (tid < s) s_pe[tid] += s_pe[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(&ene[b], s_pe[0]);
}

// ---- 3B force kernel -----------------------------------------------------
static __global__ void uf3_eval_3b_force(
  int nf, const int* __restrict__ fidx, const int* __restrict__ nat,
  const int* __restrict__ off, const int* __restrict__ typ,
  const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
  const float* __restrict__ tensor, int nc0,int nc1,int nc2,
  const float4* __restrict__ b0, int ni0, const float4* __restrict__ b1, int ni1,
  const float4* __restrict__ b2, int ni2,
  float km0,float kd0,float rc0, float km1,float kd1,float rc1,
  float km2,float kd2,float rc2, const int* __restrict__ tmap, int ntr,int nt,
  const int* __restrict__ nn_off, const int* __restrict__ nn_lst,
  const int* __restrict__ nn_frame_off,
  float* __restrict__ fx, float* __restrict__ fy, float* __restrict__ fz)
{
  int b = blockIdx.x; if (b >= nf) return;
  int tid = threadIdx.x, stride = blockDim.x;
  int fid = fidx[b], n = nat[fid], o = off[fid];
  int nn_base = nn_frame_off[fid];

  for (int i = tid; i < n; i += stride) {
    int nni = nn_off[nn_base + i + 1] - nn_off[nn_base + i];
    int nn_start = nn_off[nn_base + i];
    int ti = typ[o+i];

    for (int jj = 0; jj < nni; jj++) {
      int j = nn_lst[nn_start + jj];
      if (j <= i) continue;
      float dx12=x[o+j]-x[o+i], dy12=y[o+j]-y[o+i], dz12=z[o+j]-z[o+i];
      float r12=sqrtf(dx12*dx12+dy12*dy12+dz12*dz12); if (r12>=rc0) continue;
      int tj=typ[o+j];
      float inv12 = 1.0f/r12;

      for (int kk = jj+1; kk < nni; kk++) {
        int k = nn_lst[nn_start + kk];
        if (k <= j) continue;
        float dx13=x[o+k]-x[o+i], dy13=y[o+k]-y[o+i], dz13=z[o+k]-z[o+i];
        float r13=sqrtf(dx13*dx13+dy13*dy13+dz13*dz13); if (r13>=rc1) continue;
        float dx23=x[o+k]-x[o+j], dy23=y[o+k]-y[o+j], dz23=z[o+k]-z[o+j];
        float r23=sqrtf(dx23*dx23+dy23*dy23+dz23*dz23); if (r23>=rc2) continue;
        int tk=typ[o+k];
        float inv13=1.0f/r13, inv23=1.0f/r23;

        int m0=uf3_find_interval(r12,km0,kd0,ni0), m1=uf3_find_interval(r13,km1,kd1,ni1), m2=uf3_find_interval(r23,km2,kd2,ni2);
        float u0=(r12-(km0+m0*kd0))/kd0, u1=(r13-(km1+m1*kd1))/kd1, u2=(r23-(km2+m2*kd2))/kd2;

        // Basis values for energy
        float vb0[4],vb1[4],vb2[4];
        // Derivative basis for each dimension
        float db0[4],db1[4],db2[4];
        for(int p=0;p<4;p++){
          float4 cb = __ldg(&b0[m0*4+p]); vb0[p]=uf3_eval_cubic(cb,u0);
          db0[p] = (cb.y + u0*(2.0f*cb.z + u0*3.0f*cb.w)) / kd0;
        }
        for(int p=0;p<4;p++){
          float4 cb = __ldg(&b1[m1*4+p]); vb1[p]=uf3_eval_cubic(cb,u1);
          db1[p] = (cb.y + u1*(2.0f*cb.z + u1*3.0f*cb.w)) / kd1;
        }
        for(int p=0;p<4;p++){
          float4 cb = __ldg(&b2[m2*4+p]); vb2[p]=uf3_eval_cubic(cb,u2);
          db2[p] = (cb.y + u2*(2.0f*cb.z + u2*3.0f*cb.w)) / kd2;
        }

        int p0=m0-3;if(p0<0)p0=0; int p1=m1-3;if(p1<0)p1=0; int p2=m2-3;if(p2<0)p2=0;
        const float* C=&tensor[tmap[(ti*nt+tj)*nt+tk]*nc0*nc1*nc2];

        // Tensor contractions: dV/dr12, dV/dr13, dV/dr23
        float dv12=0, dv13=0, dv23=0;
        for(int dp=0;dp<4;dp++){int q=p0+dp;if(q>=nc0)continue;
        for(int dq=0;dq<4;dq++){int r=p1+dq;if(r>=nc1)continue;
        for(int dr=0;dr<4;dr++){int s=p2+dr;if(s>=nc2)continue;
          float Cv = C[q+r*nc0+s*nc0*nc1];
          dv12 += Cv * db0[dp] * vb1[dq] * vb2[dr];
          dv13 += Cv * vb0[dp] * db1[dq] * vb2[dr];
          dv23 += Cv * vb0[dp] * vb1[dq] * db2[dr];
        }}}

        // Convert to Cartesian forces (negative gradient convention)
        float f12 = -dv12 * inv12, f13 = -dv13 * inv13, f23 = -dv23 * inv23;

        // Force on i: -dV/dri = -(d12*r̂12 + d13*r̂13) where d12 = dV/dr12
        float fix = -(dv12*inv12*dx12 + dv13*inv13*dx13);
        float fiy = -(dv12*inv12*dy12 + dv13*inv13*dy13);
        float fiz = -(dv12*inv12*dz12 + dv13*inv13*dz13);
        atomicAdd(&fx[o+i], fix); atomicAdd(&fy[o+i], fiy); atomicAdd(&fz[o+i], fiz);

        // Force on j: -dV/drj = d12*r̂12 - d23*r̂23
        float fjx = dv12*inv12*dx12 - dv23*inv23*dx23;
        float fjy = dv12*inv12*dy12 - dv23*inv23*dy23;
        float fjz = dv12*inv12*dz12 - dv23*inv23*dz23;
        atomicAdd(&fx[o+j], fjx); atomicAdd(&fy[o+j], fjy); atomicAdd(&fz[o+j], fjz);

        // Force on k: -dV/drk = d13*r̂13 + d23*r̂23
        float fkx = dv13*inv13*dx13 + dv23*inv23*dx23;
        float fky = dv13*inv13*dy13 + dv23*inv23*dy23;
        float fkz = dv13*inv13*dz13 + dv23*inv23*dz23;
        atomicAdd(&fx[o+k], fkx); atomicAdd(&fy[o+k], fky); atomicAdd(&fz[o+k], fkz);
      }
    }
  }
}

// ---- Analytical gradient kernel (2B energy) ------------------------------
// For each frame: accumulates B_k(r_ij) values per coefficient, then
// gradient[k] += (E_pred - E_ref) * Σ B_k.  One block per coefficient.
static __global__ void uf3_grad_2b(
  int nf, int nparam,
  const int* __restrict__ fidx, const int* __restrict__ nat, const int* __restrict__ off,
  const int* __restrict__ typ,
  const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
  int np, int ncoeff, int nint, float kmin, float kd, float rc,
  const int* __restrict__ tmap, int nt,
  const float* __restrict__ energy_diff, // [nf] per-atom-normalized energy residual
  float scale_e,
  float* __restrict__ gradient)          // [nparam] output
{
  int k = blockIdx.x; if (k >= nparam) return;            // which coefficient
  int tid = threadIdx.x, stride = blockDim.x;             // threads per coefficient over frames
  int pair_idx = k / ncoeff;                              // which type pair
  int coeff_idx = k % ncoeff;                             // which coefficient within pair
  float grad = 0;

  for (int b = tid; b < nf; b += stride) {
    int fid = fidx[b], n = nat[fid], o = off[fid];
    float delta = energy_diff[b];
    float basis_sum = 0;
    for (int i = 0; i < n; i++) {
      int ti = typ[o+i];
      for (int j = i+1; j < n; j++) {
        int tj = typ[o+j];
        if (tmap[ti*nt+tj] != pair_idx) continue;
        float dx = x[o+i]-x[o+j], dy = y[o+i]-y[o+j], dz = z[o+i]-z[o+j];
        float r = sqrtf(dx*dx+dy*dy+dz*dz); if (r>=rc) continue;
        int m = uf3_find_interval(r, kmin, kd, nint);
        float u = (r - (kmin + m*kd)) / kd;
        // The basis value for coefficient coeff_idx at interval m:
        // V(u) = A+Bu+Cu^2+Du^3 where (A,B,C,D) are from the combined cubic
        // But we need the CONTRIBUTION of coefficient coeff_idx to V(u).
        // For coefficient coeff_idx at interval m, the contribution is:
        // coeff[coeff_idx] * basis_function_k(u)
        // The 4 active basis functions at interval m correspond to coeffs [m-3,m-2,m-1,m]
        int c_active = m - 3 + coeff_idx - 0; // which of the 4 active coefficients
        // Actually: active coeffs are m-3, m-2, m-1, m. So coeff m-3 maps to basis 0, m-2 to 1, etc.
        // For coeff_idx=0: active when m-3 <= idx <= m, i.e., c_active = idx - (m-3) = coeff_idx - m + 3
        int c_rel = coeff_idx - m + 3;
        if (c_rel >= 0 && c_rel < 4) {
          float basis_val = 0;
          float bu = u;
          if (c_rel == 0) basis_val = (1.0f/6) * (1-bu)*(1-bu)*(1-bu);  // B0 = (1-u)^3/6
          else if (c_rel == 1) basis_val = (3*bu*bu*bu - 6*bu*bu + 4) / 6;
          else if (c_rel == 2) basis_val = (-3*bu*bu*bu + 3*bu*bu + 3*bu + 1) / 6;
          else basis_val = bu*bu*bu / 6;
          basis_sum += basis_val;
        }
      }
    }
    grad += delta * basis_sum;
  }

  // Block reduction
  __shared__ float s_grad[64];
  s_grad[tid] = grad; __syncthreads();
  for (int s = stride/2; s > 0; s >>= 1) { if (tid < s) s_grad[tid] += s_grad[tid+s]; __syncthreads(); }
  if (tid == 0) {
    gradient[k] += scale_e * s_grad[0];
  }
}

// ---- pre-computation -----------------------------------------------------
static void precompute_2b(const std::vector<float>& cf, std::vector<float4>& out) {
  int nc=(int)cf.size(), ni=nc+3; out.resize(ni);
  for(int m=0;m<ni;m++){
    int i0=m-3,i1=m-2,i2=m-1,i3=m;
    if(i0<0)i0=0;if(i1<0)i1=0;if(i2<0)i2=0;if(i2>=nc)i2=nc-1;if(i3>=nc)i3=nc-1;
    float c0=cf[i0],c1=cf[i1],c2=cf[i2],c3=cf[i3];
    out[m]=make_float4((c0+4*c1+c2)/6,(-3*c0+3*c2)/6,(3*c0-6*c1+3*c2)/6,(-c0+3*c1-3*c2+c3)/6);
  }
}
static void precompute_3b_basis(int ni, std::vector<float4>& out) {
  out.resize(ni*4);
  float b[4][4]={{1.0f/6,-3.0f/6,3.0f/6,-1.0f/6},{4.0f/6,0,-6.0f/6,3.0f/6},{1.0f/6,3.0f/6,3.0f/6,-3.0f/6},{0,0,0,1.0f/6}};
  for(int m=0;m<ni;m++) for(int p=0;p<4;p++) out[m*4+p]=make_float4(b[p][0],b[p][1],b[p][2],b[p][3]);
}

// ---- Uf3Model ------------------------------------------------------------
Uf3Model::Uf3Model(UF3_Parameters& para)
{
  num_types_=para.num_types; elements_=para.elements;
  ncoeff_2b_=para.n_max_2b; nknots_2b_=ncoeff_2b_+4; nint_2b_=nknots_2b_-1; rc_2b_=(float)para.rc_2b;
  has_3b_=(para.n_max_3b[0]>0);
  if(has_3b_){
    for(int d=0;d<3;d++){nc_3b_[d]=para.n_max_3b[d];nk_3b_[d]=nc_3b_[d]+4;nint_3b_[d]=nk_3b_[d]-1;}
    rc_3b_[0]=(float)para.rc_3b[0];rc_3b_[1]=(float)para.rc_3b[1];rc_3b_[2]=rc_3b_[0];
    num_trips_=num_types_*num_types_*num_types_;
    num_params_3b_=num_trips_*nc_3b_[0]*nc_3b_[1]*nc_3b_[2];

  }else{num_trips_=0;num_params_3b_=0;}
  // num_params_3b_ computed above
  num_params_2b_=num_types_*num_types_*ncoeff_2b_;
  num_params_total_=num_params_2b_+num_params_3b_;
  build_knots();

  // Init coefficients (random)
  coeffs_2b_.resize(num_types_*num_types_); srand(42);
  for(size_t p=0;p<coeffs_2b_.size();p++){coeffs_2b_[p].resize(ncoeff_2b_);
    for(int c=0;c<ncoeff_2b_;c++)coeffs_2b_[p][c]=(rand()/(float)RAND_MAX-.5f)*.1f;}
  if(has_3b_){coeffs_3b_.resize(num_params_3b_);
    for(int i=0;i<num_params_3b_;i++)coeffs_3b_[i]=(rand()/(float)RAND_MAX-.5f)*.001f;}

  // Pre-allocate ALL GPU buffers ONCE
  // Use explicit cudaMalloc to avoid GPU_Vector resize() overhead
  prealloc_gpu(para);
}

void Uf3Model::prealloc_gpu(const UF3_Parameters& para)
{
  int max_atoms = para.batch * 200;  // generous per-frame estimate
  if (max_atoms < 5000) max_atoms = 5000; // floor for small batches
  gpu_max_atoms_ = max_atoms; gpu_max_batch_ = para.batch;

  d_types.resize(max_atoms);
  d_x.resize(max_atoms); d_y.resize(max_atoms); d_z.resize(max_atoms);
  d_fx.resize(max_atoms); d_fy.resize(max_atoms); d_fz.resize(max_atoms);
  d_batch_idx.resize(para.batch);
  d_bnatoms.resize(para.batch);
  d_boffsets.resize(para.batch+1);
  d_energy_buf.resize(para.batch);

  int np2 = num_types_ * num_types_;
  std::vector<int> hm(np2); for(int i=0;i<np2;i++) hm[i]=i;
  d_type_map.resize(np2); d_type_map.copy_from_host(hm.data());

  gpu_max_coeff_2b_ = np2 * nint_2b_;
  d_coeff_2b.resize(gpu_max_coeff_2b_);
  upload_2b_coeffs();

  if (has_3b_) {
    init_3b_basis();
    gpu_max_tensor_ = num_trips_ * nc_3b_[0] * nc_3b_[1] * nc_3b_[2];
    d_tensor_3b.resize(gpu_max_tensor_);
    upload_3b_coeffs();
    std::vector<int> ht(num_trips_); for(int i=0;i<num_trips_;i++) ht[i]=i;
    d_trip_map.resize(num_trips_); d_trip_map.copy_from_host(ht.data());
  }
}

void Uf3Model::ensure_batch_buffers(int batch_atoms, int batch_size)
{
  if (batch_atoms > gpu_max_atoms_) {
    gpu_max_atoms_ = batch_atoms;
    d_types.resize(batch_atoms); d_x.resize(batch_atoms); d_y.resize(batch_atoms); d_z.resize(batch_atoms);
    d_fx.resize(batch_atoms); d_fy.resize(batch_atoms); d_fz.resize(batch_atoms);
  }
  if (batch_size > gpu_max_batch_) {
    gpu_max_batch_ = batch_size;
    d_batch_idx.resize(batch_size); d_bnatoms.resize(batch_size); d_boffsets.resize(batch_size+1);
    d_energy_buf.resize(batch_size);
  }
}

template<typename T>
static void grow_copy(GPU_Vector<T>& gv, size_t n, const T* host_data) {
  if (gv.size() < (int)n) gv.resize(n);
  cudaMemcpy(gv.data(), host_data, n * sizeof(T), cudaMemcpyHostToDevice);
}

void Uf3Model::build_knots() {
  knots_2b_.resize(nknots_2b_); float d2=rc_2b_/(nknots_2b_-1);
  for(int i=0;i<nknots_2b_;i++)knots_2b_[i]=i*d2;
  for(int dim=0;dim<3;dim++){if(!has_3b_)break;
    knots_3b_[dim].resize(nk_3b_[dim]); float d=rc_3b_[dim]/(nk_3b_[dim]-1);
    for(int i=0;i<nk_3b_[dim];i++)knots_3b_[dim][i]=i*d;}
}
void Uf3Model::init_3b_basis() {
  std::vector<float4> all;
  for(int d=0;d<3;d++){
    basis_offsets_[d] = (int)all.size();
    std::vector<float4> hb; precompute_3b_basis(nint_3b_[d], hb);
    for(auto& c : hb) all.push_back(c);
  }
  d_basis_3b_all.resize(all.size());
  d_basis_3b_all.copy_from_host(all.data());
}
void Uf3Model::upload_2b_coeffs() {
  std::vector<float4> all; all.reserve(gpu_max_coeff_2b_);
  for(size_t p=0;p<coeffs_2b_.size();p++){std::vector<float4> hc;precompute_2b(coeffs_2b_[p],hc);
    for(auto& c:hc)all.push_back(c);}
  d_coeff_2b.copy_from_host(all.data());  // no resize — already pre-alloc'd
}
void Uf3Model::upload_3b_coeffs() {
  d_tensor_3b.copy_from_host(coeffs_3b_.data());  // no resize
}

void Uf3Model::get_parameters(float* params) const {
  int idx=0;
  for(size_t p=0;p<coeffs_2b_.size();p++)for(int c=0;c<ncoeff_2b_;c++)params[idx++]=coeffs_2b_[p][c];
  for(size_t i=0;i<coeffs_3b_.size();i++)params[idx++]=coeffs_3b_[i];
}
void Uf3Model::set_parameters(const float* params) {
  int idx=0;
  for(size_t p=0;p<coeffs_2b_.size();p++)for(int c=0;c<ncoeff_2b_;c++)coeffs_2b_[p][c]=params[idx++];
  for(size_t i=0;i<coeffs_3b_.size();i++)coeffs_3b_[i]=params[idx++];
  upload_2b_coeffs(); if(has_3b_)upload_3b_coeffs();
}

void Uf3Model::evaluate(
  const std::vector<Uf3Frame>& frames,
  const std::vector<int>& batch_indices,
  GPU_Vector<float>& d_energy)
{
  int B = (int)batch_indices.size();
  int total=0; int max_nn=0;
  for(int b=0;b<B;b++){const auto& f=frames[batch_indices[b]]; total+=f.num_atoms;
    if(has_3b_ && (int)f.nn_list.size()>max_nn) max_nn=(int)f.nn_list.size();}
  ensure_batch_buffers(total, B);

  // Build host batch arrays + neighbor list data for 3B
  std::vector<int> h_bnatoms(B),h_boffsets(B+1);
  std::vector<int> h_btypes(total);
  std::vector<float> h_bx(total),h_by(total),h_bz(total);
  // 3B: neighbor list flat arrays
  std::vector<int> h_nn_offset;     // per-atom offsets into nn_list
  std::vector<int> h_nn_list;       // flat neighbor indices
  std::vector<int> h_nn_frame_off;  // per-frame start in nn_offset
  if(has_3b_){ h_nn_offset.reserve(total + B); h_nn_list.reserve(max_nn); h_nn_frame_off.reserve(B+1); }

  h_boffsets[0]=0; int nn_global_off=0;
  {int off=0; for(int b=0;b<B;b++){const Uf3Frame& f = frames[batch_indices[b]];
    h_bnatoms[b]=f.num_atoms;h_boffsets[b+1]=h_boffsets[b]+f.num_atoms;
    for(int i=0;i<f.num_atoms;i++){h_btypes[off+i]=f.types[i];h_bx[off+i]=f.x[i];h_by[off+i]=f.y[i];h_bz[off+i]=f.z[i];}
    // 3B: append neighbor list for this frame (always add entry, even if empty)
    if(has_3b_){
      h_nn_frame_off.push_back((int)h_nn_offset.size());
      for(int i=0;i<f.num_atoms;i++){
        h_nn_offset.push_back(nn_global_off);
        if(!f.nn_list.empty()) {
          for(int jj=0;jj<f.nn_counts[i];jj++) h_nn_list.push_back(f.nn_list[f.nn_offset[i]+jj]);
          nn_global_off += f.nn_counts[i];
        }
      }
    }
    off+=f.num_atoms;}}
  if(has_3b_) h_nn_frame_off.push_back((int)h_nn_offset.size()); // trailing

  // Upload using grow_copy (only reallocate when growing, raw cudaMemcpy)
  grow_copy(d_types, total, h_btypes.data());
  grow_copy(d_x, total, h_bx.data()); grow_copy(d_y, total, h_by.data()); grow_copy(d_z, total, h_bz.data());
  std::vector<int> h_bidx(B); for(int b=0;b<B;b++)h_bidx[b]=b;
  grow_copy(d_batch_idx, B, h_bidx.data());
  grow_copy(d_bnatoms, B, h_bnatoms.data());
  grow_copy(d_boffsets, B+1, h_boffsets.data());

  // 2B: multi-threaded (BLOCK_SIZE threads per frame, each handles some atoms)
  const int BLK = 64;
  float kmin2=knots_2b_[0], kd2=(knots_2b_.back()-knots_2b_[0])/nint_2b_;
  uf3_eval_2b<<<B, BLK>>>(B,d_batch_idx.data(),d_bnatoms.data(),d_boffsets.data(),
    d_types.data(),d_x.data(),d_y.data(),d_z.data(),
    d_coeff_2b.data(),num_types_*num_types_,nint_2b_,kmin2,kd2,rc_2b_,
    d_type_map.data(),num_types_,d_energy_buf.data());
  GPU_CHECK_KERNEL

  // 2B forces
  uf3_eval_2b_force<<<B, BLK>>>(B,d_batch_idx.data(),d_bnatoms.data(),d_boffsets.data(),
    d_types.data(),d_x.data(),d_y.data(),d_z.data(),
    d_coeff_2b.data(),num_types_*num_types_,nint_2b_,kmin2,kd2,rc_2b_,
    d_type_map.data(),num_types_,d_fx.data(),d_fy.data(),d_fz.data());
  GPU_CHECK_KERNEL

  // 3B: neighbor-list-based, multi-threaded
  if(has_3b_ && h_nn_frame_off.size() > 1){
    grow_copy(d_nn_off, h_nn_offset.size(), h_nn_offset.data());
    grow_copy(d_nn_lst, h_nn_list.size(), h_nn_list.data());
    grow_copy(d_nn_frame_off, h_nn_frame_off.size(), h_nn_frame_off.data());
    uf3_eval_3b<<<B, BLK>>>(B,d_batch_idx.data(),d_bnatoms.data(),d_boffsets.data(),
      d_types.data(),d_x.data(),d_y.data(),d_z.data(),
      d_tensor_3b.data(),nc_3b_[0],nc_3b_[1],nc_3b_[2],
      d_basis_3b_all.data()+basis_offsets_[0],nint_3b_[0],d_basis_3b_all.data()+basis_offsets_[1],nint_3b_[1],d_basis_3b_all.data()+basis_offsets_[2],nint_3b_[2],
      knots_3b_[0][0],(knots_3b_[0].back()-knots_3b_[0][0])/nint_3b_[0],rc_3b_[0],
      knots_3b_[1][0],(knots_3b_[1].back()-knots_3b_[1][0])/nint_3b_[1],rc_3b_[1],
      knots_3b_[2][0],(knots_3b_[2].back()-knots_3b_[2][0])/nint_3b_[2],rc_3b_[2],
      d_trip_map.data(),num_trips_,num_types_,
      d_nn_off.data(),d_nn_lst.data(),d_nn_frame_off.data(),
      d_energy_buf.data());
    GPU_CHECK_KERNEL

    // 3B forces
    uf3_eval_3b_force<<<B, BLK>>>(B, d_batch_idx.data(), d_bnatoms.data(), d_boffsets.data(),
      d_types.data(), d_x.data(), d_y.data(), d_z.data(),
      d_tensor_3b.data(), nc_3b_[0], nc_3b_[1], nc_3b_[2],
      d_basis_3b_all.data()+basis_offsets_[0], nint_3b_[0],
      d_basis_3b_all.data()+basis_offsets_[1], nint_3b_[1],
      d_basis_3b_all.data()+basis_offsets_[2], nint_3b_[2],
      knots_3b_[0][0], (knots_3b_[0].back()-knots_3b_[0][0])/nint_3b_[0], rc_3b_[0],
      knots_3b_[1][0], (knots_3b_[1].back()-knots_3b_[1][0])/nint_3b_[1], rc_3b_[1],
      knots_3b_[2][0], (knots_3b_[2].back()-knots_3b_[2][0])/nint_3b_[2], rc_3b_[2],
      d_trip_map.data(), num_trips_, num_types_,
      d_nn_off.data(), d_nn_lst.data(), d_nn_frame_off.data(),
      d_fx.data(), d_fy.data(), d_fz.data());
    GPU_CHECK_KERNEL
  }

  d_energy.resize(B);
  cudaMemcpy(d_energy.data(), d_energy_buf.data(), B*sizeof(float), cudaMemcpyDeviceToDevice);
}

// ---- GPU-side per-atom normalized energy residual ---------------------------
static __global__ void gpu_energy_peratom_diff_kernel(
  int B,
  const float* d_energy,
  const float* d_ref,
  const int* d_fidx,
  const int* d_natoms,
  float* d_diff)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) {
    return;
  }
  int fid = d_fidx[b];
  float na = (float)d_natoms[fid];
  if (na < 1.0f) {
    na = 1.0f;
  }
  d_diff[b] = d_energy[b] / na - d_ref[fid] / na;
}

// ---- Analytical force-term gradient kernel (2B) -----------------------------
static __global__ void uf3_grad_2b_force(
  int nf,
  int nparam,
  const int* __restrict__ fidx,
  const int* __restrict__ nat,
  const int* __restrict__ off,
  const int* __restrict__ typ,
  const float* __restrict__ x,
  const float* __restrict__ y,
  const float* __restrict__ z,
  int np,
  int ncoeff,
  int nint,
  float kmin,
  float kd,
  float rc,
  const int* __restrict__ tmap,
  int nt,
  const float* __restrict__ fx,
  const float* __restrict__ fy,
  const float* __restrict__ fz,
  const float* __restrict__ fx_ref,
  const float* __restrict__ fy_ref,
  const float* __restrict__ fz_ref,
  float scale_f,
  float* __restrict__ gradient)
{
  int k = blockIdx.x;
  if (k >= nparam) {
    return;
  }
  int tid = threadIdx.x;
  int stride = blockDim.x;
  int pair_idx = k / ncoeff;
  int coeff_idx = k % ncoeff;
  float grad = 0.0f;

  for (int b = tid; b < nf; b += stride) {
    int fid = fidx[b];
    int n = nat[fid];
    int o = off[fid];
    for (int i = 0; i < n; i++) {
      int ti = typ[o + i];
      float rx = fx[o + i] - fx_ref[o + i];
      float ry = fy[o + i] - fy_ref[o + i];
      float rz = fz[o + i] - fz_ref[o + i];
      for (int j = 0; j < n; j++) {
        if (i == j) {
          continue;
        }
        int tj = typ[o + j];
        if (tmap[ti * nt + tj] != pair_idx) {
          continue;
        }
        float dx = x[o + i] - x[o + j];
        float dy = y[o + i] - y[o + j];
        float dz = z[o + i] - z[o + j];
        float r2 = dx * dx + dy * dy + dz * dz;
        float r = sqrtf(r2);
        if (r >= rc) {
          continue;
        }
        float inv_r = 1.0f / r;
        int m = uf3_find_interval(r, kmin, kd, nint);
        float u = (r - (kmin + m * kd)) / kd;
        int c_rel = coeff_idx - m + 3;
        if (c_rel < 0 || c_rel > 3) {
          continue;
        }
        float dbdu = 0.0f;
        if (c_rel == 0) {
          dbdu = -3.0f * (1.0f - u) * (1.0f - u) / 6.0f;
        } else if (c_rel == 1) {
          dbdu = (9.0f * u * u - 12.0f * u) / 6.0f;
        } else if (c_rel == 2) {
          dbdu = (-9.0f * u * u + 6.0f * u + 3.0f) / 6.0f;
        } else {
          dbdu = 3.0f * u * u / 6.0f;
        }
        float dbdr = dbdu / kd;
        float factor = -dbdr * inv_r;
        float gx = factor * dx;
        float gy = factor * dy;
        float gz = factor * dz;
        grad += scale_f * (rx * gx + ry * gy + rz * gz);
      }
    }
  }

  __shared__ float s_grad[64];
  s_grad[tid] = grad;
  __syncthreads();
  for (int s = stride / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_grad[tid] += s_grad[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    gradient[k] += s_grad[0];
  }
}

// ---- Analytical energy-term gradient kernel (3B) ----------------------------
static __device__ inline float uf3_basis_value(int coeff_idx, int m, float u)
{
  int c_rel = coeff_idx - m + 3;
  if (c_rel < 0 || c_rel > 3) {
    return 0.0f;
  }
  if (c_rel == 0) {
    return (1.0f / 6.0f) * (1.0f - u) * (1.0f - u) * (1.0f - u);
  }
  if (c_rel == 1) {
    return (3.0f * u * u * u - 6.0f * u * u + 4.0f) / 6.0f;
  }
  if (c_rel == 2) {
    return (-3.0f * u * u * u + 3.0f * u * u + 3.0f * u + 1.0f) / 6.0f;
  }
  return u * u * u / 6.0f;
}

static __global__ void uf3_grad_3b_energy(
  int nf,
  int nparam,
  int num_params_2b,
  const int* __restrict__ fidx,
  const int* __restrict__ nat,
  const int* __restrict__ off,
  const int* __restrict__ typ,
  const float* __restrict__ x,
  const float* __restrict__ y,
  const float* __restrict__ z,
  int nc0,
  int nc1,
  int nc2,
  int ni0,
  int ni1,
  int ni2,
  float k0,
  float kd0,
  float r0,
  float k1,
  float kd1,
  float r1,
  float k2,
  float kd2,
  float r2,
  int nt,
  int num_trips,
  const int* __restrict__ nn_off,
  const int* __restrict__ nn_lst,
  const int* __restrict__ nn_frame_off,
  const float* __restrict__ energy_diff,
  float scale_e,
  float* __restrict__ gradient)
{
  int k = blockIdx.x;
  if (k < num_params_2b || k >= nparam) {
    return;
  }
  int local = k - num_params_2b;
  int nc01 = nc0 * nc1;
  int trip = local / nc01 / nc2;
  int rem = local - trip * nc01 * nc2;
  int p = rem % nc0;
  int rem2 = rem / nc0;
  int q = rem2 % nc1;
  int r = rem2 / nc1;

  int tid = threadIdx.x;
  int stride = blockDim.x;
  float grad = 0.0f;

  for (int b = tid; b < nf; b += stride) {
    int fid = fidx[b];
    int n = nat[fid];
    int o = off[fid];
    float delta = energy_diff[b];
    for (int i = 0; i < n; i++) {
      int ti = typ[o + i];
      for (int j = i + 1; j < n; j++) {
        float dx12 = x[o + j] - x[o + i];
        float dy12 = y[o + j] - y[o + i];
        float dz12 = z[o + j] - z[o + i];
        float r12 = sqrtf(dx12 * dx12 + dy12 * dy12 + dz12 * dz12);
        if (r12 >= r0) {
          continue;
        }
        int tj = typ[o + j];
        for (int kk = j + 1; kk < n; kk++) {
          float dx13 = x[o + kk] - x[o + i];
          float dy13 = y[o + kk] - y[o + i];
          float dz13 = z[o + kk] - z[o + i];
          float r13 = sqrtf(dx13 * dx13 + dy13 * dy13 + dz13 * dz13);
          if (r13 >= r1) {
            continue;
          }
          float dx23 = x[o + kk] - x[o + j];
          float dy23 = y[o + kk] - y[o + j];
          float dz23 = z[o + kk] - z[o + j];
          float r23 = sqrtf(dx23 * dx23 + dy23 * dy23 + dz23 * dz23);
          if (r23 >= r2) {
            continue;
          }
          int tk = typ[o + kk];
          int trip_idx = (ti * nt + tj) * nt + tk;
          if (trip_idx != trip) {
            continue;
          }
          int m0 = uf3_find_interval(r12, k0, kd0, ni0);
          int m1 = uf3_find_interval(r13, k1, kd1, ni1);
          int m2 = uf3_find_interval(r23, k2, kd2, ni2);
          float u0 = (r12 - (k0 + m0 * kd0)) / kd0;
          float u1 = (r13 - (k1 + m1 * kd1)) / kd1;
          float u2 = (r23 - (k2 + m2 * kd2)) / kd2;
          float bp = uf3_basis_value(p, m0, u0);
          float bq = uf3_basis_value(q, m1, u1);
          float br = uf3_basis_value(r, m2, u2);
          grad += scale_e * delta * bp * bq * br;
        }
      }
    }
  }

  __shared__ float s_grad[64];
  s_grad[tid] = grad;
  __syncthreads();
  for (int s = stride / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_grad[tid] += s_grad[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    gradient[k] += s_grad[0];
  }
}

// ---- Fast evaluate using pre-loaded GPU dataset (NO H2D data upload) ---------
void Uf3Model::evaluate(
  const Uf3DatasetGPU& ds,
  const std::vector<int>& batch_indices,
  GPU_Vector<float>& d_energy)
{
  int B = (int)batch_indices.size();
  int total = 0;
  for (int b = 0; b < B; b++) total += ds.h_natoms[batch_indices[b]];
  ensure_batch_buffers(total, B);

  // Upload batch frame indices (global) — ONLY H2D, ~4KB
  std::vector<int> h_fidx(B);
  for (int b = 0; b < B; b++) h_fidx[b] = batch_indices[b];
  d_batch_idx.copy_from_host(h_fidx.data());

  const int BLK = 64;
  float kmin2 = knots_2b_[0], kd2 = (knots_2b_.back() - knots_2b_[0]) / nint_2b_;

  // 2B energy
  uf3_eval_2b<<<B, BLK>>>(B, d_batch_idx.data(), ds.d_natoms.data(), ds.d_offsets.data(),
    ds.d_types.data(), ds.d_x.data(), ds.d_y.data(), ds.d_z.data(),
    d_coeff_2b.data(), num_types_ * num_types_, nint_2b_, kmin2, kd2, rc_2b_,
    d_type_map.data(), num_types_, d_energy_buf.data());
  GPU_CHECK_KERNEL

  // 2B forces
  uf3_eval_2b_force<<<B, BLK>>>(B, d_batch_idx.data(), ds.d_natoms.data(), ds.d_offsets.data(),
    ds.d_types.data(), ds.d_x.data(), ds.d_y.data(), ds.d_z.data(),
    d_coeff_2b.data(), num_types_ * num_types_, nint_2b_, kmin2, kd2, rc_2b_,
    d_type_map.data(), num_types_, d_fx.data(), d_fy.data(), d_fz.data());
  GPU_CHECK_KERNEL

  // 3B energy + forces (use global neighbor lists already on GPU)
  if (has_3b_ && ds.has_3b) {
    uf3_eval_3b<<<B, BLK>>>(B, d_batch_idx.data(), ds.d_natoms.data(), ds.d_offsets.data(),
      ds.d_types.data(), ds.d_x.data(), ds.d_y.data(), ds.d_z.data(),
      d_tensor_3b.data(), nc_3b_[0], nc_3b_[1], nc_3b_[2],
      d_basis_3b_all.data() + basis_offsets_[0], nint_3b_[0],
      d_basis_3b_all.data() + basis_offsets_[1], nint_3b_[1],
      d_basis_3b_all.data() + basis_offsets_[2], nint_3b_[2],
      knots_3b_[0][0], (knots_3b_[0].back() - knots_3b_[0][0]) / nint_3b_[0], rc_3b_[0],
      knots_3b_[1][0], (knots_3b_[1].back() - knots_3b_[1][0]) / nint_3b_[1], rc_3b_[1],
      knots_3b_[2][0], (knots_3b_[2].back() - knots_3b_[2][0]) / nint_3b_[2], rc_3b_[2],
      d_trip_map.data(), num_trips_, num_types_,
      ds.d_nn_off.data(), ds.d_nn_lst.data(), ds.d_nn_frame_off.data(),
      d_energy_buf.data());
    GPU_CHECK_KERNEL

    uf3_eval_3b_force<<<B, BLK>>>(B, d_batch_idx.data(), ds.d_natoms.data(), ds.d_offsets.data(),
      ds.d_types.data(), ds.d_x.data(), ds.d_y.data(), ds.d_z.data(),
      d_tensor_3b.data(), nc_3b_[0], nc_3b_[1], nc_3b_[2],
      d_basis_3b_all.data() + basis_offsets_[0], nint_3b_[0],
      d_basis_3b_all.data() + basis_offsets_[1], nint_3b_[1],
      d_basis_3b_all.data() + basis_offsets_[2], nint_3b_[2],
      knots_3b_[0][0], (knots_3b_[0].back() - knots_3b_[0][0]) / nint_3b_[0], rc_3b_[0],
      knots_3b_[1][0], (knots_3b_[1].back() - knots_3b_[1][0]) / nint_3b_[1], rc_3b_[1],
      knots_3b_[2][0], (knots_3b_[2].back() - knots_3b_[2][0]) / nint_3b_[2], rc_3b_[2],
      d_trip_map.data(), num_trips_, num_types_,
      ds.d_nn_off.data(), ds.d_nn_lst.data(), ds.d_nn_frame_off.data(),
      d_fx.data(), d_fy.data(), d_fz.data());
    GPU_CHECK_KERNEL
  }

  d_energy.resize(B);
  cudaMemcpy(d_energy.data(), d_energy_buf.data(), B * sizeof(float), cudaMemcpyDeviceToDevice);
}

// ---- Fast unified loss gradient (matches Uf3Fitness::compute_loss) ---------
void Uf3Model::compute_loss_gradient(
  const Uf3DatasetGPU& ds,
  const std::vector<int>& batch_indices,
  float loss_e,
  float loss_f,
  float lambda_e,
  float lambda_f,
  std::vector<float>& host_gradient)
{
  int B = (int)batch_indices.size();
  int total_atoms = 0;
  for (int b = 0; b < B; b++) {
    total_atoms += ds.h_natoms[batch_indices[b]];
  }

  int nparam = num_parameters();
  GPU_Vector<float> d_grad(nparam);
  cudaMemset(d_grad.data(), 0, nparam * sizeof(float));

  GPU_Vector<float> d_ediff(B);
  gpu_energy_peratom_diff_kernel<<<(B + 63) / 64, 64>>>(
    B, d_energy_buf.data(), ds.d_energy_ref.data(), d_batch_idx.data(), ds.d_natoms.data(),
    d_ediff.data());
  GPU_CHECK_KERNEL

  float kmin2 = knots_2b_[0];
  float kd2 = (knots_2b_.back() - knots_2b_[0]) / nint_2b_;

  if (lambda_e > 0.0f && loss_e > 1e-12f && B > 0) {
    float scale_e = lambda_e / ((float)B * loss_e);
    uf3_grad_2b<<<num_params_2b_, 64>>>(
      B, nparam, d_batch_idx.data(), ds.d_natoms.data(), ds.d_offsets.data(),
      ds.d_types.data(), ds.d_x.data(), ds.d_y.data(), ds.d_z.data(),
      num_types_ * num_types_, ncoeff_2b_, nint_2b_, kmin2, kd2, rc_2b_,
      d_type_map.data(), num_types_, d_ediff.data(), scale_e, d_grad.data());
    GPU_CHECK_KERNEL
    if (has_3b_ && ds.has_3b && num_params_3b_ > 0) {
      uf3_grad_3b_energy<<<nparam, 64>>>(
        B, nparam, num_params_2b_, d_batch_idx.data(), ds.d_natoms.data(), ds.d_offsets.data(),
        ds.d_types.data(), ds.d_x.data(), ds.d_y.data(), ds.d_z.data(),
        nc_3b_[0], nc_3b_[1], nc_3b_[2], nint_3b_[0], nint_3b_[1], nint_3b_[2],
        knots_3b_[0][0], (knots_3b_[0].back() - knots_3b_[0][0]) / nint_3b_[0], rc_3b_[0],
        knots_3b_[1][0], (knots_3b_[1].back() - knots_3b_[1][0]) / nint_3b_[1], rc_3b_[1],
        knots_3b_[2][0], (knots_3b_[2].back() - knots_3b_[2][0]) / nint_3b_[2], rc_3b_[2],
        num_types_, num_trips_, ds.d_nn_off.data(), ds.d_nn_lst.data(), ds.d_nn_frame_off.data(),
        d_ediff.data(), scale_e, d_grad.data());
      GPU_CHECK_KERNEL
    }
  }

  if (lambda_f > 0.0f && loss_f > 1e-12f && total_atoms > 0) {
    float scale_f = lambda_f / ((float)(3 * total_atoms) * loss_f);
    uf3_grad_2b_force<<<num_params_2b_, 64>>>(
      B, nparam, d_batch_idx.data(), ds.d_natoms.data(), ds.d_offsets.data(),
      ds.d_types.data(), ds.d_x.data(), ds.d_y.data(), ds.d_z.data(),
      num_types_ * num_types_, ncoeff_2b_, nint_2b_, kmin2, kd2, rc_2b_,
      d_type_map.data(), num_types_, d_fx.data(), d_fy.data(), d_fz.data(),
      ds.d_fx_ref.data(), ds.d_fy_ref.data(), ds.d_fz_ref.data(), scale_f, d_grad.data());
    GPU_CHECK_KERNEL
  }

  host_gradient.resize(nparam);
  d_grad.copy_to_host(host_gradient.data());
}

// Legacy energy-only gradient (host-side frames).
void Uf3Model::compute_energy_gradient(
  const std::vector<Uf3Frame>& frames,
  const std::vector<int>& batch_indices,
  GPU_Vector<float>& d_energy_diff,
  std::vector<float>& host_gradient)
{
  int B = (int)batch_indices.size();
  evaluate(frames, batch_indices, d_energy_diff);
  std::vector<float> h_ene(B);
  d_energy_diff.copy_to_host(h_ene.data());
  std::vector<float> h_diff(B);
  for (int b = 0; b < B; b++) {
    int fidx = batch_indices[b];
    int na = frames[fidx].num_atoms;
    if (na < 1) {
      na = 1;
    }
    h_diff[b] = h_ene[b] / na - frames[fidx].energy / na;
  }
  d_energy_diff.copy_from_host(h_diff.data());

  int total = 0;
  for (int b = 0; b < B; b++) {
    total += frames[batch_indices[b]].num_atoms;
  }
  ensure_batch_buffers(total, B);

  int nparam = num_parameters();
  GPU_Vector<float> d_grad(nparam);
  float kmin2 = knots_2b_[0];
  float kd2 = (knots_2b_.back() - knots_2b_[0]) / nint_2b_;
  uf3_grad_2b<<<num_params_2b_, 64>>>(
    B, nparam, d_batch_idx.data(), d_bnatoms.data(), d_boffsets.data(),
    d_types.data(), d_x.data(), d_y.data(), d_z.data(),
    num_types_ * num_types_, ncoeff_2b_, nint_2b_, kmin2, kd2, rc_2b_,
    d_type_map.data(), num_types_,
    d_energy_diff.data(), 1.0f / B, d_grad.data());
  GPU_CHECK_KERNEL

  host_gradient.resize(nparam);
  d_grad.copy_to_host(host_gradient.data());
}
