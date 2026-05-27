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
  // Always resize to current batch size (simplest, avoids subtle sizing bugs)
  d_types.resize(batch_atoms);
  d_x.resize(batch_atoms); d_y.resize(batch_atoms); d_z.resize(batch_atoms);
  d_batch_idx.resize(batch_size);
  d_bnatoms.resize(batch_size);
  d_boffsets.resize(batch_size+1);
  d_energy_buf.resize(batch_size);
}

// Only resize GPU vector when growing (avoids repeated cudaFree/cudaMalloc)
template<typename T>
static void grow_only(GPU_Vector<T>& gv, size_t new_size) {
  if (gv.size() < (int)new_size) gv.resize(new_size);
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

  // Upload
  d_types.copy_from_host(h_btypes.data());
  d_x.copy_from_host(h_bx.data()); d_y.copy_from_host(h_by.data()); d_z.copy_from_host(h_bz.data());
  std::vector<int> h_bidx(B); for(int b=0;b<B;b++)h_bidx[b]=b;
  d_batch_idx.copy_from_host(h_bidx.data());
  d_bnatoms.copy_from_host(h_bnatoms.data());
  d_boffsets.copy_from_host(h_boffsets.data());

  // 2B: multi-threaded (BLOCK_SIZE threads per frame, each handles some atoms)
  const int BLK = 64;
  float kmin2=knots_2b_[0], kd2=(knots_2b_.back()-knots_2b_[0])/nint_2b_;
  uf3_eval_2b<<<B, BLK>>>(B,d_batch_idx.data(),d_bnatoms.data(),d_boffsets.data(),
    d_types.data(),d_x.data(),d_y.data(),d_z.data(),
    d_coeff_2b.data(),num_types_*num_types_,nint_2b_,kmin2,kd2,rc_2b_,
    d_type_map.data(),num_types_,d_energy_buf.data());
  GPU_CHECK_KERNEL

  // 3B: neighbor-list-based, multi-threaded
  if(has_3b_ && h_nn_frame_off.size() > 1){
    grow_only(d_nn_off, h_nn_offset.size());
    cudaMemcpy(d_nn_off.data(), h_nn_offset.data(), h_nn_offset.size()*sizeof(int), cudaMemcpyHostToDevice);
    grow_only(d_nn_lst, h_nn_list.size());
    cudaMemcpy(d_nn_lst.data(), h_nn_list.data(), h_nn_list.size()*sizeof(int), cudaMemcpyHostToDevice);
    grow_only(d_nn_frame_off, h_nn_frame_off.size());
    cudaMemcpy(d_nn_frame_off.data(), h_nn_frame_off.data(), h_nn_frame_off.size()*sizeof(int), cudaMemcpyHostToDevice);
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
  }

  d_energy.resize(B);
  cudaMemcpy(d_energy.data(), d_energy_buf.data(), B*sizeof(float), cudaMemcpyDeviceToDevice);
}
