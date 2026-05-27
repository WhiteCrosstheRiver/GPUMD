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

/*----------------------------------------------------------------------------80
The UF3 (Ultra-Fast Force Field) potential — inference implementation.
Ref: S. R. Xie et al., npj Comput. Mater. 9, 143 (2023).

B-spline evaluation: cubic B-spline basis functions are pre-computed at
initialization as per-interval combined cubic polynomials (float4 per
interval).  At runtime, the potential is a simple Horner evaluation of a
cubic polynomial: V(u) = A + u*(B + u*(C + u*D)).
------------------------------------------------------------------------------*/

#include "neighbor.cuh"
#include "uf3.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// ---------------------------------------------------------------------------
// B-spline pre-computation helpers (host-side)
// ---------------------------------------------------------------------------

// Expand uniform cubic B-spline: combined polynomial V(u) = A+Bu+Cu^2+Du^3
// on interval m, from coefficients co[m-3..m].  (Boundary intervals are
// clamped: coeffs beyond range use the nearest valid coefficient.)
static void precompute_2b_uniform(
  const std::vector<float>& knots,
  const std::vector<float>& coeffs,
  int leading_trim,
  std::vector<float4>& h_coeff)
{
  const int nk = (int)knots.size();
  const int nc = (int)coeffs.size();
  const int nint = nk - 1;

  h_coeff.resize(nint);
  for (int m = 0; m < nint; m++) {
    int i0 = m - 3, i1 = m - 2, i2 = m - 1, i3 = m;
    if (i0 < 0) i0 = 0;
    if (i1 < 0) i1 = 0;
    if (i2 < 0) i2 = 0;
    if (i2 >= nc) i2 = nc - 1;
    if (i3 >= nc) i3 = nc - 1;
    float c0 = coeffs[i0], c1 = coeffs[i1], c2 = coeffs[i2], c3 = coeffs[i3];
    // Uniform cubic B-spline basis on [0,1] with coeffs c0..c3
    float A = (c0 + 4.0f * c1 + c2) / 6.0f;
    float B = (-3.0f * c0 + 3.0f * c2) / 6.0f;
    float C = (3.0f * c0 - 6.0f * c1 + 3.0f * c2) / 6.0f;
    float D = (-c0 + 3.0f * c1 - 3.0f * c2 + c3) / 6.0f;
    h_coeff[m] = make_float4(A, B, C, D);
  }
}

// Build per-interval basis cubic polynomials for 3-body dimension.
// On each interval, the 4 active basis functions (cubic polynomials in u)
// are stored as float4 entries.  basis_coeff[m * 4 + p] for p=0,1,2,3.
static void precompute_3b_basis_uniform(
  const std::vector<float>& knots,
  int leading_trim,
  std::vector<float4>& h_basis)
{
  const int nk = (int)knots.size();
  const int nint = nk - 1;
  (void)leading_trim; // unused in uniform basis precomputation

  h_basis.resize(nint * 4);
  for (int m = 0; m < nint; m++) {
    // 4 active basis functions at interval m: B_{m-3}, B_{m-2}, B_{m-1}, B_m
    // In the uniform case they are simply the standard cubic B-spline basis:
    // B0(u)=(1-u)^3/6, B1(u)=(3u^3-6u^2+4)/6,
    // B2(u)=(-3u^3+3u^2+3u+1)/6, B3(u)=u^3/6
    // Pre-compute as A+Bu+Cu^2+Du^3
    float b[4][4] = {
      { 1.0f/6,  -3.0f/6,   3.0f/6,  -1.0f/6 },   // B0 = (1-u)^3
      { 4.0f/6,   0.0f,     -6.0f/6,   3.0f/6 },   // B1 = 3u^3-6u^2+4
      { 1.0f/6,   3.0f/6,   3.0f/6,  -3.0f/6 },   // B2 = -3u^3+3u^2+3u+1
      { 0.0f,     0.0f,     0.0f,     1.0f/6 }     // B3 = u^3
    };
    for (int p = 0; p < 4; p++) {
      h_basis[m * 4 + p] = make_float4(b[p][0], b[p][1], b[p][2], b[p][3]);
    }
  }
}

// ---------------------------------------------------------------------------
// Device helper: evaluate cubic polynomial
// ---------------------------------------------------------------------------
__device__ inline float eval_cubic(float4 c, float u)
{
  return c.x + u * (c.y + u * (c.z + u * c.w));
}

__device__ inline float eval_cubic_deriv(float4 c, float u)
{
  return c.y + u * (2.0f * c.z + u * 3.0f * c.w);
}

// ---------------------------------------------------------------------------
// Device helper: find knot interval (uniform)
// ---------------------------------------------------------------------------
__device__ inline int find_interval(float r, float knot_min, float knot_delta, int nint)
{
  float t = (r - knot_min) / knot_delta;
  int i = (int)t;
  if (i < 0) i = 0;
  if (i >= nint) i = nint - 1;
  return i;
}

// ---------------------------------------------------------------------------
// 2-body kernel
// ---------------------------------------------------------------------------
static __global__ void find_force_uf3_2b(
  const int N, const int N1, const int N2,
  const Box box,
  const float4* __restrict__ d_coeff,
  int nint,
  float knot_min, float knot_delta,
  double rc,
  const int* __restrict__ g_NN,
  const int* __restrict__ g_NL,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_pe,
  double* g_fx, double* g_fy, double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;

  int NN = g_NN[n1];
  double x1 = g_x[n1], y1 = g_y[n1], z1 = g_z[n1];
  float pe = 0.0f;
  float fx = 0.0f, fy = 0.0f, fz = 0.0f;
  float sxx = 0, sxy = 0, sxz = 0, syx = 0, syy = 0, syz = 0, szx = 0, szy = 0, szz = 0;

  for (int i1 = 0; i1 < NN; ++i1) {
    int n2 = g_NL[n1 + N * i1];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    float r = sqrtf(x12 * x12 + y12 * y12 + z12 * z12);
    if (r >= rc) continue;

    int m = find_interval(r, knot_min, knot_delta, nint);
    float h = knot_delta;
    float u = (r - (knot_min + m * h)) / h;
    float4 c = __ldg(&d_coeff[m]);
    float val = eval_cubic(c, u);
    float deriv = eval_cubic_deriv(c, u) / h;

    float rinv = 1.0f / r;
    float fpair = deriv * rinv;
    float f12x = fpair * x12 * 0.5f;
    float f12y = fpair * y12 * 0.5f;
    float f12z = fpair * z12 * 0.5f;

    fx += f12x; fy += f12y; fz += f12z;
    pe += val * 0.5f;

    sxx -= f12x * x12; sxy -= f12x * y12; sxz -= f12x * z12;
    syx -= f12y * x12; syy -= f12y * y12; syz -= f12y * z12;
    szx -= f12z * x12; szy -= f12z * y12; szz -= f12z * z12;
  }

  g_pe[n1] += pe;
  g_fx[n1] += fx; g_fy[n1] += fy; g_fz[n1] += fz;
  g_virial[n1 + 0 * N] += sxx;
  g_virial[n1 + 1 * N] += syy;
  g_virial[n1 + 2 * N] += szz;
  g_virial[n1 + 3 * N] += sxy;
  g_virial[n1 + 4 * N] += sxz;
  g_virial[n1 + 5 * N] += syz;
  g_virial[n1 + 6 * N] += syx;
  g_virial[n1 + 7 * N] += szx;
  g_virial[n1 + 8 * N] += szy;
}

// ---------------------------------------------------------------------------
// 3-body kernel  (Tersoff-style partial force accumulation)
// ---------------------------------------------------------------------------
static __global__ void find_force_uf3_3b(
  const int N, const int N1, const int N2,
  const Box box,
  const float* __restrict__ d_tensor,
  int nc_ij, int nc_ik, int nc_jk,
  const float4* __restrict__ d_basis_ij, int nint_ij,
  const float4* __restrict__ d_basis_ik, int nint_ik,
  const float4* __restrict__ d_basis_jk, int nint_jk,
  float knot_min_ij, float knot_delta_ij,
  float knot_min_ik, float knot_delta_ik,
  float knot_min_jk, float knot_delta_jk,
  double rc_ij, double rc_ik, double rc_jk,
  const int* __restrict__ g_NN,
  const int* __restrict__ g_NL,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  float* g_f12x, float* g_f12y, float* g_f12z,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;

  int NN = g_NN[n1];
  double x1 = g_x[n1], y1 = g_y[n1], z1 = g_z[n1];
  float pe = 0.0f;

  for (int j1 = 0; j1 < NN; ++j1) {
    int n2 = g_NL[n1 + N * j1];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    float r12 = sqrtf(x12*x12 + y12*y12 + z12*z12);
    if (r12 >= rc_ij) continue;

    for (int k1 = j1 + 1; k1 < NN; ++k1) {
      int n3 = g_NL[n1 + N * k1];
      float x13 = g_x[n3] - x1;
      float y13 = g_y[n3] - y1;
      float z13 = g_z[n3] - z1;
      apply_mic(box, x13, y13, z13);
      float r13 = sqrtf(x13*x13 + y13*y13 + z13*z13);
      if (r13 >= rc_ik) continue;

      float x23 = g_x[n3] - g_x[n2];
      float y23 = g_y[n3] - g_y[n2];
      float z23 = g_z[n3] - g_z[n2];
      apply_mic(box, x23, y23, z23);
      float r23 = sqrtf(x23*x23 + y23*y23 + z23*z23);
      if (r23 >= rc_jk) continue;

      // Evaluate basis values for each dimension
      int mi = find_interval(r12, knot_min_ij, knot_delta_ij, nint_ij);
      int mk = find_interval(r13, knot_min_ik, knot_delta_ik, nint_ik);
      int mj = find_interval(r23, knot_min_jk, knot_delta_jk, nint_jk);

      float hi = knot_delta_ij, hk = knot_delta_ik, hj = knot_delta_jk;
      float ui = (r12 - (knot_min_ij + mi * hi)) / hi;
      float uk = (r13 - (knot_min_ik + mk * hk)) / hk;
      float uj = (r23 - (knot_min_jk + mj * hj)) / hj;

      float b_ij[4], db_ij[4];
      float b_ik[4], db_ik[4];
      float b_jk[4], db_jk[4];

      for (int p = 0; p < 4; p++) {
        float4 cb = __ldg(&d_basis_ij[mi * 4 + p]);
        b_ij[p] = eval_cubic(cb, ui);
        db_ij[p] = eval_cubic_deriv(cb, ui) / hi;
      }
      for (int p = 0; p < 4; p++) {
        float4 cb = __ldg(&d_basis_ik[mk * 4 + p]);
        b_ik[p] = eval_cubic(cb, uk);
        db_ik[p] = eval_cubic_deriv(cb, uk) / hk;
      }
      for (int p = 0; p < 4; p++) {
        float4 cb = __ldg(&d_basis_jk[mj * 4 + p]);
        b_jk[p] = eval_cubic(cb, uj);
        db_jk[p] = eval_cubic_deriv(cb, uj) / hj;
      }

      // Starting indices for active basis functions on this interval
      int p0 = mi - 3; if (p0 < 0) p0 = 0;
      int q0 = mk - 3; if (q0 < 0) q0 = 0;
      int r0 = mj - 3; if (r0 < 0) r0 = 0;

      // Tensor contraction (4x4x4 = 64 terms)
      float val = 0, dv_d12 = 0, dv_d13 = 0, dv_d23 = 0;
      for (int dp = 0; dp < 4; dp++) {
        int p = p0 + dp;
        if (p >= nc_ij) continue;
        float bp = b_ij[dp], dbp = db_ij[dp];
        int p_stride = p;

        for (int dq = 0; dq < 4; dq++) {
          int q = q0 + dq;
          if (q >= nc_ik) continue;
          float bq = b_ik[dq], dbq = db_ik[dq];
          int pq_stride = p_stride + q * nc_ij;

          for (int dr = 0; dr < 4; dr++) {
            int r = r0 + dr;
            if (r >= nc_jk) continue;
            float br = b_jk[dr], dbr = db_jk[dr];
            float C = __ldg(&d_tensor[pq_stride + r * nc_ij * nc_ik]);

            val     += C * bp * bq * br;
            dv_d12 += C * dbp * bq * br;
            dv_d13 += C * bp * dbq * br;
            dv_d23 += C * bp * bq * dbr;
          }
        }
      }

      pe += val / 3.0f;  // each triplet counted 3 times (once per center)

      float inv_r12 = 1.0f / r12, inv_r13 = 1.0f / r13;
      float fij_s = dv_d12 * inv_r12 * 0.5f;
      float fik_s = dv_d13 * inv_r13 * 0.5f;

      int idx_12 = j1 * N + n1;
      g_f12x[idx_12] += fij_s * x12;
      g_f12y[idx_12] += fij_s * y12;
      g_f12z[idx_12] += fij_s * z12;

      int idx_13 = k1 * N + n1;
      g_f12x[idx_13] += fik_s * x13;
      g_f12y[idx_13] += fik_s * y13;
      g_f12z[idx_13] += fik_s * z13;
    }
  }
  g_pe[n1] += pe;
}

// ---------------------------------------------------------------------------
// UF3 class implementation
// ---------------------------------------------------------------------------

UF3::UF3(const char* filename, const int number_of_atoms, const int max_neighbor)
{
  max_neighbor_ = max_neighbor;
  has_2b = false;
  has_3b = false;
  initialize(filename, number_of_atoms);
}

UF3::~UF3(void) {}

void UF3::initialize(const char* filename, const int number_of_atoms)
{
  // ---- Parse the .uf3 file ----
  std::ifstream input(filename);
  if (!input.is_open()) {
    std::cout << "Failed to open " << filename << std::endl;
    exit(1);
  }

  std::vector<std::string> lines;
  {
    std::string line;
    while (std::getline(input, line)) {
      lines.push_back(line);
    }
  }
  input.close();

  double rc_max = 0.0;
  size_t li = 0;
  while (li < lines.size()) {
    // Skip comments, empty lines, and the GPUMD header ("uf3 N elem1 ...")
    if (lines[li].empty() || lines[li][0] == '#') { li++; continue; }
    {
      std::istringstream iss_test(lines[li]);
      std::string first;
      iss_test >> first;
      if (first == "uf3") { li++; continue; } // GPUMD header line
    }

    std::istringstream iss(lines[li]);
    std::string body_type;
    iss >> body_type;

    if (body_type == "2B") {
      // Format: 2B elem1 elem2 leading_trim trailing_trim knot_type
      std::string e1, e2;
      int leading_trim, trailing_trim;
      std::string kt_str;
      iss >> e1 >> e2 >> leading_trim >> trailing_trim >> kt_str;
      int knot_type = (kt_str == "uk") ? 1 : 0;
      (void)trailing_trim; // always 3 for cubic

      li++;
      // cutoff and knot count
      {
        std::istringstream iss2(lines[li]);
        double rc_val; int nk;
        iss2 >> rc_val >> nk;
        two_body.rc = rc_val;
        two_body.nknots = nk;
        two_body.nint = nk - 1;
        two_body.knot_type = knot_type;
        if (rc_val > rc_max) rc_max = rc_val;
      }

      li++;
      // knot vector
      std::vector<float> knots(two_body.nknots);
      {
        std::istringstream iss2(lines[li]);
        for (int i = 0; i < two_body.nknots; i++) iss2 >> knots[i];
      }

      li++;
      // coefficient count
      int ncoeff;
      {
        std::istringstream iss2(lines[li]);
        iss2 >> ncoeff;
      }

      li++;
      // coefficient vector
      std::vector<float> coeffs(ncoeff);
      {
        std::istringstream iss2(lines[li]);
        for (int i = 0; i < ncoeff; i++) iss2 >> coeffs[i];
      }

      // Pre-compute per-interval combined cubic polynomials
      {
        std::vector<float4> h_coeff;
        precompute_2b_uniform(knots, coeffs, leading_trim, h_coeff);
        two_body.d_coeff.resize(h_coeff.size());
        two_body.d_coeff.copy_from_host(h_coeff.data());
      }

      // Store knot info for GPU
      two_body.knot_min = knots[0];
      two_body.knot_delta = (knots[two_body.nknots - 1] - knots[0]) / (two_body.nknots - 1);
      {
        two_body.d_knots.resize(two_body.nknots);
        two_body.d_knots.copy_from_host(knots.data());
      }

      has_2b = true;
    }
    else if (body_type == "3B") {
      // Format: 3B elem1 elem2 elem3 leading_trim trailing_trim knot_type
      std::string e1, e2, e3;
      int leading_trim, trailing_trim;
      std::string kt_str;
      iss >> e1 >> e2 >> e3 >> leading_trim >> trailing_trim >> kt_str;
      int knot_type = (kt_str == "uk") ? 1 : 0;
      (void)trailing_trim;

      li++;
      // cutoffs and knot counts: rc_jk rc_ik rc_ij nk_jk nk_ik nk_ij
      {
        std::istringstream iss2(lines[li]);
        iss2 >> three_body.rc_jk >> three_body.rc_ik >> three_body.rc_ij
             >> three_body.nk_jk >> three_body.nk_ik >> three_body.nk_ij;
        three_body.nint_ij = three_body.nk_ij - 1;
        three_body.nint_ik = three_body.nk_ik - 1;
        three_body.nint_jk = three_body.nk_jk - 1;
        three_body.knot_type = knot_type;
        double rc3 = three_body.rc_ij;
        if (three_body.rc_ik > rc3) rc3 = three_body.rc_ik;
        if (rc3 > rc_max) rc_max = rc3;
      }

      // Read 3 knot vectors
      std::vector<float> k_ij(three_body.nk_ij);
      std::vector<float> k_ik(three_body.nk_ik);
      std::vector<float> k_jk(three_body.nk_jk);

      li++; { std::istringstream iss2(lines[li]);
        for (int i = 0; i < three_body.nk_jk; i++) iss2 >> k_jk[i]; }
      li++; { std::istringstream iss2(lines[li]);
        for (int i = 0; i < three_body.nk_ik; i++) iss2 >> k_ik[i]; }
      li++; { std::istringstream iss2(lines[li]);
        for (int i = 0; i < three_body.nk_ij; i++) iss2 >> k_ij[i]; }

      li++;
      // coefficient dimensions: dim1 dim2 dim3
      {
        std::istringstream iss2(lines[li]);
        iss2 >> three_body.nc_ij >> three_body.nc_ik >> three_body.nc_jk;
      }

      // Read coefficient tensor rows
      int total_rows = three_body.nc_ij * three_body.nc_ik;
      int row_len = three_body.nc_jk;
      std::vector<float> tensor(total_rows * row_len, 0.0f);
      for (int r = 0; r < total_rows; r++) {
        li++;
        std::istringstream iss2(lines[li]);
        int offset = r * row_len;
        for (int c = 0; c < row_len; c++) iss2 >> tensor[offset + c];
      }

      // Pre-compute basis polynomials
      {
        std::vector<float4> h_basis;
        precompute_3b_basis_uniform(k_ij, leading_trim, h_basis);
        three_body.d_basis_ij.resize(h_basis.size());
        three_body.d_basis_ij.copy_from_host(h_basis.data());
      }
      {
        std::vector<float4> h_basis;
        precompute_3b_basis_uniform(k_ik, leading_trim, h_basis);
        three_body.d_basis_ik.resize(h_basis.size());
        three_body.d_basis_ik.copy_from_host(h_basis.data());
      }
      {
        std::vector<float4> h_basis;
        precompute_3b_basis_uniform(k_jk, leading_trim, h_basis);
        three_body.d_basis_jk.resize(h_basis.size());
        three_body.d_basis_jk.copy_from_host(h_basis.data());
      }

      // Upload coefficient tensor
      three_body.d_tensor.resize(tensor.size());
      three_body.d_tensor.copy_from_host(tensor.data());

      // Store knot info
      three_body.knot_min_ij = k_ij[0];
      three_body.knot_min_ik = k_ik[0];
      three_body.knot_min_jk = k_jk[0];
      three_body.knot_delta_ij = (k_ij[three_body.nk_ij - 1] - k_ij[0]) / (three_body.nk_ij - 1);
      three_body.knot_delta_ik = (k_ik[three_body.nk_ik - 1] - k_ik[0]) / (three_body.nk_ik - 1);
      three_body.knot_delta_jk = (k_jk[three_body.nk_jk - 1] - k_jk[0]) / (three_body.nk_jk - 1);

      // Upload knot arrays
      three_body.d_knots_ij.resize(k_ij.size()); three_body.d_knots_ij.copy_from_host(k_ij.data());
      three_body.d_knots_ik.resize(k_ik.size()); three_body.d_knots_ik.copy_from_host(k_ik.data());
      three_body.d_knots_jk.resize(k_jk.size()); three_body.d_knots_jk.copy_from_host(k_jk.data());

      has_3b = true;
    }
    li++;
  }

  rc = rc_max;
  if (!has_2b && !has_3b) {
    std::cout << "UF3 potential file has no 2B or 3B blocks." << std::endl;
    exit(1);
  }

  // Allocate neighbor list and partial-force buffers
  neighbor.initialize(rc, number_of_atoms, max_neighbor_);
  if (has_3b) {
    f12x.resize(max_neighbor_ * number_of_atoms);
    f12y.resize(max_neighbor_ * number_of_atoms);
    f12z.resize(max_neighbor_ * number_of_atoms);
  }

  printf("Use UF3 potential.\n");
  if (has_2b) {
    printf("    2B: rc=%.1f A, %d intervals, %d knots\n",
           two_body.rc, two_body.nint, two_body.nknots);
  }
  if (has_3b) {
    printf("    3B: rc(ij,ik,jk)=(%.1f,%.1f,%.1f) A, coeff dims=%dx%dx%d\n",
           three_body.rc_ij, three_body.rc_ik, three_body.rc_jk,
           three_body.nc_ij, three_body.nc_ik, three_body.nc_jk);
  }
}

void UF3::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  neighbor.find_neighbor_global(rc, box, type, position_per_atom);

  if (has_2b) {
    find_force_uf3_2b<<<grid_size, BLOCK_SIZE>>>(
      N, N1, N2, box,
      two_body.d_coeff.data(), two_body.nint,
      two_body.knot_min, two_body.knot_delta, two_body.rc,
      neighbor.NN.data(), neighbor.NL.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      potential_per_atom.data(),
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data());
    GPU_CHECK_KERNEL
  }

  if (has_3b) {
    // Zero the partial force buffers
    cudaMemset(f12x.data(), 0, f12x.size() * sizeof(float));
    cudaMemset(f12y.data(), 0, f12y.size() * sizeof(float));
    cudaMemset(f12z.data(), 0, f12z.size() * sizeof(float));

    find_force_uf3_3b<<<grid_size, BLOCK_SIZE>>>(
      N, N1, N2, box,
      three_body.d_tensor.data(),
      three_body.nc_ij, three_body.nc_ik, three_body.nc_jk,
      three_body.d_basis_ij.data(), three_body.nint_ij,
      three_body.d_basis_ik.data(), three_body.nint_ik,
      three_body.d_basis_jk.data(), three_body.nint_jk,
      three_body.knot_min_ij, three_body.knot_delta_ij,
      three_body.knot_min_ik, three_body.knot_delta_ik,
      three_body.knot_min_jk, three_body.knot_delta_jk,
      three_body.rc_ij, three_body.rc_ik, three_body.rc_jk,
      neighbor.NN.data(), neighbor.NL.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      f12x.data(), f12y.data(), f12z.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL

    find_properties_many_body(
      box,
      neighbor.NN.data(),
      neighbor.NL.data(),
      f12x.data(),
      f12y.data(),
      f12z.data(),
      false, // is_dipole
      position_per_atom,
      force_per_atom,
      virial_per_atom);
  }
}
