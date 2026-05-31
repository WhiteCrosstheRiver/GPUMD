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

#pragma once

// ===========================================================================
// General (clamped / non-uniform) cubic B-spline basis as a per-interval
// polynomial table.  Reference UF3 uses scipy clamped B-splines (repeated end
// knots, interpolatory at the endpoints); the previous uniform basis left the
// edge intervals clamped to constants, which hurt the fit.  This primitive
// works for ANY knot vector (uniform or clamped).
//
// For each real (nonzero-width) knot interval [t[m], t[m+1]] the four nonzero
// cubic basis functions  N_{m-3}, N_{m-2}, N_{m-1}, N_m  are stored as cubic
// polynomials in the local coordinate u = (r - t[m]) / (t[m+1] - t[m]) in [0,1].
// p0[seg] = m-3 is the global index of the first of those four coefficients.
//
// Device side: a small struct of pointers + the segment count.  Evaluation
// searches the (short) segment-boundary array, then evaluates 4 cubics.
// ===========================================================================

#include <vector>
#include <cmath>

// ---- Host: Cox-de Boor recursion (degree 3), handles repeated knots ---------
static inline double bb_cox_deboor(const std::vector<float>& t, int i, int p, double u)
{
  if (p == 0) {
    // Half-open [t[i], t[i+1]); the caller evaluates strictly inside a segment.
    return (u >= t[i] && u < t[i + 1]) ? 1.0 : 0.0;
  }
  double d1 = (double)t[i + p] - (double)t[i];
  double d2 = (double)t[i + p + 1] - (double)t[i + 1];
  double a = (d1 > 1e-12) ? (u - t[i]) / d1 * bb_cox_deboor(t, i, p - 1, u) : 0.0;
  double b = (d2 > 1e-12) ? (t[i + p + 1] - u) / d2 * bb_cox_deboor(t, i + 1, p - 1, u) : 0.0;
  return a + b;
}

// Host-side basis table (flat arrays, ready to upload).
struct BSplineTableHost {
  int nseg = 0;
  std::vector<float> seg_lo;     // [nseg]   lower bound of each real interval
  std::vector<float> seg_invw;   // [nseg]   1 / (hi - lo)
  std::vector<int>   seg_p0;     // [nseg]   first coefficient index (m-3)
  std::vector<float> poly;       // [nseg*16] 4 basis x 4 cubic coeffs (a,b,c,d in u)
};

// Build the per-interval basis-polynomial table from a knot vector.
// ncoeff = (int)knots.size() - 4.
static inline void build_bspline_table(const std::vector<float>& knots, int ncoeff,
                                       BSplineTableHost& tab)
{
  int nk = (int)knots.size();
  // Fixed 4-point Vandermonde inverse for u = {0.125, 0.375, 0.625, 0.875}
  // (well-separated interior nodes) — exact for cubics.
  const double un[4] = {0.125, 0.375, 0.625, 0.875};
  // Build and invert V (4x4) once.
  double V[4][4], Vinv[4][4];
  for (int r = 0; r < 4; r++) { double u = un[r]; V[r][0]=1; V[r][1]=u; V[r][2]=u*u; V[r][3]=u*u*u; }
  // Gauss-Jordan inverse.
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) Vinv[i][j] = (i==j)?1.0:0.0;
  double M[4][4]; for (int i=0;i<4;i++) for(int j=0;j<4;j++) M[i][j]=V[i][j];
  for (int col = 0; col < 4; col++) {
    int piv = col; for (int r=col+1;r<4;r++) if (std::fabs(M[r][col])>std::fabs(M[piv][col])) piv=r;
    for (int j=0;j<4;j++){ std::swap(M[col][j],M[piv][j]); std::swap(Vinv[col][j],Vinv[piv][j]); }
    double d = M[col][col];
    for (int j=0;j<4;j++){ M[col][j]/=d; Vinv[col][j]/=d; }
    for (int r=0;r<4;r++) if (r!=col){ double f=M[r][col]; for(int j=0;j<4;j++){ M[r][j]-=f*M[col][j]; Vinv[r][j]-=f*Vinv[col][j]; } }
  }

  tab.nseg = 0; tab.seg_lo.clear(); tab.seg_invw.clear(); tab.seg_p0.clear(); tab.poly.clear();
  for (int m = 3; m < nk - 4; m++) {            // real intervals contributing 4 basis funcs
    double lo = knots[m], hi = knots[m + 1];
    if (hi - lo < 1e-9) continue;               // zero-width (repeated knot) — skip
    int p0 = m - 3;
    tab.seg_lo.push_back((float)lo);
    tab.seg_invw.push_back((float)(1.0 / (hi - lo)));
    tab.seg_p0.push_back(p0);
    for (int b = 0; b < 4; b++) {               // basis N_{p0+b}
      int idx = p0 + b;
      double vals[4];
      for (int s = 0; s < 4; s++) {
        double r = lo + un[s] * (hi - lo);
        vals[s] = bb_cox_deboor(knots, idx, 3, r);
      }
      // poly coeffs c = Vinv * vals
      for (int j = 0; j < 4; j++) {
        double cj = 0; for (int s = 0; s < 4; s++) cj += Vinv[j][s] * vals[s];
        tab.poly.push_back((float)cj);
      }
    }
    tab.nseg++;
  }
}

// ---- Device: table view + evaluator -----------------------------------------
struct BSplineTableDev {
  int nseg;
  const float* seg_lo;     // [nseg]
  const float* seg_invw;   // [nseg]
  const int*   seg_p0;     // [nseg]
  const float* poly;       // [nseg*16]
};

// Find the segment containing r (linear scan over the short boundary array).
// Returns segment index, clamped to [0, nseg-1] for r outside the range.
__device__ __forceinline__ int bs_find_seg(const BSplineTableDev& t, float r)
{
  int m = 0;
  // seg_lo is increasing; find last seg with seg_lo <= r.
  for (int s = 1; s < t.nseg; s++) { if (t.seg_lo[s] <= r) m = s; else break; }
  return m;
}

// Evaluate the 4 basis values B[4], derivatives dB[4] (w.r.t. r), and first
// coefficient index p0 at distance r.
__device__ __forceinline__ void bs_eval(const BSplineTableDev& t, float r,
                                        float B[4], float dB[4], int& p0)
{
  int m = bs_find_seg(t, r);
  float u = (r - t.seg_lo[m]) * t.seg_invw[m];
  const float* P = t.poly + (size_t)m * 16;
  float u2 = u * u;
  #pragma unroll
  for (int b = 0; b < 4; b++) {
    float a0 = P[b*4+0], a1 = P[b*4+1], a2 = P[b*4+2], a3 = P[b*4+3];
    B[b]  = a0 + u * (a1 + u * (a2 + u * a3));
    dB[b] = (a1 + u * (2.0f*a2 + 3.0f*a3*u)) * t.seg_invw[m]; // d/dr = d/du * 1/width
  }
  (void)u2;
  p0 = t.seg_p0[m];
}

// Value-only (no derivative) variant for energy features.
__device__ __forceinline__ void bs_eval_val(const BSplineTableDev& t, float r,
                                             float B[4], int& p0)
{
  int m = bs_find_seg(t, r);
  float u = (r - t.seg_lo[m]) * t.seg_invw[m];
  const float* P = t.poly + (size_t)m * 16;
  #pragma unroll
  for (int b = 0; b < 4; b++)
    B[b] = P[b*4+0] + u * (P[b*4+1] + u * (P[b*4+2] + u * P[b*4+3]));
  p0 = t.seg_p0[m];
}
