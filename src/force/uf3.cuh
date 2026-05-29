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
The UF3 (Ultra-Fast Force Field) potential.
Ref: S. R. Xie et al., "Ultra-fast interpretable machine-learning potentials",
     npj Comput. Mater. 9, 143 (2023).
------------------------------------------------------------------------------*/

#pragma once
#include "neighbor.cuh"
#include "potential.cuh"
#include "utilities/gpu_vector.cuh"

class UF3 : public Potential
{
public:
  using Potential::compute;

  UF3(const char* filename, const int number_of_atoms, const int max_neighbor = 400);
  virtual ~UF3(void);

  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial) override;

  virtual const GPU_Vector<int>& get_NN_radial_ptr() override { return neighbor.NN; }
  virtual const GPU_Vector<int>& get_NL_radial_ptr() override { return neighbor.NL; }

private:
  void initialize(const char* filename, const int number_of_atoms);

  // ---- 2-body data ----
  struct {
    GPU_Vector<float4> d_coeff;  // [nint] combined cubic per interval (A,B,C,D)
    GPU_Vector<float> d_knots;   // [nknots] for non-uniform knot lookup
    double rc;
    int nknots, nint;
    int knot_type;               // 0=non-uniform, 1=uniform
    float knot_min, knot_delta, inv_knot_delta;  // uniform interval lookup
  } two_body;

  // ---- 3-body data ----
  struct {
    GPU_Vector<float> d_tensor;      // flattened coefficient tensor [nc_ij*nc_ik*nc_jk]
    GPU_Vector<float4> d_basis_ij;   // [nint_ij] 4 basis cubic coefficients per interval
    GPU_Vector<float4> d_basis_ik;
    GPU_Vector<float4> d_basis_jk;
    GPU_Vector<float> d_knots_ij, d_knots_ik, d_knots_jk;
    int nc_ij, nc_ik, nc_jk;         // coefficient dimensions
    int nint_ij, nint_ik, nint_jk;   // intervals per dim
    int nk_ij, nk_ik, nk_jk;         // knot counts
    int knot_type;
    double rc_ij, rc_ik, rc_jk;
    float knot_min_ij, knot_delta_ij, inv_knot_delta_ij;
    float knot_min_ik, knot_delta_ik, inv_knot_delta_ik;
    float knot_min_jk, knot_delta_jk, inv_knot_delta_jk;
  } three_body;

  bool has_2b, has_3b;
  int max_neighbor_;
  Neighbor neighbor;

  // Partial force buffers for 3B (Tersoff-style)
  GPU_Vector<float> f12x, f12y, f12z;
  GPU_Vector<int> NN_3b, NL_3b;  // local neighbor list (filtered to 3B cutoff)
};
