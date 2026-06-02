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
#include <string>
#include <vector>

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

  // Number of element types and their symbols, parsed from the "uf3 N e1 e2 ..."
  // header.  Atom type index t corresponds to the position of the element in
  // this list (standard GPUMD convention — the potential file defines the type
  // ordering used by model.xyz / run.in).
  int num_types_ = 1;
  std::vector<std::string> elements_;

  // ---- 2-body data (per type pair) ----
  // The GPUMD trainer (main_uf3/main.cu::write_uf3_file) emits one 2B block per
  // ordered type pair (index = ti*num_types + tj) and all pairs share the same
  // uniform knot grid.  Coefficients for pair p occupy d_coeff[p*nint .. p*nint+nint).
  struct {
    GPU_Vector<float4> d_coeff;  // [num_pairs * nint] combined cubic per interval
    GPU_Vector<float> d_knots;   // [nknots] (uniform grid, shared by all pairs)
    double rc;
    int num_pairs;               // = num_types_ * num_types_
    int nknots, nint;
    int knot_type;               // 0=non-uniform, 1=uniform
    float knot_min, knot_delta, inv_knot_delta;  // uniform interval lookup
  } two_body;

  // ---- 3-body data (per type triplet) ----
  // One 3B block per type triplet (index = (ti*num_types + tj)*num_types + tk);
  // all triplets share the same grid.  Tensor for triplet t occupies
  // d_tensor[t*tensor_stride .. ] in jk-fastest layout.
  struct {
    GPU_Vector<float> d_tensor;      // [num_trips * nc_ij*nc_ik*nc_jk]
    int num_trips;                   // = num_types_^3
    int tensor_stride;               // = nc_ij*nc_ik*nc_jk
    int nc_ij, nc_ik, nc_jk;         // coefficient dimensions
    int nint_ij, nint_ik, nint_jk;   // intervals per dim (for interval clamping)
    int nk_ij, nk_ik, nk_jk;         // knot counts
    int knot_type;
    double rc_ij, rc_ik, rc_jk;
    float knot_min_ij, knot_delta_ij, inv_knot_delta_ij;
    float knot_min_ik, knot_delta_ik, inv_knot_delta_ik;
    float knot_min_jk, knot_delta_jk, inv_knot_delta_jk;
  } three_body;

  bool has_2b, has_3b;
  bool sym_3b_ = false;   // neighbour-swap symmetry (ij/ik legs share grid)
  int max_neighbor_;
  Neighbor neighbor;

  // Multi-image (ghost) neighbour list used when the cell is too small for the
  // minimum-image convention (any periodic thickness < 2*rc).  Each entry is an
  // (atom index, lattice-shift code) pair, so every periodic image within rc is
  // a distinct neighbour — matching the trainer's ghost-supercell distances.
  GPU_Vector<int> d_NN;
  GPU_Vector<int> d_NL;
  GPU_Vector<int> d_NL_shift;
  int neighbor_MN_ = 0;   // per-atom neighbour-list stride for the arrays above

  // Average 3B coefficients with their neighbour-swap partner so MD energies
  // are independent of neighbour-list ordering (mirrors main_uf3::project_3b_symmetric).
  void project_3b_symmetric(std::vector<float>& tensor) const;

  // Packed float4 positions (x,y,z,0) for coalesced gather in force kernel
  GPU_Vector<float4> d_pos_packed;

  // 1-body per-element energy offsets (loaded from .uf3 1B section)
  GPU_Vector<float> d_e0;
};
