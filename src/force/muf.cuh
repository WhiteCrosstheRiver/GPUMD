/*
    Copyright 2026 MUF development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    MUF-C (Chemical) GPU implementation — multi-element alloy support
*/

#pragma once
#include "neighbor.cuh"
#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"

// MUF model parameters (MUF-C multi-element)
struct MUF_Param {
  // Core parameters
  int K = 0;                    // B-spline coefficient count
  int L_max = 0;                // max angular channel (1..L_max)
  int num_sh_terms = 0;         // (L_max+1)^2 - 1
  int band_width = 0;           // W band width

  // Geometry
  double rc_2b = 0.0;           // 2B cutoff radius
  double rc_3b = 0.0;           // 3B cutoff radius
  double r_min_2b = 0.0;        // 2B spline lower bound
  double r_min_3b = 0.0;        // 3B spline lower bound
  double inv_kdelta_2b = 0.0;   // 1/kdelta for 2B
  double inv_kdelta_3b = 0.0;   // 1/kdelta for 3B
  double kdelta_2b = 0.0;       // 2B knot spacing
  double kdelta_3b = 0.0;       // 3B knot spacing

  // Multi-element
  int num_types = 0;            // number of element types
  int num_JJ_pairs = 0;         // num_types*(num_types+1)/2
  int num_ab_pairs = 0;         // K*(K+1)/2
  int MN = 200;                 // max neighbors per atom

  // 1B offsets per type (GPU)
  GPU_Vector<double> e0;        // [num_types]

  // 2B coefficients per JJ pair (GPU)
  GPU_Vector<double> coeff_2b;  // [num_JJ_pairs * K]

  // 3B weights: [num_types * num_JJ_pairs * num_ab_pairs * L_max] (GPU)
  GPU_Vector<double> W;
};

// MUF runtime data (GPU buffers)
struct MUF_Data {
  GPU_Vector<int> NN;           // neighbor count per atom
  GPU_Vector<int> NL;           // flat neighbor list (GPU)

  GPU_Vector<double> moments;   // type-channel descriptors
  // [N * num_types * K * num_sh_terms]
  GPU_Vector<double> dE_dA;     // gradient adjoint (same size as moments)
  GPU_Vector<double> energy_3b; // per-atom 3B energy

  // Type array (GPU) — copied from main GPUMD type vector
};

class MUF : public Potential
{
public:
  MUF_Param param;
  MUF_Data muf_data;
  Neighbor neighbor;           // GPUMD GPU neighbor list

  // Constructor: parse model file
  MUF(FILE* fid, const int num_types, const int number_of_atoms);

  // Destructor
  ~MUF() {}

  // Main compute method (implements Potential interface)
  void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom) override;

private:
  int number_of_atoms_ = -1;

  // Parse .muf file
  void parse_model_file(FILE* fid, int num_types);
};
