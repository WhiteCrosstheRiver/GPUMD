/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#pragma once
#include "utilities/gpu_vector.cuh"

// fix defines dynamics; each potential defines its own force dependency.
enum class InfluencePolicy {
  Full,        // compute all atoms (default, always correct)
  ActiveOnly,  // pair-like: centers = active atoms
  LocalRadius, // many-body: active plus a local influence region
  Custom
};

struct ComputeRequest {
  bool need_active_force = true;
  bool need_full_energy = false;
  bool need_full_virial = false;
};

struct InfluenceDomain {
  GPU_Vector<int> active_indices;
  GPU_Vector<int> influence_indices;
};
