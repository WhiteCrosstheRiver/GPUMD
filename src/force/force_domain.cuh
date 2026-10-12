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

// run.in keyword fix_skip (propagating). The defaults reproduce the skip path
// as it was before fix_skip existed.
struct SkipOptions {
  double min_frozen_fraction = 0.0; // below this frozen fraction compute all atoms
  int influence_every = 1;          // a potential may reuse its influence set for up to K steps
  double influence_skin = 0.0;      // halo margin (A); reuse is exact while no active atom moved this far
  bool cached_alloc = false;        // reuse Thrust scratch storage on the skip path
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
