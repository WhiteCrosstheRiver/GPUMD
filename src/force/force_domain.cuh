/*
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

// run.in keyword fix_skip. Without it every atom is computed, as in plain fix.
struct SkipOptions {
  bool enabled = false;
  double min_frozen_fraction = 0.0; // below this frozen fraction compute all atoms
};

enum class InfluencePolicy {
  Full,        // always compute every atom
  ActiveOnly,  // pair potentials: active forces need only active centers
  LocalRadius, // many-body potentials: centers also need atoms within the cutoff
};
