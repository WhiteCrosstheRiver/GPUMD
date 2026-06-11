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
#include <string>
#include <vector>

struct Uf3Frame {
  int num_atoms = 0;   // number of REAL atoms (energy/forces centered on these)
  int num_total = 0;   // real + periodic ghost images (neighbors only).
                       //   == num_atoms for non-periodic frames.

  // Per-atom arrays.  Indices [0, num_atoms)        -> real atoms.
  //                    Indices [num_atoms, num_total) -> ghost images.
  // x/y/z/types/parent are sized to num_total; fx/fy/fz to num_atoms (real only).
  std::vector<int> types;
  std::vector<float> x, y, z;
  std::vector<float> fx, fy, fz;
  // parent[i] = real-atom index that ghost i is an image of (parent[i]=i for
  // real atoms).  Used to scatter ghost force contributions back to real atoms.
  std::vector<int> parent;
  float energy = 0.0f;

  // Reference virial in eV, symmetrised to 6 components (xx yy zz xy xz yz).
  // Parsed from virial="..." (9 values, row-major) or stress="..." (eV/A^3,
  // virial = -stress * cell volume).  has_virial=false when neither is present;
  // such frames simply contribute no virial rows to the fit.
  float virial[6] = {};
  bool has_virial = false;

  // Lattice matrix (column-major, same convention as main_nep):
  //   box[0..8]  = H = [a|b|c], where a,b,c are lattice vectors as columns
  //   box_inv[0..8] = H^{-1} (for minimum-image convention)
  float box[9]     = {};
  float box_inv[9] = {};
  bool has_lattice = false;

  // Pre-built neighbor list within 3B cutoff.  Entries index into the expanded
  // [0, num_total) atom array, so ghost images appear as ordinary neighbors and
  // distances can be computed directly from x/y/z (no PBC math in the kernel).
  std::vector<int> nn_counts;
  std::vector<int> nn_offset;
  std::vector<int> nn_list;
};

// elements: ordered list from UF3_Parameters (e.g. {"Si","Ge"}).
// ghost_cutoff > 0 triggers periodic ghost-atom expansion (images within
//   ghost_cutoff of any real atom are appended for correct PBC neighbor finding).
// nn_cutoff > 0 additionally builds the 3B neighbor list (over expanded atoms).
std::vector<Uf3Frame> load_uf3_frames(
  const char* filename,
  const std::vector<std::string>& elements,
  float nn_cutoff = 0.0f,
  float ghost_cutoff = 0.0f);
