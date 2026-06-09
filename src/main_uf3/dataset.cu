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
Extended-XYZ loader for UF3 training data.

Design goals (matching main_nep/structure.cu):
  1. Element type mapping driven by UF3_Parameters::elements — no hardcoding.
  2. Lattice matrix parsed from Lattice="..." header token.
  3. Minimum-image convention (MIC) applied when building the 3B neighbor list.
  4. Properties= column offsets parsed so species/pos/forces can appear in any
     order; default layout species:pos:forces is handled without Properties=.
------------------------------------------------------------------------------*/

#include "dataset.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// ---------------------------------------------------------------------------
// Box helpers — column-major 3×3 (same convention as main_nep)
// ---------------------------------------------------------------------------

// Compute 3×3 matrix determinant (column-major h[col*3 + row])
static float box_det(const float h[9])
{
  return h[0] * (h[4] * h[8] - h[5] * h[7])
       - h[3] * (h[1] * h[8] - h[2] * h[7])
       + h[6] * (h[1] * h[5] - h[2] * h[4]);
}

// Compute H^{-1} and store column-major in h_inv
static void box_inverse(const float h[9], float h_inv[9])
{
  float det = box_det(h);
  float inv = 1.0f / det;
  h_inv[0] =  (h[4]*h[8] - h[5]*h[7]) * inv;
  h_inv[1] = -(h[1]*h[8] - h[2]*h[7]) * inv;
  h_inv[2] =  (h[1]*h[5] - h[2]*h[4]) * inv;
  h_inv[3] = -(h[3]*h[8] - h[5]*h[6]) * inv;
  h_inv[4] =  (h[0]*h[8] - h[2]*h[6]) * inv;
  h_inv[5] = -(h[0]*h[5] - h[2]*h[3]) * inv;
  h_inv[6] =  (h[3]*h[7] - h[4]*h[6]) * inv;
  h_inv[7] = -(h[0]*h[7] - h[1]*h[6]) * inv;
  h_inv[8] =  (h[0]*h[4] - h[1]*h[3]) * inv;
}

// Apply minimum-image convention to (dx,dy,dz) using precomputed H and H^{-1}
static void apply_mic(
  const float h[9], const float h_inv[9],
  float& dx, float& dy, float& dz)
{
  // Fractional coordinates
  float sx = h_inv[0]*dx + h_inv[3]*dy + h_inv[6]*dz;
  float sy = h_inv[1]*dx + h_inv[4]*dy + h_inv[7]*dz;
  float sz = h_inv[2]*dx + h_inv[5]*dy + h_inv[8]*dz;
  // Wrap to [-0.5, 0.5)
  sx -= std::round(sx);
  sy -= std::round(sy);
  sz -= std::round(sz);
  // Back to Cartesian
  dx = h[0]*sx + h[3]*sy + h[6]*sz;
  dy = h[1]*sx + h[4]*sy + h[7]*sz;
  dz = h[2]*sx + h[5]*sy + h[8]*sz;
}

// ---------------------------------------------------------------------------
// Header-line parsing helpers
// ---------------------------------------------------------------------------

// Tokenise a string by whitespace, respecting quoted regions.
// Quotes are used in extended-XYZ for Lattice="..." values.
static std::vector<std::string> tokenise_header(const std::string& line)
{
  std::vector<std::string> tokens;
  std::string cur;
  bool in_quote = false;
  for (char c : line) {
    if (c == '"') {
      in_quote = !in_quote;
    } else if (std::isspace((unsigned char)c) && !in_quote) {
      if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
    } else {
      cur += c;
    }
  }
  if (!cur.empty()) tokens.push_back(cur);
  return tokens;
}

// Extract value of key="value" or key=value from a token list.
// Returns empty string if not found.
static std::string find_key(const std::vector<std::string>& tokens, const std::string& key)
{
  for (const auto& t : tokens) {
    if (t.size() > key.size() && t.substr(0, key.size()) == key) {
      return t.substr(key.size());
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// Properties= column layout parser
// Determines offsets of species, pos, forces columns.
// Default: species:S:1 pos:R:3 forces:R:3
// ---------------------------------------------------------------------------
struct ColLayout {
  int species = 0;
  int pos     = 1;
  int force   = 4;
  int ncols   = 7;
  bool valid  = true;
};

static ColLayout parse_properties(const std::string& prop_str)
{
  ColLayout lay;
  if (prop_str.empty()) return lay;   // use defaults

  // Replace ':' with spaces for tokenisation
  std::string s = prop_str;
  for (char& c : s) if (c == ':') c = ' ';
  std::istringstream iss(s);
  std::vector<std::string> sub;
  std::string tok;
  while (iss >> tok) sub.push_back(tok);

  if (sub.size() % 3 != 0) return lay;   // malformed; fall back to defaults

  int col = 0, sp = -1, pp = -1, fp = -1;
  for (int k = 0; k < (int)sub.size() / 3; k++) {
    const std::string& name = sub[k * 3];
    int n = std::stoi(sub[k * 3 + 2]);
    if (name == "species") sp = col;
    else if (name == "pos")    pp = col;
    else if (name == "force" || name == "forces") fp = col;
    col += n;
  }

  if (sp < 0 || pp < 0) { lay.valid = false; return lay; }
  lay.species = sp;
  lay.pos     = pp;
  lay.force   = (fp >= 0) ? fp : -1;
  lay.ncols   = col;
  return lay;
}

// ---------------------------------------------------------------------------
// Periodic ghost-atom expansion
// ---------------------------------------------------------------------------
// Append periodic images of the real atoms that fall within `cutoff` of any
// real atom.  After this, all interactions within `cutoff` can be found by
// plain distance computation over [0, num_total) with no PBC math in kernels.
// Correct even when the cell is smaller than 2*cutoff (multiple images of the
// same atom are added), unlike minimum-image convention.
static void generate_ghosts(Uf3Frame& f, float cutoff)
{
  const float* H = f.box;            // columns a,b,c
  const float ax = H[0], ay = H[1], az = H[2];   // a = column 0
  const float bx = H[3], by = H[4], bz = H[5];   // b = column 1
  const float cx = H[6], cy = H[7], cz = H[8];   // c = column 2

  // Perpendicular spacings (interplanar distances) to bound the image range.
  auto cross = [](float ux, float uy, float uz, float vx, float vy, float vz,
                  float& rx, float& ry, float& rz) {
    rx = uy * vz - uz * vy;
    ry = uz * vx - ux * vz;
    rz = ux * vy - uy * vx;
  };
  float nx, ny, nz;
  cross(bx, by, bz, cx, cy, cz, nx, ny, nz);
  float vol = std::fabs(ax * nx + ay * ny + az * nz);
  if (vol < 1e-6f) return;  // degenerate cell

  auto perp = [&](float ux, float uy, float uz, float vx, float vy, float vz) {
    float rx, ry, rz;
    cross(ux, uy, uz, vx, vy, vz, rx, ry, rz);
    float area = std::sqrt(rx * rx + ry * ry + rz * rz);
    return vol / std::max(area, 1e-6f);
  };
  int N1 = (int)std::ceil(cutoff / perp(bx, by, bz, cx, cy, cz));
  int N2 = (int)std::ceil(cutoff / perp(ax, ay, az, cx, cy, cz));
  int N3 = (int)std::ceil(cutoff / perp(ax, ay, az, bx, by, bz));

  int n_real = f.num_atoms;
  const float cut2 = cutoff * cutoff;
  for (int i1 = -N1; i1 <= N1; i1++) {
    for (int i2 = -N2; i2 <= N2; i2++) {
      for (int i3 = -N3; i3 <= N3; i3++) {
        if (i1 == 0 && i2 == 0 && i3 == 0) continue;
        float sx = i1 * ax + i2 * bx + i3 * cx;
        float sy = i1 * ay + i2 * by + i3 * cy;
        float sz = i1 * az + i2 * bz + i3 * cz;
        for (int i = 0; i < n_real; i++) {
          float gx = f.x[i] + sx, gy = f.y[i] + sy, gz = f.z[i] + sz;
          // Keep ghost only if within cutoff of at least one real atom.
          bool keep = false;
          for (int j = 0; j < n_real; j++) {
            float dx = gx - f.x[j], dy = gy - f.y[j], dz = gz - f.z[j];
            if (dx * dx + dy * dy + dz * dz < cut2) { keep = true; break; }
          }
          if (!keep) continue;
          f.x.push_back(gx);
          f.y.push_back(gy);
          f.z.push_back(gz);
          f.types.push_back(f.types[i]);
          f.parent.push_back(i);
        }
      }
    }
  }
  f.num_total = (int)f.x.size();
}

// ---------------------------------------------------------------------------
// Main loader
// ---------------------------------------------------------------------------
std::vector<Uf3Frame> load_uf3_frames(
  const char* filename,
  const std::vector<std::string>& elements,
  float nn_cutoff,
  float ghost_cutoff)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    std::cerr << "Error: cannot open " << filename << std::endl;
    exit(1);
  }

  const float cutoff_sq = nn_cutoff * nn_cutoff;
  std::vector<Uf3Frame> frames;
  std::string line;

  while (std::getline(input, line)) {
    if (line.empty()) continue;

    int natoms = 0;
    try { natoms = std::stoi(line); }
    catch (...) { continue; }

    Uf3Frame f;
    f.num_atoms = natoms;
    f.types.resize(natoms);
    f.x.resize(natoms); f.y.resize(natoms); f.z.resize(natoms);
    f.fx.resize(natoms); f.fy.resize(natoms); f.fz.resize(natoms);

    // ---- Header line -------------------------------------------------------
    std::getline(input, line);
    auto header = tokenise_header(line);

    // Energy
    std::string ev = find_key(header, "energy=");
    if (!ev.empty()) {
      try { f.energy = std::stof(ev); } catch (...) {}
    }

    // Lattice (row-major in extxyz: rows are lattice vectors a,b,c)
    // Store column-major in f.box (same as main_nep): H = [a|b|c]
    std::string lv = find_key(header, "Lattice=");
    if (lv.empty()) lv = find_key(header, "lattice=");
    if (!lv.empty()) {
      // Strip surrounding quotes if present
      if (!lv.empty() && lv.front() == '"') lv = lv.substr(1);
      if (!lv.empty() && lv.back() == '"')  lv.pop_back();
      std::istringstream ls(lv);
      float raw[9] = {};
      for (int k = 0; k < 9; k++) ls >> raw[k];
      // raw = [a1 a2 a3 b1 b2 b3 c1 c2 c3] (row-major, each row = lattice vector)
      // Convert to column-major H: H[:,0]=a, H[:,1]=b, H[:,2]=c
      // H[row + 3*col] → col-major: H[0]=h11=a1, H[1]=h21=b1, H[2]=h31=c1, ...
      // transpose_index from main_nep: {0,3,6,1,4,7,2,5,8}
      static const int T[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
      for (int k = 0; k < 9; k++) f.box[T[k]] = raw[k];
      box_inverse(f.box, f.box_inv);
      f.has_lattice = true;
    }

    // Properties= column layout
    std::string pv = find_key(header, "Properties=");
    if (pv.empty()) pv = find_key(header, "properties=");
    ColLayout lay = parse_properties(pv);

    // ---- Atom lines --------------------------------------------------------
    bool frame_ok = true;
    for (int i = 0; i < natoms; i++) {
      if (!std::getline(input, line) || line.empty()) {
        std::cerr << "Warning: frame " << frames.size()
                  << " is truncated at atom " << i << "/" << natoms
                  << " — skipping frame.\n";
        frame_ok = false;
        break;
      }
      std::istringstream iss(line);
      std::vector<std::string> cols;
      std::string w;
      while (iss >> w) cols.push_back(w);

      // Species — match against elements list (same logic as main_nep)
      std::string elem = (lay.species < (int)cols.size()) ? cols[lay.species] : "";
      bool found = false;
      for (int t = 0; t < (int)elements.size(); t++) {
        if (elem == elements[t]) {
          f.types[i] = t;
          found = true;
          break;
        }
      }
      if (!found) {
        std::cerr << "Error: atom symbol '" << elem
                  << "' in " << filename
                  << " is not in the type list. Check the 'type' line in uf3.in.\n";
        exit(1);
      }

      // Position
      if (lay.pos + 2 < (int)cols.size()) {
        f.x[i] = std::stof(cols[lay.pos]);
        f.y[i] = std::stof(cols[lay.pos + 1]);
        f.z[i] = std::stof(cols[lay.pos + 2]);
      }

      // Forces
      if (lay.force >= 0 && lay.force + 2 < (int)cols.size()) {
        f.fx[i] = std::stof(cols[lay.force]);
        f.fy[i] = std::stof(cols[lay.force + 1]);
        f.fz[i] = std::stof(cols[lay.force + 2]);
      }
    }

    if (!frame_ok) continue;

    // ---- Periodic ghost-atom expansion ------------------------------------
    // Real atoms occupy [0, num_atoms); periodic images are appended so that
    // every interaction within the cutoff is reachable by a plain distance
    // computation (correct even for cells smaller than 2*cutoff).
    f.num_total = natoms;
    f.parent.resize(natoms);
    for (int i = 0; i < natoms; i++) f.parent[i] = i;
    float gcut = std::max(ghost_cutoff, nn_cutoff);
    if (f.has_lattice && gcut > 0.0f) {
      generate_ghosts(f, gcut);
    }

    // ---- 3B neighbor list (over expanded atoms; raw distances) ------------
    // Centers are the real atoms [0, num_atoms); neighbors range over the full
    // expanded set [0, num_total), so ghost images are ordinary neighbors.
    if (nn_cutoff > 0.0f) {
      int ntot = f.num_total;
      f.nn_counts.assign(natoms, 0);
      f.nn_offset.resize(natoms + 1, 0);

      // Pass 1: count
      for (int i = 0; i < natoms; i++) {
        for (int j = 0; j < ntot; j++) {
          if (j == i) continue;
          float dx = f.x[j] - f.x[i];
          float dy = f.y[j] - f.y[i];
          float dz = f.z[j] - f.z[i];
          if (dx*dx + dy*dy + dz*dz < cutoff_sq) f.nn_counts[i]++;
        }
      }
      for (int i = 0; i < natoms; i++)
        f.nn_offset[i + 1] = f.nn_offset[i] + f.nn_counts[i];
      f.nn_list.resize(f.nn_offset[natoms]);

      // Pass 2: fill
      std::vector<int> counters(natoms, 0);
      for (int i = 0; i < natoms; i++) {
        for (int j = 0; j < ntot; j++) {
          if (j == i) continue;
          float dx = f.x[j] - f.x[i];
          float dy = f.y[j] - f.y[i];
          float dz = f.z[j] - f.z[i];
          if (dx*dx + dy*dy + dz*dz < cutoff_sq)
            f.nn_list[f.nn_offset[i] + counters[i]++] = j;
        }
      }
    }

    frames.push_back(std::move(f));
  }

  input.close();
  return frames;
}
