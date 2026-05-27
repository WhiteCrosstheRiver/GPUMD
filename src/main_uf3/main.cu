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
UF3 (Ultra-Fast Force Field) training — main entry point.
GPU-accelerated evolutionary-strategy optimizer for B-spline coefficients.
------------------------------------------------------------------------------*/

#include "parameters.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/main_common.cuh"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// ---------------------------------------------------------------------------
// GPU: combined cubic coefficients + knot metadata (matches force/uf3.cu)
// ---------------------------------------------------------------------------
__device__ inline float eval_cubic(float4 c, float u)
{
  return c.x + u * (c.y + u * (c.z + u * c.w));
}

__device__ inline int find_interval(float r, float knot_min, float knot_delta, int nint)
{
  float t = (r - knot_min) / knot_delta;
  int i = (int)t;
  if (i < 0) i = 0;
  if (i >= nint) i = nint - 1;
  return i;
}

// GPU kernel: evaluate 2-body energy for a batch of frames
// One block per frame, single-thread per block (enough for ~128-atom frames)
static __global__ void eval_2b_batch(
  int num_frames,
  const int* __restrict__ d_frame_idx,   // [num_frames] which global frame to use
  const int* __restrict__ d_natoms,      // [total_frames] atom count per frame
  const int* __restrict__ d_offsets,     // [total_frames+1] atom offset in flat arrays
  const int* __restrict__ d_types,       // flat types array
  const float* __restrict__ d_x,         // flat position arrays
  const float* __restrict__ d_y,
  const float* __restrict__ d_z,
  const float4* __restrict__ d_coeff,    // [num_pairs * nint] combined cubic
  int num_pairs, int nint,
  float knot_min, float knot_delta, float rc,
  const int* __restrict__ d_type_map,    // [num_pairs] maps pair index to coeff offset
  float* __restrict__ d_energy)          // [num_frames] output energy
{
  int b = blockIdx.x;
  if (b >= num_frames) return;

  int fidx = d_frame_idx[b];
  int n = d_natoms[fidx];
  int off = d_offsets[fidx];
  float pe = 0.0f;

  // Single-thread per frame: iterate all pairs within cutoff
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      float dx = d_x[off + i] - d_x[off + j];
      float dy = d_y[off + i] - d_y[off + j];
      float dz = d_z[off + i] - d_z[off + j];
      float r = sqrtf(dx * dx + dy * dy + dz * dz);
      if (r >= rc) continue;
      int ti = d_types[off + i], tj = d_types[off + j];
      // type_map: (ti, tj) -> pair coefficient offset
      int num_types = (int)sqrtf((float)num_pairs + 0.5f);
      int pair_idx = d_type_map[ti * num_types + tj];
      int m = find_interval(r, knot_min, knot_delta, nint);
      float h = knot_delta;
      float u = (r - (knot_min + m * h)) / h;
      float4 c = __ldg(&d_coeff[pair_idx * nint + m]);
      pe += eval_cubic(c, u);
    }
  }
  d_energy[fidx] = pe;
}

// ---------------------------------------------------------------------------
// CPU: file loading
// ---------------------------------------------------------------------------
struct Frame {
  int num_atoms;
  std::vector<int> types;
  std::vector<float> x, y, z;
  float energy;
};

static std::vector<Frame> load_xyz(const char* filename)
{
  std::vector<Frame> frames;
  std::ifstream input(filename);
  if (!input.is_open()) {
    printf("Error: cannot open %s\n", filename);
    exit(1);
  }
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    int natoms = std::stoi(line);
    Frame f;
    f.num_atoms = natoms;
    std::getline(input, line);
    size_t pos = line.find("energy=");
    if (pos != std::string::npos) f.energy = std::stof(line.substr(pos + 7));
    f.types.resize(natoms);
    f.x.resize(natoms); f.y.resize(natoms); f.z.resize(natoms);
    for (int i = 0; i < natoms; i++) {
      std::getline(input, line);
      std::istringstream iss(line);
      std::string elem;
      float fx, fy, fz;
      iss >> elem >> f.x[i] >> f.y[i] >> f.z[i] >> fx >> fy >> fz;
      if (elem == "Si") f.types[i] = 0;
      else if (elem == "Ge") f.types[i] = 1;
      else f.types[i] = 0;
    }
    frames.push_back(f);
  }
  input.close();
  return frames;
}

// ---------------------------------------------------------------------------
// CPU: make uniform knots
// ---------------------------------------------------------------------------
static std::vector<float> make_uniform_knots(int n, float r_min, float rc)
{
  std::vector<float> knots(n);
  float delta = (rc - r_min) / (n - 1);
  for (int i = 0; i < n; i++) knots[i] = r_min + i * delta;
  return knots;
}

// CPU: precompute combined cubic coefficients from B-spline coeffs
static void precompute_2b_uniform(
  const std::vector<float>& coeffs, std::vector<float4>& h_coeff)
{
  int nc = (int)coeffs.size();
  int nint = nc + 3;  // nknots = nc + 4, nint = nknots - 1 = nc + 3
  h_coeff.resize(nint);
  for (int m = 0; m < nint; m++) {
    int i0 = m - 3, i1 = m - 2, i2 = m - 1, i3 = m;
    if (i0 < 0) i0 = 0;
    if (i1 < 0) i1 = 0;
    if (i2 < 0) i2 = 0;
    if (i2 >= nc) i2 = nc - 1;
    if (i3 >= nc) i3 = nc - 1;
    float c0 = coeffs[i0], c1 = coeffs[i1], c2 = coeffs[i2], c3 = coeffs[i3];
    float A = (c0 + 4.0f*c1 + c2) / 6.0f;
    float B = (-3.0f*c0 + 3.0f*c2) / 6.0f;
    float C = (3.0f*c0 - 6.0f*c1 + 3.0f*c2) / 6.0f;
    float D = (-c0 + 3.0f*c1 - 3.0f*c2 + c3) / 6.0f;
    h_coeff[m] = make_float4(A, B, C, D);
  }
}

// ---------------------------------------------------------------------------
static void print_welcome_information(void)
{
  printf("\n");
  printf("***************************************************************\n");
  printf("*                 Welcome to use GPUMD                        *\n");
  printf("*     (Graphics Processing Units Molecular Dynamics)          *\n");
  printf("*                     version 5.3                             *\n");
  printf("*              This is the uf3 executable                     *\n");
  printf("***************************************************************\n");
  printf("\n");
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  print_welcome_information();
  print_gpu_information();

  print_line_1();
  printf("Started running UF3 training (GPU-accelerated).\n");
  print_line_2();

  UF3_Parameters para;
  if (argc < 2) { printf("Usage: uf3 <uf3.in>\n"); return EXIT_FAILURE; }
  parse_uf3_parameters(argv[1], para);

  // Load training data
  printf("Loading training data from %s ...\n", para.train_data.c_str());
  auto train_frames = load_xyz(para.train_data.c_str());
  printf("Loaded %zu training frames.\n", train_frames.size());

  int num_pairs = para.num_types * para.num_types;
  int ncoeff = para.n_max_2b;
  int nknots = ncoeff + 4;
  int nint = nknots - 1;

  // Initialize random coefficients
  std::vector<std::vector<float>> coeffs_2b(num_pairs);
  srand(42);
  for (int p = 0; p < num_pairs; p++) {
    coeffs_2b[p].resize(ncoeff);
    for (int c = 0; c < ncoeff; c++)
      coeffs_2b[p][c] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
  }

  // Build knot vectors
  auto knots = make_uniform_knots(nknots, 0.0f, (float)para.rc_2b);

  // Type map: for Si=0, Ge=1: pair (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
  std::vector<int> h_type_map(num_pairs);
  for (int p = 0; p < num_pairs; p++) h_type_map[p] = p;

  // ---- ES training loop (GPU-accelerated, per-batch upload) ----
  int batch_size = std::min(para.batch, (int)train_frames.size());
  double best_rmse = 1e30;
  std::vector<std::vector<float>> best_coeffs = coeffs_2b;
  int max_atoms_per_frame = 0;
  for (auto& f : train_frames)
    if (f.num_atoms > max_atoms_per_frame) max_atoms_per_frame = f.num_atoms;

  printf("\nStarting GPU-ES training (%d gen, pop=%d, batch=%d)...\n",
         para.generation, para.population, batch_size);
  print_line_1();

  const int BLOCK_SIZE = 64;
  srand(12345);

  // Pre-allocate reusable GPU buffers
  GPU_Vector<int> d_type_map(num_pairs);
  d_type_map.copy_from_host(h_type_map.data());
  GPU_Vector<int> d_batch_idx(batch_size);
  GPU_Vector<int> d_bnatoms(batch_size);
  GPU_Vector<int> d_boffsets(batch_size + 1);
  GPU_Vector<int> d_types;
  GPU_Vector<float> d_x, d_y, d_z;
  GPU_Vector<float4> d_coeff(num_pairs * nint);
  GPU_Vector<float> d_energy(batch_size);
  std::vector<float> h_energy(batch_size);

  auto time_start = std::chrono::high_resolution_clock::now();

  for (int gen = 0; gen < para.generation; gen++) {
    double total_rmse = 0;
    int frames_used = 0;

    for (int pop = 0; pop < para.population; pop++) {
      // Perturb coefficients
      std::vector<std::vector<float>> trial = best_coeffs;
      for (int p = 0; p < num_pairs; p++)
        for (int c = 0; c < ncoeff; c++)
          trial[p][c] += (rand() / (float)RAND_MAX - 0.5f) * 0.02f;

      // Precompute GPU coefficients from trial
      std::vector<float4> all_coeff;
      for (int p = 0; p < num_pairs; p++) {
        std::vector<float4> hc;
        precompute_2b_uniform(trial[p], hc);
        for (auto& c : hc) all_coeff.push_back(c);
      }
      d_coeff.resize(all_coeff.size());
      d_coeff.copy_from_host(all_coeff.data());

      // Select random batch
      std::vector<int> batch_frames(batch_size);
      for (int b = 0; b < batch_size; b++)
        batch_frames[b] = rand() % train_frames.size();

      // Per-batch GPU upload: build batch data on host, copy to GPU
      int batch_atoms = 0;
      for (int b = 0; b < batch_size; b++)
        batch_atoms += train_frames[batch_frames[b]].num_atoms;

      std::vector<int>   h_btypes(batch_atoms);
      std::vector<float> h_bx(batch_atoms), h_by(batch_atoms), h_bz(batch_atoms);
      {
        int off = 0;
        for (int b = 0; b < batch_size; b++) {
          Frame& f = train_frames[batch_frames[b]];
          for (int i = 0; i < f.num_atoms; i++) {
            h_btypes[off + i] = f.types[i];
            h_bx[off + i] = f.x[i]; h_by[off + i] = f.y[i]; h_bz[off + i] = f.z[i];
          }
          off += f.num_atoms;
        }
      }
      d_types.resize(batch_atoms);
      d_types.copy_from_host(h_btypes.data());
      d_x.resize(batch_atoms);
      d_x.copy_from_host(h_bx.data());
      d_y.resize(batch_atoms);
      d_y.copy_from_host(h_by.data());
      d_z.resize(batch_atoms);
      d_z.copy_from_host(h_bz.data());

      // GPU evaluation: simple CPU-iterated per-frame (avoids complex indexing kernel)
      // We launch one block per frame, using flat batch arrays with offset tracking
      std::vector<int> h_batch_idx(batch_size);
      std::vector<int> h_batch_natoms(batch_size);
      for (int b = 0; b < batch_size; b++) {
        h_batch_idx[b] = b;  // local batch index
        h_batch_natoms[b] = train_frames[batch_frames[b]].num_atoms;
      }
      d_batch_idx.copy_from_host(h_batch_idx.data());

      // Simplified: use direct frame-indexed GPU arrays instead of offsets
      // Build natoms + offsets for this batch
      std::vector<int> h_bnatoms(batch_size), h_boffsets(batch_size + 1);
      h_boffsets[0] = 0;
      for (int b = 0; b < batch_size; b++) {
        h_bnatoms[b] = train_frames[batch_frames[b]].num_atoms;
        h_boffsets[b + 1] = h_boffsets[b] + h_bnatoms[b];
      }
      d_bnatoms.copy_from_host(h_bnatoms.data());
      d_boffsets.copy_from_host(h_boffsets.data());
      d_batch_idx.copy_from_host(h_batch_idx.data());

      int grid_size = batch_size;
      eval_2b_batch<<<grid_size, 1>>>(
        batch_size, d_batch_idx.data(),
        d_bnatoms.data(), d_boffsets.data(),
        d_types.data(), d_x.data(), d_y.data(), d_z.data(),
        d_coeff.data(), num_pairs, nint,
        knots[0], (knots.back() - knots[0]) / nint, (float)para.rc_2b,
        d_type_map.data(), d_energy.data());
      GPU_CHECK_KERNEL

      // Download energies
      d_energy.copy_to_host(h_energy.data());

      // Compute RMSE
      double rmse_sum = 0;
      for (int b = 0; b < batch_size; b++) {
        int fidx = batch_frames[b];
        rmse_sum += fabs(h_energy[b] - train_frames[fidx].energy);
      }
      double rmse = rmse_sum / batch_size;
      total_rmse += rmse;

      if (rmse < best_rmse) {
        best_rmse = rmse;
        best_coeffs = trial;
      }
    }

    if (gen % 5 == 0 || gen == para.generation - 1) {
      auto tnow = std::chrono::high_resolution_clock::now();
      double elapsed = std::chrono::duration<double>(tnow - time_start).count();
      printf("  gen %5d: avg RMSE = %.3f eV, best = %.3f eV (%.1fs)\n",
             gen, total_rmse / para.population, best_rmse, elapsed);
    }
  }

  auto time_end = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration<double>(time_end - time_start).count();

  // Write trained potential
  print_line_1();
  std::string outfile;
  for (int n = 0; n < para.num_types; n++) outfile += para.elements[n];
  outfile += ".uf3";

  printf("Writing trained potential to %s ...\n", outfile.c_str());
  {
    std::ofstream out(outfile);
    out.precision(10);
    out << "uf3 " << para.num_types;
    for (int n = 0; n < para.num_types; n++) out << " " << para.elements[n];
    out << "\n";
    for (int p = 0; p < num_pairs; p++) {
      int ti = p / para.num_types, tj = p % para.num_types;
      out << "2B " << para.elements[ti] << " " << para.elements[tj]
          << " 0 3 uk\n";
      out << para.rc_2b << " " << nknots << "\n";
      out << std::fixed;
      for (int k = 0; k < nknots; k++)
        out << knots[k] << (k < nknots-1 ? " " : "\n");
      out << ncoeff << "\n";
      for (int c = 0; c < ncoeff; c++)
        out << best_coeffs[p][c] << (c < ncoeff-1 ? " " : "\n");
      out << "#\n";
    }
    out.close();
  }
  printf("Done. Best RMSE = %.3f eV, total time = %.1f s\n", best_rmse, total_time);
  print_line_2();

  return EXIT_SUCCESS;
}
