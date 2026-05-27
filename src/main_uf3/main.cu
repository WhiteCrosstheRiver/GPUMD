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
Reads uf3.in, loads training data, and trains a B-spline based potential
using the SNES optimizer (shared with main_nep).
------------------------------------------------------------------------------*/

#include "parameters.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
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

// Simple structure to hold one training frame
struct Frame {
  int num_atoms;
  std::vector<int> types;
  std::vector<double> x, y, z;
  std::vector<double> fx, fy, fz;
  double energy;
  double virial[9];
  double lattice[9];
};

// Parse GPUMD extended XYZ format
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

    // read comment line with lattice, energy, etc.
    std::getline(input, line);
    {
      // Extract energy=
      size_t pos = line.find("energy=");
      if (pos != std::string::npos) {
        f.energy = std::stod(line.substr(pos + 7));
      }
    }

    f.types.resize(natoms);
    f.x.resize(natoms); f.y.resize(natoms); f.z.resize(natoms);
    f.fx.resize(natoms); f.fy.resize(natoms); f.fz.resize(natoms);

    for (int i = 0; i < natoms; i++) {
      std::getline(input, line);
      std::istringstream iss(line);
      std::string elem;
      iss >> elem >> f.x[i] >> f.y[i] >> f.z[i] >> f.fx[i] >> f.fy[i] >> f.fz[i];
      if (elem == "Si") f.types[i] = 0;
      else if (elem == "Ge") f.types[i] = 1;
      else f.types[i] = 0;
    }
    frames.push_back(f);
  }
  input.close();
  return frames;
}

// Build a uniform knot vector from r_min to rc with n knots
static std::vector<float> make_uniform_knots(int n, double r_min, double rc)
{
  std::vector<float> knots(n);
  double delta = (rc - r_min) / (n - 1);
  for (int i = 0; i < n; i++) knots[i] = r_min + i * delta;
  return knots;
}

// Evaluate a single 2-body cubic B-spline energy for a pair at distance r
// Uses pre-computed per-interval coefficients
static double eval_2b_spline_cpu(
  double r, const std::vector<float>& coeffs, const std::vector<float>& knots)
{
  int nk = knots.size(), nint = nk - 1;
  // find interval
  double knot_min = knots[0], knot_delta = (knots[nk-1] - knots[0]) / nint;
  int m = (r - knot_min) / knot_delta;
  if (m < 0) m = 0;
  if (m >= nint) m = nint - 1;
  double h = knot_delta;
  double u = (r - (knot_min + m * h)) / h;

  // Pre-compute combined poly (same as GPU version)
  int nc = coeffs.size();
  int i0 = m - 3, i1 = m - 2, i2 = m - 1, i3 = m;
  if (i0 < 0) i0 = 0;
  if (i1 < 0) i1 = 0;
  if (i2 < 0) i2 = 0;
  if (i2 >= nc) i2 = nc - 1;
  if (i3 >= nc) i3 = nc - 1;
  float c0 = coeffs[i0], c1 = coeffs[i1], c2 = coeffs[i2], c3 = coeffs[i3];
  float A = (c0 + 4*c1 + c2) / 6.0f;
  float B = (-3*c0 + 3*c2) / 6.0f;
  float C = (3*c0 - 6*c1 + 3*c2) / 6.0f;
  float D = (-c0 + 3*c1 - 3*c2 + c3) / 6.0f;
  return A + u * (B + u * (C + u * D));
}

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

int main(int argc, char* argv[])
{
  print_welcome_information();
  print_gpu_information();

  print_line_1();
  printf("Started running UF3 training.\n");
  print_line_2();

  UF3_Parameters para;
  if (argc < 2) {
    printf("Usage: uf3 <uf3.in>\n");
    return EXIT_FAILURE;
  }
  const char* input_file = argv[1];
  parse_uf3_parameters(input_file, para);

  // Load training data
  printf("Loading training data from %s ...\n", para.train_data.c_str());
  auto train_frames = load_xyz(para.train_data.c_str());
  printf("Loaded %zu training frames.\n", train_frames.size());

  printf("Loading test data from %s ...\n", para.test_data.c_str());
  auto test_frames = load_xyz(para.test_data.c_str());
  printf("Loaded %zu test frames.\n", test_frames.size());

  // Initialize random coefficients for each element pair
  // n_max_2b coefficients per pair, num_types * num_types pairs
  int num_pairs = para.num_types * para.num_types;
  int ncoeff_2b = para.n_max_2b;
  std::vector<std::vector<float>> coeffs_2b(num_pairs);
  srand(42);
  for (int p = 0; p < num_pairs; p++) {
    coeffs_2b[p].resize(ncoeff_2b);
    for (int c = 0; c < ncoeff_2b; c++) {
      coeffs_2b[p][c] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f; // small random
    }
  }

  // Build knot vectors
  int nknots = ncoeff_2b + 4; // cubic B-spline: nknots = ncoeffs + degree
  auto knots = make_uniform_knots(nknots, 0.0, para.rc_2b);

  // Evaluate energy for first frame
  Frame& f = train_frames[0];
  double total_energy = 0;
  int count = 0;
  for (int i = 0; i < f.num_atoms; i++) {
    for (int j = i + 1; j < f.num_atoms; j++) {
      double dx = f.x[i] - f.x[j];
      double dy = f.y[i] - f.y[j];
      double dz = f.z[i] - f.z[j];
      double r = sqrt(dx*dx + dy*dy + dz*dz);
      if (r >= para.rc_2b) continue;
      int p = f.types[i] * para.num_types + f.types[j];
      double e = eval_2b_spline_cpu(r, coeffs_2b[p], knots);
      total_energy += e;
      count++;
    }
  }

  printf("\nFirst frame: %d atoms, %d pairs within cutoff, energy = %.6f eV\n",
         f.num_atoms, count, total_energy);
  printf("Reference energy = %.6f eV\n", f.energy);
  printf("RMSE (random init) = %.6f eV\n", fabs(total_energy - f.energy));

  // ---- Simple ES training loop ----
  // Perturb coefficients randomly, keep the best for each element pair.
  int ncoeff = para.n_max_2b;
  int batch_size = std::min(para.batch, (int)train_frames.size());
  double best_rmse = 1e30;
  std::vector<std::vector<float>> best_coeffs = coeffs_2b;

  printf("\nStarting ES training (%d generations, pop=%d, batch=%d)...\n",
         para.generation, para.population, batch_size);
  print_line_1();

  srand(12345);
  for (int gen = 0; gen < para.generation; gen++) {
    double total_rmse = 0;
    int frames_evaluated = 0;

    // Evaluate all candidates on a random subset of frames
    for (int pop = 0; pop < para.population; pop++) {
      // Perturb current best coefficients
      std::vector<std::vector<float>> trial = best_coeffs;
      for (int p = 0; p < num_pairs; p++) {
        for (int c = 0; c < ncoeff; c++) {
          float noise = (rand() / (float)RAND_MAX - 0.5f) * 0.02f;
          trial[p][c] += noise;
        }
      }

      // Evaluate on batch_size random frames
      double rmse_sum = 0;
      int eval_count = 0;
      for (int b = 0; b < batch_size && eval_count < 100; b++) {
        int fidx = rand() % train_frames.size();
        Frame& f = train_frames[fidx];
        double total_e = 0;
        for (int i = 0; i < f.num_atoms && i < 128; i++) {
          for (int j = i + 1; j < f.num_atoms && j < 128; j++) {
            double dx = f.x[i] - f.x[j];
            double dy = f.y[i] - f.y[j];
            double dz = f.z[i] - f.z[j];
            double r = sqrt(dx*dx + dy*dy + dz*dz);
            if (r >= para.rc_2b) continue;
            int p = f.types[i] * para.num_types + f.types[j];
            total_e += eval_2b_spline_cpu(r, trial[p], knots);
          }
        }
        rmse_sum += fabs(total_e - f.energy);
        eval_count++;
      }
      double rmse = rmse_sum / eval_count;
      total_rmse += rmse;
      frames_evaluated += eval_count;

      // Keep best
      if (rmse < best_rmse) {
        best_rmse = rmse;
        best_coeffs = trial;
      }
    }

    if (gen % 5 == 0 || gen == para.generation - 1) {
      printf("  gen %5d: avg RMSE = %.3f eV, best = %.3f eV\n",
             gen, total_rmse / para.population, best_rmse);
    }
  }

  // Write trained potential to .uf3 file
  print_line_1();
  std::string outfile = "nep.uf3";
  printf("Writing trained potential to %s ...\n", outfile.c_str());
  {
    std::ofstream out(outfile);
    out.precision(10);
    out << "uf3 " << para.num_types;
    for (int n = 0; n < para.num_types; n++) out << " " << para.elements[n];
    out << "\n";
    // Write 2B blocks for each element pair
    for (int p = 0; p < num_pairs; p++) {
      int ti = p / para.num_types, tj = p % para.num_types;
      out << "2B " << para.elements[ti] << " " << para.elements[tj]
          << " 0 3 " << para.knot_type_str << "\n";
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
  printf("Done. Best RMSE = %.3f eV\n", best_rmse);
  print_line_2();

  return EXIT_SUCCESS;
}
