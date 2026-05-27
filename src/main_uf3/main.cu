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
GPU-accelerated B-spline coefficient optimization with selectable optimizer.
------------------------------------------------------------------------------*/

#include "adam.cuh"
#include "dataset.cuh"
#include "es.cuh"
#include "fitness.cuh"
#include "parameters.cuh"
#include "snes.cuh"
#include "uf3.cuh"
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

static void write_uf3_file(UF3_Parameters& para, Uf3Model* model)
{
  std::string outfile;
  for (int n = 0; n < para.num_types; n++) outfile += para.elements[n];
  outfile += ".uf3";

  std::ofstream out(outfile);
  out.precision(10);
  out << "uf3 " << para.num_types;
  for (int n = 0; n < para.num_types; n++) out << " " << para.elements[n];
  out << "\n";

  int nk = model->nknots_2b(), nc = model->ncoeff_2b();
  int np = model->num_pairs(), nt = model->num_types();
  auto& elements = model->elements();
  auto& knots = model->knots_2b();

  std::vector<float> coeffs(nc * np);
  model->get_parameters(coeffs.data());

  for (int p = 0; p < np; p++) {
    int ti = p / nt, tj = p % nt;
    out << "2B " << elements[ti] << " " << elements[tj] << " 0 3 uk\n";
    out << model->rc_2b() << " " << nk << "\n";
    out << std::fixed;
    for (int k = 0; k < nk; k++) out << knots[k] << (k < nk - 1 ? " " : "\n");
    out << nc << "\n";
    for (int c = 0; c < nc; c++)
      out << coeffs[p * nc + c] << (c < nc - 1 ? " " : "\n");
    out << "#\n";
  }
  // 3B blocks
  if (model->has_3b()) {
    int num_2b = np * nc;
    for (int t = 0; t < model->num_triplets(); t++) {
      int ti = t / (nt * nt), tj = (t / nt) % nt, tk = t % nt;
      out << "3B " << elements[ti] << " " << elements[tj] << " " << elements[tk]
          << " 0 3 uk\n";
      float rc_jk = model->rc_3b(0) * 2.0f;
      out << std::fixed << rc_jk << " " << model->rc_3b(1) << " " << model->rc_3b(0)
          << " " << model->nknots_3b(0) << " " << model->nknots_3b(1) << " " << model->nknots_3b(2) << "\n";
      auto& k0 = model->knots_3b(0), &k1 = model->knots_3b(1), &k2 = model->knots_3b(2);
      for (size_t i = 0; i < k0.size(); i++) out << k0[i] << (i<k0.size()-1?" ":"\n");
      for (size_t i = 0; i < k1.size(); i++) out << k1[i] << (i<k1.size()-1?" ":"\n");
      for (size_t i = 0; i < k2.size(); i++) out << k2[i] << (i<k2.size()-1?" ":"\n");
      out << model->ncoeff_3b(0) << " " << model->ncoeff_3b(1) << " " << model->ncoeff_3b(2) << "\n";
      int nc0 = model->ncoeff_3b(0), nc1 = model->ncoeff_3b(1), nc2 = model->ncoeff_3b(2);
      int t_off = t * nc0 * nc1 * nc2;
      for (int i = 0; i < nc0; i++) {
        for (int j = 0; j < nc1; j++) {
          for (int k = 0; k < nc2; k++) {
            out << coeffs[num_2b + t_off + i + j * nc0 + k * nc0 * nc1] << (k<nc2-1?" ":"");
          }
          out << "\n";
        }
      }
      out << "#\n";
    }
  }

  out.close();
  printf("Wrote %s\n", outfile.c_str());
}

int main(int argc, char* argv[])
{
  print_welcome_information();
  print_gpu_information();

  print_line_1();
  printf("Started running UF3 training.\n");
  print_line_2();

  if (argc < 2) { printf("Usage: uf3 <uf3.in>\n"); return EXIT_FAILURE; }

  UF3_Parameters para;
  parse_uf3_parameters(argv[1], para);

  // Load data
  printf("Loading data...\n");
  float nn_cutoff = para.n_max_3b[0] > 0 ? (float)std::max(para.rc_3b[0], para.rc_3b[1]) : 0.0f;
  auto train_frames = load_uf3_frames(para.train_data.c_str(), nn_cutoff);
  printf("Loaded %zu training frames.\n", train_frames.size());

  // Create model and fitness
  Uf3Model model(para);
  Uf3Fitness fitness(para, &model, train_frames);

  printf("Model: %d params, %d pairs, %d coeffs/pair\n",
         model.num_parameters(), model.num_pairs(), model.ncoeff_2b());

  // Dispatch optimizer
  auto t0 = std::chrono::high_resolution_clock::now();

  if (para.optimizer == "adam") {
    printf("\n=== Adam optimizer ===\n");
    run_adam(para, fitness);
  } else if (para.optimizer == "snes") {
    printf("\n=== SNES optimizer ===\n");
    run_snes(para, fitness);
  } else {
    printf("\n=== ES optimizer (default) ===\n");
    run_es(para, fitness);
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double total = std::chrono::duration<double>(t1 - t0).count();

  // Save result
  print_line_1();
  write_uf3_file(para, &model);
  printf("Total time = %.1f s\n", total);
  print_line_2();

  return EXIT_SUCCESS;
}
