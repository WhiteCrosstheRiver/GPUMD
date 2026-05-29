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

struct UF3_OptimizerStage {
  std::string name = "adam";
  int generation = 0;
  int population = 0;
  int batch = 0;           // >0 for non-lstsq; lstsq can use batch full
  bool full_batch = false; // lstsq: use all training frames
};

struct UF3_Parameters {
  int n_max_2b = 10;
  int n_max_3b[3] = {0, 0, 0};
  double rc_2b = 6.0;
  double rc_3b[2] = {0, 0};
  int knot_type = 1;
  int num_types = 1;
  std::string knot_type_str = "uk";

  int batch = 1000;  // computed from stage max batch
  double lambda_e = 1.0;
  double lambda_f = 1.0;
  double lambda_v = 0.1;
  double lambda_1 = 0.0;
  double lambda_2 = 0.0;
  std::string train_data = "train.xyz";
  std::string test_data = "test.xyz";
  std::vector<std::string> elements;

  std::vector<UF3_OptimizerStage> stages;
};

void parse_uf3_parameters(const char* input_file, UF3_Parameters& para);
void normalize_uf3_optimizer_stages(UF3_Parameters& para);
