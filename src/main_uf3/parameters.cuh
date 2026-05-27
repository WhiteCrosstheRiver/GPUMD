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

struct UF3_Parameters {
  int n_max_2b = 10;         // number of 2B basis functions per pair
  int n_max_3b[3] = {0,0,0}; // 3B basis dimensions (optional)
  double rc_2b = 6.0;
  double rc_3b[2] = {0,0};   // 3B cutoffs (ij, ik)
  int knot_type = 1;         // 0=non-uniform, 1=uniform
  int num_types = 1;
  std::string knot_type_str = "uk";

  int batch = 1000;
  int population = 50;
  int generation = 5000;
  double lambda_e = 1.0;
  double lambda_f = 1.0;
  double lambda_v = 0.1;
  std::string train_data = "train.xyz";
  std::string test_data = "test.xyz";
  std::vector<std::string> elements;
};

void parse_uf3_parameters(const char* input_file, UF3_Parameters& para);
