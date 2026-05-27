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

#include "parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

void parse_uf3_parameters(const char* input_file, UF3_Parameters& para)
{
  std::ifstream input(input_file);
  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Cannot open uf3.in.");
  }

  std::string line;
  while (std::getline(input, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() == 0) continue;
    if (tokens[0][0] == '#') continue;

    if (tokens[0] == "n_max_2b") {
      para.n_max_2b = get_int_from_token(tokens[1], __FILE__, __LINE__);
    } else if (tokens[0] == "n_max_3b") {
      if (tokens.size() >= 4) {
        para.n_max_3b[0] = get_int_from_token(tokens[1], __FILE__, __LINE__);
        para.n_max_3b[1] = get_int_from_token(tokens[2], __FILE__, __LINE__);
        para.n_max_3b[2] = get_int_from_token(tokens[3], __FILE__, __LINE__);
      }
    } else if (tokens[0] == "rc_2b") {
      para.rc_2b = get_double_from_token(tokens[1], __FILE__, __LINE__);
    } else if (tokens[0] == "rc_3b") {
      if (tokens.size() >= 3) {
        para.rc_3b[0] = get_double_from_token(tokens[1], __FILE__, __LINE__);
        para.rc_3b[1] = get_double_from_token(tokens[2], __FILE__, __LINE__);
      }
    } else if (tokens[0] == "knot_type") {
      para.knot_type_str = tokens[1];
      para.knot_type = (tokens[1] == "uk") ? 1 : 0;
    } else if (tokens[0] == "type") {
      para.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
      for (int n = 0; n < para.num_types; n++) {
        para.elements.push_back(tokens[2 + n]);
      }
    } else if (tokens[0] == "batch") {
      para.batch = get_int_from_token(tokens[1], __FILE__, __LINE__);
    } else if (tokens[0] == "population") {
      para.population = get_int_from_token(tokens[1], __FILE__, __LINE__);
    } else if (tokens[0] == "generation") {
      para.generation = get_int_from_token(tokens[1], __FILE__, __LINE__);
    } else if (tokens[0] == "lambda_e") {
      para.lambda_e = get_double_from_token(tokens[1], __FILE__, __LINE__);
    } else if (tokens[0] == "lambda_f") {
      para.lambda_f = get_double_from_token(tokens[1], __FILE__, __LINE__);
    } else if (tokens[0] == "lambda_v") {
      para.lambda_v = get_double_from_token(tokens[1], __FILE__, __LINE__);
    } else if (tokens[0] == "optimizer") {
      para.optimizer = tokens[1];
    } else if (tokens[0] == "train_data") {
      para.train_data = tokens[1];
    } else if (tokens[0] == "test_data") {
      para.test_data = tokens[1];
    }
  }
  input.close();

  printf("UF3 training parameters:\n");
  printf("  n_max_2b = %d\n", para.n_max_2b);
  printf("  rc_2b = %g A\n", para.rc_2b);
  printf("  knot_type = %s\n", para.knot_type_str.c_str());
  printf("  num_types = %d (", para.num_types);
  for (int n = 0; n < para.num_types; n++) {
    printf("%s%s", para.elements[n].c_str(), n < para.num_types - 1 ? " " : "");
  }
  printf(")\n");
  printf("  optimizer = %s\n", para.optimizer.c_str());
  printf("  training data = %s\n", para.train_data.c_str());
  printf("  test data = %s\n", para.test_data.c_str());
}
