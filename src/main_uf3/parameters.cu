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

static void apply_stage_token(UF3_OptimizerStage& stage, const std::vector<std::string>& tokens)
{
  if (tokens[0] == "batch") {
    if (tokens.size() >= 2 && tokens[1] == "full") {
      stage.full_batch = true;
      stage.batch = -1;
    } else {
      stage.full_batch = false;
      stage.batch = get_int_from_token(tokens[1], __FILE__, __LINE__);
    }
  } else if (tokens[0] == "population") {
    stage.population = get_int_from_token(tokens[1], __FILE__, __LINE__);
  } else if (tokens[0] == "generation") {
    stage.generation = get_int_from_token(tokens[1], __FILE__, __LINE__);
  }
}

static void apply_global_token(UF3_Parameters& para, const std::vector<std::string>& tokens)
{
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
  } else if (tokens[0] == "lambda_1") {
    para.lambda_1 = get_double_from_token(tokens[1], __FILE__, __LINE__);
  } else if (tokens[0] == "lambda_2") {
    para.lambda_2 = get_double_from_token(tokens[1], __FILE__, __LINE__);
  } else if (tokens[0] == "optimizer") {
    if (tokens.size() >= 3 && tokens[1] == "start") {
      PRINT_INPUT_ERROR("optimizer start must be closed with optimizer end.");
    } else if (tokens.size() >= 2) {
      para.optimizer = tokens[1];
    }
  } else if (tokens[0] == "train_data") {
    para.train_data = tokens[1];
  } else if (tokens[0] == "test_data") {
    para.test_data = tokens[1];
  }
}

void finalize_uf3_optimizer_stages(UF3_Parameters& para)
{
  if (!para.stages.empty()) {
    return;
  }
  UF3_OptimizerStage stage;
  stage.name = para.optimizer;
  stage.generation = para.generation;
  stage.population = para.population;
  stage.batch = para.batch;
  if (stage.name == "lstsq") {
    stage.full_batch = true;
    stage.generation = 1;
  }
  para.stages.push_back(stage);
}

static void print_optimizer_stage_line(
  size_t index, const UF3_OptimizerStage& stage, int global_batch)
{
  printf("  [%zu] %s: generation=%d population=%d",
         index, stage.name.c_str(), stage.generation, stage.population);
  if (stage.full_batch) {
    printf(" batch=full");
  } else if (stage.batch >= 0) {
    printf(" batch=%d", stage.batch);
  } else {
    printf(" batch=global(%d)", global_batch);
  }
  printf("\n");
}

static void print_optimizer_stages(const char* title, const UF3_Parameters& para)
{
  printf("%s\n", title);
  for (size_t s = 0; s < para.stages.size(); s++) {
    print_optimizer_stage_line(s, para.stages[s], para.batch);
  }
}

void normalize_uf3_optimizer_stages(UF3_Parameters& para)
{
  if (para.stages.empty()) {
    return;
  }

  int first_lstsq = -1;
  for (size_t i = 0; i < para.stages.size(); i++) {
    if (para.stages[i].name == "lstsq") {
      if (first_lstsq < 0) {
        first_lstsq = (int)i;
      }
    }
  }
  if (first_lstsq < 0) {
    return;
  }

  const bool was_first = (first_lstsq == 0);
  bool dropped_duplicate = false;
  UF3_OptimizerStage lstsq_stage = para.stages[first_lstsq];
  std::vector<UF3_OptimizerStage> ordered;
  ordered.reserve(para.stages.size());
  ordered.push_back(lstsq_stage);

  for (size_t i = 0; i < para.stages.size(); i++) {
    if (para.stages[i].name == "lstsq") {
      if ((int)i != first_lstsq) {
        dropped_duplicate = true;
      }
      continue;
    }
    ordered.push_back(para.stages[i]);
  }
  para.stages = std::move(ordered);

  if (!was_first) {
    printf("Note: lstsq moved to stage 0 (UF3 runs lstsq first when configured).\n");
  }
  if (dropped_duplicate) {
    printf("Warning: duplicate lstsq blocks ignored; using the first lstsq block.\n");
  }
}

void parse_uf3_parameters(const char* input_file, UF3_Parameters& para)
{
  std::ifstream input(input_file);
  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Cannot open UF3 input file.");
  }

  bool in_block = false;
  UF3_OptimizerStage current;

  std::string line;
  while (std::getline(input, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() == 0) {
      continue;
    }
    if (tokens[0][0] == '#') {
      continue;
    }

    if (tokens[0] == "optimizer" && tokens.size() >= 3 && tokens[1] == "start") {
      in_block = true;
      current = UF3_OptimizerStage{};
      current.name = tokens[2];
      current.generation = para.generation;
      current.population = para.population;
      current.batch = -1;
      current.full_batch = (current.name == "lstsq");
      if (current.name == "lstsq") {
        current.generation = 1;
      }
      continue;
    }

    if (tokens[0] == "optimizer" && tokens.size() >= 3 && tokens[1] == "end") {
      if (!in_block) {
        PRINT_INPUT_ERROR("optimizer end without matching optimizer start.");
      }
      if (current.name != tokens[2]) {
        printf("Warning: optimizer end %s does not match start %s.\n",
               tokens[2].c_str(), current.name.c_str());
      }
      para.stages.push_back(current);
      in_block = false;
      continue;
    }

    if (in_block) {
      apply_stage_token(current, tokens);
    } else {
      apply_global_token(para, tokens);
    }
  }
  input.close();

  if (in_block) {
    PRINT_INPUT_ERROR("Unclosed optimizer block (missing optimizer end).");
  }

  finalize_uf3_optimizer_stages(para);
  normalize_uf3_optimizer_stages(para);

  printf("UF3 training parameters:\n");
  printf("  n_max_2b = %d\n", para.n_max_2b);
  printf("  rc_2b = %g A\n", para.rc_2b);
  printf("  knot_type = %s\n", para.knot_type_str.c_str());
  printf("  num_types = %d (", para.num_types);
  for (int n = 0; n < para.num_types; n++) {
    printf("%s%s", para.elements[n].c_str(), n < para.num_types - 1 ? " " : "");
  }
  printf(")\n");
  printf("  global batch = %d\n", para.batch);
  printf("  training data = %s\n", para.train_data.c_str());
  printf("  test data = %s\n", para.test_data.c_str());
  print_optimizer_stages("  optimizer stages (execution order):", para);
}
