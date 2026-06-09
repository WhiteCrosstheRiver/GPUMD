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

#include "optimizer.cuh"
#include "adam.cuh"
#include "es.cuh"
#include "lbfgs.cuh"
#include "lstsq.cuh"
#include "snes.cuh"
#include "utilities/error.cuh"
#include <cstdio>

static void run_stage(
  const UF3_Parameters& para,
  const UF3_OptimizerStage& stage,
  int stage_id,
  int gen_offset,
  Uf3Fitness& fitness)
{
  printf("\n=== Stage %d: %s ===\n", stage_id, stage.name.c_str());

  if (stage.name == "adam") {
    run_adam(para, stage, stage_id, gen_offset, fitness);
  } else if (stage.name == "lstsq") {
    run_lstsq(para, stage, stage_id, gen_offset, fitness);
  } else if (stage.name == "lbfgs") {
    run_lbfgs(para, stage, stage_id, gen_offset, fitness);
  } else if (stage.name == "snes") {
    run_snes(para, stage, stage_id, gen_offset, fitness);
  } else if (stage.name == "es") {
    run_es(para, stage, stage_id, gen_offset, fitness);
  } else {
    PRINT_INPUT_ERROR("Unknown optimizer stage.");
  }
}

void run_optimizer_pipeline(UF3_Parameters& para, Uf3Fitness& fitness)
{
  int gen_offset = 0;
  for (size_t s = 0; s < para.stages.size(); s++) {
    const auto& stage = para.stages[s];
    run_stage(para, stage, (int)s, gen_offset, fitness);
    if (stage.name == "lstsq") {
      gen_offset += 1;
    } else {
      gen_offset += stage.generation;
    }
  }
}
