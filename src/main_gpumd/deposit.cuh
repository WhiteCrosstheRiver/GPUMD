/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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

#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include <map>
#include <random>
#include <string>
#include <vector>

class Force;
class VariableScope;

struct DepositEvent
{
  bool active = false;
  bool pulse = false;
  int first_atom = 0;
  int count = 0;
  double normal[3] = {0.0, 0.0, 1.0};
  double entry_height = 0.0;
  int relax_steps = 0;
  int max_wait = 0;
  double omitted_time_min_fs = 0.0;
  double omitted_time_max_fs = 0.0;
};

class DepositRegistry
{
public:
  DepositEvent execute(
    const std::vector<std::string>& tokens,
    Box& box,
    Atom& atoms,
    std::vector<Group>& groups,
    GPU_Vector<double>& thermo,
    Force& force,
    VariableScope* variables);

  bool skips_flight(const std::string& name) const;

private:
  struct Recipe
  {
    std::vector<std::string> particle;
    std::vector<std::string> placement;
    std::vector<std::string> region;
    std::vector<std::string> velocity;
    std::vector<std::string> flight;
    std::vector<std::string> exclusion;
    std::mt19937 rng;
    bool rng_initialized = false;
  };
  std::map<std::string, Recipe> recipes_;
};
