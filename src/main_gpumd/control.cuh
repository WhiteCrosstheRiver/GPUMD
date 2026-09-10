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

#include "variable.cuh"
#include <memory>
#include <string>
#include <vector>

class Run;

struct Node
{
  virtual ~Node() = default;
  virtual void execute(Run& run, VariableScope& scope) = 0;
};

struct CommandNode : Node
{
  std::vector<std::string> tokens;
  void execute(Run& run, VariableScope& scope) override;
};

struct ForNode : Node
{
  std::string variable;
  std::vector<std::string> values;
  std::vector<std::unique_ptr<Node>> body;
  void execute(Run& run, VariableScope& scope) override;
};

struct IfNode : Node
{
  std::string lhs;
  std::string op;
  std::string rhs;
  std::vector<std::unique_ptr<Node>> then_body;
  std::vector<std::unique_ptr<Node>> else_body;
  void execute(Run& run, VariableScope& scope) override;
};

void execute_control_script(Run& run, const char* filename);
