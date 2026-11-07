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

#include <map>
#include <random>
#include <string>
#include <vector>

class Box;

bool is_variable_identifier(const std::string& s);
bool is_delayed_variable_token(const std::string& token);
std::string delayed_variable_name(const std::string& token);
std::string format_variable_number(double value);

class VariableScope
{
public:
  void push(const std::string& name, const std::string& value);
  void pop();
  bool has_loop_variable(const std::string& name) const;
  void define_equal(const std::vector<std::string>& tokens);

  std::string expand_text(const std::string& text, const Box& box);
  double eval_name(const std::string& name, const Box& box);
  double eval_formula(const std::string& formula, const Box& box);
  double resolve_real(const std::string& token, const Box& box, const char* what);

private:
  std::string expand_name(const std::string& name, const Box& box);
  const std::string* find_loop(const std::string& name) const;

  std::vector<std::pair<std::string, std::string>> stack_;
  std::map<std::string, std::string> equal_;
  std::vector<std::string> evaluating_;
  std::mt19937 rng_;
  bool rng_initialized_ = false;

  friend class FormulaParser;
};
