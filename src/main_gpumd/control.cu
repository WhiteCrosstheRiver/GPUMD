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
Minimal control-flow layer for run.in: for / end / ${variable}.
Ordinary GPUMD commands are expanded then passed to parse_one_keyword.
------------------------------------------------------------------------------*/

#include "control.cuh"
#include "run.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

void VariableScope::push(const std::string& name, const std::string& value)
{
  stack_.emplace_back(name, value);
}

void VariableScope::pop()
{
  stack_.pop_back();
}

const std::string& VariableScope::lookup(const std::string& name) const
{
  for (int i = static_cast<int>(stack_.size()) - 1; i >= 0; --i) {
    if (stack_[i].first == name) {
      return stack_[i].second;
    }
  }
  std::string msg = "Undefined variable '" + name + "'.";
  PRINT_INPUT_ERROR(msg.c_str());
  static const std::string empty;
  return empty;
}

static bool is_ident_start(char c)
{
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

static bool is_ident_char(char c)
{
  return is_ident_start(c) || (c >= '0' && c <= '9');
}

static bool is_identifier(const std::string& s)
{
  if (s.empty() || !is_ident_start(s[0])) {
    return false;
  }
  for (size_t i = 1; i < s.size(); ++i) {
    if (!is_ident_char(s[i])) {
      return false;
    }
  }
  return true;
}

// If text[i] is "${", parse ${ident} and set name plus index after '}'.
static void parse_variable_ref(const std::string& text, size_t i, std::string& name, size_t& end)
{
  size_t j = i + 2;
  if (j >= text.size() || text[j] == '}') {
    PRINT_INPUT_ERROR("Empty variable name in ${}.");
  }
  if (!is_ident_start(text[j])) {
    PRINT_INPUT_ERROR("Invalid variable name in ${}.");
  }
  const size_t start = j;
  ++j;
  while (j < text.size() && is_ident_char(text[j])) {
    ++j;
  }
  if (j >= text.size() || text[j] != '}') {
    PRINT_INPUT_ERROR("Unclosed ${variable} reference.");
  }
  name = text.substr(start, j - start);
  end = j + 1;
}

static std::string expand_variables(const std::string& text, const VariableScope& scope)
{
  std::string out;
  for (size_t i = 0; i < text.size();) {
    if (text[i] == '$' && i + 1 < text.size() && text[i + 1] == '{') {
      std::string name;
      size_t end = 0;
      parse_variable_ref(text, i, name, end);
      out += scope.lookup(name);
      i = end;
    } else {
      out += text[i];
      ++i;
    }
  }
  return out;
}

static void check_variable_refs(
  const std::vector<std::string>& tokens, const std::vector<std::string>& enclosing)
{
  for (const auto& token : tokens) {
    for (size_t i = 0; i < token.size();) {
      if (token[i] == '$' && i + 1 < token.size() && token[i + 1] == '{') {
        std::string name;
        size_t end = 0;
        parse_variable_ref(token, i, name, end);
        bool found = false;
        for (const auto& v : enclosing) {
          if (v == name) {
            found = true;
            break;
          }
        }
        if (!found) {
          std::string msg = "Undefined variable '" + name + "'.";
          PRINT_INPUT_ERROR(msg.c_str());
        }
        i = end;
      } else {
        ++i;
      }
    }
  }
}

static bool next_statement(std::ifstream& input, std::vector<std::string>& tokens)
{
  while (input.peek() != EOF) {
    std::vector<std::string> raw = get_tokens(input);
    tokens.clear();
    for (const auto& t : raw) {
      if (t[0] != '#') {
        tokens.emplace_back(t);
      } else {
        break;
      }
    }
    if (!tokens.empty()) {
      return true;
    }
  }
  return false;
}

static std::vector<std::string> make_range(int start, int stop, int step)
{
  if (step == 0) {
    PRINT_INPUT_ERROR("for range step cannot be 0.");
  }
  if (start > stop && step > 0) {
    PRINT_INPUT_ERROR("for range has start > stop but step > 0.");
  }
  if (start < stop && step < 0) {
    PRINT_INPUT_ERROR("for range has start < stop but step < 0.");
  }

  std::vector<std::string> values;
  for (long long v = start; step > 0 ? v <= stop : v >= stop; v += step) {
    values.emplace_back(std::to_string(static_cast<int>(v)));
  }
  return values;
}

static std::vector<std::unique_ptr<Node>> parse_statements(
  std::ifstream& input, const std::vector<std::string>& enclosing, bool inside_for);

static std::unique_ptr<ForNode> parse_for(
  std::ifstream& input,
  const std::vector<std::string>& header,
  const std::vector<std::string>& enclosing)
{
  for (size_t i = 1; i < header.size(); ++i) {
    if (header[i].find("${") != std::string::npos) {
      PRINT_INPUT_ERROR("Variable substitution ${name} is not allowed in a for header.");
    }
  }

  if (header.size() < 4) {
    PRINT_INPUT_ERROR("Invalid for header. Use: for <variable> range|values ...");
  }

  const std::string& variable = header[1];
  if (!is_identifier(variable)) {
    PRINT_INPUT_ERROR("for variable must be an identifier [A-Za-z_][A-Za-z0-9_]*.");
  }
  for (const auto& v : enclosing) {
    if (v == variable) {
      std::string msg = "Nested for cannot reuse outer variable '" + variable + "'.";
      PRINT_INPUT_ERROR(msg.c_str());
    }
  }

  auto node = std::make_unique<ForNode>();
  node->variable = variable;

  const std::string& kind = header[2];
  if (kind == "range") {
    if (header.size() != 5 && header.size() != 6) {
      PRINT_INPUT_ERROR("for range syntax: for <variable> range <start> <stop> [step]");
    }
    int start = 0;
    int stop = 0;
    int step = 1;
    if (!is_valid_int(header[3].c_str(), &start) || !is_valid_int(header[4].c_str(), &stop)) {
      PRINT_INPUT_ERROR("for range start and stop must be integers.");
    }
    if (header.size() == 6) {
      if (!is_valid_int(header[5].c_str(), &step)) {
        PRINT_INPUT_ERROR("for range step must be an integer.");
      }
    }
    node->values = make_range(start, stop, step);
  } else if (kind == "values") {
    for (size_t i = 3; i < header.size(); ++i) {
      node->values.emplace_back(header[i]);
    }
  } else {
    PRINT_INPUT_ERROR("for header must use 'range' or 'values'.");
  }

  std::vector<std::string> inner = enclosing;
  inner.emplace_back(variable);
  node->body = parse_statements(input, inner, true);
  return node;
}

static std::vector<std::unique_ptr<Node>> parse_statements(
  std::ifstream& input, const std::vector<std::string>& enclosing, bool inside_for)
{
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::string> tokens;
  while (next_statement(input, tokens)) {
    if (tokens[0] == "end") {
      if (tokens.size() != 1) {
        PRINT_INPUT_ERROR("end takes no arguments.");
      }
      if (!inside_for) {
        PRINT_INPUT_ERROR("end without matching for.");
      }
      return nodes;
    }
    if (tokens[0] == "for") {
      nodes.emplace_back(parse_for(input, tokens, enclosing));
    } else {
      check_variable_refs(tokens, enclosing);
      auto cmd = std::make_unique<CommandNode>();
      cmd->tokens = std::move(tokens);
      nodes.emplace_back(std::move(cmd));
    }
  }
  if (inside_for) {
    PRINT_INPUT_ERROR("for is missing a matching end.");
  }
  return nodes;
}

void CommandNode::execute(Run& run, VariableScope& scope)
{
  std::vector<std::string> expanded;
  expanded.reserve(tokens.size());
  for (const auto& token : tokens) {
    expanded.emplace_back(expand_variables(token, scope));
  }
  run.parse_one_keyword(expanded);
}

void ForNode::execute(Run& run, VariableScope& scope)
{
  for (const auto& value : values) {
    scope.push(variable, value);
    for (auto& node : body) {
      node->execute(run, scope);
    }
    scope.pop();
  }
}

void execute_control_script(Run& run, const char* filename)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    std::cout << "Failed to open " << filename << "." << std::endl;
    exit(1);
  }

  std::vector<std::string> enclosing;
  auto nodes = parse_statements(input, enclosing, false);
  VariableScope scope;
  for (auto& node : nodes) {
    node->execute(run, scope);
  }
}
