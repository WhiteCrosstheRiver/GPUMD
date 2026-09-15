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
Minimal control-flow layer for run.in: for / if / else / end / variable /
${name} / $(formula). v_name is left intact for commands that evaluate it later.
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

static bool is_ident_start(char c)
{
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

static bool is_ident_char(char c)
{
  return is_ident_start(c) || (c >= '0' && c <= '9');
}

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

static void check_variable_refs(
  const std::vector<std::string>& tokens, const std::vector<std::string>& enclosing)
{
  (void)enclosing;
  for (const auto& token : tokens) {
    for (size_t i = 0; i < token.size();) {
      if (token[i] == '$' && i + 1 < token.size() && token[i + 1] == '{') {
        std::string name;
        size_t end = 0;
        parse_variable_ref(token, i, name, end);
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

enum class ParseMode { Top, For, IfThen, IfElse };
enum class CloseKind { None, End, Else };

static std::vector<std::unique_ptr<Node>> parse_statements(
  std::ifstream& input,
  const std::vector<std::string>& enclosing,
  ParseMode mode,
  CloseKind* close);

static std::unique_ptr<ForNode> parse_for(
  std::ifstream& input,
  const std::vector<std::string>& header,
  const std::vector<std::string>& enclosing);

static std::unique_ptr<IfNode> parse_if(
  std::ifstream& input,
  const std::vector<std::string>& header,
  const std::vector<std::string>& enclosing);

static std::unique_ptr<ForNode> parse_for(
  std::ifstream& input,
  const std::vector<std::string>& header,
  const std::vector<std::string>& enclosing)
{
  for (size_t i = 1; i < header.size(); ++i) {
    if (header[i].find("${") != std::string::npos || header[i].find("$(") != std::string::npos) {
      PRINT_INPUT_ERROR("Variable substitution is not allowed in a for header.");
    }
  }

  if (header.size() < 4) {
    PRINT_INPUT_ERROR("Invalid for header. Use: for <variable> range|values ...");
  }

  const std::string& variable = header[1];
  if (!is_variable_identifier(variable)) {
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
  CloseKind close = CloseKind::None;
  node->body = parse_statements(input, inner, ParseMode::For, &close);
  if (close != CloseKind::End) {
    PRINT_INPUT_ERROR("for is missing a matching end.");
  }
  return node;
}

static bool is_compare_op(const std::string& op)
{
  return op == "==" || op == "!=" || op == "<" || op == "<=" || op == ">" || op == ">=";
}

static std::unique_ptr<IfNode> parse_if(
  std::ifstream& input,
  const std::vector<std::string>& header,
  const std::vector<std::string>& enclosing)
{
  if (header.size() != 4) {
    PRINT_INPUT_ERROR("Usage: if <a> <op> <b>");
  }
  if (!is_compare_op(header[2])) {
    PRINT_INPUT_ERROR("if operator must be == != < <= > >=.");
  }
  auto node = std::make_unique<IfNode>();
  node->lhs = header[1];
  node->op = header[2];
  node->rhs = header[3];
  CloseKind close = CloseKind::None;
  node->then_body = parse_statements(input, enclosing, ParseMode::IfThen, &close);
  if (close == CloseKind::Else) {
    node->else_body = parse_statements(input, enclosing, ParseMode::IfElse, &close);
  }
  if (close != CloseKind::End) {
    PRINT_INPUT_ERROR("if is missing a matching end.");
  }
  return node;
}

static std::vector<std::unique_ptr<Node>> parse_statements(
  std::ifstream& input,
  const std::vector<std::string>& enclosing,
  ParseMode mode,
  CloseKind* close)
{
  if (close) {
    *close = CloseKind::None;
  }
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::string> tokens;
  while (next_statement(input, tokens)) {
    if (tokens[0] == "end") {
      if (tokens.size() != 1) {
        PRINT_INPUT_ERROR("end takes no arguments.");
      }
      if (mode == ParseMode::Top) {
        PRINT_INPUT_ERROR("end without matching for or if.");
      }
      if (close) {
        *close = CloseKind::End;
      }
      return nodes;
    }
    if (tokens[0] == "else") {
      if (tokens.size() != 1) {
        PRINT_INPUT_ERROR("else takes no arguments.");
      }
      if (mode != ParseMode::IfThen) {
        PRINT_INPUT_ERROR("else without matching if.");
      }
      if (close) {
        *close = CloseKind::Else;
      }
      return nodes;
    }
    if (tokens[0] == "for") {
      nodes.emplace_back(parse_for(input, tokens, enclosing));
    } else if (tokens[0] == "if") {
      nodes.emplace_back(parse_if(input, tokens, enclosing));
    } else {
      check_variable_refs(tokens, enclosing);
      auto cmd = std::make_unique<CommandNode>();
      cmd->tokens = std::move(tokens);
      nodes.emplace_back(std::move(cmd));
    }
  }
  if (mode == ParseMode::For) {
    PRINT_INPUT_ERROR("for is missing a matching end.");
  }
  if (mode == ParseMode::IfThen || mode == ParseMode::IfElse) {
    PRINT_INPUT_ERROR("if is missing a matching end.");
  }
  return nodes;
}

static std::vector<std::string> expand_tokens(
  const std::vector<std::string>& tokens, VariableScope& scope, const Box& box)
{
  std::vector<std::string> expanded;
  expanded.reserve(tokens.size());
  for (const auto& token : tokens) {
    expanded.emplace_back(scope.expand_text(token, box));
  }
  return expanded;
}

void CommandNode::execute(Run& run, VariableScope& scope)
{
  run.variables = &scope;
  std::vector<std::string> expanded = expand_tokens(tokens, scope, run.current_box());
  if (!expanded.empty() && expanded[0] == "variable") {
    scope.define_equal(expanded);
    return;
  }
  run.parse_one_keyword(expanded);
}

static bool is_delete_isolated_command(
  const Node& node, VariableScope& scope, const Box& box)
{
  const auto* cmd = dynamic_cast<const CommandNode*>(&node);
  if (cmd == nullptr || cmd->tokens.size() < 2) {
    return false;
  }
  const std::string a = scope.expand_text(cmd->tokens[0], box);
  const std::string b = scope.expand_text(cmd->tokens[1], box);
  return a == "delete" && b == "isolated";
}

static bool is_deposit_command(const Node& node, VariableScope& scope, const Box& box)
{
  const auto* cmd = dynamic_cast<const CommandNode*>(&node);
  if (cmd == nullptr || cmd->tokens.empty()) {
    return false;
  }
  return scope.expand_text(cmd->tokens[0], box) == "deposit";
}

static void execute_node_list(
  Run& run, std::vector<std::unique_ptr<Node>>& nodes, VariableScope& scope)
{
  size_t i = 0;
  while (i < nodes.size()) {
    if (is_delete_isolated_command(*nodes[i], scope, run.current_box())) {
      std::vector<std::vector<std::string>> batch;
      while (i < nodes.size() && is_delete_isolated_command(*nodes[i], scope, run.current_box())) {
        const auto* cmd = static_cast<const CommandNode*>(nodes[i].get());
        batch.push_back(expand_tokens(cmd->tokens, scope, run.current_box()));
        ++i;
      }
      run.variables = &scope;
      run.delete_isolated_batch(batch);
    } else if (is_deposit_command(*nodes[i], scope, run.current_box())) {
      std::vector<std::vector<std::string>> batch;
      while (i < nodes.size() && is_deposit_command(*nodes[i], scope, run.current_box())) {
        const auto* cmd = static_cast<const CommandNode*>(nodes[i].get());
        batch.push_back(expand_tokens(cmd->tokens, scope, run.current_box()));
        ++i;
      }
      run.variables = &scope;
      run.deposit_sequence(batch);
    } else {
      nodes[i]->execute(run, scope);
      ++i;
    }
  }
}

void ForNode::execute(Run& run, VariableScope& scope)
{
  for (const auto& value : values) {
    scope.push(variable, value);
    execute_node_list(run, body, scope);
    scope.pop();
  }
}

static bool eval_if_condition(const IfNode& node, VariableScope& scope, const Box& box)
{
  const std::string a = scope.expand_text(node.lhs, box);
  const std::string b = scope.expand_text(node.rhs, box);
  double xa = 0.0;
  double xb = 0.0;
  if (is_valid_real(a.c_str(), &xa) && is_valid_real(b.c_str(), &xb)) {
    if (node.op == "==") {
      return xa == xb;
    }
    if (node.op == "!=") {
      return xa != xb;
    }
    if (node.op == "<") {
      return xa < xb;
    }
    if (node.op == "<=") {
      return xa <= xb;
    }
    if (node.op == ">") {
      return xa > xb;
    }
    return xa >= xb;
  }
  if (node.op == "==") {
    return a == b;
  }
  if (node.op == "!=") {
    return a != b;
  }
  PRINT_INPUT_ERROR("if relational operator needs numeric operands.");
  return false;
}

void IfNode::execute(Run& run, VariableScope& scope)
{
  if (eval_if_condition(*this, scope, run.current_box())) {
    execute_node_list(run, then_body, scope);
  } else {
    execute_node_list(run, else_body, scope);
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
  auto nodes = parse_statements(input, enclosing, ParseMode::Top, nullptr);
  VariableScope scope;
  run.variables = &scope;
  execute_node_list(run, nodes, scope);
}
