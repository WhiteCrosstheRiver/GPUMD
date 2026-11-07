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
Equal-style variables for run.in: store a formula, evaluate on use.
${name} / $(formula) are immediate; v_name is delayed until resolve_real.
------------------------------------------------------------------------------*/

#include "variable.cuh"
#include "model/box.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>

static bool is_ident_start(char c)
{
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

static bool is_ident_char(char c)
{
  return is_ident_start(c) || (c >= '0' && c <= '9');
}

bool is_variable_identifier(const std::string& s)
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

bool is_delayed_variable_token(const std::string& token)
{
  return token.size() > 2 && token[0] == 'v' && token[1] == '_' &&
         is_variable_identifier(token.substr(2));
}

std::string delayed_variable_name(const std::string& token)
{
  return token.substr(2);
}

std::string format_variable_number(double value)
{
  char buf[64];
  snprintf(buf, sizeof(buf), "%.15g", value);
  return std::string(buf);
}

void VariableScope::push(const std::string& name, const std::string& value)
{
  stack_.emplace_back(name, value);
}

void VariableScope::pop()
{
  stack_.pop_back();
}

bool VariableScope::has_loop_variable(const std::string& name) const
{
  return find_loop(name) != nullptr;
}

const std::string* VariableScope::find_loop(const std::string& name) const
{
  for (int i = static_cast<int>(stack_.size()) - 1; i >= 0; --i) {
    if (stack_[i].first == name) {
      return &stack_[i].second;
    }
  }
  return nullptr;
}

void VariableScope::define_equal(const std::vector<std::string>& tokens)
{
  if (tokens.size() < 4) {
    PRINT_INPUT_ERROR("Usage: variable <name> equal <formula>");
  }
  if (!is_variable_identifier(tokens[1])) {
    PRINT_INPUT_ERROR("variable name must be an identifier [A-Za-z_][A-Za-z0-9_]*.");
  }
  if (tokens[2] != "equal") {
    PRINT_INPUT_ERROR("Only 'variable <name> equal <formula>' is supported.");
  }
  if (has_loop_variable(tokens[1])) {
    std::string msg = "Cannot define equal-style variable '" + tokens[1] +
                      "' while a for loop uses the same name.";
    PRINT_INPUT_ERROR(msg.c_str());
  }
  std::string formula;
  for (size_t i = 3; i < tokens.size(); ++i) {
    formula += tokens[i];
  }
  if (formula.empty()) {
    PRINT_INPUT_ERROR("variable equal needs a formula.");
  }
  equal_[tokens[1]] = formula;
  printf("Equal-style variable %s equal %s\n", tokens[1].c_str(), formula.c_str());
}

class FormulaParser
{
public:
  FormulaParser(VariableScope& scope, const Box& box, const std::string& text)
    : scope_(scope), box_(box), text_(text), i_(0)
  {
    next();
  }

  double parse()
  {
    const double v = parse_expr();
    if (kind_ != End) {
      PRINT_INPUT_ERROR("Unexpected token in variable formula.");
    }
    return v;
  }

private:
  enum Kind
  {
    End,
    Number,
    Ident,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Lp,
    Rp,
    Comma
  };

  VariableScope& scope_;
  const Box& box_;
  const std::string& text_;
  size_t i_;
  Kind kind_ = End;
  double number_ = 0.0;
  std::string ident_;

  void skip_space()
  {
    while (i_ < text_.size() && (text_[i_] == ' ' || text_[i_] == '\t')) {
      ++i_;
    }
  }

  void next()
  {
    skip_space();
    if (i_ >= text_.size()) {
      kind_ = End;
      return;
    }
    const char c = text_[i_];
    if (c == '+') {
      kind_ = Plus;
      ++i_;
      return;
    }
    if (c == '-') {
      kind_ = Minus;
      ++i_;
      return;
    }
    if (c == '*') {
      kind_ = Star;
      ++i_;
      return;
    }
    if (c == '/') {
      kind_ = Slash;
      ++i_;
      return;
    }
    if (c == '^') {
      kind_ = Caret;
      ++i_;
      return;
    }
    if (c == '(') {
      kind_ = Lp;
      ++i_;
      return;
    }
    if (c == ')') {
      kind_ = Rp;
      ++i_;
      return;
    }
    if (c == ',') {
      kind_ = Comma;
      ++i_;
      return;
    }
    if (is_ident_start(c)) {
      const size_t start = i_;
      ++i_;
      while (i_ < text_.size() && is_ident_char(text_[i_])) {
        ++i_;
      }
      ident_ = text_.substr(start, i_ - start);
      kind_ = Ident;
      return;
    }
    if ((c >= '0' && c <= '9') || c == '.') {
      const size_t start = i_;
      while (i_ < text_.size() && text_[i_] >= '0' && text_[i_] <= '9') {
        ++i_;
      }
      if (i_ < text_.size() && text_[i_] == '.') {
        ++i_;
        while (i_ < text_.size() && text_[i_] >= '0' && text_[i_] <= '9') {
          ++i_;
        }
      }
      if (i_ < text_.size() && (text_[i_] == 'e' || text_[i_] == 'E')) {
        ++i_;
        if (i_ < text_.size() && (text_[i_] == '+' || text_[i_] == '-')) {
          ++i_;
        }
        while (i_ < text_.size() && text_[i_] >= '0' && text_[i_] <= '9') {
          ++i_;
        }
      }
      if (!is_valid_real(text_.substr(start, i_ - start).c_str(), &number_)) {
        PRINT_INPUT_ERROR("Invalid number in variable formula.");
      }
      kind_ = Number;
      return;
    }
    PRINT_INPUT_ERROR("Invalid character in variable formula.");
  }

  double parse_expr()
  {
    double v = parse_term();
    while (kind_ == Plus || kind_ == Minus) {
      const Kind op = kind_;
      next();
      const double r = parse_term();
      v = (op == Plus) ? v + r : v - r;
    }
    return v;
  }

  double parse_term()
  {
    double v = parse_power();
    while (kind_ == Star || kind_ == Slash) {
      const Kind op = kind_;
      next();
      const double r = parse_power();
      if (op == Star) {
        v *= r;
      } else {
        if (r == 0.0) {
          PRINT_INPUT_ERROR("Division by zero in variable formula.");
        }
        v /= r;
      }
    }
    return v;
  }

  double parse_power()
  {
    double v = parse_unary();
    while (kind_ == Caret) {
      next();
      v = std::pow(v, parse_unary());
    }
    return v;
  }

  double parse_unary()
  {
    if (kind_ == Minus) {
      next();
      return -parse_unary();
    }
    if (kind_ == Plus) {
      next();
      return parse_unary();
    }
    return parse_primary();
  }

  double vector_length(int col) const
  {
    const double x = box_.cpu_h[col];
    const double y = box_.cpu_h[3 + col];
    const double z = box_.cpu_h[6 + col];
    return sqrt(x * x + y * y + z * z);
  }

  double parse_primary()
  {
    if (kind_ == Number) {
      const double v = number_;
      next();
      return v;
    }
    if (kind_ == Lp) {
      next();
      const double v = parse_expr();
      if (kind_ != Rp) {
        PRINT_INPUT_ERROR("Missing ')' in variable formula.");
      }
      next();
      return v;
    }
    if (kind_ != Ident) {
      PRINT_INPUT_ERROR("Expected a number or identifier in variable formula.");
    }
    const std::string id = ident_;
    next();
    if (kind_ == Lp) {
      if (id != "random") {
        std::string msg = "Unknown function '" + id + "' in variable formula.";
        PRINT_INPUT_ERROR(msg.c_str());
      }
      next();
      const double lo = parse_expr();
      if (kind_ != Comma) {
        PRINT_INPUT_ERROR("Usage: random(lo,hi,seed)");
      }
      next();
      const double hi = parse_expr();
      if (kind_ != Comma) {
        PRINT_INPUT_ERROR("Usage: random(lo,hi,seed)");
      }
      next();
      const double seed = parse_expr();
      if (kind_ != Rp) {
        PRINT_INPUT_ERROR("Usage: random(lo,hi,seed)");
      }
      next();
      if (lo > hi) {
        PRINT_INPUT_ERROR("random lo should be <= hi.");
      }
      if (!scope_.rng_initialized_) {
        scope_.rng_ = std::mt19937(static_cast<unsigned>(seed));
        scope_.rng_initialized_ = true;
      }
      if (lo == hi) {
        return lo;
      }
      std::uniform_real_distribution<double> dist(lo, hi);
      return dist(scope_.rng_);
    }
    if (id == "PI") {
      return 3.14159265358979323846;
    }
    if (id == "lx") {
      return vector_length(0);
    }
    if (id == "ly") {
      return vector_length(1);
    }
    if (id == "lz") {
      return vector_length(2);
    }
    if (id == "vol") {
      return box_.get_volume();
    }
    if (id.size() > 2 && id[0] == 'v' && id[1] == '_' && is_variable_identifier(id.substr(2))) {
      return scope_.eval_name(id.substr(2), box_);
    }
    std::string msg = "Unknown identifier '" + id + "' in variable formula.";
    PRINT_INPUT_ERROR(msg.c_str());
    return 0.0;
  }
};

double VariableScope::eval_formula(const std::string& formula, const Box& box)
{
  FormulaParser parser(*this, box, formula);
  return parser.parse();
}

double VariableScope::eval_name(const std::string& name, const Box& box)
{
  if (const std::string* loop = find_loop(name)) {
    double value = 0.0;
    if (!is_valid_real(loop->c_str(), &value)) {
      std::string msg = "for variable '" + name + "' is not a number.";
      PRINT_INPUT_ERROR(msg.c_str());
    }
    return value;
  }
  const auto it = equal_.find(name);
  if (it == equal_.end()) {
    std::string msg = "Undefined variable '" + name + "'.";
    PRINT_INPUT_ERROR(msg.c_str());
  }
  for (const auto& active : evaluating_) {
    if (active == name) {
      std::string msg = "Cyclic variable reference involving '" + name + "'.";
      PRINT_INPUT_ERROR(msg.c_str());
    }
  }
  evaluating_.push_back(name);
  const double value = eval_formula(it->second, box);
  evaluating_.pop_back();
  return value;
}

std::string VariableScope::expand_name(const std::string& name, const Box& box)
{
  if (const std::string* loop = find_loop(name)) {
    return *loop;
  }
  return format_variable_number(eval_name(name, box));
}

static void parse_brace_name(const std::string& text, size_t i, std::string& name, size_t& end)
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

static void parse_immediate_formula(
  const std::string& text, size_t i, std::string& formula, size_t& end)
{
  size_t j = i + 2;
  int depth = 1;
  while (j < text.size() && depth > 0) {
    if (text[j] == '(') {
      ++depth;
    } else if (text[j] == ')') {
      --depth;
    }
    if (depth > 0) {
      ++j;
    }
  }
  if (depth != 0) {
    PRINT_INPUT_ERROR("Unclosed $(formula) reference.");
  }
  formula = text.substr(i + 2, j - (i + 2));
  end = j + 1;
}

std::string VariableScope::expand_text(const std::string& text, const Box& box)
{
  std::string out;
  for (size_t i = 0; i < text.size();) {
    if (text[i] == '$' && i + 1 < text.size() && text[i + 1] == '{') {
      std::string name;
      size_t end = 0;
      parse_brace_name(text, i, name, end);
      out += expand_name(name, box);
      i = end;
    } else if (text[i] == '$' && i + 1 < text.size() && text[i + 1] == '(') {
      std::string formula;
      size_t end = 0;
      parse_immediate_formula(text, i, formula, end);
      out += format_variable_number(eval_formula(formula, box));
      i = end;
    } else {
      out += text[i];
      ++i;
    }
  }
  return out;
}

double VariableScope::resolve_real(const std::string& token, const Box& box, const char* what)
{
  if (is_delayed_variable_token(token)) {
    return eval_name(delayed_variable_name(token), box);
  }
  double value = 0.0;
  if (!is_valid_real(token.c_str(), &value)) {
    PRINT_INPUT_ERROR(what);
  }
  return value;
}
