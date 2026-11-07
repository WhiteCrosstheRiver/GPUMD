/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#pragma once

#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

enum class CoordCmp { LT, LE, GT, GE, EQ, NE };

enum class IsolatedMode { Only, Full, Selected };

struct CoordExpr
{
  enum Kind { CMP, AND, OR, NOT } kind = CMP;
  std::vector<std::string> species; // {"X"} = any neighbor; {"Si","Ge"} = Si+Ge count
  CoordCmp cmp = CoordCmp::LT;
  int value = 0;
  std::unique_ptr<CoordExpr> a;
  std::unique_ptr<CoordExpr> b;
};

inline void split_isolated_token(const std::string& s, std::vector<std::string>& out)
{
  const int n = static_cast<int>(s.size());
  int i = 0;
  while (i < n) {
    if (s[i] == '(' || s[i] == ')') {
      out.emplace_back(1, s[i]);
      ++i;
      continue;
    }
    if (i + 1 < n) {
      const char c0 = s[i];
      const char c1 = s[i + 1];
      if ((c0 == '<' && c1 == '=') || (c0 == '>' && c1 == '=') || (c0 == '=' && c1 == '=') ||
          (c0 == '!' && c1 == '=')) {
        out.emplace_back(s.substr(i, 2));
        i += 2;
        continue;
      }
    }
    if (s[i] == '<' || s[i] == '>') {
      out.emplace_back(1, s[i]);
      ++i;
      continue;
    }
    if (s[i] == '=' || s[i] == '!') {
      PRINT_INPUT_ERROR("delete isolated: use == or != in the coordination expression.");
    }
    int j = i;
    while (j < n && s[j] != '(' && s[j] != ')' && s[j] != '<' && s[j] != '>' && s[j] != '=' &&
           s[j] != '!') {
      ++j;
    }
    if (j == i) {
      PRINT_INPUT_ERROR("delete isolated: invalid character in expression.");
    }
    out.push_back(s.substr(i, j - i));
    i = j;
  }
}

inline bool is_isolated_cmd_keyword(const std::string& t)
{
  return t == "cutoff" || t == "type" || t == "only" || t == "full" || t == "selected";
}

inline bool is_isolated_expr_start(const std::string& t)
{
  return t == "coord" || t == "not" || t == "(";
}

inline bool is_wildcard_species(const std::string& s) { return s == "X" || s == "x"; }

inline bool is_coord_cmp_token(const std::string& t)
{
  return t == "<" || t == "<=" || t == ">" || t == ">=" || t == "==" || t == "!=";
}

struct CoordTokenReader
{
  const std::vector<std::string>& t;
  int i = 0;
  bool at_end() const { return i >= (int)t.size(); }
  const std::string& peek() const
  {
    if (at_end()) {
      PRINT_INPUT_ERROR("delete isolated: unexpected end of coordination expression.");
    }
    return t[i];
  }
  bool peek_is(const char* w) const { return !at_end() && t[i] == w; }
  std::string get()
  {
    const std::string s = peek();
    ++i;
    return s;
  }
  bool accept(const char* w)
  {
    if (peek_is(w)) {
      ++i;
      return true;
    }
    return false;
  }
};

inline bool is_coord_species_stop(const CoordTokenReader& r)
{
  if (r.at_end()) {
    return true;
  }
  const std::string& t = r.t[r.i];
  if (is_coord_cmp_token(t) || t == "or" || t == "and" || t == "not" || t == "coord" || t == ")" ||
      is_isolated_cmd_keyword(t)) {
    return true;
  }
  int dummy = 0;
  return is_valid_int(t.c_str(), &dummy) != 0;
}

inline CoordCmp parse_coord_cmp(CoordTokenReader& r)
{
  if (r.accept("<")) {
    return CoordCmp::LT;
  }
  if (r.accept("<=")) {
    return CoordCmp::LE;
  }
  if (r.accept(">")) {
    return CoordCmp::GT;
  }
  if (r.accept(">=")) {
    return CoordCmp::GE;
  }
  if (r.accept("==")) {
    return CoordCmp::EQ;
  }
  if (r.accept("!=")) {
    return CoordCmp::NE;
  }
  PRINT_INPUT_ERROR("delete isolated: expected <, <=, >, >=, ==, or != after coord species.");
  return CoordCmp::LT;
}

inline std::unique_ptr<CoordExpr> parse_coord_or(CoordTokenReader& r);

inline std::unique_ptr<CoordExpr> parse_coord_atom(CoordTokenReader& r)
{
  if (r.accept("(")) {
    auto inner = parse_coord_or(r);
    if (!r.accept(")")) {
      PRINT_INPUT_ERROR("delete isolated: missing ')' in coordination expression.");
    }
    return inner;
  }
  if (!r.accept("coord")) {
    PRINT_INPUT_ERROR("delete isolated: expected coord, not, or '(' in expression.");
  }
  auto node = std::make_unique<CoordExpr>();
  node->kind = CoordExpr::CMP;
  if (r.at_end()) {
    PRINT_INPUT_ERROR("delete isolated: Usage: coord <species|X> [species ...] <op> <n>");
  }
  int shorthand = 0;
  if (is_valid_int(r.peek().c_str(), &shorthand)) {
    if (shorthand < 0) {
      PRINT_INPUT_ERROR("delete isolated coord count should be >= 0.");
    }
    r.get();
    node->species = {"X"};
    node->cmp = CoordCmp::LT;
    node->value = shorthand;
    return node;
  }
  bool saw_wildcard = false;
  while (!is_coord_species_stop(r)) {
    std::string s = r.get();
    if (s.empty() || is_isolated_cmd_keyword(s) || s == "or" || s == "and" || s == "not" ||
        s == "coord") {
      PRINT_INPUT_ERROR("delete isolated: coord needs a species or X.");
    }
    if (is_wildcard_species(s)) {
      saw_wildcard = true;
    } else {
      node->species.push_back(s);
    }
  }
  if (saw_wildcard) {
    node->species = {"X"};
  }
  if (node->species.empty()) {
    PRINT_INPUT_ERROR("delete isolated: coord needs a species or X.");
  }
  if (r.at_end()) {
    PRINT_INPUT_ERROR("delete isolated: Usage: coord <species|X> [species ...] <op> <n>");
  }
  int value_only = 0;
  if (is_valid_int(r.peek().c_str(), &value_only)) {
    r.get();
    node->cmp = CoordCmp::LT;
    node->value = value_only;
  } else {
    node->cmp = parse_coord_cmp(r);
    if (r.at_end() || !is_valid_int(r.peek().c_str(), &node->value)) {
      PRINT_INPUT_ERROR("delete isolated: coord comparison needs an integer.");
    }
    r.get();
  }
  if (node->value < 0) {
    PRINT_INPUT_ERROR("delete isolated coord count should be >= 0.");
  }
  return node;
}

inline std::unique_ptr<CoordExpr> parse_coord_not(CoordTokenReader& r)
{
  if (r.accept("not")) {
    auto node = std::make_unique<CoordExpr>();
    node->kind = CoordExpr::NOT;
    node->a = parse_coord_not(r);
    return node;
  }
  return parse_coord_atom(r);
}

inline std::unique_ptr<CoordExpr> parse_coord_and(CoordTokenReader& r)
{
  auto left = parse_coord_not(r);
  while (r.peek_is("and")) {
    r.get();
    auto node = std::make_unique<CoordExpr>();
    node->kind = CoordExpr::AND;
    node->a = std::move(left);
    node->b = parse_coord_not(r);
    left = std::move(node);
  }
  return left;
}

inline std::unique_ptr<CoordExpr> parse_coord_or(CoordTokenReader& r)
{
  auto left = parse_coord_and(r);
  while (r.peek_is("or")) {
    r.get();
    auto node = std::make_unique<CoordExpr>();
    node->kind = CoordExpr::OR;
    node->a = std::move(left);
    node->b = parse_coord_and(r);
    left = std::move(node);
  }
  return left;
}

inline std::unique_ptr<CoordExpr> parse_coord_expr(const std::vector<std::string>& tokens, int& i)
{
  if (i >= (int)tokens.size() || is_isolated_cmd_keyword(tokens[i])) {
    auto node = std::make_unique<CoordExpr>();
    node->kind = CoordExpr::CMP;
    node->species = {"X"};
    node->cmp = CoordCmp::LT;
    node->value = 1;
    return node;
  }
  CoordTokenReader r{tokens, i};
  auto expr = parse_coord_or(r);
  i = r.i;
  return expr;
}

inline bool eval_coord_cmp(int n, CoordCmp cmp, int value)
{
  switch (cmp) {
    case CoordCmp::LT:
      return n < value;
    case CoordCmp::LE:
      return n <= value;
    case CoordCmp::GT:
      return n > value;
    case CoordCmp::GE:
      return n >= value;
    case CoordCmp::EQ:
      return n == value;
    case CoordCmp::NE:
      return n != value;
  }
  return false;
}

inline int coord_count_of(
  const std::vector<std::string>& species, int total, const std::unordered_map<std::string, int>& by_sym)
{
  if (species.size() == 1 && species[0] == "X") {
    return total;
  }
  int n = 0;
  for (const auto& s : species) {
    auto it = by_sym.find(s);
    if (it != by_sym.end()) {
      n += it->second;
    }
  }
  return n;
}

inline bool eval_coord_expr(
  const CoordExpr& e, int total, const std::unordered_map<std::string, int>& by_sym)
{
  switch (e.kind) {
    case CoordExpr::CMP:
      return eval_coord_cmp(coord_count_of(e.species, total, by_sym), e.cmp, e.value);
    case CoordExpr::AND:
      return eval_coord_expr(*e.a, total, by_sym) && eval_coord_expr(*e.b, total, by_sym);
    case CoordExpr::OR:
      return eval_coord_expr(*e.a, total, by_sym) || eval_coord_expr(*e.b, total, by_sym);
    case CoordExpr::NOT:
      return !eval_coord_expr(*e.a, total, by_sym);
  }
  return false;
}

// true if any leaf counts a named species (not just X); by_sym map only needed then
inline bool coord_expr_needs_by_sym(const CoordExpr& e)
{
  switch (e.kind) {
    case CoordExpr::CMP:
      for (const auto& s : e.species) {
        if (!is_wildcard_species(s)) {
          return true;
        }
      }
      return false;
    case CoordExpr::AND:
    case CoordExpr::OR:
      return coord_expr_needs_by_sym(*e.a) || coord_expr_needs_by_sym(*e.b);
    case CoordExpr::NOT:
      return coord_expr_needs_by_sym(*e.a);
  }
  return false;
}

inline std::string coord_cmp_str(CoordCmp cmp)
{
  switch (cmp) {
    case CoordCmp::LT:
      return "<";
    case CoordCmp::LE:
      return "<=";
    case CoordCmp::GT:
      return ">";
    case CoordCmp::GE:
      return ">=";
    case CoordCmp::EQ:
      return "==";
    case CoordCmp::NE:
      return "!=";
  }
  return "?";
}

inline std::string format_coord_species(const std::vector<std::string>& species)
{
  std::string out;
  for (size_t k = 0; k < species.size(); ++k) {
    if (k > 0) {
      out += " ";
    }
    out += species[k];
  }
  return out;
}

inline std::string format_coord_expr(const CoordExpr& e)
{
  if (e.kind == CoordExpr::CMP) {
    return "coord " + format_coord_species(e.species) + " " + coord_cmp_str(e.cmp) + " " +
           std::to_string(e.value);
  }
  if (e.kind == CoordExpr::NOT) {
    return "not ( " + format_coord_expr(*e.a) + " )";
  }
  const char* op = (e.kind == CoordExpr::AND) ? " and " : " or ";
  return "( " + format_coord_expr(*e.a) + op + format_coord_expr(*e.b) + " )";
}
