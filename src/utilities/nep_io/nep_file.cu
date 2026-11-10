/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "nep_file.cuh"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <istream>
#include <limits>
#include <locale>
#include <ostream>
#include <set>
#include <sstream>
#include <utility>

static std::string lower(std::string text)
{
  for (char& c : text) {
    if (c >= 'A' && c <= 'Z') {
      c += 'a' - 'A';
    }
  }
  return text;
}

static bool identifier(const std::string& text)
{
  if (text.empty()) {
    return false;
  }
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' ||
          (i > 0 && c >= '0' && c <= '9'))) {
      return false;
    }
  }
  return true;
}

static bool token_valid(const std::string& token)
{
  if (token.empty()) {
    return false;
  }
  for (const unsigned char c : token) {
    if (c <= 32 || c == 127 || c == '#') {
      return false;
    }
  }
  return true;
}

static bool fail(std::string& error, const std::string& message, std::size_t line = 0)
{
  error = "NEP file" + (line ? " line " + std::to_string(line) : "") + ": " + message;
  return false;
}

static std::string element_symbol(const std::string& token)
{
  static const char* const symbols[] = {
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
    "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};
  const std::string name = lower(token);
  for (const char* symbol : symbols) {
    if (lower(symbol) == name) {
      return symbol;
    }
  }
  return "";
}

static int next_line(std::istream& input, std::string& line)
{
  line.clear();
  try {
    std::getline(input, line);
  } catch (const std::ios_base::failure&) {
    if (!input.eof() || input.bad()) {
      return -1;
    }
  } catch (...) {
    if (!input.bad()) {
      throw;
    }
    return -1;
  }
  if (input.bad() || (input.fail() && !input.eof())) {
    return -1;
  }
  return input.fail() ? 0 : 1;
}

static bool tokenize(std::string line, std::vector<std::string>& tokens)
{
  tokens.clear();
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  const std::size_t comment = line.find('#');
  if (comment != std::string::npos) {
    line.resize(comment);
  }
  for (const unsigned char c : line) {
    if ((c < 32 && c != '\t') || c == 127) {
      return false;
    }
  }
  std::istringstream input(line);
  input.imbue(std::locale::classic());
  std::string token;
  while (input >> token) {
    tokens.push_back(token);
  }
  return true;
}

bool NEP_File::validate(std::string& error)
{
  if (elements.empty()) {
    return fail(error, "empty element list.", 1);
  }
  std::set<std::string> seen_elements;
  for (std::string& element : elements) {
    const std::string symbol = element_symbol(element);
    if (symbol.empty()) {
      return fail(error, "unknown element '" + element + "'.", 1);
    }
    if (!seen_elements.insert(symbol).second) {
      return fail(error, "duplicate element '" + symbol + "'.", 1);
    }
    element = symbol;
  }
  std::set<std::string> names;
  for (Field& field : fields) {
    field.name = lower(field.name);
    if (!identifier(field.name)) {
      return fail(error, "invalid field name.", field.line);
    }
    if (!names.insert(field.name).second) {
      return fail(error, "duplicate record '" + field.name + "'.", field.line);
    }
    if (field.values.empty()) {
      return fail(error, "empty field '" + field.name + "'.", field.line);
    }
    for (const std::string& value : field.values) {
      if (!token_valid(value)) {
        return fail(error, "invalid token in '" + field.name + "'.", field.line);
      }
    }
    if (field.name == "model") {
      if (field.values.size() != 1 || !identifier(field.values[0])) {
        return fail(error, "model requires one identifier.", field.line);
      }
      field.values[0] = lower(field.values[0]);
    }
  }
  for (Block& block : blocks) {
    block.name = lower(block.name);
    if (!identifier(block.name)) {
      return fail(error, "invalid block name.", block.line);
    }
    if (!names.insert(block.name).second) {
      return fail(error, "duplicate record '" + block.name + "'.", block.line);
    }
    for (const double value : block.values) {
      if (!std::isfinite(value)) {
        return fail(error, "non-finite value in '" + block.name + "'.", block.line);
      }
    }
  }
  const Field* version = find_field("format_version");
  if (version == nullptr) {
    return fail(error, "missing format_version.");
  }
  if (version->values.size() != 1 || version->values[0] != "1") {
    return fail(error, "unsupported format_version (expected 1).", version->line);
  }
  if (find_field("model") == nullptr) {
    return fail(error, "missing model.");
  }
  return true;
}

static bool contains(const std::vector<std::string>& names, const std::string& name)
{
  for (const std::string& entry : names) {
    if (lower(entry) == lower(name)) {
      return true;
    }
  }
  return false;
}

static void put_line(std::ostream& output, std::ostringstream& line)
{
  line << '\n';
  const std::string text = line.str();
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  line.str("");
}

static bool compare_fields(const NEP_File::Field& a, const NEP_File::Field& b)
{
  if (a.name == b.name) {
    return false;
  }
  if (a.name == "format_version" || b.name == "format_version") {
    return a.name == "format_version";
  }
  if (a.name == "model" || b.name == "model") {
    return a.name == "model";
  }
  return a.name < b.name;
}

static bool compare_blocks(const NEP_File::Block& a, const NEP_File::Block& b)
{
  return a.name < b.name;
}

const NEP_File::Field* NEP_File::find_field(const std::string& name) const
{
  for (const Field& field : fields) {
    if (lower(field.name) == lower(name)) {
      return &field;
    }
  }
  return nullptr;
}

const NEP_File::Block* NEP_File::find_block(const std::string& name) const
{
  for (const Block& block : blocks) {
    if (lower(block.name) == lower(name)) {
      return &block;
    }
  }
  return nullptr;
}

bool NEP_File::parse_size(const std::string& token, std::size_t& value)
{
  if (token.empty()) {
    return false;
  }
  std::size_t parsed = 0;
  for (const char c : token) {
    if (c < '0' || c > '9') {
      return false;
    }
    const unsigned int digit = static_cast<unsigned int>(c - '0');
    if (parsed > (std::numeric_limits<std::size_t>::max() - digit) / 10) {
      return false;
    }
    parsed = parsed * 10 + digit;
  }
  value = parsed;
  return true;
}

bool NEP_File::parse_double(const std::string& token, double& value)
{
  std::size_t i = 0;
  if (i < token.size() && (token[i] == '+' || token[i] == '-')) {
    ++i;
  }
  bool digits = false;
  bool nonzero = false;
  bool point = false;
  for (; i < token.size(); ++i) {
    const char c = token[i];
    if (c >= '0' && c <= '9') {
      digits = true;
      nonzero = nonzero || c != '0';
    } else if (c == '.' && !point) {
      point = true;
    } else {
      break;
    }
  }
  if (!digits) {
    return false;
  }
  if (i < token.size() && (token[i] == 'e' || token[i] == 'E')) {
    ++i;
    if (i < token.size() && (token[i] == '+' || token[i] == '-')) {
      ++i;
    }
    const std::size_t start = i;
    while (i < token.size() && token[i] >= '0' && token[i] <= '9') {
      ++i;
    }
    if (i == start) {
      return false;
    }
  }
  if (i != token.size()) {
    return false;
  }
  std::istringstream input(token);
  input.imbue(std::locale::classic());
  double parsed;
  if (!(input >> parsed) || !std::isfinite(parsed) || (parsed == 0.0 && nonzero)) {
    return false;
  }
  value = parsed;
  return true;
}

static bool is_nep5_header(const std::string& token)
{
  const std::string name = lower(token);
  return name == "nep5" || name == "nep5_zbl";
}

bool NEP_File::is_nep_header(const std::string& token)
{
  return lower(token) == "nep" || is_nep5_header(token);
}

// gnep writes NEP5 files: the legacy NEP4 text layout plus one output bias per element
// after each element's [w0 b0 w1] block. That bias is a constant energy per atom, so it maps
// exactly onto energy_reference and the rest is a plain "nep" model. first holds the
// tokens of the header line, which has already been read from input.
static bool translate_nep5(
  std::istream& input, const std::vector<std::string>& first, std::string& text, std::string& error)
{
  if (first.size() < 3) {
    return fail(error, "NEP5 header requires: nep5 <num_types> <elements...>.", 1);
  }
  std::size_t num_types;
  if (!NEP_File::parse_size(first[1], num_types) || num_types == 0 ||
      num_types != first.size() - 2) {
    return fail(error, "element count does not match first line.", 1);
  }
  const int T = static_cast<int>(num_types);
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "nep " << T;
  for (std::size_t i = 2; i < first.size(); ++i) {
    out << ' ' << first[i];
  }
  out << "\nformat_version 1\nmodel nep\n";

  std::string line;
  std::vector<std::string> tokens;
  std::size_t line_number = 1;
  int n_max_radial = -1, n_max_angular = -1, basis_radial = -1, basis_angular = -1;
  int l_max = -1, l_max_4 = 0, l_max_5 = 0, neurons = -1;
  bool flexible_zbl = false;
  auto to_int = [](const std::string& token, int& value) {
    std::size_t parsed;
    if (!NEP_File::parse_size(token, parsed) || parsed > 100000) {
      return false;
    }
    value = static_cast<int>(parsed);
    return true;
  };
  while (neurons < 0) {
    const int status = next_line(input, line);
    ++line_number;
    if (status <= 0) {
      return fail(error, "NEP5 model ended before its ANN line.", line_number);
    }
    if (!tokenize(line, tokens)) {
      return fail(error, "invalid control character.", line_number);
    }
    if (tokens.empty()) {
      continue;
    }
    const std::string key = lower(tokens[0]);
    bool ok = true;
    if (key == "zbl" && (tokens.size() == 3 || tokens.size() == 4)) {
      double inner, outer;
      ok = NEP_File::parse_double(tokens[1], inner) && NEP_File::parse_double(tokens[2], outer);
      flexible_zbl = ok && inner == 0.0 && outer == 0.0;
      out << line << '\n';
    } else if (key == "cutoff") {
      out << line << '\n';
    } else if (key == "n_max" && tokens.size() == 3) {
      ok = to_int(tokens[1], n_max_radial) && to_int(tokens[2], n_max_angular);
      out << line << '\n';
    } else if (key == "basis_size" && tokens.size() == 3) {
      ok = to_int(tokens[1], basis_radial) && to_int(tokens[2], basis_angular);
      out << line << '\n';
    } else if (key == "l_max" && tokens.size() >= 2 && tokens.size() <= 4) {
      ok = to_int(tokens[1], l_max) && (tokens.size() < 3 || to_int(tokens[2], l_max_4)) &&
           (tokens.size() < 4 || to_int(tokens[3], l_max_5));
      out << "l_max " << l_max << ' ' << l_max_4 << ' ' << l_max_5 << '\n';
    } else if (key == "ann" && tokens.size() == 3) {
      int second = -1;
      ok = to_int(tokens[1], neurons) && to_int(tokens[2], second);
      if (ok && second != 0) {
        return fail(error, "NEP5 models with two hidden layers are not supported.", line_number);
      }
      ok = ok && neurons >= 1;
      out << "ann " << neurons << " 0\n";
    } else {
      return fail(error, "unexpected NEP5 header line '" + tokens[0] + "'.", line_number);
    }
    if (!ok) {
      return fail(error, "invalid value in NEP5 header line '" + tokens[0] + "'.", line_number);
    }
  }
  if (n_max_radial < 0 || n_max_angular < 0 || basis_radial < 0 || basis_angular < 0 ||
      l_max < 0) {
    return fail(error, "NEP5 model has an incomplete header.", line_number);
  }

  std::vector<std::string> values;
  int status;
  while ((status = next_line(input, line)) > 0) {
    ++line_number;
    if (!tokenize(line, tokens)) {
      return fail(error, "invalid control character.", line_number);
    }
    values.insert(values.end(), tokens.begin(), tokens.end());
  }
  if (status < 0) {
    return fail(error, "input stream failure.", line_number + 1);
  }

  const int num_l = l_max + (l_max_4 == 2 ? 1 : 0) + (l_max_5 == 1 ? 1 : 0);
  const std::size_t dim = n_max_radial + 1 + std::size_t(n_max_angular + 1) * num_l;
  const std::size_t block = (dim + 2) * neurons + 1;
  const std::size_t descriptor =
    num_types * num_types *
    (std::size_t(n_max_radial + 1) * (basis_radial + 1) +
     std::size_t(n_max_angular + 1) * (basis_angular + 1));
  const std::size_t zbl = flexible_zbl ? 10 * num_types * (num_types + 1) / 2 : 0;
  const std::size_t expected = block * num_types + 1 + descriptor + dim + zbl;
  if (values.size() != expected) {
    return fail(
      error,
      "NEP5 model has " + std::to_string(values.size()) + " values after its header, expected " +
        std::to_string(expected) + ".",
      line_number);
  }
  std::vector<double> numbers(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (!NEP_File::parse_double(values[i], numbers[i])) {
      return fail(error, "invalid value '" + values[i] + "' in NEP5 model.");
    }
  }

  // E_atom = ANN(q) - b_t - b_shared for NEP5; for "nep" it is ANN(q) - b + reference[t].
  const double shared_bias = numbers[block * num_types];
  out << std::setprecision(std::numeric_limits<double>::max_digits10) << "energy_reference";
  for (int t = 0; t < T; ++t) {
    out << ' ' << -(numbers[t * block + block - 1] + shared_bias);
  }
  out << '\n';
  out << "@ parameters " << (block - 1) * num_types + 1 + descriptor << '\n';
  for (int t = 0; t < T; ++t) {
    for (std::size_t n = 0; n + 1 < block; ++n) {
      out << values[t * block + n] << '\n';
    }
  }
  out << "0\n";
  std::size_t index = block * num_types + 1;
  for (std::size_t n = 0; n < descriptor; ++n) {
    out << values[index++] << '\n';
  }
  out << "@ q_scaler " << dim << '\n';
  for (std::size_t n = 0; n < dim; ++n) {
    out << values[index++] << '\n';
  }
  if (zbl) {
    out << "@ zbl_parameters " << zbl << '\n';
    for (std::size_t n = 0; n < zbl; ++n) {
      out << values[index++] << '\n';
    }
  }
  text = out.str();
  return true;
}

bool NEP_File::read(std::istream& input, std::string& error)
{
  return read_impl(input, error, false);
}

bool NEP_File::read_model_name(std::istream& input, std::string& name, std::string& error)
{
  name.clear();
  NEP_File file;
  if (!file.read_impl(input, error, true)) {
    return false;
  }
  name = file.find_field("model")->values[0];
  return true;
}

bool NEP_File::read_impl(std::istream& input, std::string& error, bool model_only)
{
  *this = NEP_File();
  error.clear();
  NEP_File parsed;
  std::string line;
  std::vector<std::string> tokens;
  int status = next_line(input, line);
  if (status < 0) {
    return fail(error, "input stream failure.", 1);
  }
  if (status == 0 || !tokenize(line, tokens) || tokens.size() < 3 || !is_nep_header(tokens[0])) {
    return fail(error, "expected first line: nep <num_types> <elements...>.", 1);
  }
  if (is_nep5_header(tokens[0])) {
    std::string text;
    if (!translate_nep5(input, tokens, text, error)) {
      return false;
    }
    std::istringstream translated(text);
    translated.imbue(std::locale::classic());
    return read_impl(translated, error, model_only);
  }
  std::size_t num_types;
  if (!parse_size(tokens[1], num_types) || num_types == 0 || num_types != tokens.size() - 2) {
    return fail(error, "element count does not match first line.", 1);
  }
  parsed.elements.assign(tokens.begin() + 2, tokens.end());
  std::size_t line_number = 1;
  std::size_t remaining = 0;
  while ((status = next_line(input, line)) > 0) {
    ++line_number;
    if (!tokenize(line, tokens)) {
      return fail(error, "invalid control character.", line_number);
    }
    if (tokens.empty()) {
      continue;
    }
    if (remaining != 0) {
      Block& block = parsed.blocks.back();
      if (tokens.size() > remaining) {
        return fail(error, "too many values in block '" + block.name + "'.", line_number);
      }
      if (!model_only) {
        for (const std::string& token : tokens) {
          double value;
          if (!parse_double(token, value)) {
            return fail(
              error, "expected finite decimal value in block '" + block.name + "'.", line_number);
          }
          block.values.push_back(value);
        }
      }
      remaining -= tokens.size();
    } else if (tokens[0] == "@") {
      if (tokens.size() != 3 || !identifier(tokens[1]) || !parse_size(tokens[2], remaining)) {
        return fail(error, "expected block header: @ <name> <count>.", line_number);
      }
      Block block;
      block.name = lower(tokens[1]);
      block.line = line_number;
      if (remaining > block.values.max_size()) {
        return fail(error, "block count too large.", line_number);
      }

      parsed.blocks.push_back(std::move(block));
    } else {
      if (tokens.size() < 2 || !identifier(tokens[0])) {
        return fail(error, "expected field: <name> <values...>.", line_number);
      }
      Field field;
      field.name = lower(tokens[0]);
      field.values.assign(tokens.begin() + 1, tokens.end());
      field.line = line_number;
      if (model_only && field.name == "model") {
        if (field.values.size() != 1 || !identifier(field.values[0])) {
          return fail(error, "model requires one identifier.", line_number);
        }
        field.values[0] = lower(field.values[0]);
        parsed.fields.push_back(std::move(field));
        *this = std::move(parsed);
        return true;
      }
      parsed.fields.push_back(std::move(field));
    }
  }
  if (status < 0) {
    return fail(error, "input stream failure.", line_number + 1);
  }
  if (remaining != 0) {
    return fail(
      error, "truncated block '" + parsed.blocks.back().name + "'.", parsed.blocks.back().line);
  }
  if (!parsed.validate(error)) {
    return false;
  }
  *this = std::move(parsed);
  return true;
}

bool NEP_File::write(std::ostream& output, std::string& error) const
{
  error.clear();
  NEP_File normalized = *this;
  if (!normalized.validate(error)) {
    return false;
  }

  std::sort(normalized.fields.begin(), normalized.fields.end(), compare_fields);
  std::sort(normalized.blocks.begin(), normalized.blocks.end(), compare_blocks);
  std::ostringstream line;
  line.imbue(std::locale::classic());
  line << std::setprecision(std::numeric_limits<double>::max_digits10);
  try {
    line << "nep " << normalized.elements.size();
    for (const std::string& element : normalized.elements) {
      line << ' ' << element;
    }
    put_line(output, line);
    for (const Field& field : normalized.fields) {
      line << field.name;
      for (const std::string& value : field.values) {
        line << ' ' << value;
      }
      put_line(output, line);
    }
    for (const Block& block : normalized.blocks) {
      line << "@ " << block.name << ' ' << block.values.size();
      put_line(output, line);
      for (std::size_t i = 0; i < block.values.size(); ++i) {
        if (i % 5 != 0) {
          line << ' ';
        }
        line << block.values[i];
        if (i % 5 == 4 || i + 1 == block.values.size()) {
          put_line(output, line);
        }
      }
    }
  } catch (const std::ios_base::failure&) {
    return fail(error, "output stream failure.");
  } catch (...) {
    if (!output.bad()) {
      throw;
    }
    return fail(error, "output stream failure.");
  }
  if (!output) {
    return fail(error, "output stream failure.");
  }
  return true;
}

bool NEP_File::check_unhandled(
  const std::vector<std::string>& handled_fields,
  const std::vector<std::string>& handled_blocks,
  std::string& error) const
{
  error.clear();
  for (const Field& field : fields) {
    if (!contains(handled_fields, field.name)) {
      return fail(error, "unhandled field '" + field.name + "'.", field.line);
    }
  }
  for (const Block& block : blocks) {
    if (!contains(handled_blocks, block.name)) {
      return fail(error, "unhandled block '" + block.name + "'.", block.line);
    }
  }
  return true;
}
