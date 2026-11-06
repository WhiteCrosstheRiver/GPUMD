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

#include "utilities/error.cuh"
#include "nep_model_file.cuh"
#include "nep_model_input.cuh"
#include <sstream>
#include <string>

bool NEP_Model_Input::is_new_format(const std::string& token)
{
  return token.size() == 3 && (token[0] == 'n' || token[0] == 'N') &&
         (token[1] == 'e' || token[1] == 'E') && (token[2] == 'p' || token[2] == 'P');
}

// NEP5 differs from NEP4 only by one output bias per element. Each bias is a constant
// energy per atom, so it maps exactly onto energy_reference and the MD kernels and forces
// are unchanged. File: per-type [w0 b0 w1 b_t] blocks, a shared bias, then the same
// descriptor, scaler and ZBL data as a legacy NEP4 file.
void NEP_Model_Input::translate_nep5()
{
  std::string line;
  std::ostringstream output;
  std::getline(file, line);
  std::vector<std::string> tokens = ::get_tokens(line);
  if (tokens.size() < 3) {
    PRINT_INPUT_ERROR("The first line of a NEP5 model is incomplete.");
  }
  const int num_types = std::stoi(tokens[1]);
  if (num_types < 1 || int(tokens.size()) != 2 + num_types) {
    PRINT_INPUT_ERROR("The first line of a NEP5 model has the wrong number of elements.");
  }
  output << (tokens[0] == "nep5_zbl" ? "nep4_zbl" : "nep4");
  for (size_t n = 1; n < tokens.size(); ++n) {
    output << ' ' << tokens[n];
  }
  output << '\n';

  int n_max_radial = -1, n_max_angular = -1, l_max = -1, l_max_4 = 0, l_max_5 = 0, neurons = -1;
  while (neurons < 0) {
    if (!std::getline(file, line)) {
      PRINT_INPUT_ERROR("A NEP5 model ended before its ANN line.");
    }
    tokens = ::get_tokens(line);
    if (tokens.empty()) {
      continue;
    }
    if (tokens[0] == "n_max" && tokens.size() == 3) {
      n_max_radial = std::stoi(tokens[1]);
      n_max_angular = std::stoi(tokens[2]);
    } else if (tokens[0] == "l_max" && tokens.size() >= 2) {
      l_max = std::stoi(tokens[1]);
      l_max_4 = tokens.size() > 2 ? std::stoi(tokens[2]) : 0;
      l_max_5 = tokens.size() > 3 ? std::stoi(tokens[3]) : 0;
    } else if (tokens[0] == "ANN") {
      if (tokens.size() != 3 || tokens[2] != "0") {
        PRINT_INPUT_ERROR("Native MD supports only one hidden layer.");
      }
      neurons = std::stoi(tokens[1]);
    }
    output << line << '\n';
  }
  if (n_max_radial < 0 || n_max_angular < 0 || l_max < 0 || neurons < 1) {
    PRINT_INPUT_ERROR("A NEP5 model has an incomplete header.");
  }

  const int num_l = l_max + (l_max_4 == 2) + (l_max_5 == 1);
  const int dim = n_max_radial + 1 + (n_max_angular + 1) * num_l;
  const size_t block = size_t(dim + 2) * neurons + 1;
  std::vector<std::string> values;
  while (std::getline(file, line)) {
    tokens = ::get_tokens(line);
    if (!tokens.empty()) {
      values.push_back(tokens[0]);
    }
  }
  if (values.size() < block * num_types + 1) {
    PRINT_INPUT_ERROR("A NEP5 model has too few parameters.");
  }
  const double shared_bias = std::stod(values[block * num_types]);
  reference.resize(num_types);
  for (int t = 0; t < num_types; ++t) {
    for (size_t n = 0; n + 1 < block; ++n) {
      output << values[t * block + n] << '\n';
    }
    reference[t] = -(std::stod(values[t * block + block - 1]) + shared_bias);
  }
  output << "0\n";
  for (size_t n = block * num_types + 1; n < values.size(); ++n) {
    output << values[n] << '\n';
  }
  converted.str(output.str());
  translated_nep5 = true;
}

NEP_Model_Input::NEP_Model_Input(const std::string& filename) : file(filename)
{
  if (!file.is_open()) {
    const std::string error = "Cannot open potential file " + filename + ".";
    PRINT_INPUT_ERROR(error.c_str());
  }
  const std::vector<std::string> header = ::get_tokens(file);
  file.clear();
  file.seekg(0);
  if (!header.empty() && (header[0] == "nep5" || header[0] == "nep5_zbl")) {
    translate_nep5();
    return;
  }
  new_format = !header.empty() && is_new_format(header[0]);
  if (!new_format) {
    return;
  }

  NEP_Model_File model;
  std::string error;
  if (!model.read(file, error)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  std::string input;
  if (!model.get_md_input(input, error)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  converted.str(input);
  reference.swap(model.energy_reference);
}

std::vector<std::string> NEP_Model_Input::get_tokens()
{
  std::string line;
  std::getline(
    new_format || translated_nep5 ? static_cast<std::istream&>(converted) : file, line);
  return ::get_tokens(line);
}

const std::vector<double>& NEP_Model_Input::energy_reference() const { return reference; }

std::vector<float> NEP_Model_Input::get_descriptor_parameters_type_pair(
  const std::vector<float>& parameters,
  const int num_para_ann,
  const int num_types,
  const int n_max_radial,
  const int n_max_angular,
  const int basis_size_radial,
  const int basis_size_angular) const
{
  const int num_types_sq = num_types * num_types;
  const int num_radial_basis = (n_max_radial + 1) * (basis_size_radial + 1);
  const int num_angular_basis = (n_max_angular + 1) * (basis_size_angular + 1);
  const int num_c_radial = num_types_sq * num_radial_basis;

  // New files already use the type-pair-major layout required by MD kernels.
  if (new_format) {
    return std::vector<float>(
      parameters.begin() + num_para_ann,
      parameters.begin() + num_para_ann + num_c_radial + num_types_sq * num_angular_basis);
  }

  // Legacy files use basis-major order, separately for radial and angular parameters.
  std::vector<float> descriptor_parameters(
    num_c_radial + num_types_sq * num_angular_basis);

  for (int type_pair = 0; type_pair < num_types_sq; ++type_pair) {
    for (int basis = 0; basis < num_radial_basis; ++basis) {
      descriptor_parameters[type_pair * num_radial_basis + basis] =
        parameters[num_para_ann + basis * num_types_sq + type_pair];
    }
    for (int basis = 0; basis < num_angular_basis; ++basis) {
      descriptor_parameters[num_c_radial + type_pair * num_angular_basis + basis] =
        parameters[num_para_ann + num_c_radial + basis * num_types_sq + type_pair];
    }
  }

  return descriptor_parameters;
}
