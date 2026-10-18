/*----------------------------------------------------------------------------80
SOAP parameter groups for the native NEP parameter layout.
------------------------------------------------------------------------------*/

#pragma once

#include "experimental_soap.cuh"
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpumd_soap
{

inline std::vector<Group> nep_parameter_groups(
  int types,
  int neurons,
  int dimension,
  int n_radial,
  int basis_radial,
  int n_angular,
  int basis_angular,
  std::size_t parameter_count,
  bool use_preconditioners = true)
{
  if (
    types < 1 || neurons < 1 || dimension < 1 || n_radial < 0 || basis_radial < 0 ||
    n_angular < 0 || basis_angular < 0) {
    throw std::invalid_argument("SOAP requires positive NEP dimensions and nonnegative orders.");
  }
  const auto mul = [](std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
      throw std::invalid_argument("SOAP NEP dimensions overflow size_t.");
    }
    return a * b;
  };
  const auto add = [](std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
      throw std::invalid_argument("SOAP NEP dimensions overflow size_t.");
    }
    return a + b;
  };
  const std::size_t t = types;
  const std::size_t h = neurons;
  const std::size_t d = dimension;
  const std::size_t nr = std::size_t(n_radial) + 1;
  const std::size_t br = std::size_t(basis_radial) + 1;
  const std::size_t na = std::size_t(n_angular) + 1;
  const std::size_t ba = std::size_t(basis_angular) + 1;
  const std::size_t ann_block = add(mul(add(d, 2), h), 1);
  const std::size_t ann_count = mul(t, ann_block);
  const std::size_t pairs = mul(t, t);
  const std::size_t radial_count = mul(mul(nr, br), pairs);
  const std::size_t angular_count = mul(mul(na, ba), pairs);
  if (add(add(ann_count, radial_count), angular_count) != parameter_count) {
    throw std::invalid_argument("SOAP NEP parameter count does not match native layout.");
  }

  std::vector<Group> groups;
  const auto contiguous = [&](
                            const std::string& name,
                            std::size_t start,
                            std::size_t rows,
                            std::size_t cols,
                            bool precondition) {
    Group group;
    group.name = name;
    group.rows = rows;
    group.cols = cols;
    group.precondition = precondition && use_preconditioners;
    group.indices.resize(mul(rows, cols));
    for (std::size_t i = 0; i < group.indices.size(); ++i) {
      group.indices[i] = start + i;
    }
    groups.push_back(std::move(group));
  };
  for (std::size_t type = 0; type < t; ++type) {
    const std::string prefix = "type_" + std::to_string(type);
    const std::size_t start = type * ann_block;
    contiguous(prefix + "_w0", start, h, d, true);
    contiguous(prefix + "_b0", start + h * d, h, 1, false);
    contiguous(prefix + "_w1", start + h * (d + 1), h, 1, false);
    contiguous(prefix + "_b1", start + h * (d + 2), 1, 1, false);
  }
  // Native descriptor coefficients interleave the ordered element pairs.
  const auto descriptor =
    [&](const std::string& name, std::size_t start, std::size_t rows, std::size_t cols) {
      for (std::size_t central = 0; central < t; ++central) {
        for (std::size_t neighbor = 0; neighbor < t; ++neighbor) {
          Group group;
          group.name = name + "_" + std::to_string(central) + "_" + std::to_string(neighbor);
          group.rows = rows;
          group.cols = cols;
          group.precondition = use_preconditioners;
          group.indices.resize(mul(rows, cols));
          for (std::size_t i = 0; i < group.indices.size(); ++i) {
            group.indices[i] = start + i * pairs + central * t + neighbor;
          }
          groups.push_back(std::move(group));
        }
      }
    };
  descriptor("radial", ann_count, nr, br);
  descriptor("angular", ann_count + radial_count, na, ba);
  return groups;
}

} // namespace gpumd_soap
