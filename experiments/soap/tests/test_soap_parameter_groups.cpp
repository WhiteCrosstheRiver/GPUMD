#include "soap_parameter_groups.cuh"
#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check_layout(int types, int neurons, int dim, int nr, int br, int na, int ba)
{
  const std::size_t ann_block = (std::size_t(dim) + 2) * neurons + 1;
  const std::size_t ann = ann_block * types;
  const std::size_t pairs = std::size_t(types) * types;
  const std::size_t radial = (std::size_t(nr) + 1) * (br + 1) * pairs;
  const std::size_t count = ann + radial + (std::size_t(na) + 1) * (ba + 1) * pairs;
  const auto groups = gpumd_soap::nep_parameter_groups(types, neurons, dim, nr, br, na, ba, count);
  assert(groups.size() == std::size_t(4 * types) + 2 * pairs);
  std::vector<int> coverage(count);
  for (const auto& group : groups) {
    assert(group.indices.size() == group.rows * group.cols);
    for (auto index : group.indices) {
      assert(index < count);
      ++coverage[index];
    }
  }
  assert(std::all_of(coverage.begin(), coverage.end(), [](int n) { return n == 1; }));
  for (int type = 0; type < types; ++type) {
    const auto& w = groups[4 * type];
    assert(w.rows == std::size_t(neurons) && w.cols == std::size_t(dim) && w.precondition);
    for (int h = 0; h < neurons; ++h)
      for (int q = 0; q < dim; ++q)
        assert(w.indices[h * dim + q] == type * ann_block + h * dim + q);
    const auto& b0 = groups[4 * type + 1];
    const auto& w1 = groups[4 * type + 2];
    const auto& b1 = groups[4 * type + 3];
    assert(!b0.precondition && !w1.precondition && !b1.precondition);
    assert(b0.indices.front() == type * ann_block + neurons * dim);
    assert(w1.indices.front() == type * ann_block + neurons * (dim + 1));
    assert(b1.indices.front() == type * ann_block + neurons * (dim + 2));
  }
  for (int central = 0; central < types; ++central) {
    for (int neighbor = 0; neighbor < types; ++neighbor) {
      const std::size_t pair = central * types + neighbor;
      const auto& r = groups[4 * types + pair];
      const auto& a = groups[4 * types + pairs + pair];
      assert(r.precondition && a.precondition);
      for (int n = 0; n <= nr; ++n)
        for (int k = 0; k <= br; ++k)
          assert(r.indices[n * (br + 1) + k] == ann + (n * (br + 1) + k) * pairs + pair);
      for (int n = 0; n <= na; ++n)
        for (int k = 0; k <= ba; ++k)
          assert(a.indices[n * (ba + 1) + k] == ann + radial + (n * (ba + 1) + k) * pairs + pair);
    }
  }
  const auto control =
    gpumd_soap::nep_parameter_groups(types, neurons, dim, nr, br, na, ba, count, false);
  assert(control.size() == groups.size());
  for (std::size_t i = 0; i < control.size(); ++i) {
    assert(!control[i].precondition);
    assert(control[i].indices == groups[i].indices);
  }
}

int main()
{
  check_layout(1, 1, 2, 0, 0, 0, 0);
  check_layout(2, 40, 42, 5, 8, 5, 8);
  check_layout(3, 7, 23, 2, 4, 3, 6);
  check_layout(4, 200, 80, 9, 19, 9, 19);
  check_layout(16, 3, 5, 1, 2, 0, 1);
  int rejected = 0;
  try {
    gpumd_soap::nep_parameter_groups(0, 1, 2, 0, 0, 0, 0, 7);
  } catch (const std::invalid_argument&) {
    ++rejected;
  }
  try {
    gpumd_soap::nep_parameter_groups(1, 1, 2, -1, 0, 0, 0, 7);
  } catch (const std::invalid_argument&) {
    ++rejected;
  }
  try {
    gpumd_soap::nep_parameter_groups(1, 1, 2, 0, 0, 0, 0, 8);
  } catch (const std::invalid_argument&) {
    ++rejected;
  }
  try {
    gpumd_soap::nep_parameter_groups(
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, 1);
  } catch (const std::invalid_argument&) {
    ++rejected;
  }
  assert(rejected == 4);
  std::cout << "SOAP NEP group layout: 5 valid cases and 4 invalid cases passed.\n";
}
