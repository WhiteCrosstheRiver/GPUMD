#include "experimental_soap.cuh"
#include <cstdlib>
#include <iostream>
#include <sstream>

namespace
{
using gpumd_soap::Config;
using gpumd_soap::Group;
using gpumd_soap::SOAP;

void require(bool value, const char* message)
{
  if (!value)
    throw std::runtime_error(message);
}

template <typename Function>
void rejects(Function function, const char* message)
{
  bool rejected = false;
  try {
    function();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, message);
}

Group group(std::string name, std::size_t rows, std::size_t cols, bool precondition)
{
  Group result;
  result.name = std::move(name);
  result.rows = rows;
  result.cols = cols;
  result.precondition = precondition;
  result.indices.resize(rows * cols);
  std::iota(result.indices.begin(), result.indices.end(), std::size_t(0));
  return result;
}

void unit_tests()
{
  Config config;
  const auto vector_group = group("vector", 3, 1, false);
  rejects([&] { SOAP invalid(0, {}); }, "empty shape accepted");
  rejects(
    [&] {
      Config c;
      c.beta1 = 1.0;
      SOAP invalid(3, {vector_group}, c);
    },
    "bad beta accepted");
  rejects(
    [&] {
      Config c;
      c.beta2 = -0.1;
      SOAP invalid(3, {vector_group}, c);
    },
    "negative beta accepted");
  rejects(
    [&] {
      Config c;
      c.learning_rate = std::nan("");
      SOAP invalid(3, {vector_group}, c);
    },
    "NaN lr accepted");
  rejects(
    [&] {
      Config c;
      c.eps = 0.0;
      SOAP invalid(3, {vector_group}, c);
    },
    "bad eps accepted");
  rejects(
    [&] {
      Config c;
      c.precondition_frequency = 0;
      SOAP invalid(3, {vector_group}, c);
    },
    "bad frequency accepted");
  rejects(
    [&] {
      auto g = vector_group;
      g.indices[1] = 0;
      SOAP invalid(3, {g});
    },
    "duplicate index accepted");
  rejects(
    [&] {
      auto g = vector_group;
      g.indices[1] = 9;
      SOAP invalid(3, {g});
    },
    "out-of-range index accepted");
  rejects([&] { SOAP invalid(4, {vector_group}); }, "missing index accepted");
  rejects(
    [&] {
      auto g = vector_group;
      g.cols = 2;
      SOAP invalid(3, {g});
    },
    "bad group shape accepted");
  rejects(
    [&] {
      Config c;
      c.max_precond_dim = 2;
      SOAP invalid(9, {group("large", 3, 3, true)}, c);
    },
    "dimension guard missing");

  SOAP vector_optimizer(3, {vector_group}, config);
  std::vector<double> parameters{1.0, -2.0, 3.0};
  const auto original = parameters;
  std::vector<double> gradient{0.4, -0.3, 1.2};
  require(!vector_optimizer.step(parameters, gradient), "initial step must skip parameter update");
  require(
    parameters == original && vector_optimizer.updates() == 0, "initialization changed parameters");
  std::vector<double> expected = original, m(3, 0.0), v(3, 0.0);
  for (int step = 1; step <= 50; ++step) {
    for (std::size_t j = 0; j < 3; ++j) {
      gradient[j] = std::sin(0.37 * step + 0.71 * j) + 0.03 * step;
      m[j] = config.beta1 * m[j] + (1.0 - config.beta1) * gradient[j];
      v[j] = config.beta2 * v[j] + (1.0 - config.beta2) * gradient[j] * gradient[j];
      const double lr = config.learning_rate * std::sqrt(1.0 - std::pow(config.beta2, step)) /
                        (1.0 - std::pow(config.beta1, step));
      expected[j] -= lr * m[j] / (std::sqrt(v[j]) + config.eps);
    }
    require(vector_optimizer.step(parameters, gradient), "normal vector step was skipped");
    for (std::size_t j = 0; j < 3; ++j) {
      require(
        std::abs(parameters[j] - expected[j]) < 1.0e-14, "vector SOAP differs from reference Adam");
    }
  }
  require(vector_optimizer.calls() == 51 && vector_optimizer.updates() == 50, "counter mismatch");
  const auto saved = parameters;
  rejects([&] { vector_optimizer.step(parameters, {1.0}); }, "bad gradient length accepted");
  rejects(
    [&] {
      auto bad = gradient;
      bad[1] = std::numeric_limits<double>::infinity();
      vector_optimizer.step(parameters, bad);
    },
    "nonfinite gradient accepted");
  require(parameters == saved, "invalid input mutated parameters");
  rejects([&] { vector_optimizer.set_learning_rate(-0.1); }, "negative lr accepted");
  vector_optimizer.set_learning_rate(0.0);
  vector_optimizer.step(parameters, gradient);
  require(parameters == saved, "zero learning rate changed parameters");

  Config quadratic_config;
  quadratic_config.learning_rate = 0.02;
  SOAP quadratic(9, {group("weights", 3, 3, true)}, quadratic_config);
  std::vector<double> point{1.0, -0.5, 0.2, 0.4, 0.8, -1.2, -0.6, 0.3, 0.9};
  const auto loss = [](const std::vector<double>& x) {
    double result = 0.0;
    for (std::size_t j = 0; j < x.size(); ++j)
      result += (1.0 + j * 0.25) * x[j] * x[j];
    return result;
  };
  const double initial_loss = loss(point);
  gradient.resize(point.size());
  for (int step = 0; step < 500; ++step) {
    for (std::size_t j = 0; j < point.size(); ++j)
      gradient[j] = 2.0 * (1.0 + j * 0.25) * point[j];
    quadratic.step(point, gradient);
  }
  require(loss(point) < initial_loss * 1.0e-3, "matrix quadratic did not decrease sufficiently");

  SOAP rank_deficient(12, {group("rectangular", 4, 3, true)}, config);
  point.assign(12, 1.0);
  gradient.assign(12, 0.0);
  rank_deficient.step(point, gradient);
  for (int step = 0; step < 35; ++step) {
    for (std::size_t j = 0; j < gradient.size(); ++j)
      gradient[j] = (j % 3 + 1.0) * std::sin(step + 0.3);
    rank_deficient.step(point, gradient);
  }
  for (const auto value : point)
    require(std::isfinite(value), "rank-deficient updates became nonfinite");

  SOAP deterministic_a(4, {group("square", 2, 2, true)}, config);
  SOAP deterministic_b(4, {group("square", 2, 2, true)}, config);
  std::vector<double> a{0.1, -0.2, 0.3, -0.4}, b = a;
  gradient.resize(4);
  for (int step = 0; step < 35; ++step) {
    for (std::size_t j = 0; j < gradient.size(); ++j)
      gradient[j] = std::sin(step * 0.3 + j) + 0.5 * std::cos(step + 0.2 * j);
    deterministic_a.step(a, gradient);
    deterministic_b.step(b, gradient);
  }
  std::ostringstream trace_a, trace_b;
  deterministic_a.write_trace_json(trace_a, a);
  deterministic_b.write_trace_json(trace_b, b);
  require(trace_a.str() == trace_b.str(), "determinism test failed");

  auto strided = group("strided", 2, 2, true);
  strided.indices = {0, 2, 4, 6};
  auto remainder = group("remainder", 3, 1, false);
  remainder.indices = {1, 3, 5};
  SOAP mixed(7, {strided, remainder}, config);
  point.assign(7, 1.0);
  gradient.assign(7, 1.0);
  mixed.step(point, gradient);
  mixed.step(point, gradient);
  require(mixed.groups().size() == 2, "mixed groups were lost");

  std::cout << "SOAP core tests passed: validation, Adam equivalence, quadratic, rank-deficient "
               "bases, deterministic refresh, strided groups\n";
}

void trace()
{
  std::size_t n, ngroups, nsteps;
  Config config;
  if (!(std::cin >> n >> ngroups >> nsteps >> config.learning_rate >> config.beta1 >>
        config.beta2 >> config.shampoo_beta >> config.eps >> config.weight_decay >>
        config.precondition_frequency)) {
    throw std::runtime_error("trace input header is invalid");
  }
  if (n == 0 || n > 1000000 || ngroups == 0 || ngroups > n || nsteps > 1000000) {
    throw std::runtime_error("trace dimensions outside test limits");
  }
  std::vector<Group> groups;
  for (std::size_t i = 0; i < ngroups; ++i) {
    Group g;
    int enabled;
    if (
      !(std::cin >> g.name >> g.rows >> g.cols >> enabled) || g.rows == 0 || g.cols == 0 ||
      g.rows > n / g.cols || (enabled != 0 && enabled != 1)) {
      throw std::runtime_error("trace group is invalid");
    }
    g.precondition = enabled == 1;
    g.indices.resize(g.rows * g.cols);
    for (auto& index : g.indices)
      if (!(std::cin >> index))
        throw std::runtime_error("missing index");
    groups.push_back(std::move(g));
  }
  std::vector<double> parameters(n), gradient(n);
  for (auto& value : parameters)
    if (!(std::cin >> value))
      throw std::runtime_error("missing parameter");
  SOAP optimizer(n, std::move(groups), config);
  optimizer.write_trace_json(std::cout, parameters);
  std::cout << '\n';
  for (std::size_t i = 0; i < nsteps; ++i) {
    for (auto& value : gradient)
      if (!(std::cin >> value))
        throw std::runtime_error("missing gradient");
    optimizer.step(parameters, gradient);
    optimizer.write_trace_json(std::cout, parameters);
    std::cout << '\n';
  }
  std::string unexpected;
  if (std::cin >> unexpected)
    throw std::runtime_error("unexpected trailing trace input");
}
} // namespace

int main(int argc, char** argv)
{
  try {
    if (argc == 2 && std::string(argv[1]) == "--trace")
      trace();
    else if (argc == 1)
      unit_tests();
    else
      throw std::runtime_error("usage: test_soap_core [--trace]");
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
