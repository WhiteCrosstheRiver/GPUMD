// SPDX-License-Identifier: GPL-3.0-or-later
#include "experimental_lmma.cuh"
#include <cstdio>
#include <iostream>
#include <numeric>
#include <sstream>

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function> static void require_throw(Function function, const char* message) {
  bool caught = false;
  try { function(); } catch (const std::exception&) { caught = true; }
  require(caught, message);
}

static std::vector<int> rank_sphere(const std::vector<float>& candidates, std::size_t n, int population) {
  std::vector<double> loss(population, 0.0);
  std::vector<int> order(population);
  std::iota(order.begin(), order.end(), 0);
  for (int i = 0; i < population; ++i)
    for (std::size_t k = 0; k < n; ++k) {
      const double x = candidates[static_cast<std::size_t>(i) * n + k];
      loss[i] += x * x;
    }
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return loss[a] < loss[b]; });
  return order;
}

static double squared_norm(const std::vector<double>& values) {
  double value = 0.0;
  for (double x : values) value += x * x;
  return value;
}

// Independent Python reference uses the same deterministic (not sampled) z.
static std::vector<double> deterministic_z(int generation, std::size_t n, int population) {
  std::vector<double> values(n * population);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = std::sin((generation + 1.0) * 0.7 + (i + 1.0) * 0.11)
                + std::cos((generation + 1.0) * 0.13 + (i + 1.0) * 0.017);
  return values;
}

static void trace(int dimension, int population, int steps) {
  std::vector<float> mean(dimension);
  for (int k = 0; k < dimension; ++k) mean[k] = static_cast<float>(0.2 * std::sin((k + 1.0) * 0.3));
  ExperimentalLMMA optimizer(mean, 0.1, population, 271828);
  optimizer.write_state_json(std::cout);
  std::cout << '\n';
  for (int generation = 0; generation < steps; ++generation) {
    const auto candidates = optimizer.ask_from_standard_normals(deterministic_z(generation, dimension, population));
    const auto order = rank_sphere(candidates, dimension, population);
    optimizer.tell(order.data());
    optimizer.write_state_json(std::cout);
    std::cout << '\n';
  }
}

static void tests() {
  require(ExperimentalLMMA::parse_seed("0") == 0, "Zero seed rejected");
  require(ExperimentalLMMA::parse_seed("18446744073709551615") ==
          std::numeric_limits<std::uint64_t>::max(), "Maximum seed rejected");
  const char* invalid_seeds[] = {nullptr, "", "-1", "+1", " 1", "1 ", "1x",
                                "18446744073709551616"};
  for (const char* text : invalid_seeds)
    require_throw([&] { ExperimentalLMMA::parse_seed(text); }, "Invalid seed accepted");
  float losses[] = {0.0f, 1.0f, -1.0f};
  require(ExperimentalLMMA::finite_fitness(losses, 3), "Finite fitness rejected");
  losses[0] = std::numeric_limits<float>::quiet_NaN();
  require(!ExperimentalLMMA::finite_fitness(losses, 3), "NaN fitness accepted");
  losses[0] = std::numeric_limits<float>::infinity();
  require(!ExperimentalLMMA::finite_fitness(losses, 3), "Infinite fitness accepted");
  require(!ExperimentalLMMA::finite_fitness(nullptr, 3), "Null fitness accepted");
  require(!ExperimentalLMMA::finite_fitness(losses, 0), "Empty fitness accepted");
  const int n = 128, population = 10;
  std::vector<float> mean(n, 1.0f);
  require_throw([&] { ExperimentalLMMA bad({}, 0.1, population, 1); }, "Empty mean accepted");
  require_throw([&] { ExperimentalLMMA bad(mean, 0.0, population, 1); }, "Zero sigma accepted");
  require_throw([&] { ExperimentalLMMA bad(mean, 0.1, 64, 1); }, "Invalid c_sigma accepted");
  std::vector<float> nonfinite(mean);
  nonfinite[0] = std::numeric_limits<float>::infinity();
  require_throw([&] { ExperimentalLMMA bad(nonfinite, 0.1, population, 1); }, "Nonfinite mean accepted");

  ExperimentalLMMA optimizer(mean, 0.1, population, 1234);
  require(optimizer.memory_size() == 18, "Incorrect default memory length");
  require_throw([&] { optimizer.tell(nullptr); }, "Tell before ask accepted");
  require_throw([&] { optimizer.ask_from_standard_normals({1.0}); }, "Wrong z size accepted");
  const auto normals = deterministic_z(0, n, population);
  const auto initial = optimizer.ask_from_standard_normals(normals);
  for (std::size_t i = 0; i < initial.size(); ++i)
    require(initial[i] == static_cast<float>(1.0 + 0.1 * normals[i]), "First generation is not isotropic");
  require_throw([&] { optimizer.ask(); }, "Double ask accepted");
  std::vector<int> invalid(population, 0);
  require_throw([&] { optimizer.tell(invalid.data()); }, "Duplicate ranking accepted");
  auto order = rank_sphere(initial, n, population);
  optimizer.tell(order.data());
  require(optimizer.generation() == 1, "Iteration not incremented");
  require(optimizer.sigma() > 0.0 && std::isfinite(optimizer.sigma()), "Invalid updated sigma");
  optimizer.write_checkpoint("test_lmma_checkpoint.tmp");
  std::ifstream checkpoint("test_lmma_checkpoint.tmp");
  std::stringstream checkpoint_contents;
  checkpoint_contents << checkpoint.rdbuf();
  checkpoint.close();
  require(checkpoint_contents.str().find("experimental_lmma_checkpoint 1") == 0, "Checkpoint magic missing");
  require(checkpoint_contents.str().find("rng_mt19937_64 ") != std::string::npos, "Checkpoint RNG missing");
  require(checkpoint_contents.str().find("normal_distribution ") != std::string::npos, "Checkpoint distribution missing");
  std::remove("test_lmma_checkpoint.tmp");

  ExperimentalLMMA a(mean, 0.1, population, 42), b(mean, 0.1, population, 42);
  for (int generation = 0; generation < 30; ++generation) {
    const auto candidates_a = a.ask(), candidates_b = b.ask();
    require(candidates_a == candidates_b, "Seeded reproducibility failed");
    order = rank_sphere(candidates_a, n, population);
    a.tell(order.data());
    b.tell(order.data());
    require(a.mean() == b.mean() && a.memory() == b.memory() && a.sigma() == b.sigma(),
            "Seeded state reproducibility failed");
  }
  ExperimentalLMMA sphere(mean, 0.1, population, 777);
  for (int generation = 0; generation < 1000; ++generation) {
    const auto candidates = sphere.ask();
    order = rank_sphere(candidates, n, population);
    sphere.tell(order.data());
  }
  require(squared_norm(sphere.mean()) < n * 0.01, "Sphere optimization failed to reduce loss 100x");
  std::cout << "PASS LMMA tests; sphere final squared norm=" << squared_norm(sphere.mean()) << '\n';
}

int main(int argc, char** argv) {
  try {
    if (argc > 1 && std::string(argv[1]) == "--trace") {
      trace(argc > 2 ? std::stoi(argv[2]) : 128,
            argc > 3 ? std::stoi(argv[3]) : 10,
            argc > 4 ? std::stoi(argv[4]) : 40);
    } else {
      tests();
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
