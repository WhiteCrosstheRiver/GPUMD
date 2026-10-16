// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef EXPERIMENTAL_LMMA_CUH
#define EXPERIMENTAL_LMMA_CUH

// Independent implementation of Algorithm 1 (not the appendix variant) in
// Loshchilov et al., "Limited-Memory Matrix Adaptation for Large Scale Black-box
// Optimization", arXiv:1705.06693. The population size is caller-selected;
// the memory length retains the paper's dimension-dependent default.
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

class ExperimentalLMMA {
 public:
  static std::uint64_t parse_seed(const char* text) {
    if (!text || !*text)
      throw std::invalid_argument("LM-MA-ES requires GPUMD_EXPERIMENT_SEED.");
    for (const char* digit = text; *digit; ++digit)
      if (*digit < '0' || *digit > '9')
        throw std::invalid_argument("LM-MA-ES seed must contain decimal digits only.");
    errno = 0;
    char* end = nullptr;
    const unsigned long long seed = std::strtoull(text, &end, 10);
    if (errno == ERANGE || *end || seed > std::numeric_limits<std::uint64_t>::max())
      throw std::invalid_argument("LM-MA-ES seed is outside the uint64 range.");
    return static_cast<std::uint64_t>(seed);
  }

  static bool finite_fitness(const float* values, std::size_t count) {
    if (!values || count == 0) return false;
    for (std::size_t i = 0; i < count; ++i)
      if (!std::isfinite(values[i])) return false;
    return true;
  }

  ExperimentalLMMA(const std::vector<float>& initial_mean, double initial_sigma,
                   int population, std::uint64_t seed)
      : n_(initial_mean.size()), lambda_(population), sigma_(initial_sigma), rng_(seed) {
    if (n_ == 0 || population < 2 || !std::isfinite(sigma_) || sigma_ <= 0.0)
      throw std::invalid_argument("LM-MA-ES requires nonempty mean, lambda >= 2, positive finite sigma");
    if (static_cast<double>(population) * 2.0 >= static_cast<double>(n_))
      throw std::invalid_argument("LM-MA-ES Algorithm 1 requires 2 * lambda < dimension");
    if (n_ > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(lambda_))
      throw std::invalid_argument("LM-MA-ES population allocation overflows");
    for (float value : initial_mean)
      if (!std::isfinite(value)) throw std::invalid_argument("LM-MA-ES initial mean is non-finite");
    mean_.assign(initial_mean.begin(), initial_mean.end());
    mu_ = lambda_ / 2;
    memory_count_ = 4 + static_cast<int>(std::floor(3.0 * std::log(static_cast<double>(n_))));
    weights_.resize(mu_);
    double total = 0.0;
    for (int i = 0; i < mu_; ++i) {
      weights_[i] = std::log(mu_ + 0.5) - std::log(i + 1.0);
      total += weights_[i];
    }
    double squared_weights = 0.0;
    for (double& weight : weights_) {
      weight /= total;
      squared_weights += weight * weight;
    }
    mu_eff_ = 1.0 / squared_weights;
    c_sigma_ = 2.0 * lambda_ / static_cast<double>(n_);
    path_.assign(n_, 0.0);
    memory_.assign(static_cast<std::size_t>(memory_count_) * n_, 0.0);
    c_d_.resize(memory_count_);
    c_c_.resize(memory_count_);
    for (int j = 0; j < memory_count_; ++j) {
      c_d_[j] = 1.0 / (std::pow(1.5, j) * static_cast<double>(n_));
      c_c_[j] = lambda_ / (std::pow(4.0, j) * static_cast<double>(n_));
    }
  }

  std::vector<float> ask() {
    ensure_ready_to_ask();
    std::vector<double> normals(static_cast<std::size_t>(lambda_) * n_);
    for (double& value : normals) value = normal_(rng_);
    return ask_from_standard_normals(normals);
  }

  // Supplying z explicitly permits independent, deterministic recurrence tests.
  std::vector<float> ask_from_standard_normals(const std::vector<double>& normals) {
    ensure_ready_to_ask();
    if (normals.size() != static_cast<std::size_t>(lambda_) * n_)
      throw std::invalid_argument("LM-MA-ES standard-normal vector has wrong size");
    for (double value : normals)
      if (!std::isfinite(value)) throw std::invalid_argument("LM-MA-ES z is non-finite");
    z_ = normals;
    directions_ = normals;
    std::vector<float> candidates(normals.size());
    const int active_memory = static_cast<int>(std::min<std::uint64_t>(iteration_, memory_count_));
    for (int candidate = 0; candidate < lambda_; ++candidate) {
      double* direction = directions_.data() + static_cast<std::size_t>(candidate) * n_;
      for (int j = 0; j < active_memory; ++j) {
        const double* memory = memory_.data() + static_cast<std::size_t>(j) * n_;
        double dot = 0.0;
        for (std::size_t k = 0; k < n_; ++k) dot += memory[k] * direction[k];
        // The dot product uses the progressively transformed direction.
        for (std::size_t k = 0; k < n_; ++k)
          direction[k] = (1.0 - c_d_[j]) * direction[k] + c_d_[j] * memory[k] * dot;
      }
      for (std::size_t k = 0; k < n_; ++k) {
        const double value = mean_[k] + sigma_ * direction[k];
        if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
          throw std::runtime_error("LM-MA-ES candidate exceeds finite float range");
        candidates[static_cast<std::size_t>(candidate) * n_ + k] = static_cast<float>(value);
      }
    }
    awaiting_tell_ = true;
    return candidates;
  }

  // order contains the complete candidate permutation, ascending by global loss.
  void tell(const int* order) {
    if (!awaiting_tell_) throw std::logic_error("LM-MA-ES tell requires an outstanding ask");
    if (!order) throw std::invalid_argument("LM-MA-ES ranking pointer is null");
    std::vector<bool> seen(lambda_, false);
    for (int i = 0; i < lambda_; ++i) {
      if (order[i] < 0 || order[i] >= lambda_ || seen[order[i]])
        throw std::invalid_argument("LM-MA-ES ranking is not a permutation");
      seen[order[i]] = true;
    }
    std::vector<double> weighted_z(n_, 0.0), weighted_d(n_, 0.0);
    for (int i = 0; i < mu_; ++i) {
      const std::size_t offset = static_cast<std::size_t>(order[i]) * n_;
      for (std::size_t k = 0; k < n_; ++k) {
        weighted_z[k] += weights_[i] * z_[offset + k];
        weighted_d[k] += weights_[i] * directions_[offset + k];
      }
    }
    const double path_scale = std::sqrt(mu_eff_ * c_sigma_ * (2.0 - c_sigma_));
    double path_squared = 0.0;
    for (std::size_t k = 0; k < n_; ++k) {
      mean_[k] += sigma_ * weighted_d[k];
      path_[k] = (1.0 - c_sigma_) * path_[k] + path_scale * weighted_z[k];
      if (!std::isfinite(mean_[k]) || !std::isfinite(path_[k]))
        throw std::runtime_error("LM-MA-ES mean or path became non-finite");
      path_squared += path_[k] * path_[k];
    }
    // All paths update from the original z, including currently unused paths.
    for (int j = 0; j < memory_count_; ++j) {
      double* memory = memory_.data() + static_cast<std::size_t>(j) * n_;
      const double memory_scale = std::sqrt(mu_eff_ * c_c_[j] * (2.0 - c_c_[j]));
      for (std::size_t k = 0; k < n_; ++k) {
        memory[k] = (1.0 - c_c_[j]) * memory[k] + memory_scale * weighted_z[k];
        if (!std::isfinite(memory[k])) throw std::runtime_error("LM-MA-ES memory became non-finite");
      }
    }
    sigma_ *= std::exp(c_sigma_ / 2.0 * (path_squared / static_cast<double>(n_) - 1.0));
    if (!std::isfinite(sigma_) || sigma_ <= 0.0)
      throw std::runtime_error("LM-MA-ES sigma became non-finite or underflowed");
    ++iteration_;
    awaiting_tell_ = false;
  }

  void write_checkpoint(const char* filename) const {
    if (awaiting_tell_) throw std::logic_error("LM-MA-ES checkpoint must follow tell");
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot open LM-MA-ES checkpoint");
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "experimental_lmma_checkpoint 1\n";
    out << "dimension " << n_ << "\npopulation " << lambda_ << "\nparents " << mu_
        << "\nmemory_count " << memory_count_ << "\niteration " << iteration_
        << "\nsigma " << sigma_ << "\nmu_eff " << mu_eff_ << "\nc_sigma " << c_sigma_ << '\n';
    write_text_vector(out, "weights", weights_);
    write_text_vector(out, "c_d", c_d_);
    write_text_vector(out, "c_c", c_c_);
    write_text_vector(out, "mean", mean_);
    write_text_vector(out, "path", path_);
    write_text_vector(out, "memory_row_major", memory_);
    out << "rng_mt19937_64 " << rng_ << '\n';
    out << "normal_distribution " << normal_ << '\n';
    out.flush();
    if (!out) throw std::runtime_error("Failed writing LM-MA-ES checkpoint");
  }

  void write_state_json(std::ostream& out) const {
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\"dimension\":" << n_ << ",\"population\":" << lambda_ << ",\"parents\":" << mu_
        << ",\"memory_count\":" << memory_count_ << ",\"iteration\":" << iteration_
        << ",\"sigma\":" << sigma_ << ",\"mu_eff\":" << mu_eff_ << ",\"c_sigma\":" << c_sigma_;
    write_json_vector(out, "weights", weights_);
    write_json_vector(out, "c_d", c_d_);
    write_json_vector(out, "c_c", c_c_);
    write_json_vector(out, "mean", mean_);
    write_json_vector(out, "path", path_);
    write_json_vector(out, "memory", memory_);
    out << '}';
  }

  void write_state_json(const char* filename) const {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot open LM-MA-ES JSON state");
    write_state_json(out);
    out << '\n';
    out.flush();
    if (!out) throw std::runtime_error("Failed writing LM-MA-ES JSON state");
  }

  std::size_t dimension() const { return n_; }
  int population_size() const { return lambda_; }
  int memory_size() const { return memory_count_; }
  std::uint64_t generation() const { return iteration_; }
  double sigma() const { return sigma_; }
  const std::vector<double>& mean() const { return mean_; }
  const std::vector<double>& path() const { return path_; }
  const std::vector<double>& memory() const { return memory_; }

 private:
  void ensure_ready_to_ask() const {
    if (awaiting_tell_) throw std::logic_error("LM-MA-ES ask requires completion of prior tell");
  }
  static void write_text_vector(std::ostream& out, const char* name, const std::vector<double>& values) {
    out << name << ' ' << values.size();
    for (double value : values) out << ' ' << value;
    out << '\n';
  }
  static void write_json_vector(std::ostream& out, const char* name, const std::vector<double>& values) {
    out << ",\"" << name << "\":[";
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i) out << ',';
      out << values[i];
    }
    out << ']';
  }
  std::size_t n_;
  int lambda_, mu_, memory_count_;
  std::uint64_t iteration_ = 0;
  double sigma_, mu_eff_, c_sigma_;
  bool awaiting_tell_ = false;
  std::vector<double> mean_, path_, memory_, weights_, c_d_, c_c_, z_, directions_;
  std::mt19937_64 rng_;
  std::normal_distribution<double> normal_{0.0, 1.0};
};

#endif
