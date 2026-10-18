/*----------------------------------------------------------------------------80
Experimental host SOAP optimizer, independently implemented in C++/Eigen.
Ref: N. Vyas et al., SOAP, https://arxiv.org/abs/2409.11321.
Reference implementation (MIT): https://github.com/nikhilvyas/SOAP/blob/
a1e553530fde97d0e6b307d7c82ac6d38b072340/soap.py
This implementation uses double-precision linear algebra throughout, rather
than the reference's float32 eigensolver/QR casts. Tensor groups have one or
two axes; vectors use Adam and matrices use both Shampoo eigenbases.
No normalization, dimension merging, or checkpoint loading is implemented.
------------------------------------------------------------------------------*/

#ifndef GPUMD_EXPERIMENTAL_SOAP_CUH
#define GPUMD_EXPERIMENTAL_SOAP_CUH

// SOAP's Eigen operations are host-only, including when this header is read by NVCC.
#ifndef EIGEN_NO_CUDA
#define EIGEN_NO_CUDA
#endif
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpumd_soap
{

struct Group {
  std::string name;
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<std::size_t> indices;
  bool precondition = false;
};

struct Config {
  double learning_rate = 0.003;
  double beta1 = 0.95;
  double beta2 = 0.95;
  double shampoo_beta = 0.95;
  double eps = 1.0e-8;
  double weight_decay = 0.0;
  std::size_t precondition_frequency = 10;
  std::size_t max_precond_dim = 2048;
};

class SOAP
{
public:
  SOAP(std::size_t parameter_count, std::vector<Group> groups, Config config = Config())
    : parameter_count_(parameter_count), groups_(std::move(groups)), config_(config)
  {
    validate_config(config_);
    if (parameter_count_ == 0 || groups_.empty()) {
      throw std::invalid_argument("SOAP requires parameters and nonempty groups");
    }
    std::vector<unsigned char> coverage(parameter_count_, 0);
    for (const auto& group : groups_) {
      if (
        group.name.empty() || group.rows == 0 || group.cols == 0 ||
        group.rows > parameter_count_ / group.cols ||
        group.rows * group.cols != group.indices.size()) {
        throw std::invalid_argument("SOAP group name or shape is invalid");
      }
      if (
        group.rows > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()) ||
        group.cols > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max())) {
        throw std::invalid_argument("SOAP group shape exceeds Eigen index range");
      }
      if (
        group.precondition &&
        (group.rows > config_.max_precond_dim || group.cols > config_.max_precond_dim)) {
        throw std::invalid_argument("SOAP matrix exceeds configured preconditioner dimension");
      }
      for (const auto index : group.indices) {
        if (index >= parameter_count_ || coverage[index] != 0) {
          throw std::invalid_argument("SOAP indices must cover every parameter exactly once");
        }
        coverage[index] = 1;
      }
      State state;
      state.m = Matrix::Zero(group.rows, group.cols);
      state.v = Matrix::Zero(group.rows, group.cols);
      if (group.precondition) {
        state.gg_left = Matrix::Zero(group.rows, group.rows);
        state.gg_right = Matrix::Zero(group.cols, group.cols);
      }
      states_.push_back(std::move(state));
    }
    if (std::find(coverage.begin(), coverage.end(), 0) != coverage.end()) {
      throw std::invalid_argument("SOAP indices omit parameters");
    }
  }

  // Returns false only for the mandatory first-call preconditioner initialization.
  bool step(std::vector<double>& parameters, const std::vector<double>& gradient)
  {
    validate_vector(parameters, "parameters");
    validate_vector(gradient, "gradient");
    if (calls_ == std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error("SOAP step counter overflow");
    }
    if (calls_ == 0) {
      for (std::size_t k = 0; k < groups_.size(); ++k) {
        if (groups_[k].precondition) {
          update_covariances(gather(gradient, groups_[k]), states_[k]);
          states_[k].q_left = eigenbasis(states_[k].gg_left);
          states_[k].q_right = eigenbasis(states_[k].gg_right);
        }
      }
      ++calls_;
      return false;
    }

    const auto next_update = updates_ + 1;
    const double correction = std::sqrt(1.0 - std::pow(config_.beta2, next_update)) /
                              (1.0 - std::pow(config_.beta1, next_update));
    const double step_size = config_.learning_rate * correction;
    const double decay = 1.0 - config_.learning_rate * config_.weight_decay;
    for (std::size_t k = 0; k < groups_.size(); ++k) {
      const auto& group = groups_[k];
      auto& state = states_[k];
      const Matrix g = gather(gradient, group);
      const Matrix projected = project(g, state, group.precondition);
      state.m = config_.beta1 * state.m + (1.0 - config_.beta1) * projected;
      state.v =
        config_.beta2 * state.v + (1.0 - config_.beta2) * projected.array().square().matrix();
      const Matrix direction = (state.m.array() / (state.v.array().sqrt() + config_.eps)).matrix();
      const Matrix physical_direction = project_back(direction, state, group.precondition);
      Matrix values = (gather(parameters, group) - step_size * physical_direction) * decay;
      require_finite(values, "parameter update");
      scatter(values, group, parameters);

      if (group.precondition) {
        // Reference SOAP transports m through physical space on every update,
        // while v is only permuted (not rotated) during a basis refresh.
        const Matrix physical_m = project_back(state.m, state, true);
        update_covariances(g, state);
        if (next_update % config_.precondition_frequency == 0) {
          refresh_basis(state.gg_left, state.q_left, state.v, true);
          refresh_basis(state.gg_right, state.q_right, state.v, false);
        }
        state.m = project(physical_m, state, true);
      }
      require_finite(state.m, "first moment");
      require_finite(state.v, "second moment");
    }
    ++updates_;
    ++calls_;
    return true;
  }

  void set_learning_rate(double learning_rate)
  {
    Config next = config_;
    next.learning_rate = learning_rate;
    validate_config(next);
    config_ = next;
  }

  std::uint64_t calls() const
  {
    return calls_;
  }

  std::uint64_t updates() const
  {
    return updates_;
  }

  const Config& config() const
  {
    return config_;
  }

  const std::vector<Group>& groups() const
  {
    return groups_;
  }

  std::size_t parameter_count() const
  {
    return parameter_count_;
  }

  // Diagnostic serialization is complete but is not a resumable checkpoint API.
  void write_trace_json(std::ostream& out, const std::vector<double>& parameters) const
  {
    validate_vector(parameters, "parameters");
    const auto old_precision = out.precision();
    out << std::setprecision(17);
    out << "{\"calls\":" << calls_ << ",\"updates\":" << updates_ << ",\"params\":";
    write_vector(out, parameters);
    out << ",\"config\":{\"learning_rate\":" << config_.learning_rate
        << ",\"beta1\":" << config_.beta1 << ",\"beta2\":" << config_.beta2
        << ",\"shampoo_beta\":" << config_.shampoo_beta << ",\"eps\":" << config_.eps
        << ",\"weight_decay\":" << config_.weight_decay
        << ",\"precondition_frequency\":" << config_.precondition_frequency
        << ",\"max_precond_dim\":" << config_.max_precond_dim << "},\"groups\":[";
    for (std::size_t k = 0; k < groups_.size(); ++k) {
      if (k != 0) {
        out << ',';
      }
      const auto& group = groups_[k];
      const auto& state = states_[k];
      out << "{\"name\":";
      write_string(out, group.name);
      out << ",\"rows\":" << group.rows << ",\"cols\":" << group.cols
          << ",\"precondition\":" << (group.precondition ? "true" : "false") << ",\"indices\":";
      write_vector(out, group.indices);
      out << ",\"m\":";
      write_matrix(out, state.m);
      out << ",\"v\":";
      write_matrix(out, state.v);
      out << ",\"gg_left\":";
      write_matrix(out, state.gg_left);
      out << ",\"gg_right\":";
      write_matrix(out, state.gg_right);
      out << ",\"q_left\":";
      write_matrix(out, state.q_left);
      out << ",\"q_right\":";
      write_matrix(out, state.q_right);
      out << '}';
    }
    out << "]}";
    out.precision(old_precision);
  }

private:
  using Matrix = Eigen::MatrixXd;
  struct State {
    Matrix m;
    Matrix v;
    Matrix gg_left;
    Matrix gg_right;
    Matrix q_left;
    Matrix q_right;
  };

  static void validate_config(const Config& config)
  {
    const auto is_beta = [](double value) {
      return std::isfinite(value) && value >= 0.0 && value < 1.0;
    };
    if (
      !std::isfinite(config.learning_rate) || config.learning_rate < 0.0 ||
      !is_beta(config.beta1) || !is_beta(config.beta2) || !is_beta(config.shampoo_beta) ||
      !std::isfinite(config.eps) || config.eps <= 0.0 || !std::isfinite(config.weight_decay) ||
      config.weight_decay < 0.0 || config.precondition_frequency == 0 ||
      config.max_precond_dim == 0 || !std::isfinite(config.learning_rate * config.weight_decay) ||
      config.learning_rate * config.weight_decay >= 1.0) {
      throw std::invalid_argument("Invalid SOAP optimizer configuration");
    }
  }

  void validate_vector(const std::vector<double>& values, const char* label) const
  {
    if (values.size() != parameter_count_) {
      throw std::invalid_argument(std::string("SOAP ") + label + " size mismatch");
    }
    for (const auto value : values) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string("SOAP ") + label + " contains a nonfinite value");
      }
    }
  }

  static void require_finite(const Matrix& matrix, const char* label)
  {
    if (!matrix.allFinite()) {
      throw std::runtime_error(std::string("SOAP nonfinite ") + label);
    }
  }

  static Matrix gather(const std::vector<double>& values, const Group& group)
  {
    Matrix matrix(group.rows, group.cols);
    for (std::size_t row = 0; row < group.rows; ++row) {
      for (std::size_t col = 0; col < group.cols; ++col) {
        matrix(row, col) = values[group.indices[row * group.cols + col]];
      }
    }
    return matrix;
  }

  static void scatter(const Matrix& matrix, const Group& group, std::vector<double>& values)
  {
    for (std::size_t row = 0; row < group.rows; ++row) {
      for (std::size_t col = 0; col < group.cols; ++col) {
        values[group.indices[row * group.cols + col]] = matrix(row, col);
      }
    }
  }

  static Matrix project(const Matrix& matrix, const State& state, bool enabled)
  {
    if (!enabled) {
      return matrix;
    }
    return state.q_left.transpose() * matrix * state.q_right;
  }

  static Matrix project_back(const Matrix& matrix, const State& state, bool enabled)
  {
    if (!enabled) {
      return matrix;
    }
    return state.q_left * matrix * state.q_right.transpose();
  }

  void update_covariances(const Matrix& gradient, State& state) const
  {
    state.gg_left = config_.shampoo_beta * state.gg_left +
                    (1.0 - config_.shampoo_beta) * (gradient * gradient.transpose());
    state.gg_right = config_.shampoo_beta * state.gg_right +
                     (1.0 - config_.shampoo_beta) * (gradient.transpose() * gradient);
    require_finite(state.gg_left, "left gradient covariance");
    require_finite(state.gg_right, "right gradient covariance");
  }

  static Matrix eigenbasis(const Matrix& covariance)
  {
    Matrix regularized = covariance;
    regularized.diagonal().array() += 1.0e-30;
    Eigen::SelfAdjointEigenSolver<Matrix> solver(regularized);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("SOAP initial eigendecomposition failed");
    }
    Matrix basis(covariance.rows(), covariance.cols());
    for (Eigen::Index col = 0; col < covariance.cols(); ++col) {
      basis.col(col) = solver.eigenvectors().col(covariance.cols() - col - 1);
    }
    require_finite(basis, "initial eigenbasis");
    return basis;
  }

  static void refresh_basis(const Matrix& covariance, Matrix& basis, Matrix& v, bool left)
  {
    const Matrix transformed = basis.transpose() * covariance * basis;
    require_finite(transformed, "eigenvalue estimate");
    std::vector<Eigen::Index> order(static_cast<std::size_t>(basis.cols()));
    std::iota(order.begin(), order.end(), Eigen::Index(0));
    std::stable_sort(order.begin(), order.end(), [&](Eigen::Index a, Eigen::Index b) {
      return transformed(a, a) > transformed(b, b);
    });
    Matrix sorted_basis = basis;
    Matrix sorted_v = v;
    for (Eigen::Index col = 0; col < basis.cols(); ++col) {
      sorted_basis.col(col) = basis.col(order[col]);
      if (left) {
        sorted_v.row(col) = v.row(order[col]);
      } else {
        sorted_v.col(col) = v.col(order[col]);
      }
    }
    const Matrix power_iteration = covariance * sorted_basis;
    require_finite(power_iteration, "basis power iteration");
    Eigen::HouseholderQR<Matrix> qr(power_iteration);
    basis = qr.householderQ() * Matrix::Identity(covariance.rows(), covariance.cols());
    require_finite(basis, "refreshed eigenbasis");
    v = std::move(sorted_v);
  }

  template <typename T>
  static void write_vector(std::ostream& out, const std::vector<T>& values)
  {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i != 0) {
        out << ',';
      }
      out << values[i];
    }
    out << ']';
  }

  static void write_matrix(std::ostream& out, const Matrix& matrix)
  {
    out << '[';
    bool first = true;
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
      for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
        if (!first) {
          out << ',';
        }
        first = false;
        out << matrix(row, col);
      }
    }
    out << ']';
  }

  static void write_string(std::ostream& out, const std::string& value)
  {
    const char* hex = "0123456789abcdef";
    out << '"';
    for (const unsigned char c : value) {
      if (c == '"' || c == '\\') {
        out << '\\' << c;
      } else if (c < 0x20) {
        out << "\\u00" << hex[c >> 4] << hex[c & 15];
      } else {
        out << c;
      }
    }
    out << '"';
  }

  std::size_t parameter_count_;
  std::vector<Group> groups_;
  Config config_;
  std::vector<State> states_;
  std::uint64_t calls_ = 0;
  std::uint64_t updates_ = 0;
};

} // namespace gpumd_soap

#endif
