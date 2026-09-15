/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#include "deposit.cuh"
#include "atom_mutation.cuh"
#include "variable.cuh"
#include "model/group.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

extern double get_mass_from_symbol(const std::string& symbol);
extern void read_xyz_in_line_3(
  std::ifstream& input,
  const int N,
  const int has_velocity_in_xyz,
  const bool has_mass,
  const bool has_charge,
  const int num_columns,
  const int* property_offset,
  int& number_of_types,
  std::vector<std::string>& atom_symbols,
  std::vector<std::string>& cpu_atom_symbol,
  std::vector<int>& cpu_type,
  std::vector<double>& cpu_mass,
  std::vector<float>& cpu_charge,
  std::vector<double>& cpu_position_per_atom,
  std::vector<double>& cpu_velocity_per_atom,
  std::vector<Group>& group);

struct DepositConfig
{
  enum Style
  {
    Grid,
    Random,
    Gaussian,
    Point
  };
  enum Entity
  {
    Atom,
    Molecule
  };
  enum Direction
  {
    Axis,
    Vector,
    Target
  };
  enum Surface
  {
    Fixed,
    Global,
    Local
  };
  enum Velocity
  {
    Constant,
    Uniform
  };
  enum Select
  {
    Sequential,
    RandomSelect
  };

  Style style = Point;
  Entity entity = Atom;
  std::string source;

  bool number_all = false;
  bool has_number = false;
  int number = 1;

  bool has_direction = false;
  Direction direction_kind = Axis;
  double flight[3] = {0.0, 0.0, -1.0};
  double target[3] = {0.0, 0.0, 0.0};

  bool has_normal = false;
  double normal[3] = {0.0, 0.0, 1.0};

  bool has_basis = false;
  double basis[3] = {0.0, 0.0, 0.0};

  bool has_surface = false;
  Surface surface_kind = Fixed;
  double surface_H = 0.0;
  double surface_gap = 0.0;
  double surface_radius = 0.0;

  bool has_region = false;
  double region[4] = {0.0, 0.0, 0.0, 0.0};

  bool has_spacing = false;
  double spacing[2] = {0.0, 0.0};

  bool has_origin = false;
  double origin[2] = {0.0, 0.0};

  bool has_sigma = false;
  double sigma = 0.0;

  bool has_spread = false;
  double spread_sigma_deg = 0.0;

  bool has_velocity = false;
  Velocity velocity_kind = Constant;
  double speed = 0.0;
  double speed_min = 0.0;
  double speed_max = 0.0;

  bool has_near = false;
  double near_r = 0.0;

  bool has_select = false;
  Select select = RandomSelect;

  bool has_attempt = false;
  int attempt = 10;

  bool has_seed = false;
  unsigned seed = 0;
};

struct DepositFrame
{
  double u[3] = {1.0, 0.0, 0.0};
  double v[3] = {0.0, 1.0, 0.0};
  double n[3] = {0.0, 0.0, 1.0};
  double d[3] = {0.0, 0.0, -1.0};
  int cartesian_n = -1;
};

struct DepositUV
{
  double u = 0.0;
  double v = 0.0;
};

struct EntityTemplate
{
  int n_atoms = 0;
  std::vector<std::string> symbol;
  std::vector<int> type;
  std::vector<double> mass;
  std::vector<double> rel_pos;
  double radius = 0.0;
};

static int require_index(int i, int num_param, const char* what)
{
  if (i >= num_param) {
    PRINT_INPUT_ERROR(what);
  }
  return i;
}

static double require_real(
  const char** param,
  int num_param,
  int i,
  const Box& box,
  VariableScope* variables,
  const char* what)
{
  require_index(i, num_param, what);
  if (variables != nullptr) {
    return variables->resolve_real(param[i], box, what);
  }
  double value = 0.0;
  if (!is_valid_real(param[i], &value)) {
    PRINT_INPUT_ERROR(what);
  }
  return value;
}

static int require_int(const char** param, int num_param, int i, const char* what)
{
  require_index(i, num_param, what);
  int value = 0;
  if (!is_valid_int(param[i], &value)) {
    PRINT_INPUT_ERROR(what);
  }
  return value;
}

static bool parse_axis(const char* token, double d[3])
{
  int sign = 1;
  const char* p = token;
  if (p[0] == '+') {
    p += 1;
  } else if (p[0] == '-') {
    sign = -1;
    p += 1;
  }
  d[0] = d[1] = d[2] = 0.0;
  if (strcmp(p, "x") == 0) {
    d[0] = static_cast<double>(sign);
    return true;
  }
  if (strcmp(p, "y") == 0) {
    d[1] = static_cast<double>(sign);
    return true;
  }
  if (strcmp(p, "z") == 0) {
    d[2] = static_cast<double>(sign);
    return true;
  }
  return false;
}

static void normalize3(double v[3], const char* what)
{
  const double n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  if (n2 <= 1.0e-30) {
    PRINT_INPUT_ERROR(what);
  }
  const double inv = 1.0 / sqrt(n2);
  v[0] *= inv;
  v[1] *= inv;
  v[2] *= inv;
}

static bool deposit_needs_rng(const DepositConfig& cfg)
{
  if (cfg.style == DepositConfig::Random || cfg.style == DepositConfig::Gaussian) {
    return true;
  }
  if (cfg.style == DepositConfig::Grid && !cfg.number_all && cfg.select == DepositConfig::RandomSelect) {
    return true;
  }
  if (cfg.has_spread) {
    return true;
  }
  if (cfg.velocity_kind == DepositConfig::Uniform) {
    return true;
  }
  return false;
}

static void parse_deposit_config(
  const char** param,
  int num_param,
  const Box& box,
  VariableScope* variables,
  DepositConfig& cfg)
{
  if (num_param < 4) {
    PRINT_INPUT_ERROR("Usage: deposit <style> <entity> <source> keyword values ...");
  }

  if (strcmp(param[1], "grid") == 0) {
    cfg.style = DepositConfig::Grid;
  } else if (strcmp(param[1], "random") == 0) {
    cfg.style = DepositConfig::Random;
  } else if (strcmp(param[1], "gaussian") == 0) {
    cfg.style = DepositConfig::Gaussian;
  } else if (strcmp(param[1], "point") == 0) {
    cfg.style = DepositConfig::Point;
  } else {
    PRINT_INPUT_ERROR("deposit style must be grid, random, gaussian, or point.");
  }

  if (strcmp(param[2], "atom") == 0) {
    cfg.entity = DepositConfig::Atom;
  } else if (strcmp(param[2], "molecule") == 0) {
    cfg.entity = DepositConfig::Molecule;
  } else {
    PRINT_INPUT_ERROR("deposit entity must be atom or molecule.");
  }
  cfg.source = param[3];

  auto real_at = [&](int index, const char* what) {
    return require_real(param, num_param, index, box, variables, what);
  };

  int i = 4;
  while (i < num_param) {
    if (strcmp(param[i], "number") == 0) {
      require_index(i + 1, num_param, "Usage: number <N> or number all");
      if (strcmp(param[i + 1], "all") == 0) {
        cfg.number_all = true;
        cfg.has_number = true;
      } else {
        cfg.number = require_int(param, num_param, i + 1, "number N should be an integer.");
        if (cfg.number < 1) {
          PRINT_INPUT_ERROR("deposit number should be >= 1.");
        }
        cfg.has_number = true;
      }
      i += 2;
    } else if (strcmp(param[i], "direction") == 0) {
      require_index(i + 1, num_param, "Usage: direction axis <±x|±y|±z> or vector vx vy vz or target tx ty tz");
      if (strcmp(param[i + 1], "axis") == 0) {
        require_index(i + 2, num_param, "Usage: direction axis <±x|±y|±z>");
        if (!parse_axis(param[i + 2], cfg.flight)) {
          PRINT_INPUT_ERROR("direction axis must be x, y, z, -x, -y, or -z.");
        }
        cfg.direction_kind = DepositConfig::Axis;
        i += 3;
      } else if (strcmp(param[i + 1], "vector") == 0) {
        cfg.flight[0] = real_at(i + 2, "direction vector values should be numbers.");
        cfg.flight[1] = real_at(i + 3, "direction vector values should be numbers.");
        cfg.flight[2] = real_at(i + 4, "direction vector values should be numbers.");
        normalize3(cfg.flight, "direction vector must be non-zero.");
        cfg.direction_kind = DepositConfig::Vector;
        i += 5;
      } else if (strcmp(param[i + 1], "target") == 0) {
        cfg.target[0] = real_at(i + 2, "direction target values should be numbers.");
        cfg.target[1] = real_at(i + 3, "direction target values should be numbers.");
        cfg.target[2] = real_at(i + 4, "direction target values should be numbers.");
        cfg.direction_kind = DepositConfig::Target;
        i += 5;
      } else {
        PRINT_INPUT_ERROR("direction must be axis, vector, or target.");
      }
      cfg.has_direction = true;
    } else if (strcmp(param[i], "normal") == 0) {
      require_index(i + 1, num_param, "Usage: normal axis <±x|±y|±z> or vector nx ny nz");
      if (strcmp(param[i + 1], "axis") == 0) {
        require_index(i + 2, num_param, "Usage: normal axis <±x|±y|±z>");
        if (!parse_axis(param[i + 2], cfg.normal)) {
          PRINT_INPUT_ERROR("normal axis must be x, y, z, -x, -y, or -z.");
        }
        i += 3;
      } else if (strcmp(param[i + 1], "vector") == 0) {
        cfg.normal[0] = real_at(i + 2, "normal vector values should be numbers.");
        cfg.normal[1] = real_at(i + 3, "normal vector values should be numbers.");
        cfg.normal[2] = real_at(i + 4, "normal vector values should be numbers.");
        normalize3(cfg.normal, "normal vector must be non-zero.");
        i += 5;
      } else {
        PRINT_INPUT_ERROR("normal must be axis or vector.");
      }
      cfg.has_normal = true;
    } else if (strcmp(param[i], "basis") == 0) {
      cfg.basis[0] = real_at(i + 1, "basis values should be numbers.");
      cfg.basis[1] = real_at(i + 2, "basis values should be numbers.");
      cfg.basis[2] = real_at(i + 3, "basis values should be numbers.");
      cfg.has_basis = true;
      i += 4;
    } else if (strcmp(param[i], "surface") == 0) {
      require_index(i + 1, num_param, "Usage: surface fixed H, global gap D, or local radius R gap D");
      if (strcmp(param[i + 1], "fixed") == 0) {
        cfg.surface_H = real_at(i + 2, "surface fixed H should be a number.");
        cfg.surface_kind = DepositConfig::Fixed;
        i += 3;
      } else if (strcmp(param[i + 1], "global") == 0) {
        if (i + 2 >= num_param || strcmp(param[i + 2], "gap") != 0) {
          PRINT_INPUT_ERROR("Usage: surface global gap <D>");
        }
        cfg.surface_gap = real_at(i + 3, "surface gap should be a number.");
        if (cfg.surface_gap < 0.0) {
          PRINT_INPUT_ERROR("surface gap should be >= 0.");
        }
        cfg.surface_kind = DepositConfig::Global;
        i += 4;
      } else if (strcmp(param[i + 1], "local") == 0) {
        if (i + 4 >= num_param || strcmp(param[i + 2], "radius") != 0 ||
            strcmp(param[i + 4], "gap") != 0) {
          PRINT_INPUT_ERROR("Usage: surface local radius <R> gap <D>");
        }
        cfg.surface_radius = real_at(i + 3, "surface local radius should be a number.");
        cfg.surface_gap = real_at(i + 5, "surface gap should be a number.");
        if (cfg.surface_radius <= 0.0) {
          PRINT_INPUT_ERROR("surface local radius should be > 0.");
        }
        if (cfg.surface_gap < 0.0) {
          PRINT_INPUT_ERROR("surface gap should be >= 0.");
        }
        cfg.surface_kind = DepositConfig::Local;
        i += 6;
      } else {
        PRINT_INPUT_ERROR("surface must be fixed, global, or local.");
      }
      cfg.has_surface = true;
    } else if (strcmp(param[i], "region") == 0) {
      for (int k = 0; k < 4; ++k) {
        cfg.region[k] = real_at(i + 1 + k, "region values should be numbers.");
      }
      if (cfg.region[0] > cfg.region[1] || cfg.region[2] > cfg.region[3]) {
        PRINT_INPUT_ERROR("region min should be <= max.");
      }
      cfg.has_region = true;
      i += 5;
    } else if (strcmp(param[i], "spacing") == 0) {
      cfg.spacing[0] = real_at(i + 1, "spacing values should be numbers.");
      cfg.spacing[1] = real_at(i + 2, "spacing values should be numbers.");
      if (cfg.spacing[0] <= 0.0 || cfg.spacing[1] <= 0.0) {
        PRINT_INPUT_ERROR("spacing should be > 0.");
      }
      cfg.has_spacing = true;
      i += 3;
    } else if (strcmp(param[i], "origin") == 0) {
      cfg.origin[0] = real_at(i + 1, "origin values should be numbers.");
      cfg.origin[1] = real_at(i + 2, "origin values should be numbers.");
      cfg.has_origin = true;
      i += 3;
    } else if (strcmp(param[i], "sigma") == 0) {
      cfg.sigma = real_at(i + 1, "sigma should be a number.");
      if (cfg.sigma < 0.0) {
        PRINT_INPUT_ERROR("sigma should be >= 0.");
      }
      cfg.has_sigma = true;
      i += 2;
    } else if (strcmp(param[i], "spread") == 0) {
      require_index(i + 1, num_param, "Usage: spread gaussian <sigma_deg>");
      if (strcmp(param[i + 1], "gaussian") != 0) {
        PRINT_INPUT_ERROR("Usage: spread gaussian <sigma_deg>");
      }
      cfg.spread_sigma_deg = real_at(i + 2, "spread gaussian sigma should be a number.");
      if (cfg.spread_sigma_deg < 0.0) {
        PRINT_INPUT_ERROR("spread gaussian sigma should be >= 0.");
      }
      cfg.has_spread = true;
      i += 3;
    } else if (strcmp(param[i], "velocity") == 0) {
      require_index(i + 1, num_param, "Usage: velocity constant V or uniform Vmin Vmax");
      if (strcmp(param[i + 1], "constant") == 0) {
        cfg.speed = real_at(i + 2, "velocity constant V should be a number.");
        if (cfg.speed < 0.0) {
          PRINT_INPUT_ERROR("velocity should be >= 0.");
        }
        cfg.velocity_kind = DepositConfig::Constant;
        i += 3;
      } else if (strcmp(param[i + 1], "uniform") == 0) {
        cfg.speed_min = real_at(i + 2, "velocity uniform values should be numbers.");
        cfg.speed_max = real_at(i + 3, "velocity uniform values should be numbers.");
        if (cfg.speed_min < 0.0 || cfg.speed_max < cfg.speed_min) {
          PRINT_INPUT_ERROR("velocity uniform requires 0 <= Vmin <= Vmax.");
        }
        cfg.velocity_kind = DepositConfig::Uniform;
        i += 4;
      } else {
        PRINT_INPUT_ERROR("velocity must be constant or uniform.");
      }
      cfg.has_velocity = true;
    } else if (strcmp(param[i], "near") == 0) {
      cfg.near_r = real_at(i + 1, "near R should be a number.");
      if (cfg.near_r <= 0.0) {
        PRINT_INPUT_ERROR("near R should be > 0.");
      }
      cfg.has_near = true;
      i += 2;
    } else if (strcmp(param[i], "select") == 0) {
      require_index(i + 1, num_param, "Usage: select sequential or select random");
      if (strcmp(param[i + 1], "sequential") == 0) {
        cfg.select = DepositConfig::Sequential;
      } else if (strcmp(param[i + 1], "random") == 0) {
        cfg.select = DepositConfig::RandomSelect;
      } else {
        PRINT_INPUT_ERROR("select must be sequential or random.");
      }
      cfg.has_select = true;
      i += 2;
    } else if (strcmp(param[i], "attempt") == 0) {
      cfg.attempt = require_int(param, num_param, i + 1, "attempt Q should be an integer.");
      if (cfg.attempt < 1) {
        PRINT_INPUT_ERROR("attempt should be >= 1.");
      }
      cfg.has_attempt = true;
      i += 2;
    } else if (strcmp(param[i], "seed") == 0) {
      const int seed = require_int(param, num_param, i + 1, "seed should be an integer.");
      if (seed < 0) {
        PRINT_INPUT_ERROR("seed should be >= 0.");
      }
      cfg.seed = static_cast<unsigned>(seed);
      cfg.has_seed = true;
      i += 2;
    } else {
      PRINT_INPUT_ERROR("Unknown keyword in deposit.");
    }
  }
}

static void validate_deposit_config(const DepositConfig& cfg)
{
  if (!cfg.has_direction) {
    PRINT_INPUT_ERROR("deposit requires direction.");
  }
  if (cfg.direction_kind == DepositConfig::Target && !cfg.has_normal) {
    PRINT_INPUT_ERROR("direction target requires normal.");
  }
  if (!cfg.has_surface) {
    PRINT_INPUT_ERROR("deposit requires surface.");
  }
  if (!cfg.has_velocity) {
    PRINT_INPUT_ERROR("deposit requires velocity.");
  }
  if (cfg.number_all && cfg.style != DepositConfig::Grid) {
    PRINT_INPUT_ERROR("number all is only valid for grid.");
  }
  if (cfg.has_select && cfg.style != DepositConfig::Grid) {
    PRINT_INPUT_ERROR("select is only valid for grid.");
  }
  if (cfg.has_attempt && cfg.style != DepositConfig::Random && cfg.style != DepositConfig::Gaussian) {
    PRINT_INPUT_ERROR("attempt is only valid for random or gaussian.");
  }
  if (cfg.has_spacing && cfg.style != DepositConfig::Grid) {
    PRINT_INPUT_ERROR("spacing is only valid for grid.");
  }
  if (cfg.has_sigma && cfg.style != DepositConfig::Gaussian) {
    PRINT_INPUT_ERROR("sigma is only valid for gaussian.");
  }

  if (cfg.style == DepositConfig::Grid) {
    if (!cfg.has_region) {
      PRINT_INPUT_ERROR("grid requires region <u_min> <u_max> <v_min> <v_max>.");
    }
    if (!cfg.has_spacing) {
      PRINT_INPUT_ERROR("grid requires spacing <du> <dv>.");
    }
    if (!cfg.has_number) {
      PRINT_INPUT_ERROR("grid requires number <N> or number all.");
    }
  } else if (cfg.style == DepositConfig::Random) {
    if (!cfg.has_region) {
      PRINT_INPUT_ERROR("random requires region <u_min> <u_max> <v_min> <v_max>.");
    }
    if (!cfg.has_number || cfg.number_all) {
      PRINT_INPUT_ERROR("random requires number <N>.");
    }
  } else if (cfg.style == DepositConfig::Gaussian) {
    if (!cfg.has_origin) {
      PRINT_INPUT_ERROR("gaussian requires origin <u> <v>.");
    }
    if (!cfg.has_sigma) {
      PRINT_INPUT_ERROR("gaussian requires sigma <s>.");
    }
    if (!cfg.has_number || cfg.number_all) {
      PRINT_INPUT_ERROR("gaussian requires number <N>.");
    }
  } else {
    if (!cfg.has_origin) {
      PRINT_INPUT_ERROR("point requires origin <u> <v>.");
    }
  }

  if (deposit_needs_rng(cfg) && !cfg.has_seed) {
    PRINT_INPUT_ERROR("deposit random behavior requires an explicit seed.");
  }
}

static double norm3(const double v[3])
{
  return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static void normalize_or_die(double v[3], const char* what)
{
  const double n = norm3(v);
  if (n <= 1.0e-30) {
    PRINT_INPUT_ERROR(what);
  }
  v[0] /= n;
  v[1] /= n;
  v[2] /= n;
}

static double dot3(const double a[3], const double b[3])
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void cross3(const double a[3], const double b[3], double c[3])
{
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

static int cartesian_axis(const double n[3])
{
  for (int a = 0; a < 3; ++a) {
    if (fabs(fabs(n[a]) - 1.0) < 1.0e-12 && fabs(n[(a + 1) % 3]) < 1.0e-12 &&
        fabs(n[(a + 2) % 3]) < 1.0e-12) {
      return a;
    }
  }
  return -1;
}

static void build_deposit_frame(const DepositConfig& cfg, DepositFrame& frame)
{
  if (cfg.has_normal) {
    frame.n[0] = cfg.normal[0];
    frame.n[1] = cfg.normal[1];
    frame.n[2] = cfg.normal[2];
  } else {
    frame.n[0] = -cfg.flight[0];
    frame.n[1] = -cfg.flight[1];
    frame.n[2] = -cfg.flight[2];
  }
  normalize_or_die(frame.n, "deposit normal must be non-zero.");
  frame.cartesian_n = cartesian_axis(frame.n);

  if (cfg.direction_kind == DepositConfig::Target) {
    frame.d[0] = -frame.n[0];
    frame.d[1] = -frame.n[1];
    frame.d[2] = -frame.n[2];
  } else {
    frame.d[0] = cfg.flight[0];
    frame.d[1] = cfg.flight[1];
    frame.d[2] = cfg.flight[2];
    normalize_or_die(frame.d, "deposit direction must be non-zero.");
  }

  double a[3];
  if (cfg.has_basis) {
    a[0] = cfg.basis[0];
    a[1] = cfg.basis[1];
    a[2] = cfg.basis[2];
  } else {
    int k = 0;
    if (fabs(frame.n[1]) < fabs(frame.n[k])) {
      k = 1;
    }
    if (fabs(frame.n[2]) < fabs(frame.n[k])) {
      k = 2;
    }
    a[0] = a[1] = a[2] = 0.0;
    a[k] = 1.0;
  }

  const double adotn = dot3(a, frame.n);
  frame.u[0] = a[0] - adotn * frame.n[0];
  frame.u[1] = a[1] - adotn * frame.n[1];
  frame.u[2] = a[2] - adotn * frame.n[2];
  normalize_or_die(frame.u, "basis is parallel to the surface normal.");
  cross3(frame.n, frame.u, frame.v);
}

static void print_deposit_frame(const DepositFrame& frame)
{
  printf("Deposition basis:\n");
  printf("  u = %.8f %.8f %.8f\n", frame.u[0], frame.u[1], frame.u[2]);
  printf("  v = %.8f %.8f %.8f\n", frame.v[0], frame.v[1], frame.v[2]);
  printf("  n = %.8f %.8f %.8f\n", frame.n[0], frame.n[1], frame.n[2]);
}

static void uvH_to_xyz(const DepositFrame& frame, double u, double v, double H, double r[3])
{
  for (int d = 0; d < 3; ++d) {
    r[d] = u * frame.u[d] + v * frame.v[d] + H * frame.n[d];
  }
}

static double wrap01(double s)
{
  return s - floor(s);
}

static void wrap_lateral(const Box& box, const DepositFrame& frame, double r[3])
{
  if (frame.cartesian_n < 0) {
    return;
  }
  double s[3];
  s[0] = box.cpu_h[9] * r[0] + box.cpu_h[10] * r[1] + box.cpu_h[11] * r[2];
  s[1] = box.cpu_h[12] * r[0] + box.cpu_h[13] * r[1] + box.cpu_h[14] * r[2];
  s[2] = box.cpu_h[15] * r[0] + box.cpu_h[16] * r[1] + box.cpu_h[17] * r[2];
  const int pbc[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
  for (int a = 0; a < 3; ++a) {
    if (a == frame.cartesian_n || pbc[a] != 1) {
      continue;
    }
    s[a] = wrap01(s[a]);
  }
  r[0] = box.cpu_h[0] * s[0] + box.cpu_h[1] * s[1] + box.cpu_h[2] * s[2];
  r[1] = box.cpu_h[3] * s[0] + box.cpu_h[4] * s[1] + box.cpu_h[5] * s[2];
  r[2] = box.cpu_h[6] * s[0] + box.cpu_h[7] * s[1] + box.cpu_h[8] * s[2];
}

static int wrap_cell(int i, int n)
{
  int rem = i % n;
  return rem < 0 ? rem + n : rem;
}

static double min_image_1d(double d, double L, int pbc)
{
  if (pbc != 1 || L <= 0.0) {
    return d;
  }
  d -= L * nearbyint(d / L);
  return d;
}

struct SurfaceIndex
{
  double cell = 1.0;
  int nu = 1;
  int nv = 1;
  double Lu = 1.0;
  double Lv = 1.0;
  double u0 = 0.0;
  double v0 = 0.0;
  int pbc_u = 0;
  int pbc_v = 0;
  double global_H = 0.0;
  std::vector<int> head;
  std::vector<int> next;
  std::vector<double> u;
  std::vector<double> v;
  std::vector<double> H;

  void reduce_global(const Atom& atom, const DepositFrame& frame)
  {
    const int N = atom.number_of_atoms;
    if (N < 1) {
      PRINT_INPUT_ERROR("surface global/local needs existing atoms.");
    }
    const double* r0 = atom.cpu_position_per_atom.data();
    global_H = r0[0] * frame.n[0] + r0[N] * frame.n[1] + r0[2 * N] * frame.n[2];
    for (int i = 1; i < N; ++i) {
      const double h = r0[i] * frame.n[0] + r0[i + N] * frame.n[1] + r0[i + 2 * N] * frame.n[2];
      if (h > global_H) {
        global_H = h;
      }
    }
  }

  void build(const Atom& atom, const Box& box, const DepositFrame& frame, double radius)
  {
    const int N = atom.number_of_atoms;
    if (N < 1) {
      PRINT_INPUT_ERROR("surface global/local needs existing atoms.");
    }
    const double* r0 = atom.cpu_position_per_atom.data();
    u.resize(N);
    v.resize(N);
    H.resize(N);
    global_H = r0[0] * frame.n[0] + r0[N] * frame.n[1] + r0[2 * N] * frame.n[2];
    for (int i = 0; i < N; ++i) {
      const double xyz[3] = {r0[i], r0[N + i], r0[2 * N + i]};
      u[i] = dot3(xyz, frame.u);
      v[i] = dot3(xyz, frame.v);
      H[i] = dot3(xyz, frame.n);
      if (H[i] > global_H) {
        global_H = H[i];
      }
    }

    if (frame.cartesian_n == 2) {
      Lu = box.cpu_h[0];
      Lv = box.cpu_h[4];
      pbc_u = box.pbc_x;
      pbc_v = box.pbc_y;
    } else if (frame.cartesian_n == 0) {
      Lu = box.cpu_h[4];
      Lv = box.cpu_h[8];
      pbc_u = box.pbc_y;
      pbc_v = box.pbc_z;
    } else if (frame.cartesian_n == 1) {
      Lu = box.cpu_h[0];
      Lv = box.cpu_h[8];
      pbc_u = box.pbc_x;
      pbc_v = box.pbc_z;
    } else {
      Lu = Lv = 0.0;
      pbc_u = pbc_v = 0;
    }

    cell = radius > 0.0 ? radius : 1.0;
    if (pbc_u && Lu > 0.0) {
      nu = std::max(1, (int)ceil(Lu / cell));
      cell = Lu / nu;
    } else {
      nu = 1;
    }
    if (pbc_v && Lv > 0.0) {
      nv = std::max(1, (int)ceil(Lv / cell));
      } else {
      nv = 1;
    }
    if (pbc_u) {
      for (int i = 0; i < N; ++i) {
        u[i] = wrap01(u[i] / Lu) * Lu;
      }
    }
    if (pbc_v) {
      for (int i = 0; i < N; ++i) {
        v[i] = wrap01(v[i] / Lv) * Lv;
      }
    }
    if (!pbc_u || !pbc_v) {
      double umin = u[0], umax = u[0], vmin = v[0], vmax = v[0];
      for (int i = 1; i < N; ++i) {
        umin = std::min(umin, u[i]);
        umax = std::max(umax, u[i]);
        vmin = std::min(vmin, v[i]);
        vmax = std::max(vmax, v[i]);
      }
      if (!pbc_u) {
        u0 = umin - radius;
        Lu = std::max(umax - umin, cell);
        nu = std::max(1, (int)ceil((umax - umin + 2.0 * radius) / cell));
        for (int i = 0; i < N; ++i) {
          u[i] -= u0;
        }
      }
      if (!pbc_v) {
        v0 = vmin - radius;
        Lv = std::max(vmax - vmin, cell);
        nv = std::max(1, (int)ceil((vmax - vmin + 2.0 * radius) / cell));
        for (int i = 0; i < N; ++i) {
          v[i] -= v0;
        }
      }
    }

    head.assign(static_cast<size_t>(nu) * nv, -1);
    next.assign(N, -1);
    for (int i = 0; i < N; ++i) {
      int iu = (int)floor(u[i] / cell);
      int iv = (int)floor(v[i] / cell);
      if (pbc_u) {
        iu = wrap_cell(iu, nu);
    } else {
        iu = std::min(nu - 1, std::max(0, iu));
      }
      if (pbc_v) {
        iv = wrap_cell(iv, nv);
      } else {
        iv = std::min(nv - 1, std::max(0, iv));
      }
      const int c = iu + nu * iv;
      next[i] = head[c];
      head[c] = i;
    }
  }

  double query(double qu, double qv, double radius) const
  {
    const double r2 = radius * radius;
    if (pbc_u && Lu > 0.0) {
      qu = wrap01(qu / Lu) * Lu;
    } else {
      qu -= u0;
    }
    if (pbc_v && Lv > 0.0) {
      qv = wrap01(qv / Lv) * Lv;
    } else {
      qv -= v0;
    }
    const int span = (int)ceil(radius / cell);
    const int iu0 = (int)floor(qu / cell);
    const int iv0 = (int)floor(qv / cell);
    bool found = false;
    double best = 0.0;
    for (int du = -span; du <= span; ++du) {
      for (int dv = -span; dv <= span; ++dv) {
        int iu = iu0 + du;
        int iv = iv0 + dv;
        if (pbc_u) {
          iu = wrap_cell(iu, nu);
        } else if (iu < 0 || iu >= nu) {
          continue;
        }
        if (pbc_v) {
          iv = wrap_cell(iv, nv);
        } else if (iv < 0 || iv >= nv) {
          continue;
        }
        for (int a = head[iu + nu * iv]; a >= 0; a = next[a]) {
          const double wu = min_image_1d(u[a] - qu, Lu, pbc_u);
          const double wv = min_image_1d(v[a] - qv, Lv, pbc_v);
          if (wu * wu + wv * wv <= r2) {
            if (!found || H[a] > best) {
              best = H[a];
              found = true;
            }
          }
        }
      }
    }
    return found ? best : global_H;
  }
};

struct NearIndex
{
  double cell = 1.0;
  int nx = 1, ny = 1, nz = 1;
  int pbc[3] = {0, 0, 0};
  double L[3] = {1.0, 1.0, 1.0};
  const Box* box = nullptr;
  int overflow = -1;
  std::vector<int> head;
  std::vector<int> next;
  std::vector<double> x, y, z;

  int id(int ix, int iy, int iz) const { return ix + nx * (iy + ny * iz); }

  bool cell_of(double px, double py, double pz, int& ix, int& iy, int& iz) const
  {
    ix = (int)floor(px / cell);
    iy = (int)floor(py / cell);
    iz = (int)floor(pz / cell);
    if (pbc[0]) {
      ix = wrap_cell(ix, nx);
    } else if (ix < 0 || ix >= nx) {
      return false;
    }
    if (pbc[1]) {
      iy = wrap_cell(iy, ny);
    } else if (iy < 0 || iy >= ny) {
      return false;
    }
    if (pbc[2]) {
      iz = wrap_cell(iz, nz);
    } else if (iz < 0 || iz >= nz) {
      return false;
    }
    return true;
  }

  void link_atom(int i)
  {
    int ix, iy, iz;
    if (!cell_of(x[i], y[i], z[i], ix, iy, iz)) {
      next[i] = overflow;
      overflow = i;
      return;
    }
    const int c = id(ix, iy, iz);
    next[i] = head[c];
    head[c] = i;
  }

  void rebuild_cells()
  {
    overflow = -1;
    head.assign(static_cast<size_t>(nx) * ny * nz, -1);
    next.assign(x.size(), -1);
    for (int i = 0; i < (int)x.size(); ++i) {
      link_atom(i);
    }
  }

  void build(const Atom& atom, const Box& box_, double cutoff)
  {
    box = &box_;
    cell = cutoff > 0.0 ? cutoff : 1.0;
    L[0] = box_.cpu_h[0];
    L[1] = box_.cpu_h[4];
    L[2] = box_.cpu_h[8];
    pbc[0] = box_.pbc_x;
    pbc[1] = box_.pbc_y;
    pbc[2] = box_.pbc_z;
    nx = pbc[0] && L[0] > 0.0 ? std::max(1, (int)floor(L[0] / cell)) : std::max(1, (int)ceil(L[0] / cell));
    ny = pbc[1] && L[1] > 0.0 ? std::max(1, (int)floor(L[1] / cell)) : std::max(1, (int)ceil(L[1] / cell));
    nz = pbc[2] && L[2] > 0.0 ? std::max(1, (int)floor(L[2] / cell)) : std::max(1, (int)ceil(L[2] / cell));
    if (pbc[0] && L[0] > 0.0) {
      cell = std::min(cell, L[0] / nx);
    }
    const int N = atom.number_of_atoms;
    const double* r0 = atom.cpu_position_per_atom.data();
    x.resize(N);
    y.resize(N);
    z.resize(N);
    for (int i = 0; i < N; ++i) {
      x[i] = r0[i];
      y[i] = r0[i + N];
      z[i] = r0[i + 2 * N];
      if (pbc[0]) {
        x[i] = wrap01(x[i] / L[0]) * L[0];
      }
      if (pbc[1]) {
        y[i] = wrap01(y[i] / L[1]) * L[1];
      }
      if (pbc[2]) {
        z[i] = wrap01(z[i] / L[2]) * L[2];
      }
    }
    rebuild_cells();
  }

  void insert_one(double px, double py, double pz)
  {
    x.push_back(px);
    y.push_back(py);
    z.push_back(pz);
    next.push_back(-1);
    link_atom((int)x.size() - 1);
  }

  void insert(const double* xyz, int n_atoms)
  {
    for (int a = 0; a < n_atoms; ++a) {
      insert_one(xyz[a * 3], xyz[a * 3 + 1], xyz[a * 3 + 2]);
    }
  }

  bool overlaps(const double* xyz, int n_atoms, double cutoff) const
  {
    const double cut2 = cutoff * cutoff;
    const int span = 1;
    for (int a = 0; a < n_atoms; ++a) {
      const double px = xyz[a * 3];
      const double py = xyz[a * 3 + 1];
      const double pz = xyz[a * 3 + 2];
      int ix0 = (int)floor(px / cell);
      int iy0 = (int)floor(py / cell);
      int iz0 = (int)floor(pz / cell);
      for (int dx = -span; dx <= span; ++dx) {
        for (int dy = -span; dy <= span; ++dy) {
          for (int dz = -span; dz <= span; ++dz) {
            int ix = ix0 + dx;
            int iy = iy0 + dy;
            int iz = iz0 + dz;
            if (pbc[0]) {
              ix = wrap_cell(ix, nx);
            } else if (ix < 0 || ix >= nx) {
              continue;
            }
            if (pbc[1]) {
              iy = wrap_cell(iy, ny);
            } else if (iy < 0 || iy >= ny) {
              continue;
            }
            if (pbc[2]) {
              iz = wrap_cell(iz, nz);
            } else if (iz < 0 || iz >= nz) {
              continue;
            }
            for (int j = head[id(ix, iy, iz)]; j >= 0; j = next[j]) {
              double rx = px - x[j];
              double ry = py - y[j];
              double rz = pz - z[j];
              apply_mic(*box, rx, ry, rz);
              if (rx * rx + ry * ry + rz * rz < cut2) {
                return true;
              }
            }
          }
        }
      }
      for (int j = overflow; j >= 0; j = next[j]) {
        double rx = px - x[j];
        double ry = py - y[j];
        double rz = pz - z[j];
        apply_mic(*box, rx, ry, rz);
        if (rx * rx + ry * ry + rz * rz < cut2) {
          return true;
        }
      }
    }
    return false;
  }
};

struct GridView
{
  int nu = 0;
  int nv = 0;
  double umin = 0.0;
  double vmin = 0.0;
  double du = 1.0;
  double dv = 1.0;
  int size() const { return nu * nv; }
  DepositUV at(int s) const
  {
    return {umin + (s / nv) * du, vmin + (s % nv) * dv};
  }
};

static int count_grid_axis(double lo, double hi, double d)
{
  int n = 0;
  const double eps = 1.0e-10;
  for (;;) {
    if (lo + n * d > hi + eps) {
      break;
    }
    ++n;
    if (n > 100000000) {
      PRINT_INPUT_ERROR("grid spacing produced too many sites.");
    }
  }
  return n;
}

static GridView make_grid_view(const DepositConfig& cfg)
{
  GridView g;
  g.umin = cfg.region[0];
  g.vmin = cfg.region[2];
  g.du = cfg.spacing[0];
  g.dv = cfg.spacing[1];
  g.nu = count_grid_axis(cfg.region[0], cfg.region[1], cfg.spacing[0]);
  g.nv = count_grid_axis(cfg.region[2], cfg.region[3], cfg.spacing[1]);
  if (g.size() < 1) {
    PRINT_INPUT_ERROR("grid region/spacing produced no sites.");
  }
  return g;
}

static int map_perm(std::unordered_map<int, int>& swapped, int i)
{
  auto it = swapped.find(i);
  return it == swapped.end() ? i : it->second;
}

struct DepositSampler
{
  std::mt19937 rng;
  std::normal_distribution<double> normal{0.0, 1.0};
  std::uniform_real_distribution<double> phi{0.0, 2.0 * 3.14159265358979};
  std::uniform_real_distribution<double> speed{0.0, 1.0};
  std::uniform_real_distribution<double> uu{0.0, 1.0};
  std::uniform_real_distribution<double> vv{0.0, 1.0};
  bool has_speed = false;
  bool has_uv = false;
};

static void prepare_sampler(const DepositConfig& cfg, DepositSampler& s)
{
  if (deposit_needs_rng(cfg)) {
    s.rng.seed(cfg.seed);
  }
  if (cfg.velocity_kind == DepositConfig::Uniform && cfg.speed_min < cfg.speed_max) {
    s.speed = std::uniform_real_distribution<double>(cfg.speed_min, cfg.speed_max);
    s.has_speed = true;
  }
  if (cfg.style == DepositConfig::Random) {
    if (cfg.region[0] < cfg.region[1]) {
      s.uu = std::uniform_real_distribution<double>(cfg.region[0], cfg.region[1]);
    }
    if (cfg.region[2] < cfg.region[3]) {
      s.vv = std::uniform_real_distribution<double>(cfg.region[2], cfg.region[3]);
    }
    s.has_uv = true;
  }
}

static DepositUV sample_style_uv(const DepositConfig& cfg, DepositSampler& s)
{
  DepositUV uv;
  if (cfg.style == DepositConfig::Point) {
    uv.u = cfg.origin[0];
    uv.v = cfg.origin[1];
    return uv;
  }
  if (cfg.style == DepositConfig::Gaussian) {
    uv.u = cfg.origin[0] + s.normal(s.rng) * cfg.sigma;
    uv.v = cfg.origin[1] + s.normal(s.rng) * cfg.sigma;
    return uv;
  }
  uv.u = (cfg.region[0] < cfg.region[1]) ? s.uu(s.rng) : cfg.region[0];
  uv.v = (cfg.region[2] < cfg.region[3]) ? s.vv(s.rng) : cfg.region[2];
  return uv;
}

static void sample_flight_direction(
  const DepositConfig& cfg,
  const DepositFrame& frame,
  const double r[3],
  DepositSampler& s,
  double dir[3])
{
  double d[3];
  if (cfg.direction_kind == DepositConfig::Target) {
    d[0] = cfg.target[0] - r[0];
    d[1] = cfg.target[1] - r[1];
    d[2] = cfg.target[2] - r[2];
    normalize_or_die(d, "direction target coincides with a launch position.");
  } else {
    d[0] = frame.d[0];
    d[1] = frame.d[1];
    d[2] = frame.d[2];
  }
  if (!cfg.has_spread) {
    dir[0] = d[0];
    dir[1] = d[1];
    dir[2] = d[2];
    return;
  }
  double e1[3];
  const double d_dot_u = dot3(d, frame.u);
  e1[0] = frame.u[0] - d_dot_u * d[0];
  e1[1] = frame.u[1] - d_dot_u * d[1];
  e1[2] = frame.u[2] - d_dot_u * d[2];
  if (norm3(e1) <= 1.0e-12) {
    e1[0] = frame.v[0];
    e1[1] = frame.v[1];
    e1[2] = frame.v[2];
  }
  normalize_or_die(e1, "cannot build angular spread frame.");
  double e2[3];
  cross3(d, e1, e2);
  const double theta = fabs(s.normal(s.rng) * cfg.spread_sigma_deg * 3.14159265358979 / 180.0);
  const double phi = s.phi(s.rng);
  const double st = sin(theta);
  const double ct = cos(theta);
  for (int k = 0; k < 3; ++k) {
    dir[k] = st * cos(phi) * e1[k] + st * sin(phi) * e2[k] + ct * d[k];
  }
}

static double sample_speed(const DepositConfig& cfg, DepositSampler& s)
{
  if (cfg.velocity_kind == DepositConfig::Constant) {
    return cfg.speed;
  }
  if (!s.has_speed) {
    return cfg.speed_min;
  }
  return s.speed(s.rng);
}

static void sample_velocity(
  const DepositConfig& cfg,
  const DepositFrame& frame,
  const double r[3],
  DepositSampler& s,
  double vel[3])
{
  double dir[3];
  sample_flight_direction(cfg, frame, r, s, dir);
  const double speed = sample_speed(cfg, s);
  vel[0] = speed * dir[0];
  vel[1] = speed * dir[1];
  vel[2] = speed * dir[2];
}

static double resolve_launch_H(
  const DepositConfig& cfg,
  const SurfaceIndex& surface,
  double u,
  double v)
{
  if (cfg.surface_kind == DepositConfig::Fixed) {
    return cfg.surface_H;
  }
  if (cfg.surface_kind == DepositConfig::Global) {
    return surface.global_H + cfg.surface_gap;
  }
  return surface.query(u, v, cfg.surface_radius) + cfg.surface_gap;
}

static void fill_entity_xyz(const EntityTemplate& tmpl, const double r[3], double* xyz)
{
  for (int n = 0; n < tmpl.n_atoms; ++n) {
    for (int d = 0; d < 3; ++d) {
      xyz[n * 3 + d] = r[d] + tmpl.rel_pos[n + tmpl.n_atoms * d];
    }
  }
}

static void place_entity(
  NewAtoms& added,
  int n_total,
  int first,
  const EntityTemplate& tmpl,
  const double r[3],
  const double vel[3])
{
  for (int n = 0; n < tmpl.n_atoms; ++n) {
    const int i = first + n;
    added.symbol[i] = tmpl.symbol[n];
    added.type[i] = tmpl.type[n];
    added.mass[i] = tmpl.mass[n];
    added.charge[i] = 0.0f;
    for (int d = 0; d < 3; ++d) {
      added.position[i + n_total * d] = r[d] + tmpl.rel_pos[n + tmpl.n_atoms * d];
      added.velocity[i + n_total * d] = vel[d];
    }
  }
}

static NewAtoms make_batch(int n_atoms)
{
  NewAtoms added;
  added.n = n_atoms;
  added.type.resize(n_atoms);
  added.mass.resize(n_atoms);
  added.charge.resize(n_atoms, 0.0f);
  added.symbol.resize(n_atoms);
  added.position.resize(n_atoms * 3);
  added.velocity.resize(n_atoms * 3);
  return added;
}

static std::vector<std::string> symbols_from_atom(const Atom& atoms)
{
  std::vector<std::string> by_type(atoms.cpu_type_size.size());
  for (int i = 0; i < atoms.number_of_atoms; ++i) {
    const int t = atoms.cpu_type[i];
    if (t >= 0 && t < (int)by_type.size() && by_type[t].empty()) {
      by_type[t] = atoms.cpu_atom_symbol[i];
    }
  }
  return by_type;
}

static std::vector<std::string> symbols_from_potential()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }
  std::string filename_potential;
  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() >= 2 && tokens[0] == "potential") {
      filename_potential = tokens[1];
      break;
    }
  }
  if (filename_potential.empty()) {
    PRINT_INPUT_ERROR("There is no 'potential' keyword in run.in.");
  }
  std::ifstream input_potential(filename_potential);
  if (!input_potential.is_open()) {
    PRINT_INPUT_ERROR("Cannot open potential file.");
  }
  std::vector<std::string> tokens = get_tokens(input_potential);
  if (tokens.size() < 3) {
    PRINT_INPUT_ERROR("The first line of the potential file should have at least 3 items.");
  }
  const int number_of_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if ((int)tokens.size() != 2 + number_of_types) {
    PRINT_INPUT_ERROR("The first line of the potential file should have the correct number of atom symbols.");
  }
  std::vector<std::string> atom_symbols(number_of_types);
  for (int n = 0; n < number_of_types; ++n) {
    atom_symbols[n] = tokens[2 + n];
  }
  return atom_symbols;
}

static int type_of_symbol(const std::string& symbol, const std::vector<std::string>& allowed)
{
  for (size_t t = 0; t < allowed.size(); ++t) {
    if (symbol == allowed[t]) {
      return (int)t;
    }
  }
  PRINT_INPUT_ERROR("Atom symbol is not allowed by the used potential.");
  return -1;
}

static std::vector<std::string> resolve_allowed_symbols(const Atom& atoms, const std::string& needed)
{
  std::vector<std::string> allowed = symbols_from_atom(atoms);
  for (const auto& s : allowed) {
    if (s == needed) {
      return allowed;
    }
  }
  return symbols_from_potential();
}

static void read_molecule_xyz(
  const std::string& filename,
  const std::vector<std::string>& allowed_symbols,
  std::vector<std::string>& molecule_symbols,
  std::vector<double>& molecule_positions,
  std::vector<double>& molecule_masses)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Failed to open molecule xyz file.");
  }
  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() != 1) {
    PRINT_INPUT_ERROR("The first line for the xyz file should have one value.");
  }
  const int N = get_int_from_token(tokens[0], __FILE__, __LINE__);
  if (N < 1) {
    PRINT_INPUT_ERROR("Number of atoms should >= 1.");
  }
  std::string line;
  std::getline(input, line);
  int has_velocity_in_xyz = 0;
  bool has_mass = false;
  bool has_charge = false;
  int num_columns = 4;
  int property_offset[5] = {0, 1, -1, -1, -1};
  int number_of_types = (int)allowed_symbols.size();
  std::vector<Group> empty_group;
  molecule_symbols.resize(N);
  molecule_positions.resize(N * 3);
  molecule_masses.resize(N);
  std::vector<int> cpu_type(N);
  std::vector<float> cpu_charge(N);
  std::vector<double> cpu_velocity_per_atom(N * 3);
  std::vector<std::string> atom_symbols_copy = allowed_symbols;
  read_xyz_in_line_3(
    input,
    N,
    has_velocity_in_xyz,
    has_mass,
    has_charge,
    num_columns,
    property_offset,
    number_of_types,
    atom_symbols_copy,
    molecule_symbols,
    cpu_type,
    molecule_masses,
    cpu_charge,
    molecule_positions,
    cpu_velocity_per_atom,
    empty_group);
}

static void finish_entity_radius(EntityTemplate& tmpl)
{
  tmpl.radius = 0.0;
  for (int n = 0; n < tmpl.n_atoms; ++n) {
    double r2 = 0.0;
    for (int d = 0; d < 3; ++d) {
      const double x = tmpl.rel_pos[n + tmpl.n_atoms * d];
      r2 += x * x;
    }
    tmpl.radius = std::max(tmpl.radius, sqrt(r2));
  }
}

static void resolve_entity_template(const DepositConfig& cfg, const Atom& atoms, EntityTemplate& tmpl)
{
  if (cfg.entity == DepositConfig::Atom) {
    const std::vector<std::string> allowed = resolve_allowed_symbols(atoms, cfg.source);
    tmpl.n_atoms = 1;
    tmpl.symbol = {cfg.source};
    tmpl.type = {type_of_symbol(cfg.source, allowed)};
    tmpl.mass = {get_mass_from_symbol(cfg.source)};
    tmpl.rel_pos = {0.0, 0.0, 0.0};
    tmpl.radius = 0.0;
    return;
  }

  const std::vector<std::string> allowed = symbols_from_potential();
  std::vector<std::string> symbols;
  std::vector<double> positions;
  std::vector<double> masses;
  read_molecule_xyz(cfg.source, allowed, symbols, positions, masses);
  tmpl.n_atoms = (int)symbols.size();
  tmpl.symbol = symbols;
  tmpl.mass = masses;
  tmpl.type.resize(tmpl.n_atoms);
  for (int n = 0; n < tmpl.n_atoms; ++n) {
    tmpl.type[n] = type_of_symbol(symbols[n], allowed);
  }
  double com[3] = {0.0, 0.0, 0.0};
  double total_mass = 0.0;
  for (int n = 0; n < tmpl.n_atoms; ++n) {
    total_mass += masses[n];
    for (int d = 0; d < 3; ++d) {
      com[d] += masses[n] * positions[n + tmpl.n_atoms * d];
    }
  }
  if (total_mass <= 0.0) {
    PRINT_INPUT_ERROR("molecule mass must be > 0.");
  }
  for (int d = 0; d < 3; ++d) {
    com[d] /= total_mass;
  }
  tmpl.rel_pos.resize(tmpl.n_atoms * 3);
  for (int n = 0; n < tmpl.n_atoms; ++n) {
    for (int d = 0; d < 3; ++d) {
      tmpl.rel_pos[n + tmpl.n_atoms * d] = positions[n + tmpl.n_atoms * d] - com[d];
    }
  }
  finish_entity_radius(tmpl);
  printf("Read molecule from %s (%d atoms, mass %.6f).\n", cfg.source.c_str(), tmpl.n_atoms, total_mass);
}

struct Launch
{
  double r[3];
  double vel[3];
};

static bool try_place(
  const DepositConfig& cfg,
  const EntityTemplate& tmpl,
  const DepositFrame& frame,
  const Box& box,
  const SurfaceIndex& surface,
  NearIndex* near,
  bool skip_new_new,
  DepositSampler& sampler,
  const DepositUV& uv,
  std::vector<double>& cand,
  std::vector<Launch>& launches)
{
  const double H = resolve_launch_H(cfg, surface, uv.u, uv.v);
  double r[3];
  uvH_to_xyz(frame, uv.u, uv.v, H, r);
  wrap_lateral(box, frame, r);
  fill_entity_xyz(tmpl, r, cand.data());
  if (cfg.has_near && near->overlaps(cand.data(), tmpl.n_atoms, cfg.near_r)) {
      return false;
    }
  Launch L;
  L.r[0] = r[0];
  L.r[1] = r[1];
  L.r[2] = r[2];
  sample_velocity(cfg, frame, r, sampler, L.vel);
  launches.push_back(L);
  if (cfg.has_near && !skip_new_new) {
    near->insert(cand.data(), tmpl.n_atoms);
  }
  return true;
}

static NewAtoms launches_to_atoms(const EntityTemplate& tmpl, const std::vector<Launch>& launches)
{
  const int n_atoms = (int)launches.size() * tmpl.n_atoms;
  NewAtoms added = make_batch(n_atoms);
  for (int k = 0; k < (int)launches.size(); ++k) {
    place_entity(added, n_atoms, k * tmpl.n_atoms, tmpl, launches[k].r, launches[k].vel);
  }
  return added;
}

static const char* style_name(DepositConfig::Style style)
{
  switch (style) {
    case DepositConfig::Grid:
      return "grid";
    case DepositConfig::Random:
      return "random";
    case DepositConfig::Gaussian:
      return "gaussian";
    default:
      return "point";
  }
}

static NewAtoms deposit_grid(
  const DepositConfig& cfg,
  const EntityTemplate& tmpl,
  const DepositFrame& frame,
  const Box& box,
  const SurfaceIndex& surface,
  NearIndex* near,
  bool skip_new_new,
  DepositSampler& sampler,
  std::vector<double>& cand)
{
  const GridView grid = make_grid_view(cfg);
  const int S = grid.size();
  const int need = cfg.number_all ? S : cfg.number;
  const bool random_order = !cfg.number_all && cfg.select == DepositConfig::RandomSelect;
  std::vector<Launch> launches;
  launches.reserve(need);
  std::unordered_map<int, int> swapped;

  for (int t = 0; t < S && (int)launches.size() < need; ++t) {
    int s = t;
    if (random_order) {
      std::uniform_int_distribution<int> dist(t, S - 1);
      const int j = dist(sampler.rng);
      s = map_perm(swapped, j);
      swapped[j] = map_perm(swapped, t);
    }
    try_place(cfg, tmpl, frame, box, surface, near, skip_new_new, sampler, grid.at(s), cand, launches);
  }

  if (launches.empty() || (!cfg.number_all && (int)launches.size() < need)) {
    char msg[256];
    snprintf(
      msg,
      sizeof(msg),
      "Requested entities: %d. Valid grid sites: %d.",
      cfg.number_all ? S : cfg.number,
      (int)launches.size());
    PRINT_INPUT_ERROR(msg);
  }
  printf("Requested entities : %d\n", cfg.number_all ? (int)launches.size() : cfg.number);
  printf("Grid sites         : %d\n", S);
  printf("Selected entities  : %d\n", (int)launches.size());
  return launches_to_atoms(tmpl, launches);
}

static NewAtoms deposit_sampled(
  const DepositConfig& cfg,
  const EntityTemplate& tmpl,
  const DepositFrame& frame,
  const Box& box,
  const SurfaceIndex& surface,
  NearIndex* near,
  DepositSampler& sampler,
  std::vector<double>& cand)
{
  const int tries = (cfg.style == DepositConfig::Point) ? 1 : cfg.attempt;
  std::vector<Launch> launches;
  launches.reserve(cfg.number);
  for (int k = 0; k < cfg.number; ++k) {
    bool ok = false;
    for (int t = 0; t < tries; ++t) {
      const DepositUV uv = sample_style_uv(cfg, sampler);
      if (try_place(cfg, tmpl, frame, box, surface, near, false, sampler, uv, cand, launches)) {
        ok = true;
        break;
      }
    }
    if (!ok) {
      PRINT_INPUT_ERROR("deposit failed: no valid insertion after attempt retries.");
    }
  }
  return launches_to_atoms(tmpl, launches);
}

void Deposit(
  const char** param,
  int num_param,
  Box& box,
  Atom& atoms,
  std::vector<Group>& groups,
  GPU_Vector<double>& thermo,
  Force& force,
  VariableScope* variables)
{
  const auto time_begin = std::chrono::high_resolution_clock::now();

  DepositConfig cfg;
  parse_deposit_config(param, num_param, box, variables, cfg);
  validate_deposit_config(cfg);

  EntityTemplate tmpl;
  resolve_entity_template(cfg, atoms, tmpl);

  DepositFrame frame;
  build_deposit_frame(cfg, frame);

  SurfaceIndex surface;
  if (cfg.surface_kind == DepositConfig::Global) {
    surface.reduce_global(atoms, frame);
  } else if (cfg.surface_kind == DepositConfig::Local) {
    surface.build(atoms, box, frame, cfg.surface_radius);
  }

  NearIndex near;
  NearIndex* near_ptr = nullptr;
  if (cfg.has_near) {
    near.build(atoms, box, cfg.near_r);
    near_ptr = &near;
  }

  bool skip_new_new = false;
  if (cfg.style == DepositConfig::Grid && cfg.has_near) {
    const double dmin = std::min(cfg.spacing[0], cfg.spacing[1]);
    skip_new_new = dmin >= (2.0 * tmpl.radius + cfg.near_r);
  }

  DepositSampler sampler;
  prepare_sampler(cfg, sampler);
  std::vector<double> cand((size_t)tmpl.n_atoms * 3);

  print_line_1();
  printf(
    "deposit %s %s %s\n",
    style_name(cfg.style),
    cfg.entity == DepositConfig::Atom ? "atom" : "molecule",
    cfg.source.c_str());
  print_deposit_frame(frame);

  NewAtoms added = (cfg.style == DepositConfig::Grid)
                     ? deposit_grid(cfg, tmpl, frame, box, surface, near_ptr, skip_new_new, sampler, cand)
                     : deposit_sampled(cfg, tmpl, frame, box, surface, near_ptr, sampler, cand);

  int max_type = -1;
  for (int t : tmpl.type) {
    if (t > max_type) {
      max_type = t;
    }
  }
  if ((int)atoms.cpu_type_size.size() < max_type + 1) {
    atoms.cpu_type_size.resize(max_type + 1, 0);
  }
  AtomMutation::append_atoms(atoms, groups, thermo, force, added);

  printf("Deposited %d atoms. Total number of atoms: %d\n", added.n, atoms.number_of_atoms);
  print_line_2();

  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_used = time_finish - time_begin;
  printf("Time used for deposit = %g second.\n", time_used.count());
}
