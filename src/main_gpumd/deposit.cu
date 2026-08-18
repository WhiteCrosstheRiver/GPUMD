/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "deposit.cuh"
#include "atom_mutation.cuh"
#include "model/read_xyz.cuh"
#include "utilities/common.cuh"
#include "../../extensions/common/sampling/gaussian_position.cuh"
#include "../../extensions/common/sampling/gaussian_velocity.cuh"
#include "../../extensions/common/surface/local_surface.cuh"
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <optional>
#include <random>
#include <string>

extern double get_mass_from_symbol(const std::string& symbol);

// Forward declaration of read_xyz_in_line_3 from read_xyz.cu
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

// Helper function to read a molecule from xyz file (reusing read_xyz.cu functions)
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

  // Read first line: number of atoms (same pattern as read_xyz_line_1)
  int N;
  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() != 1) {
    PRINT_INPUT_ERROR("The first line for the xyz file should have one value.");
  }
  N = get_int_from_token(tokens[0], __FILE__, __LINE__);
  if (N < 1) {
    PRINT_INPUT_ERROR("Number of atoms should >= 1.");
  }

  // Skip second line (comment line, standard xyz format)
  std::string line;
  std::getline(input, line);

  // Prepare parameters for read_xyz_in_line_3 (standard xyz format: symbol x y z)
  int has_velocity_in_xyz = 0;
  bool has_mass = false; // Use MASS_TABLE from read_xyz.cu
  bool has_charge = false; // No charge in standard xyz format
  int num_columns = 4;   // symbol x y z
  int property_offset[5] = {0, 1, -1, -1, -1}; // species, pos, no mass, no vel, no group
  int number_of_types = allowed_symbols.size();
  std::vector<Group> empty_group; // No groups needed for molecule

  // Resize output vectors
  molecule_symbols.resize(N);
  molecule_positions.resize(N * 3);
  molecule_masses.resize(N);
  std::vector<int> cpu_type(N);
  std::vector<float> cpu_charge(N);
  std::vector<double> cpu_velocity_per_atom(N * 3);

  // Use read_xyz_in_line_3 to read atoms (reusing code from read_xyz.cu)
  // Note: atom_symbols parameter is modified by read_xyz_in_line_3, so we need a non-const copy
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

  input.close();
}

// Helper function to get atom type from symbol
static int get_atom_type(
  const std::string& symbol, const std::vector<std::string>& allowed_symbols)
{
  for (size_t t = 0; t < allowed_symbols.size(); ++t) {
    if (symbol == allowed_symbols[t]) {
      return t;
    }
  }
  PRINT_INPUT_ERROR("Atom symbol in molecule is not allowed in the used potential.");
  return -1; // never reached
}

// Helper function to get potential filename from run.in
static std::string get_potential_filename()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }

  std::string line;
  std::string filename_potential;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() >= 2) {
      if (tokens[0] == "potential") {
        filename_potential = tokens[1];
        break;
      }
    }
  }
  input_run.close();
  if (filename_potential.size() == 0) {
    PRINT_INPUT_ERROR("There is no 'potential' keyword in run.in.");
  }
  return filename_potential;
}

// Helper function to get allowed atom symbols from potential
static std::vector<std::string> get_allowed_symbols()
{
  std::string filename_potential = get_potential_filename();
  std::ifstream input_potential(filename_potential);
  if (!input_potential.is_open()) {
    PRINT_INPUT_ERROR("Cannot open potential file.");
  }

  std::vector<std::string> tokens = get_tokens(input_potential);
  if (tokens.size() < 3) {
    PRINT_INPUT_ERROR("The first line of the potential file should have at least 3 items.");
  }

  int number_of_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + number_of_types) {
    PRINT_INPUT_ERROR("The first line of the potential file should have correct number of atom symbols.");
  }

  std::vector<std::string> atom_symbols(number_of_types);
  for (int n = 0; n < number_of_types; ++n) {
    atom_symbols[n] = tokens[2 + n];
  }

  input_potential.close();
  return atom_symbols;
}

static void append_one_deposited_atom(
  Atom& atoms,
  std::vector<Group>& groups,
  GPU_Vector<double>& thermo,
  Force& force,
  const std::string& symbol,
  double x,
  double y,
  double z,
  double vx,
  double vy,
  double vz)
{
  const std::vector<std::string> allowed_symbols = get_allowed_symbols();
  NewAtoms added;
  added.n = 1;
  added.type = {get_atom_type(symbol, allowed_symbols)};
  added.mass = {get_mass_from_symbol(symbol)};
  added.charge = {0.0f};
  added.symbol = {symbol};
  added.position = {x, y, z};
  added.velocity = {vx, vy, vz};

  if ((int)atoms.cpu_type_size.size() < (int)allowed_symbols.size()) {
    atoms.cpu_type_size.resize(allowed_symbols.size(), 0);
  }
  AtomMutation::append_atoms(atoms, groups, thermo, force, added);

  print_line_1();
  printf(
    "Deposited 1 atom of type %s at (%.6f, %.6f, %.6f) with velocity (%.8f, %.8f, %.8f).\n",
    symbol.c_str(),
    x,
    y,
    z,
    vx,
    vy,
    vz);
  printf("Total number of atoms: %d\n", atoms.number_of_atoms);
  print_line_2();
}

static bool parse_real_token(const char* token, double* value, const char* what)
{
  if (!is_valid_real(token, value)) {
    PRINT_INPUT_ERROR(what);
  }
  return true;
}

static void deposit_one_atom_xyz(
  const char** param,
  int num_param,
  Atom& atoms,
  std::vector<Group>& groups,
  GPU_Vector<double>& thermo,
  Force& force)
{
  if (num_param != 8) {
    PRINT_INPUT_ERROR("Usage: deposit <symbol> <x> <y> <z> <vx> <vy> <vz>");
  }
  const std::string symbol = param[1];
  double x, y, z, vx, vy, vz;
  parse_real_token(param[2], &x, "x should be a number.");
  parse_real_token(param[3], &y, "y should be a number.");
  parse_real_token(param[4], &z, "z should be a number.");
  parse_real_token(param[5], &vx, "vx should be a number.");
  parse_real_token(param[6], &vy, "vy should be a number.");
  parse_real_token(param[7], &vz, "vz should be a number.");
  append_one_deposited_atom(atoms, groups, thermo, force, symbol, x, y, z, vx, vy, vz);
}

static void deposit_one_atom_keywords(
  const char** param,
  int num_param,
  Box& box,
  Atom& atoms,
  std::vector<Group>& groups,
  GPU_Vector<double>& thermo,
  Force& force)
{
  const std::string symbol = param[1];
  bool has_pos_xyz = false;
  bool has_pos_gaussian = false;
  bool has_vel_xyz = false;
  bool has_vel_gaussian = false;
  bool has_surface = false;
  bool has_offset = false;
  double pos[3] = {0.0, 0.0, 0.0};
  double xy0[2] = {0.0, 0.0};
  double xy_sigma = 0.0;
  double vel[3] = {0.0, 0.0, 0.0};
  double v_mag = 0.0;
  double theta_sigma_deg = 0.0;
  double surface_radius = 0.0;
  double offset_sep = 0.0;

  int i = 2;
  while (i < num_param) {
    if (strcmp(param[i], "position") == 0) {
      if (i + 1 >= num_param) {
        PRINT_INPUT_ERROR("Missing arguments after position.");
      }
      if (strcmp(param[i + 1], "gaussian") == 0) {
        if (i + 4 >= num_param) {
          PRINT_INPUT_ERROR("Usage: position gaussian <x0> <y0> <sigma>");
        }
        parse_real_token(param[i + 2], &xy0[0], "position gaussian x0 should be a number.");
        parse_real_token(param[i + 3], &xy0[1], "position gaussian y0 should be a number.");
        parse_real_token(param[i + 4], &xy_sigma, "position gaussian sigma should be a number.");
        if (xy_sigma < 0.0) {
          PRINT_INPUT_ERROR("position gaussian sigma should be >= 0.");
        }
        has_pos_gaussian = true;
        i += 5;
      } else {
        if (i + 3 >= num_param) {
          PRINT_INPUT_ERROR("Usage: position <x> <y> <z>");
        }
        parse_real_token(param[i + 1], &pos[0], "position x should be a number.");
        parse_real_token(param[i + 2], &pos[1], "position y should be a number.");
        parse_real_token(param[i + 3], &pos[2], "position z should be a number.");
        has_pos_xyz = true;
        i += 4;
      }
    } else if (strcmp(param[i], "velocity") == 0) {
      if (i + 1 >= num_param) {
        PRINT_INPUT_ERROR("Missing arguments after velocity.");
      }
      if (strcmp(param[i + 1], "gaussian") == 0) {
        if (i + 3 >= num_param) {
          PRINT_INPUT_ERROR("Usage: velocity gaussian <v> <theta_sigma_deg>");
        }
        parse_real_token(param[i + 2], &v_mag, "velocity gaussian v should be a number.");
        parse_real_token(
          param[i + 3], &theta_sigma_deg, "velocity gaussian theta_sigma_deg should be a number.");
        if (v_mag < 0.0) {
          PRINT_INPUT_ERROR("velocity gaussian v should be >= 0.");
        }
        if (theta_sigma_deg < 0.0) {
          PRINT_INPUT_ERROR("velocity gaussian theta_sigma_deg should be >= 0.");
        }
        has_vel_gaussian = true;
        i += 4;
      } else {
        if (i + 3 >= num_param) {
          PRINT_INPUT_ERROR("Usage: velocity <vx> <vy> <vz>");
        }
        parse_real_token(param[i + 1], &vel[0], "velocity vx should be a number.");
        parse_real_token(param[i + 2], &vel[1], "velocity vy should be a number.");
        parse_real_token(param[i + 3], &vel[2], "velocity vz should be a number.");
        has_vel_xyz = true;
        i += 4;
      }
    } else if (strcmp(param[i], "surface") == 0) {
      if (i + 2 >= num_param || strcmp(param[i + 1], "local") != 0) {
        PRINT_INPUT_ERROR("Usage: surface local <radius>");
      }
      parse_real_token(param[i + 2], &surface_radius, "surface local radius should be a number.");
      if (surface_radius <= 0.0) {
        PRINT_INPUT_ERROR("surface local radius should be > 0.");
      }
      has_surface = true;
      i += 3;
    } else if (strcmp(param[i], "offset") == 0) {
      if (i + 2 >= num_param || strcmp(param[i + 1], "antivel") != 0) {
        PRINT_INPUT_ERROR("Usage: offset antivel <sep>");
      }
      parse_real_token(param[i + 2], &offset_sep, "offset antivel sep should be a number.");
      if (offset_sep < 0.0) {
        PRINT_INPUT_ERROR("offset antivel sep should be >= 0.");
      }
      has_offset = true;
      i += 3;
    } else {
      PRINT_INPUT_ERROR("Unknown keyword in deposit. Expected position, velocity, surface, or offset.");
    }
  }

  if (has_pos_xyz && has_pos_gaussian) {
    PRINT_INPUT_ERROR("Specify only one of: position <x> <y> <z> or position gaussian ...");
  }
  if (has_vel_xyz && has_vel_gaussian) {
    PRINT_INPUT_ERROR("Specify only one of: velocity <vx> <vy> <vz> or velocity gaussian ...");
  }
  if (!has_pos_xyz && !has_pos_gaussian) {
    PRINT_INPUT_ERROR("deposit needs a position specification.");
  }
  if (!has_vel_xyz && !has_vel_gaussian) {
    PRINT_INPUT_ERROR("deposit needs a velocity specification.");
  }
  if (has_pos_gaussian && !has_surface) {
    PRINT_INPUT_ERROR("position gaussian requires surface local <radius>.");
  }
  if (atoms.number_of_atoms < 1 && has_surface) {
    PRINT_INPUT_ERROR("surface local needs existing atoms.");
  }

  std::random_device rd;
  std::mt19937 gen(rd());

  if (has_vel_gaussian) {
    sample_gaussian_beam_velocity(v_mag, theta_sigma_deg, gen, vel);
  }

  if (has_pos_gaussian) {
    sample_gaussian_xy(xy0[0], xy0[1], xy_sigma, gen, pos[0], pos[1]);
    wrap_xy_into_box(box, pos[0], pos[1]);
  }

  if (has_surface) {
    query_local_zmax(atoms, box, pos[0], pos[1], surface_radius, pos[0], pos[1], pos[2]);
  }

  if (has_offset) {
    const double vnorm = sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
    if (vnorm <= 1.0e-30) {
      PRINT_INPUT_ERROR("offset antivel needs a non-zero velocity.");
    }
    pos[0] -= offset_sep * vel[0] / vnorm;
    pos[1] -= offset_sep * vel[1] / vnorm;
    pos[2] -= offset_sep * vel[2] / vnorm;
    wrap_xy_into_box(box, pos[0], pos[1]);
  }

  append_one_deposited_atom(
    atoms, groups, thermo, force, symbol, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]);
}

static void deposit_from_molecule_file(
  const char** param,
  int num_param,
  Atom& atoms,
  std::vector<Group>& groups,
  GPU_Vector<double>& thermo,
  Force& force)
{
  // Format: deposit filename.xyz number N velocity vx_min vx_max vy_min vy_max vz_min vz_max position x_min x_max y_min y_max z_min z_max
  if (num_param < 16) {
    PRINT_INPUT_ERROR(
      "Deposit should have parameters: filename.xyz number N velocity vx_min vx_max vy_min vy_max "
      "vz_min vz_max position x_min x_max y_min y_max z_min z_max");
  }

  // Parse filename
  std::string filename = param[1];

  // Parse number
  int num_molecules = 0;
  if (strcmp(param[2], "number") != 0) {
    PRINT_INPUT_ERROR("Expected 'number' keyword after filename.");
  }
  if (!is_valid_int(param[3], &num_molecules)) {
    PRINT_INPUT_ERROR("Number of molecules should be an integer.");
  }
  if (num_molecules <= 0) {
    PRINT_INPUT_ERROR("Number of molecules should be > 0.");
  }

  // Parse velocity ranges
  double velocity_ranges[6];
  if (strcmp(param[4], "velocity") != 0) {
    PRINT_INPUT_ERROR("Expected 'velocity' keyword after number.");
  }
  for (int i = 0; i < 6; i++) {
    if (!is_valid_real(param[5 + i], velocity_ranges + i)) {
      PRINT_INPUT_ERROR("Velocity range values should be numbers.");
    }
  }
  // Check velocity ranges: min <= max
  if (velocity_ranges[0] > velocity_ranges[1] || velocity_ranges[2] > velocity_ranges[3] ||
      velocity_ranges[4] > velocity_ranges[5]) {
    PRINT_INPUT_ERROR("Velocity min should be <= max for each component.");
  }

  // Parse position ranges
  double position_ranges[6];
  if (strcmp(param[11], "position") != 0) {
    PRINT_INPUT_ERROR("Expected 'position' keyword after velocity.");
  }
  for (int i = 0; i < 6; i++) {
    if (!is_valid_real(param[12 + i], position_ranges + i)) {
      PRINT_INPUT_ERROR("Position range values should be numbers.");
    }
  }
  // Check position ranges: min <= max
  if (position_ranges[0] > position_ranges[1] || position_ranges[2] > position_ranges[3] ||
      position_ranges[4] > position_ranges[5]) {
    PRINT_INPUT_ERROR("Position min should be <= max for each component.");
  }

  // Get allowed atom symbols from potential (needed for read_molecule_xyz)
  std::vector<std::string> allowed_symbols = get_allowed_symbols();

  // Read molecule from xyz file (reusing read_xyz.cu functions)
  std::vector<std::string> molecule_symbols;
  std::vector<double> molecule_positions;
  std::vector<double> molecule_masses;
  read_molecule_xyz(filename, allowed_symbols, molecule_symbols, molecule_positions, molecule_masses);
  int atoms_per_molecule = molecule_symbols.size();

  // Debug output: verify molecule reading
  print_line_1();
  printf("Successfully read molecule from %s:\n", filename.c_str());
  printf("  Number of atoms in molecule: %d\n", atoms_per_molecule);
  printf("  Atom details:\n");
  for (int n = 0; n < atoms_per_molecule; n++) {
    printf("    Atom %d: %s, mass = %.6f, position = (%.6f, %.6f, %.6f)\n",
           n,
           molecule_symbols[n].c_str(),
           molecule_masses[n],
           molecule_positions[n + atoms_per_molecule * 0],
           molecule_positions[n + atoms_per_molecule * 1],
           molecule_positions[n + atoms_per_molecule * 2]);
  }
  print_line_2();

  // Calculate center of mass of molecule (relative to first atom)
  double com[3] = {0.0, 0.0, 0.0};
  double total_mass = 0.0;
  for (int n = 0; n < atoms_per_molecule; n++) {
    total_mass += molecule_masses[n];
    for (int d = 0; d < 3; d++) {
      com[d] += molecule_masses[n] * molecule_positions[n + atoms_per_molecule * d];
    }
  }
  for (int d = 0; d < 3; d++) {
    com[d] /= total_mass;
  }

  // Debug output: center of mass
  print_line_1();
  printf("Molecule center of mass: (%.6f, %.6f, %.6f)\n", com[0], com[1], com[2]);
  printf("Total molecule mass: %.6f\n", total_mass);
  print_line_2();

  // Calculate relative positions (relative to center of mass)
  std::vector<double> relative_positions(atoms_per_molecule * 3);
  for (int n = 0; n < atoms_per_molecule; n++) {
    for (int d = 0; d < 3; d++) {
      relative_positions[n + atoms_per_molecule * d] =
        molecule_positions[n + atoms_per_molecule * d] - com[d];
    }
  }

  // Debug output: relative positions
  print_line_1();
  printf("Relative positions (relative to center of mass):\n");
  for (int n = 0; n < atoms_per_molecule; n++) {
    printf("  Atom %d (%s): (%.6f, %.6f, %.6f)\n",
           n,
           molecule_symbols[n].c_str(),
           relative_positions[n + atoms_per_molecule * 0],
           relative_positions[n + atoms_per_molecule * 1],
           relative_positions[n + atoms_per_molecule * 2]);
  }
  print_line_2();

  const int atoms_added = num_molecules * atoms_per_molecule;
  NewAtoms added;
  added.n = atoms_added;
  added.type.resize(atoms_added);
  added.mass.resize(atoms_added);
  added.charge.resize(atoms_added, 0.0f);
  added.symbol.resize(atoms_added);
  added.position.resize(atoms_added * 3);
  added.velocity.resize(atoms_added * 3);

  std::random_device rd;
  std::mt19937 gen(rd());

  std::optional<std::uniform_real_distribution<double>> dist_x_pos;
  std::optional<std::uniform_real_distribution<double>> dist_y_pos;
  std::optional<std::uniform_real_distribution<double>> dist_z_pos;
  std::optional<std::uniform_real_distribution<double>> dist_vx;
  std::optional<std::uniform_real_distribution<double>> dist_vy;
  std::optional<std::uniform_real_distribution<double>> dist_vz;

  if (position_ranges[0] < position_ranges[1]) {
    dist_x_pos.emplace(position_ranges[0], position_ranges[1]);
  }
  if (position_ranges[2] < position_ranges[3]) {
    dist_y_pos.emplace(position_ranges[2], position_ranges[3]);
  }
  if (position_ranges[4] < position_ranges[5]) {
    dist_z_pos.emplace(position_ranges[4], position_ranges[5]);
  }
  if (velocity_ranges[0] < velocity_ranges[1]) {
    dist_vx.emplace(velocity_ranges[0], velocity_ranges[1]);
  }
  if (velocity_ranges[2] < velocity_ranges[3]) {
    dist_vy.emplace(velocity_ranges[2], velocity_ranges[3]);
  }
  if (velocity_ranges[4] < velocity_ranges[5]) {
    dist_vz.emplace(velocity_ranges[4], velocity_ranges[5]);
  }

  int cur = 0;
  for (int mol = 0; mol < num_molecules; mol++) {
    double center_pos[3] = {
      dist_x_pos ? (*dist_x_pos)(gen) : position_ranges[0],
      dist_y_pos ? (*dist_y_pos)(gen) : position_ranges[2],
      dist_z_pos ? (*dist_z_pos)(gen) : position_ranges[4]};
    double mol_velocity[3] = {
      dist_vx ? (*dist_vx)(gen) : velocity_ranges[0],
      dist_vy ? (*dist_vy)(gen) : velocity_ranges[2],
      dist_vz ? (*dist_vz)(gen) : velocity_ranges[4]};

    for (int n = 0; n < atoms_per_molecule; n++) {
      added.symbol[cur] = molecule_symbols[n];
      added.type[cur] = get_atom_type(molecule_symbols[n], allowed_symbols);
      added.mass[cur] = molecule_masses[n];
      added.charge[cur] = 0.0f;
      for (int d = 0; d < 3; d++) {
        added.position[cur + atoms_added * d] =
          center_pos[d] + relative_positions[n + atoms_per_molecule * d];
        added.velocity[cur + atoms_added * d] = mol_velocity[d];
      }
      cur++;
    }
  }

  if ((int)atoms.cpu_type_size.size() < (int)allowed_symbols.size()) {
    atoms.cpu_type_size.resize(allowed_symbols.size(), 0);
  }

  AtomMutation::append_atoms(atoms, groups, thermo, force, added);
  const int N_new = atoms.number_of_atoms;

  print_line_1();
  printf("Deposited %d molecules from %s.\n", num_molecules, filename.c_str());
  printf("Number of atoms per molecule: %d\n", atoms_per_molecule);
  printf("Total atoms added: %d\n", atoms_added);
  printf("Total number of atoms: %d\n", N_new);
  printf("Position range: x[%.2f, %.2f] y[%.2f, %.2f] z[%.2f, %.2f]\n",
         position_ranges[0],
         position_ranges[1],
         position_ranges[2],
         position_ranges[3],
         position_ranges[4],
         position_ranges[5]);
  printf("Velocity range: vx[%.8f, %.8f] vy[%.8f, %.8f] vz[%.8f, %.8f]\n",
         velocity_ranges[0],
         velocity_ranges[1],
         velocity_ranges[2],
         velocity_ranges[3],
         velocity_ranges[4],
         velocity_ranges[5]);
  int number_of_types = atoms.cpu_type_size.size();
  if (number_of_types == 1) {
    printf("There is only one atom type.\n");
  } else {
    printf("There are %d atom types.\n", number_of_types);
  }
  for (int m = 0; m < number_of_types; m++) {
    printf("    %d atoms of type %d.\n", atoms.cpu_type_size[m], m);
  }
  print_line_2();
}

static bool is_six_reals(const char** param)
{
  double dummy;
  for (int i = 2; i < 8; ++i) {
    if (!is_valid_real(param[i], &dummy)) {
      return false;
    }
  }
  return true;
}

void Deposit(
  const char** param,
  int num_param,
  Box& box,
  Atom& atoms,
  std::vector<Group>& groups,
  GPU_Vector<double>& thermo,
  Force& force)
{
  const auto time_begin = std::chrono::high_resolution_clock::now();
  if (num_param < 2) {
    PRINT_INPUT_ERROR("deposit needs at least a species or a molecule filename.");
  }

  if (num_param == 8 && is_six_reals(param)) {
    deposit_one_atom_xyz(param, num_param, atoms, groups, thermo, force);
  } else if (
    num_param >= 3 &&
    (strcmp(param[2], "position") == 0 || strcmp(param[2], "velocity") == 0 ||
     strcmp(param[2], "surface") == 0 || strcmp(param[2], "offset") == 0)) {
    deposit_one_atom_keywords(param, num_param, box, atoms, groups, thermo, force);
  } else {
    deposit_from_molecule_file(param, num_param, atoms, groups, thermo, force);
  }

  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_used = time_finish - time_begin;
  printf("Time used for deposit = %g second.\n", time_used.count());
}

