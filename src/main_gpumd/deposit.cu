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
#include "model/read_xyz.cuh"
#include "utilities/common.cuh"
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <optional>
#include <random>
#include <string>

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

void Deposit(
  const char** param, int num_param,  Atom& atoms, std::vector<Group>& groups)
{
  // Parse parameters
  // Format: deposit filename.xyz number N velocity vx_min vx_max vy_min vy_max vz_min vz_max position x_min x_max y_min y_max z_min z_max
  // Expected: deposit CF4.xyz number 10 velocity 0.1 0.2 0.2 0.3 -0.4 -0.3 position 5 10 5 10 30 35

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

  // Get current number of atoms
  int N_old = atoms.number_of_atoms;
  int N_new = N_old + num_molecules * atoms_per_molecule;

  // Create temporary atom object (same pattern as replicate.cu)
  Atom new_atoms;
  new_atoms.number_of_atoms = N_new;
  new_atoms.cpu_type.resize(N_new);
  new_atoms.cpu_mass.resize(N_new);
  new_atoms.cpu_atom_symbol.resize(N_new);
  new_atoms.cpu_position_per_atom.resize(N_new * 3);
  new_atoms.cpu_velocity_per_atom.resize(N_new * 3);

  // Copy existing atoms to new_atoms (same pattern as replicate.cu)
  for (int n = 0; n < N_old; n++) {
    new_atoms.cpu_type[n] = atoms.cpu_type[n];
    new_atoms.cpu_mass[n] = atoms.cpu_mass[n];
    new_atoms.cpu_atom_symbol[n] = atoms.cpu_atom_symbol[n];
    for (int d = 0; d < 3; d++) {
      new_atoms.cpu_position_per_atom[n + N_new * d] =
        atoms.cpu_position_per_atom[n + N_old * d];
      new_atoms.cpu_velocity_per_atom[n + N_new * d] =
        atoms.cpu_velocity_per_atom[n + N_old * d];
    }
  }

  // Resize groups (same pattern as replicate.cu)
  std::vector<Group> new_groups;
  new_groups.resize(groups.size());
  for (size_t m = 0; m < groups.size(); m++) {
    new_groups[m].number = groups[m].number;
    new_groups[m].cpu_label.resize(N_new);
    // Copy existing group labels
    for (int n = 0; n < N_old; n++) {
      new_groups[m].cpu_label[n] = groups[m].cpu_label[n];
    }
  }

  // Initialize random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  
  // Create distributions only when min != max, otherwise use fixed values
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

  // Add molecules to new_atoms (same pattern as replicate.cu)
  int cur = N_old;
  for (int mol = 0; mol < num_molecules; mol++) {
    // Random center position for this molecule (or fixed if min == max)
    double center_pos[3] = {
      dist_x_pos ? (*dist_x_pos)(gen) : position_ranges[0],
      dist_y_pos ? (*dist_y_pos)(gen) : position_ranges[2],
      dist_z_pos ? (*dist_z_pos)(gen) : position_ranges[4]
    };

    // Random velocity for this molecule (same for all atoms in the molecule) (or fixed if min == max)
    double mol_velocity[3] = {
      dist_vx ? (*dist_vx)(gen) : velocity_ranges[0],
      dist_vy ? (*dist_vy)(gen) : velocity_ranges[2],
      dist_vz ? (*dist_vz)(gen) : velocity_ranges[4]
    };

    // Add atoms of this molecule
    for (int n = 0; n < atoms_per_molecule; n++) {
      new_atoms.cpu_atom_symbol[cur] = molecule_symbols[n];
      new_atoms.cpu_type[cur] = get_atom_type(molecule_symbols[n], allowed_symbols);
      new_atoms.cpu_mass[cur] = molecule_masses[n];

      // Position = center + relative position
      for (int d = 0; d < 3; d++) {
        new_atoms.cpu_position_per_atom[cur + N_new * d] =
          center_pos[d] + relative_positions[n + atoms_per_molecule * d];
        new_atoms.cpu_velocity_per_atom[cur + N_new * d] = mol_velocity[d];
      }

      // Set new atoms to group 0 by default
      for (size_t m = 0; m < groups.size(); m++) {
        new_groups[m].cpu_label[cur] = 0;
      }
      cur++;
    }
  }

  // Copy to old (same pattern as replicate.cu)
  for (size_t m = 0; m < groups.size(); m++) {
    groups[m].number = new_groups[m].number;
    groups[m].cpu_label.assign(new_groups[m].cpu_label.begin(), new_groups[m].cpu_label.end());
    groups[m].find_size(N_new, m);
    groups[m].find_contents(N_new);
  }
  atoms.number_of_atoms = N_new;
  atoms.cpu_type.assign(new_atoms.cpu_type.begin(), new_atoms.cpu_type.end());
  atoms.cpu_mass.assign(new_atoms.cpu_mass.begin(), new_atoms.cpu_mass.end());
  atoms.cpu_atom_symbol.assign(new_atoms.cpu_atom_symbol.begin(), new_atoms.cpu_atom_symbol.end());
  atoms.cpu_position_per_atom.assign(
    new_atoms.cpu_position_per_atom.begin(), new_atoms.cpu_position_per_atom.end());
  atoms.cpu_velocity_per_atom.assign(
    new_atoms.cpu_velocity_per_atom.begin(), new_atoms.cpu_velocity_per_atom.end());

  // Update type sizes (same pattern as replicate.cu: keep size, recalculate)
  atoms.cpu_type_size.assign(atoms.cpu_type_size.begin(), atoms.cpu_type_size.end());
  // Recalculate type sizes for all atoms
  for (int m = 0; m < (int)atoms.cpu_type_size.size(); m++) {
    atoms.cpu_type_size[m] = 0;
  }
  for (int n = 0; n < N_new; n++) {
    if (atoms.cpu_type[n] >= 0 && atoms.cpu_type[n] < (int)atoms.cpu_type_size.size()) {
      atoms.cpu_type_size[atoms.cpu_type[n]]++;
    }
  }

  // Print information
  print_line_1();
  printf("Deposited %d molecules from %s.\n", num_molecules, filename.c_str());
  printf("Number of atoms per molecule: %d\n", atoms_per_molecule);
  printf("Total atoms added: %d\n", num_molecules * atoms_per_molecule);
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

