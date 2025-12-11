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

#include "delete.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>

void Delete(const char** param, int num_param, Atom& atoms, std::vector<Group>& groups)
{
  if (num_param < 2) {
    PRINT_INPUT_ERROR("Delete should have at least 2 parameters: style and style-specific parameters.");
  }

  int N_old = atoms.number_of_atoms;
  if (N_old == 0) {
    PRINT_INPUT_ERROR("No atoms to delete.");
  }

  // Determine which atoms to delete
  std::vector<bool> to_delete(N_old, false);
  int num_to_delete = 0;

  // Parse style
  std::string style = param[1];

  if (style == "element") {
    // Format: delete element C
    if (num_param != 3) {
      PRINT_INPUT_ERROR("Delete element should have 3 parameters: delete element element_symbol.");
    }
    std::string element_symbol = param[2];

    // Mark atoms with matching element symbol for deletion
    for (int n = 0; n < N_old; n++) {
      if (atoms.cpu_atom_symbol[n] == element_symbol) {
        to_delete[n] = true;
        num_to_delete++;
      }
    }

    print_line_1();
    printf("Deleting atoms with element symbol: %s\n", element_symbol.c_str());
    printf("Number of atoms to delete: %d\n", num_to_delete);
    print_line_2();
  } else if (style == "cubic") {
    // Format: delete cubic xmin xmax ymin ymax zmin zmax
    if (num_param != 8) {
      PRINT_INPUT_ERROR(
        "Delete cubic should have 7 parameters: cubic xmin xmax ymin ymax zmin zmax");
    }

    double bounds[6];
    for (int i = 0; i < 6; i++) {
      if (!is_valid_real(param[2 + i], bounds + i)) {
        PRINT_INPUT_ERROR("Cubic bounds should be numbers.");
      }
    }
    // Check bounds: min <= max
    if (bounds[0] > bounds[1] || bounds[2] > bounds[3] || bounds[4] > bounds[5]) {
      PRINT_INPUT_ERROR("Cubic min should be <= max for each component.");
    }

    // Mark atoms within the cubic region for deletion
    for (int n = 0; n < N_old; n++) {
      double x = atoms.cpu_position_per_atom[n + N_old * 0];
      double y = atoms.cpu_position_per_atom[n + N_old * 1];
      double z = atoms.cpu_position_per_atom[n + N_old * 2];

      if (x >= bounds[0] && x <= bounds[1] && y >= bounds[2] && y <= bounds[3] &&
          z >= bounds[4] && z <= bounds[5]) {
        to_delete[n] = true;
        num_to_delete++;
      }
    }

    print_line_1();
    printf("Deleting atoms in cubic region:\n");
    printf("  x: [%.6f, %.6f]\n", bounds[0], bounds[1]);
    printf("  y: [%.6f, %.6f]\n", bounds[2], bounds[3]);
    printf("  z: [%.6f, %.6f]\n", bounds[4], bounds[5]);
    printf("Number of atoms to delete: %d\n", num_to_delete);
    print_line_2();
  } else {
    PRINT_INPUT_ERROR("Unknown delete style. Supported styles: element, cubic");
  }

  if (num_to_delete == 0) {
    print_line_1();
    printf("No atoms to delete. System unchanged.\n");
    print_line_2();
    return;
  }

  if (num_to_delete >= N_old) {
    PRINT_INPUT_ERROR("Cannot delete all atoms. At least one atom must remain.");
  }

  // Create new atom object with remaining atoms
  int N_new = N_old - num_to_delete;
  Atom new_atoms;
  new_atoms.number_of_atoms = N_new;
  new_atoms.cpu_type.resize(N_new);
  new_atoms.cpu_mass.resize(N_new);
  new_atoms.cpu_charge.resize(N_new);
  new_atoms.cpu_atom_symbol.resize(N_new);
  new_atoms.cpu_position_per_atom.resize(N_new * 3);
  new_atoms.cpu_velocity_per_atom.resize(N_new * 3);

  // Create new groups
  std::vector<Group> new_groups;
  new_groups.resize(groups.size());
  for (size_t m = 0; m < groups.size(); m++) {
    new_groups[m].number = groups[m].number;
    new_groups[m].cpu_label.resize(N_new);
  }

  // Copy atoms that are not marked for deletion
  int cur = 0;
  for (int n = 0; n < N_old; n++) {
    if (!to_delete[n]) {
      new_atoms.cpu_type[cur] = atoms.cpu_type[n];
      new_atoms.cpu_mass[cur] = atoms.cpu_mass[n];
      new_atoms.cpu_charge[cur] = atoms.cpu_charge[n];
      new_atoms.cpu_atom_symbol[cur] = atoms.cpu_atom_symbol[n];
      for (int d = 0; d < 3; d++) {
        new_atoms.cpu_position_per_atom[cur + N_new * d] =
          atoms.cpu_position_per_atom[n + N_old * d];
        new_atoms.cpu_velocity_per_atom[cur + N_new * d] =
          atoms.cpu_velocity_per_atom[n + N_old * d];
      }
      // Copy group labels
      for (size_t m = 0; m < groups.size(); m++) {
        new_groups[m].cpu_label[cur] = groups[m].cpu_label[n];
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
  atoms.cpu_charge.assign(new_atoms.cpu_charge.begin(), new_atoms.cpu_charge.end());
  atoms.cpu_atom_symbol.assign(new_atoms.cpu_atom_symbol.begin(), new_atoms.cpu_atom_symbol.end());
  atoms.cpu_position_per_atom.assign(
    new_atoms.cpu_position_per_atom.begin(), new_atoms.cpu_position_per_atom.end());
  atoms.cpu_velocity_per_atom.assign(
    new_atoms.cpu_velocity_per_atom.begin(), new_atoms.cpu_velocity_per_atom.end());

  // Update type sizes (same pattern as replicate.cu: recalculate)
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
  printf("Deleted %d atoms.\n", num_to_delete);
  printf("Remaining number of atoms: %d\n", N_new);
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

