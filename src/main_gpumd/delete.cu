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
#include "atom_mutation.cuh"
#include "../../extensions/common/topology/connectivity_bfs.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>

void Delete(
  const char** param,
  int num_param,
  Box& box,
  Atom& atoms,
  std::vector<Group>& groups,
  GPU_Vector<double>& thermo,
  Force& force)
{
  if (num_param < 2) {
    PRINT_INPUT_ERROR("Delete should have at least 2 parameters: style and style-specific parameters.");
  }

  int N_old = atoms.number_of_atoms;
  if (N_old == 0) {
    PRINT_INPUT_ERROR("No atoms to delete.");
  }

  std::vector<char> to_delete(N_old, 0);
  int num_to_delete = 0;
  std::string style = param[1];

  if (style == "element") {
    if (num_param != 3) {
      PRINT_INPUT_ERROR("Delete element should have 3 parameters: delete element element_symbol.");
    }
    std::string element_symbol = param[2];
    for (int n = 0; n < N_old; n++) {
      if (atoms.cpu_atom_symbol[n] == element_symbol) {
        to_delete[n] = 1;
        num_to_delete++;
      }
    }

    print_line_1();
    printf("Deleting atoms with element symbol: %s\n", element_symbol.c_str());
    printf("Number of atoms to delete: %d\n", num_to_delete);
    print_line_2();
  } else if (style == "cubic") {
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
    if (bounds[0] > bounds[1] || bounds[2] > bounds[3] || bounds[4] > bounds[5]) {
      PRINT_INPUT_ERROR("Cubic min should be <= max for each component.");
    }

    for (int n = 0; n < N_old; n++) {
      double x = atoms.cpu_position_per_atom[n + N_old * 0];
      double y = atoms.cpu_position_per_atom[n + N_old * 1];
      double z = atoms.cpu_position_per_atom[n + N_old * 2];
      if (x >= bounds[0] && x <= bounds[1] && y >= bounds[2] && y <= bounds[3] &&
          z >= bounds[4] && z <= bounds[5]) {
        to_delete[n] = 1;
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
  } else if (style == "disconnected") {
    if (num_param != 4 || strcmp(param[2], "cutoff") != 0) {
      PRINT_INPUT_ERROR("Usage: delete disconnected cutoff <r>");
    }
    double cutoff = 0.0;
    if (!is_valid_real(param[3], &cutoff)) {
      PRINT_INPUT_ERROR("delete disconnected cutoff should be a number.");
    }
    if (cutoff <= 0.0) {
      PRINT_INPUT_ERROR("delete disconnected cutoff should be > 0.");
    }

    std::vector<char> keep;
    find_main_component_from_min_z(atoms, box, cutoff, keep);
    for (int n = 0; n < N_old; n++) {
      if (!keep[n]) {
        to_delete[n] = 1;
        num_to_delete++;
      }
    }

    print_line_1();
    printf("Deleting atoms disconnected from the lowest-z component.\n");
    printf("Cutoff = %.6f\n", cutoff);
    printf("Number of atoms to delete: %d\n", num_to_delete);
    print_line_2();
  } else {
    PRINT_INPUT_ERROR("Unknown delete style. Supported styles: element, cubic, disconnected");
  }

  if (num_to_delete == 0) {
    print_line_1();
    printf("No atoms to delete. System unchanged.\n");
    print_line_2();
    return;
  }

  AtomMutation::remove_atoms(atoms, groups, thermo, force, to_delete);
  const int N_new = atoms.number_of_atoms;

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
