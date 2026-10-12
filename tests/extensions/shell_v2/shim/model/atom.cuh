// Test shim: the CPU-side fields of Atom (src/model/atom.cuh) that the
// shell code reads. GPU members are omitted so the headers build with g++.
#pragma once
#include <cmath> // box.cuh uses nearbyint
#include <string>
#include <vector>

class Atom
{
public:
  int number_of_atoms = 0;
  std::vector<int> cpu_type;
  std::vector<int> cpu_type_size;
  std::vector<double> cpu_position_per_atom;
  std::vector<double> cpu_velocity_per_atom;
  std::vector<std::string> cpu_atom_symbol;
  std::vector<char> cpu_fixed;
  std::vector<char> cpu_shell_frozen;
};
