#include "extensions/common/surface/substrate_shell_v2.cuh"
#include <cassert>
#include <cmath>

static Box make_box(double lz)
{
  Box box{};
  box.pbc_z = 0;
  box.cpu_h[0] = box.cpu_h[4] = 10.0;
  box.cpu_h[8] = lz;
  box.cpu_h[9] = box.cpu_h[13] = 0.1;
  box.cpu_h[17] = 1.0 / lz;
  return box;
}

int main()
{
  Atom atom;
  atom.number_of_atoms = 3;
  atom.cpu_position_per_atom = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0};
  shell2::Params p;
  p.axis = 2;
  p.cutoff = 1.3;
  p.offset = 1.1;
  p.mode = shell2::Mode::Grow;
  p.connect = shell2::Connect::Frozen;
  p.every = 3;
  shell2::GrowState st;
  shell2::Report rep;

  Box box = make_box(10.0);
  shell2::update_grow(atom, box, p, nullptr, st, rep);
  assert(rep.kind == shell2::Report::Full);
  assert(atom.cpu_shell_frozen[0] && !atom.cpu_shell_frozen[1]);

  atom.cpu_position_per_atom[7] = 1.2;
  atom.cpu_position_per_atom[8] = 2.4;
  box = make_box(12.0);
  shell2::update_grow(atom, box, p, nullptr, st, rep);
  assert(rep.kind == shell2::Report::Full);
  assert(atom.cpu_shell_frozen[0] && atom.cpu_shell_frozen[1]);
  assert(std::fabs(st.frozen_top[st.grid.cell_of(1.0, 1.0)] - 1.2) < 1.0e-12);

  atom.cpu_position_per_atom[7] = 1.0;
  atom.cpu_position_per_atom[8] = 2.0;
  box = make_box(10.0);
  shell2::update_grow(atom, box, p, nullptr, st, rep);
  assert(rep.kind == shell2::Report::Full);
  assert(atom.cpu_shell_frozen[1]);
  return 0;
}
