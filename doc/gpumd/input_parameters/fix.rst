.. _kw_fix:
.. index::
   single: fix (keyword in run.in)

:attr:`fix`
===========

This keyword can be used to fix (freeze) a group of atoms

Syntax
------
Freeze grouping-method-0 labels (one or more). Several ``fix`` lines in the same run block are a **union**::

  fix <group_id> [group_id ...]

Freeze by species::

  fix type <symbol> [symbol ...]

Both on one line::

  fix 1 2 type Si

Recompute a receding shear from the current surface (special-purpose; etching / sputtering)::

  fix shell substrate <x|y|z> cutoff <r> offset <d> [type <symbol> ...] [region cubic <xmin> <xmax> <ymin> <ymax> <zmin> <zmax>]

``shell`` / union rewrites grouping method 0 for the integrator: group 0 is unfixed, group 1 is the frozen union.
Model-file group labels are kept internally so later ``fix 1 2`` in a :ref:`for <kw_for>` loop still sees the original sides.
It is never run from :ref:`deposit <kw_deposit>` or :ref:`delete <kw_delete>`.
:ref:`delete <kw_delete>` never removes atoms frozen by ``fix``; the freeze mask is kept after ``run``.

* ``substrate <x|y|z>``: BFS starts from the lowest-coordinate atom along that axis (same connectivity idea as :ref:`delete disconnected <kw_delete>`). The two transverse directions use PBC; the substrate axis does not.
* ``cutoff <r>``: graph edge length for the BFS (Å).
* ``offset <d>``: unfix a shear of thickness :math:`d` (Å) below the local surface of the substrate. Typical values are around 5 Å.
* ``type <symbol> ...``: only these species are the substrate (like :ref:`delete element <kw_delete>`). Species not listed, including incident F, stay unfixed.
* ``region cubic ...``: optional window (same bounds as :ref:`delete cubic <kw_delete>`). Substrate atoms outside the box are frozen; atoms that are not in the substrate stay unfixed.

The local surface is a height map of the substrate in the plane perpendicular to the axis, so a groove along the beam direction keeps its floor in the unfixed shell.

Etch loop::

  for i range 1 100
      deposit gaussian atom F number 1 origin 32.662 40.828 sigma 30 direction axis -z surface local radius 5.0 gap 2.5 spread gaussian 5 velocity constant 0.007 seed 1
      ensemble nvt_ber 300 300 100
      fix 1 2
      fix shell substrate z cutoff 2.4 offset 5.0 type Si Ge
      dump_xyz 0 0 100 surface.xyz
      run 100
      delete disconnected cutoff 2.4
  end

How it is implemented
---------------------
``fix`` only decides which atoms are dynamical degrees of freedom (active)
versus frozen. Integrators skip position updates for the frozen group and
the output forces on those atoms are set to zero.

Each potential decides which extra atoms it still has to evaluate in order
to get correct forces on the active atoms (its influence domain). Pair
potentials such as LJ only need active atoms as force centers. Local
many-body potentials such as NEP also evaluate a thin halo around the
active region. Potentials that do not implement this still compute the
full system, which is always correct.

Frozen atoms remain valid neighbors / geometry sources. Skipping them as
force centers does not mean they are deleted from the neighbor list.

If a run needs a full-system virial (NPT or NEMD heat current), GPUMD
falls back to the full compute so that pressure is not silently wrong.

With the default NVE/NVT path the skip stays on. Active forces are
still correct. Thermodynamic sums are not the full-system values:

* Many-body site energies of deep frozen atoms are omitted. Their
  neighborhood is frozen, so this is a constant offset: :math:`dE/dt`
  of the mobile region is unchanged, but the absolute potential in
  ``thermo.out`` is shifted.
* Pair potentials (LJ) store :math:`u_{ij}/2` on each center. Frozen–frozen
  pairs are omitted and active–frozen pairs are counted only on the
  active atom, so the missing energy can vary with time.
* Per-atom potential in ``dump_xyz`` / ``dump_exyz`` is 0 for deep
  frozen atoms. Output forces on frozen atoms are 0 by construction.

Temperature already excludes the frozen group. Do not use the skipped
NVE/NVT path when you need a conserved full-system energy or a
full-system pressure.

Caveats
-------
This keyword is not propagating, which means that it only affects the simulation within the run it belongs to.
Put ``fix shell`` inside :ref:`for <kw_for>` so the frozen set follows a receding surface.
Do not call it every MD step: it runs a full-system BFS each time it is issued.

``fix shell`` replaces grouping method 0. Extra groups for ``compute`` / heat baths should use grouping method 1 or higher.
