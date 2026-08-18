.. _kw_fix:
.. index::
   single: fix (keyword in run.in)

:attr:`fix`
===========

This keyword can be used to fix (freeze) a group of atoms

Syntax
------
This keyword requires a single parameter which is the label of the group in which the atoms are to be fixed (velocities and forces are always set to zero such that the atoms in the group do not move).
The full command reads::

  fix <group_label>

Here, the :attr:`group_label` refers to the grouping method 0 defined in the :ref:`simulation model file <model_xyz>`.

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
