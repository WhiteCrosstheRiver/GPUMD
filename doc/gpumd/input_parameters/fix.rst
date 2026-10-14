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

Growth mode and other v2 options
--------------------------------
These keywords are opt-in. Without them ``fix shell`` runs exactly as above (v1)::

  fix shell substrate <x|y|z> cutoff <r> offset <d> [type ...] [region cubic ...]
            [surface radius <R>] [cell <c>] [mode recede|grow]
            [connect full|frozen|none] [every <K>] [reset]

* ``surface radius <R>``: the local surface of an atom is the highest substrate atom whose
  height-map cell lies within :math:`R` (Å) of the atom's cell. ``R = 0`` is the v1 rule (the
  atom's own ``cell`` column only). With thermal displacement the v1 column rule leaves
  isolated atoms unfrozen deep in the bulk (a 1 Å column that happens to hold no higher atom);
  :math:`R \ge 2` Å removes them. Default: 0 in ``mode recede``, 3 in ``mode grow``.
* ``cell <c>``: height-map cell size (Å). Default 1.0 (v1).
* ``mode recede`` (default): classify from scratch at every call; atoms are released when the
  surface recedes (etching).
* ``mode grow``: ratchet for deposition. An atom frozen by an earlier grow call stays frozen;
  only mobile atoms are examined, so the cost scales with the mobile shell, not with N.
  The state lives in ``atom.cpu_shell_frozen`` (appended as 0 by ``deposit``, compacted by
  ``delete``). It is dropped automatically after any ``run`` whose fixed group did not include
  it (for example a final relaxation with only ``fix 1``), because those atoms may then move.
* ``connect full``: substrate = flood fill from the lowest atom (as v1, with lateral cells at
  least ``cutoff`` wide so bonds across the periodic seam are found).
  ``connect frozen`` (grow default): flood fill from the frozen band through mobile atoms only.
  ``connect none``: every eligible atom is substrate (use right after ``delete disconnected``).
* ``every <K>`` (grow only): classify on every K-th call; the other calls re-apply the cached
  frozen set. At 5 atoms per pulse on 42.7 nm\ :sup:`2` the surface rises about 0.02 Å per pulse,
  so ``every 10`` lags the freeze front by about 0.2 Å.
* ``reset`` (grow only): drop the ratchet and classify from scratch at this call.

``fix shell`` now syncs CPU positions from the GPU before classifying (v1 and v2). Previously it
used the positions of the last ``deposit``/``delete``, which are stale after a ``run``.

Deposition loop with a sliding frozen zone::

  deposit beam particle atom Si
  deposit beam placement grid spacing 4 4 select random
  deposit beam region surface normal 0 0 1 bounds 0 $(lx) 0 $(ly) height local radius 6 gap 3 scope all
  deposit beam velocity vector 0 0 -0.0078581525 units angstrom/fs
  deposit beam exclusion distance 2.5
  deposit beam seed 1
  for i range 1 2352
      deposit beam emit number 5
      ensemble nvt_ber 800 800 40.0
      fix 1
      fix shell substrate z cutoff 2.9 offset 20 surface radius 3 mode grow every 10
      dump_thermo 500
      run 500
  end

Choose ``offset`` from the mobility profile of a full-dynamics reference run (atoms deeper
than ``offset`` should not move beyond thermal vibration during the remaining growth), and
validate against that reference before production.

Etch loop::

  deposit etch particle atom F
  deposit etch placement gaussian center $(lx/2) $(ly/2) sigma 30 attempt 100
  deposit etch region surface normal 0 0 1 bounds 0 $(lx) 0 $(ly) height local radius 5 gap 2.5 scope all
  deposit etch velocity vector 0 0 -0.00068759 units angstrom/fs
  deposit etch seed 1
  for i range 1 100
      deposit etch emit number 1
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
Do not call it every MD step: ``mode recede`` runs a full-system flood fill each time it is
issued. ``mode grow`` examines only mobile atoms after its first call.

``fix shell`` replaces grouping method 0. Extra groups for ``compute`` / heat baths should use grouping method 1 or higher.
