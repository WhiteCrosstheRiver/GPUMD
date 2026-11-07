.. _kw_fix:
.. index::
   single: fix (keyword in run.in)

:attr:`fix`
===========

This keyword can be used to fix (freeze) a group of atoms

Syntax
------
This keyword accepts 1 or 2 parameters.
The atoms in the specified group will be fixed (velocities and forces are always set to zero such that the atoms do not move).
The full command reads::

  fix <group_label>
  fix <grouping_method> <group_label>

- If only :attr:`group_label` is given, grouping method 0 is used by default.
- If :attr:`grouping_method` is also specified, the given grouping method defined in the :ref:`simulation model file <model_xyz>` will be used.

Frozen sets from several sources
--------------------------------
These forms build a frozen set, and several of them in one run block form a union::

  fix group <group_label> [group_label ...] [type <symbol> ...]
  fix type <symbol> [symbol ...]
  fix shell substrate <x|y|z> cutoff <r> offset <d> [type <symbol> ...]
            [region cubic <xmin> <xmax> <ymin> <ymax> <zmin> <zmax>]
            [surface radius <R>] [cell <c>] [mode recede|grow]
            [connect full|frozen|none] [every <K>] [reset]

Use ``fix group 1 2`` for a union of groups: ``fix 1 2`` means group 2 of grouping method 1.
A plain ``fix <group_label>`` placed before these forms joins the union.
A plain ``fix`` placed after them is an error.

During the run, grouping method 0 holds the frozen set: group 0 is unfixed and group 1 is frozen.
Its labels from the model file return after the run.
Dump them with ``dump_xyz <interval> <file> group_labels``.

:attr:`fix shell` finds the substrate and freezes it, leaving a shell of thickness :attr:`d` unfixed below the local surface.

* ``substrate <x|y|z>``: the flood fill starts from the lowest atom along that axis.
* ``cutoff <r>``: bond length of the flood fill (Angstrom).
* ``offset <d>``: thickness of the unfixed shell (Angstrom).
* ``type <symbol> ...``: only these species can be substrate. Other species stay unfixed.
* ``region cubic ...``: substrate atoms outside this window are frozen.
* ``surface radius <R>``: the local surface is the highest substrate atom within :attr:`R` of the height-map cell of an atom.
  :attr:`R` = 0 uses the cell of the atom alone. Values of 2 Angstrom or more stop thermal motion from leaving unfrozen atoms deep in the bulk.
  The default is 0 for ``mode recede`` and 3 for ``mode grow``.
* ``cell <c>``: height-map cell size. The default is 1 Angstrom.
* ``mode recede`` (default): classify from scratch at every call, so atoms are released when the surface recedes.
* ``mode grow``: a frozen atom stays frozen, and only mobile atoms are examined after the first call.
  The state is dropped after any run whose fixed group did not include those atoms, because they may have moved.
* ``connect full``: flood fill from the lowest atom.
  ``connect frozen`` (the default for ``mode grow``): flood fill from the frozen band through mobile atoms.
  ``connect none``: every eligible atom is substrate.
* ``every <K>`` (grow only): classify on every :attr:`K`-th call and reuse the frozen set otherwise.
* ``reset`` (grow only): drop the state and classify from scratch.

Choose :attr:`d` from the mobility profile of a reference run with no shell, and check the results against that run before production.

Example
-------

Fix group 0 using the default grouping method 0::

  fix 0

Fix group 2 using grouping method 1::

  fix 1 2

Freeze groups 1 and 2 of grouping method 0 together with all silicon atoms::

  fix group 1 2 type Si

Freeze the substrate of a slab growing along z, leaving a shell of 20 Angstrom::

  fix 1
  fix shell substrate z cutoff 2.9 offset 20 surface radius 3 mode grow every 10

Caveats
-------
* This keyword is not propagating, which means that it only affects the simulation within the run it belongs to.
* When both :attr:`fix` and :ref:`move <kw_move>` are used, they must use the same grouping method.
* :attr:`move` cannot be used with ``fix group``, ``fix type`` or ``fix shell``.
* ``fix group``, ``fix type`` and ``fix shell`` rewrite grouping method 0 during the run, so :ref:`compute <kw_compute>` and heat baths should use grouping method 1 or higher.
* ``fix shell mode recede`` runs a full-system flood fill at every call, so do not issue it for short run blocks.
