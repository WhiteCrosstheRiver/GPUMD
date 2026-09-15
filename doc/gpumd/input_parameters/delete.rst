.. _kw_delete:
.. index::
   single: delete (keyword in run.in)

:attr:`delete`
==============

This keyword removes atoms from the current configuration.

The ``element`` and ``cubic`` styles are general.
The ``disconnected`` and ``isolated`` styles are only for special deposition or sputtering setups that need to drop flying atoms.
Ordinary MD simulations do not need them.

``delete`` never removes atoms frozen by the most recent :ref:`fix <kw_fix>` (the mask is kept after ``run``).

Syntax
------

By element symbol::

  delete element <symbol>

By cubic region::

  delete cubic <xmin> <xmax> <ymin> <ymax> <zmin> <zmax>

Drop atoms that are not connected to the substrate (special-purpose; off unless written)::

  delete disconnected cutoff <r>

Drop undercoordinated atoms without a full-system BFS (fast; special-purpose)::

  delete isolated cutoff <r> [type <symbol> ...] [only|full|selected <symbol> ...] [expr]

``expr`` is a Boolean combination of neighbor counts inside the cutoff (XY periodic, :math:`z` open).
``X`` means any neighbor. Parentheses, ``and``, ``or``, and ``not`` are allowed.
Spaces around ``(``, ``)``, and the comparison operators are safest.

::

  coord <species|X> [species ...] <op> <n>
  coord Si Ge < 2     # Si and Ge neighbors counted together
  coord <n>           # same as coord X < n
  cutoff <n> <r>      # same as cutoff <r> coord X < n  (no other expr)

``op`` is one of ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``.
If ``op`` is omitted (``coord Si 2``), it means ``<``.

Who is deleted after an atom matches ``expr``:

* ``only`` (default): that atom.
* ``full``: that atom and **every** neighbor in the cutoff.
* ``selected <symbol> ...``: that atom and neighbors of those species. ``selected X`` is the same as ``full``.

``type`` limits **candidates** that are tested against ``expr``. Neighbors of every species still count in ``coord``.
Use ``selected F`` when a Si still has two Si neighbors but many F: delete the matching Si and the nearby F, not the neighboring crystal Si.

``disconnected`` builds a graph of atom pairs within distance :math:`r` (XY periodic, :math:`z` not periodic), keeps the component that contains the lowest-:math:`z` atom, and deletes the rest.
It is never run from :ref:`deposit <kw_deposit>`; write it explicitly if you need it.

``isolated`` counts neighbors within :math:`r` and deletes according to ``expr`` and ``only`` / ``full`` / ``selected``.
Default ``expr`` is ``coord X < 1`` (zero neighbors).
Use this when sputtered atoms are already isolated and a BFS of the whole slab is too slow.
A flying molecule whose every atom fails ``expr`` is not removed in ``only`` mode; use ``full``, ``selected``, or ``disconnected``.

Examples
--------

::

  delete element C
  delete cubic 8.7 8.9 8.7 8.9 15.9 16.1
  delete isolated type Si cutoff 2.8 full coord Si Ge < 2
  delete isolated type Ge cutoff 2.8 full coord Si Ge < 2

Deposition loop that also drops disconnected atoms (special-purpose)::

  for i range 1 500
      deposit gaussian atom Ge number 1 origin 0 0 sigma 8.33 direction axis -z surface local radius 5.0 gap 2.6 spread gaussian 5 velocity constant 0.007 seed 1
      run 100
      delete disconnected cutoff 2.9
  end

Caveats
-------
* Changing the number of atoms while PIMD beads exist is not supported.
* :ref:`electron_stop <kw_electron_stop>` and ``add_random_force`` size some buffers when they are parsed.
  Issue those keywords after ``delete`` in the same run block, or they will not see the new number of atoms.
* ``delete disconnected`` runs a full-system BFS (cell-list neighbor search plus a flood fill from the lowest-:math:`z` atom) **every time** the command is issued.
  The cost grows with the number of atoms and is much larger than a single-atom :ref:`deposit <kw_deposit>`.
  Do not call it every MD step.
  If you use it, put it after a block of :ref:`run <kw_run>` steps, not inside a tight inner loop.
* ``delete isolated`` only builds a cell list and counts neighbors of the candidate atoms. It is the cheap exhaust for atoms that are already isolated.
