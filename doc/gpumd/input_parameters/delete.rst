.. _kw_delete:
.. index::
   single: delete (keyword in run.in)

:attr:`delete`
==============

This keyword removes atoms from the current configuration.

The ``element`` and ``cubic`` styles are general.
The ``disconnected`` style is only for special deposition or sputtering setups that need to drop flying or isolated atoms.
Ordinary MD simulations do not need ``disconnected``.

Syntax
------

By element symbol::

  delete element <symbol>

By cubic region::

  delete cubic <xmin> <xmax> <ymin> <ymax> <zmin> <zmax>

Drop atoms that are not connected to the substrate (special-purpose; off unless written)::

  delete disconnected cutoff <r>

``disconnected`` builds a graph of atom pairs within distance :math:`r` (XY periodic, :math:`z` not periodic), keeps the component that contains the lowest-:math:`z` atom, and deletes the rest.
It is never run from :ref:`deposit <kw_deposit>`; write it explicitly if you need it.

Examples
--------

::

  delete element C
  delete cubic 8.7 8.9 8.7 8.9 15.9 16.1

Deposition loop that also drops disconnected atoms (special-purpose)::

  for i range 1 500
      deposit Ge position gaussian 0 0 8.33 velocity gaussian 0.007 5 surface local 5.0 offset antivel 2.6
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
