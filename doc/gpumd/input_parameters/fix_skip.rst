.. _kw_fix_skip:
.. index::
   single: fix_skip (keyword in run.in)

:attr:`fix_skip`
================

Speed up runs with a :ref:`fixed group <kw_fix>`: potentials that support it (large-box NEP on one
GPU, and LJ) compute only unfixed atoms and the atoms that can reach them. Forces on unfixed atoms
stay exact. Without this keyword every atom is computed.

Syntax
------
::

  fix_skip [off] [min_frozen <f>]

* ``min_frozen <f>``: compute all atoms when fewer than a fraction :math:`f` of them are frozen.
  The default is 0.
* ``off`` turns the keyword off again.

This keyword is propagating: the settings stay until the next ``fix_skip`` line.

While it is active, per-atom energies and virials, and therefore the potential energy and pressure
in ``thermo.out``, omit frozen atoms that are out of reach, and forces on frozen atoms read zero.
It applies only to NVE and standard NVT runs; other ensembles compute every atom automatically. Use
:ref:`full_observables <kw_full_observables>` for a run that needs complete values. Do not combine
it with heat-current or other measurements that read per-atom virials of frozen atoms.
