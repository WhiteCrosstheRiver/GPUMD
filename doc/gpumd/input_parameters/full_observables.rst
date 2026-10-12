.. _kw_full_observables:
.. index::
   single: full_observables (keyword in run.in)

:attr:`full_observables`
========================

Make the next run compute full-system per-atom energies and virials, i.e. switch the
:ref:`fix <kw_fix>` skip path off for that run. Use it for runs whose outputs need them
(``dump_xyz`` with potential, virial or stress, and ``thermo.out`` potential energy and pressure),
for example a final relaxation or a snapshot export after a deposition loop that used the skip
path.

Syntax
------
::

  full_observables

This keyword is not propagating: it only affects the run that follows it.
