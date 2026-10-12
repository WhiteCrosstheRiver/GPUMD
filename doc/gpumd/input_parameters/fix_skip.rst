.. _kw_fix_skip:
.. index::
   single: fix_skip (keyword in run.in)

:attr:`fix_skip`
================

Tune the skip path that :ref:`fix <kw_fix>` enables: potentials that support it (NEP, LJ)
compute forces only for unfixed atoms and their influence halo. The defaults reproduce the
behaviour before this keyword existed.

Syntax
------
::

  fix_skip [min_frozen <f>] [rebuild <K>] [skin <s>] [cache_alloc 0|1]

* ``min_frozen <f>``: when fewer than a fraction :math:`f` of the atoms are frozen, compute
  all atoms instead (same active forces, less bookkeeping). Default 0.
* ``rebuild <K>`` and ``skin <s>`` (NEP): build the influence halo with radius
  :math:`r_c + s` and reuse it for up to :math:`K` steps. Every step a small kernel checks the
  largest displacement of an unfixed atom since the build; the halo is rebuilt as soon as it
  reaches :math:`s`, so the reused halo always contains every atom the active forces depend
  on. ``rebuild > 1`` requires ``skin > 0``. Default ``rebuild 1``, ``skin 0`` (rebuild every
  step).
* ``cache_alloc 1``: keep the Thrust scratch storage of the skip path between calls instead of
  ``cudaMalloc``/``cudaFree`` (which synchronizes the device) on every call. Default 0.

The active set itself is built once per run (group labels and the atom count cannot change
inside a run).

This keyword is propagating: the settings stay until the next ``fix_skip`` line.

Example
-------
Deposition with a sliding frozen zone (:ref:`fix shell mode grow <kw_fix>`)::

  fix_skip min_frozen 0.3 rebuild 20 skin 1.0 cache_alloc 1
