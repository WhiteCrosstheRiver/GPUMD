.. _kw_full_observables:
.. index::
   single: full_observables (keyword in run.in)

:attr:`full_observables`
========================

Compute every atom in the next run, even when :ref:`fix_skip <kw_fix_skip>` is on.

Syntax
------
::

  full_observables

This keyword is not propagating: it only affects the run that follows it.
