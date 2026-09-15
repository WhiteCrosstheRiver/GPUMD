.. _kw_if:
.. index::
   single: if (keyword in run.in)

:attr:`if`
==========

This keyword runs a block of GPUMD commands when a comparison is true.
Ordinary MD scripts do not need it.

It is meant for special input-script branching, such as choosing a deposited species.
:ref:`for <kw_for>` / :ref:`variable <kw_variable>` already cover repetition and arithmetic.

GPUMD does not support ``\`` line continuation; write one command on one line.

Syntax
------

::

  if <a> <op> <b>
      <GPUMD commands>
  else
      <GPUMD commands>
  end

``else`` is optional. ``end`` closes the innermost :ref:`for <kw_for>` or ``if``.
Nested ``if`` and ``for`` are allowed.

``op`` is one of ``== != < <= > >=``.
``a`` and ``b`` may be a number, ``${name}``, ``$(formula)``, or a :ref:`for <kw_for>` string.
The comparison is evaluated when the ``if`` is **executed**, not when the file is parsed.

If both sides become numbers after expansion, the comparison is numeric.
Otherwise only ``==`` and ``!=`` are allowed, and they compare strings.

Spaces around ``op`` are required: write ``if ${u} < 0.5``, not ``if ${u}<0.5``.

V1 does not support ``elif``, ``&&``, ``||``, ``while``, ``break``, or ``continue``.

Examples
--------

Numeric then / else::

  if 1 < 2
      dump_xyz -1 0 1 then.xyz
  else
      dump_xyz -1 0 1 else.xyz
  end
  run 1

Random species (binomial, not a fixed 50/50 count)::

  variable u equal random(0,1,12345)
  for p range 1 100
      variable u equal random(0,1,12345)
      if ${u} < 0.5
          deposit gaussian atom Si number 1 origin 0 0 sigma 6.0 direction axis -z surface local radius 5.0 gap 2.5 spread gaussian 5 velocity constant 0.015 seed 12345
      else
          deposit gaussian atom Ge number 1 origin 0 0 sigma 6.0 direction axis -z surface local radius 5.0 gap 2.5 spread gaussian 5 velocity constant 0.015 seed 12345
      end
      run 100
  end

String compare with a for-loop value::

  for sp values Si Ge
      if ${sp} == Si
          deposit gaussian atom Si number 1 origin 0 0 sigma 6.0 direction axis -z surface local radius 5.0 gap 2.5 spread gaussian 5 velocity constant 0.015 seed 12345
      else
          deposit gaussian atom Ge number 1 origin 0 0 sigma 6.0 direction axis -z surface local radius 5.0 gap 2.5 spread gaussian 5 velocity constant 0.015 seed 12345
      end
      run 100
  end

Caveats
-------
* ``< <= > >=`` on a non-numeric operand is an error.
* A lone ``else`` or ``end`` outside ``if`` / ``for`` is an error.
* ``potential`` must stay outside ``if`` and must not use ``${variable}``. Some setup code scans ``run.in`` before expansion.
