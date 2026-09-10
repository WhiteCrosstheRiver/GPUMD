.. _kw_for:
.. index::
   single: for (keyword in run.in)

:attr:`for`
===========

This keyword repeats a block of GPUMD commands.
Ordinary commands are unchanged and still go through the original command parser after ``${variable}`` expansion.

It is optional. Ordinary MD scripts can keep writing :ref:`ensemble <kw_ensemble>` and :ref:`run <kw_run>` by hand.
Combining ``for`` with :ref:`deposit <kw_deposit>` or :ref:`delete <kw_delete>` is a special deposition workflow and is not needed for normal MD.

Syntax
------

There are two forms. ``end`` closes the nearest ``for`` block.

Integer range (``stop`` is inclusive; ``step`` defaults to ``1``)::

  for <variable> range <start> <stop> [step]
      <GPUMD commands>
  end

Explicit values (integer, float, or string tokens, kept as written)::

  for <variable> values <value1> <value2> ...
      <GPUMD commands>
  end

Inside the block, substitute with ``${variable}``. Nested ``for`` is allowed. Inner blocks may read outer variables. Reusing the same variable name in a nested ``for`` is an error.

``${name}`` also works for :ref:`variable <kw_variable>` equal-style names: the formula is evaluated immediately and the number is substituted. ``$(formula)`` is an anonymous immediate formula. ``v_name`` is not expanded here; see :ref:`variable <kw_variable>`.

Rules for ``range``:

* ``start``, ``stop``, and ``step`` must be integers.
* Omitted ``step`` is ``1`` (it is not inferred from ``start > stop``).
* ``step`` cannot be ``0``.
* ``start > stop`` with ``step > 0``, or ``start < stop`` with ``step < 0``, is an error.

V1 does not support ``while``, ``break``, ``continue``, or assignments. See :ref:`if <kw_if>` for branching.

Examples
--------

Repeat a run::

  for i range 1 10
      run 1000
  end

Reverse range::

  for i range 10 1 -1
      run 1000
  end

Temperature scan::

  for T values 300 500 700
      ensemble nvt_ber ${T} ${T} 100
      run 10000
  end

Nested loops::

  for T values 300 500
      ensemble nvt_ber ${T} ${T} 100
      for cycle range 1 20
          run 5000
      end
  end

A filename token may contain a variable::

  for n range 1 3
      dump_xyz -1 0 1 position_${n}.xyz
      run 100
  end

Repeated deposition (special-purpose; see :ref:`deposit <kw_deposit>`)::

  for i range 1 500
      deposit Ge position gaussian 0 0 8.33 velocity gaussian 0.007 5 surface local 5.0 offset antivel 2.6
      run 100
  end

Caveats
-------
* ``potential`` must stay outside ``for`` and must not use ``${variable}``. Some setup code scans ``run.in`` before variable expansion.
* Dump keywords such as :ref:`dump_thermo <kw_dump_thermo>` are not propagating. Put them inside the loop, before each :ref:`run <kw_run>`, if every iteration should dump.
* ``${name}`` is expanded before the original command parser runs. Equal-style names from :ref:`variable <kw_variable>` are allowed. ``$i`` without braces is not recognized; ``$(formula)`` is.
