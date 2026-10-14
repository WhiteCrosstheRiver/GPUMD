.. _kw_variable:
.. index::
   single: variable (keyword in run.in)

:attr:`variable`
================

This keyword defines an equal-style variable: a named formula that is **not** evaluated when the line is read.

It is meant for input-script arithmetic, box lengths, and random numbers that later commands can either snapshot or re-evaluate.
Ordinary MD scripts do not need this keyword.

GPUMD does not support ``\`` line continuation; write one command on one line.

Syntax
------

::

  variable <name> equal <formula>

``name`` is ``[A-Za-z_][A-Za-z0-9_]*``. Equal-style variables may be redefined. A name that is currently used by an enclosing :ref:`for <kw_for>` loop is an error.

The formula is the rest of the line with whitespace removed. V1 supports:

* numbers, ``()``, ``+ - * / ^``, unary minus (``-2^2`` is ``4``)
* ``PI``, ``lx``, ``ly``, ``lz``, ``vol``
  (``lx`` is the length of lattice vector :math:`\mathbf{a}`, likewise ``ly``/``lz``)
* ``v_name``: another equal-style variable, or a numeric :ref:`for <kw_for>` value
* ``random(lo,hi,seed)``: uniform in ``[lo, hi)``. One ``mt19937`` for the whole script; the first seed initializes it and later seeds are ignored.

Immediate vs delayed
--------------------

This is the LAMMPS ``$`` / ``v_`` split.

* ``${name}`` and ``$(formula)`` are expanded **before** the original command parser runs. The command only sees a number string. For a for-loop name, ``${i}`` is the current string. For an equal-style name, ``${x}`` evaluates the formula **now** and freezes that number.
* ``v_name`` is **not** expanded by the control layer. Only commands that accept a delayed slot evaluate it, and they do so at use time.

Numeric arguments in named ``deposit`` source configuration accept delayed ``v_name``
tokens (placement coordinates, region bounds, velocity, and so on). Each token is
evaluated when that configuration line executes. A later ``emit`` does not
re-evaluate configuration values; repeat the configuration line inside a loop
if its value should change for each pulse.

Examples
--------

Snapshot vs live box length::

  variable tmp equal lx
  variable L0 equal ${tmp}
  variable L1 equal v_tmp
  change_box 1.0 0 0
  deposit snap placement point at ${L0} 0
  deposit live placement point at v_L1 0

After ``change_box``, ``${L0}`` is still the original ``lx``; ``v_L1`` uses
the new ``lx`` when the ``live`` placement line executes. These lines only set
placement; define the other required source facets before emitting.

New random XY for each pulse::

  variable rx equal random(0,lx,12345)
  variable ry equal random(0,ly,12345)
  deposit beam particle atom Si
  deposit beam region surface normal 0 0 1 bounds 0 $(lx) 0 $(ly) height global gap 20 scope all
  deposit beam velocity vector 0 0 -0.001 units angstrom/fs
  for p range 1 500
      deposit beam placement point at v_rx v_ry
      deposit beam emit number 1
      ensemble nvt_ber 800 800 40
      run 400
  end

Both ``${rx}`` and ``v_rx`` resolve once per placement configuration, not once
per emitted atom. Use ``placement random`` or ``placement gaussian`` when
particles within one emit need separately sampled positions. The source RNG
persists across emits; configure its ``seed`` once rather than resetting it
inside the loop.

Anonymous immediate formula::

  deposit beam placement point at $(lx/2) $(ly/2)

Caveats
-------

* ``potential`` must not use ``${}`` / ``$()``; some setup code scans ``run.in`` before expansion.
* ``$i`` without braces is not recognized. Use ``${i}``.
* Most GPUMD keywords still only accept numbers after ``${}`` expansion. ``v_name`` outside named ``deposit`` numeric slots above is passed through as a literal token and will fail as "not a number".
* atom / vector / python / file / loop / index styles are not implemented.
