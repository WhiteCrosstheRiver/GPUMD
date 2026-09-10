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

``deposit`` V1 delayed slots are ``position <x> <y> <z>`` and ``velocity <vx> <vy> <vz>``. ``position gaussian`` / ``velocity gaussian`` arguments must still be numbers or ``${}``.

Examples
--------

Snapshot vs live box length::

  variable tmp equal lx
  variable L0 equal ${tmp}
  variable L1 equal v_tmp
  change_box 1.0 0 0
  deposit C position ${L0} 0 20 velocity 0 0 -0.001
  deposit C position v_L1 0 20 velocity 0 0 -0.001

After ``change_box``, ``${L0}`` is still the original ``lx``; ``v_L1`` follows the new ``lx``.

Random XY in one pulse (one GPU rebuild)::

  variable rx equal random(0,lx,12345)
  variable ry equal random(0,ly,12345)
  for p range 1 500
      deposit Si number 64 position v_rx v_ry 0 velocity gaussian 0.015 5 surface local 5.0 offset antivel 2.5
      run 400
  end

Do **not** write ``position ${rx} ${ry}`` with ``number > 1``: ``${rx}`` is one number, so every atom in that pulse sits at the same XY.

Anonymous immediate formula::

  deposit C position $(lx/2) $(ly/2) 20 velocity 0 0 -0.001

Caveats
-------

* ``potential`` must not use ``${}`` / ``$()``; some setup code scans ``run.in`` before expansion.
* ``$i`` without braces is not recognized. Use ``${i}``.
* Most GPUMD keywords still only accept numbers after ``${}`` expansion. ``v_name`` outside the ``deposit`` slots above is passed through as a literal token and will fail as "not a number".
* atom / vector / python / file / loop / index styles are not implemented.
