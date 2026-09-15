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

``deposit`` numeric values accept delayed ``v_name`` tokens (``origin``, ``region``, ``surface``, ``velocity``, and so on). Each token is evaluated once when that ``deposit`` command runs.

Examples
--------

Snapshot vs live box length::

  variable tmp equal lx
  variable L0 equal ${tmp}
  variable L1 equal v_tmp
  change_box 1.0 0 0
  deposit point atom C origin ${L0} 0 direction axis -z surface fixed 20 velocity constant 0.001
  deposit point atom C origin v_L1 0 direction axis -z surface fixed 20 velocity constant 0.001

After ``change_box``, ``${L0}`` is still the original ``lx``; ``v_L1`` follows the new ``lx``.

Random XY in one pulse (one GPU rebuild)::

  variable rx equal random(0,lx,12345)
  variable ry equal random(0,ly,12345)
  for p range 1 500
      deposit gaussian atom Si number 64 origin v_rx v_ry sigma 0 direction axis -z surface local radius 5.0 gap 2.5 spread gaussian 5 velocity constant 0.015 seed 12345
      run 400
  end

Do **not** write ``origin ${rx} ${ry}`` with ``number > 1``: ``${rx}`` is one number, so every entity in that pulse uses the same center.

Anonymous immediate formula::

  deposit point atom C origin $(lx/2) $(ly/2) direction axis -z surface fixed 20 velocity constant 0.001

Caveats
-------

* ``potential`` must not use ``${}`` / ``$()``; some setup code scans ``run.in`` before expansion.
* ``$i`` without braces is not recognized. Use ``${i}``.
* Most GPUMD keywords still only accept numbers after ``${}`` expansion. ``v_name`` outside the ``deposit`` slots above is passed through as a literal token and will fail as "not a number".
* atom / vector / python / file / loop / index styles are not implemented.
