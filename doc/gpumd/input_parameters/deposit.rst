.. _kw_deposit:
.. index::
   single: deposit (keyword in run.in)

:attr:`deposit`
===============

This keyword inserts atom(s) or molecule(s) into the current configuration.

It is meant for special simulations that grow or inject atoms during MD, such as molecular-beam deposition.
Ordinary MD simulations do not need this keyword.

GPUMD does not support ``\`` line continuation; write one command on one line.

Syntax
------

::

  deposit <style> <entity> <source> keyword values ...

``style`` is ``grid``, ``random``, ``gaussian``, or ``point``.
``entity`` is ``atom`` or ``molecule``.
``source`` is an element symbol or a molecule xyz filename.

The six layers of the command are independent:

* ``style``: where candidate sites are generated in the surface plane
* ``entity``: what is deposited
* ``direction``: the center flight direction
* ``surface``: where along the surface normal the particle is created
* ``velocity``: the speed
* ``near`` / ``select`` / ``attempt`` / ``seed``: constraints and sampling

Direction, surface, and velocity are always required.

Deposition frame
----------------

The surface plane is defined by a unit normal :math:`\hat{\mathbf n}`.
Grid, random, gaussian, and point sites live in the :math:`u`-:math:`v` plane perpendicular to :math:`\hat{\mathbf n}`.

By default :math:`\hat{\mathbf n}=-\hat{\mathbf d}`, so a simple beam still sets the frame::

  direction axis -z

gives :math:`\hat{\mathbf n}=+z`. Override this with an explicit normal when the beam is not anti-parallel to the surface::

  normal axis +z
  normal axis +x
  normal vector nx ny nz

``direction`` only sets particle velocity::

  direction axis -z
  direction axis -x
  direction vector 0.3 0 -1
  direction target tx ty tz

``direction target`` aims each particle's velocity at ``(tx, ty, tz)`` from its own launch position.
It does not define the surface frame, so ``normal`` is required with ``target``.

If ``basis ax ay az`` is given, that vector is projected onto the plane perpendicular to :math:`\hat{\mathbf n}` to make :math:`\hat{\mathbf u}`, then :math:`\hat{\mathbf v}=\hat{\mathbf n}\times\hat{\mathbf u}`.
Otherwise the Cartesian axis least parallel to :math:`\hat{\mathbf n}` is used.
The final ``u``, ``v``, and ``n`` vectors are always printed.

For ``normal axis +z`` (the default of ``direction axis -z``) this reduces to :math:`u=x`, :math:`v=y`, :math:`n=z`.
A launch coordinate is

.. math::

   \mathbf r = u\hat{\mathbf u} + v\hat{\mathbf v} + H\hat{\mathbf n}

Lateral PBC wrapping of launch positions is applied when :math:`\hat{\mathbf n}` is a Cartesian axis (:math:`\pm x`, :math:`\pm y`, or :math:`\pm z`).
``normal vector`` does not wrap.

Surface
-------

``gap`` is always measured along :math:`\hat{\mathbf n}`, never along a sampled instantaneous velocity.

::

  surface fixed H
  surface global gap D
  surface local radius R gap D

* ``fixed``: :math:`H` is given, so :math:`\mathbf r\cdot\hat{\mathbf n}=H`.
* ``global``: :math:`H_{\max}=\max_i(\mathbf r_i\cdot\hat{\mathbf n})`, then :math:`H=H_{\max}+D`.
* ``local``: among existing atoms whose projection onto the :math:`u`-:math:`v` plane lies within radius ``R`` of the candidate site, take the largest :math:`\mathbf r_i\cdot\hat{\mathbf n}`; if none, use the global maximum. Then add ``D``.

Styles
------

``grid``
  ``region u_min u_max v_min v_max`` and ``spacing du dv``.
  Sites are the inclusive lattice :math:`u=u_{\min}+i\,du` while :math:`u\le u_{\max}` (same for :math:`v`).
  ``number N`` deposits ``N`` entities; ``number all`` deposits one entity on every valid site.
  ``select sequential`` or ``select random`` (default) chooses among valid sites when ``number N`` is used.
  Sites are visited until ``N`` entities are accepted (or until the grid is exhausted).
  Grid does not retry failed sites; if too few sites pass ``near``, the command errors.

``random``
  Sample ``(u,v)`` uniformly in ``region``. ``number N`` is required.
  ``attempt Q`` (default 10) retries a failed ``near`` check.

``gaussian``
  ``origin u v`` and ``sigma s``. :math:`u=u_0+\mathcal{N}(0,s)`, :math:`v=v_0+\mathcal{N}(0,s)`.
  ``number N`` is required. ``attempt`` is allowed.

``point``
  ``origin u v``. One deterministic site; ``number`` defaults to 1.

Velocity and spread
-------------------

::

  velocity constant V
  velocity uniform Vmin Vmax
  spread gaussian sigma_deg

``velocity`` sets only the speed :math:`|\mathbf v|` (Å/fs).
The flight direction is ``direction``, optionally broadened by ``spread gaussian`` with angular standard deviation in degrees.

Other keywords
--------------

* ``near R``: every atom of a candidate entity must stay at least ``R`` Å from existing atoms and from already accepted deposited entities (PBC minimum image). Atoms inside the same entity are not checked against each other. Off if omitted.
* ``seed S``: required whenever the command uses randomness (``random`` / ``gaussian`` styles, ``select random``, ``spread gaussian``, or ``velocity uniform``).
* ``attempt Q``: only for ``random`` and ``gaussian``.

Examples
--------

One carbon atom at a fixed point, flying along :math:`-z`::

  deposit point atom C origin 8.8 8.8 direction axis -z surface fixed 16.0 velocity constant 0.001

Every site of an XY grid::

  deposit grid atom Si number all region 0 100 0 100 spacing 10 10 direction axis -z surface fixed 120 velocity constant 0.05

MBE onto a global surface::

  deposit grid atom Si number 100 region 0 100 0 100 spacing 5 5 direction axis -z surface global gap 20 velocity constant 0.05 select random seed 12345

Oblique MBE onto an XY substrate (grid stays in the XY plane; the beam is tilted)::

  deposit grid atom Si number 100 region 0 100 0 100 spacing 5 5 normal axis +z direction vector 0.2 0 -1 surface local radius 5 gap 20 near 2.5 spread gaussian 4 velocity constant 0.05 select random seed 12345

Gaussian beam::

  deposit gaussian atom C number 5 origin 1.785 1.785 sigma 1.0 direction axis -z surface local radius 5 gap 2.0 spread gaussian 5 velocity constant 0.001 seed 1

Random molecules on a plane::

  deposit random molecule O3.xyz number 10 region 0 272 210 230 direction axis -z surface fixed 20 velocity constant 0.0002 seed 1

This keyword only creates atoms. It does not delete atoms and does not run a connectivity search.
To drop flying or isolated atoms, issue :ref:`delete <kw_delete>` separately.

Caveats
-------
* The species must be allowed by the :ref:`potential <kw_potential>`.
* Changing the number of atoms while PIMD beads exist is not supported.
* :ref:`electron_stop <kw_electron_stop>` and ``add_random_force`` size some buffers when they are parsed.
  Issue those keywords after ``deposit`` in the same run block, or they will not see the new number of atoms.
