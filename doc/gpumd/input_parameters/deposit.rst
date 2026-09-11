.. _kw_deposit:
.. index::
   single: deposit (keyword in run.in)

:attr:`deposit`
===============

This keyword inserts atom(s) into the current configuration.

It is meant for special simulations that grow or inject atoms during MD, such as molecular-beam or cluster deposition.
Ordinary MD simulations do not need this keyword.

GPUMD does not support ``\`` line continuation; write one command on one line.

Syntax
------

There are three forms.

Insert one atom at a given position and velocity::

  deposit <symbol> <x> <y> <z> <vx> <vy> <vz>

Insert atoms using keywords. ``position`` and ``velocity`` are required and may appear in any order.
For non-gaussian forms, both take six numbers (min/max per component) and sample uniformly in those ranges.
``number`` may be used with the box ranges. ``surface`` and ``offset`` are only for beam-deposition placement::

  deposit <symbol> number <N> velocity <vx_min> <vx_max> <vy_min> <vy_max> <vz_min> <vz_max> position <x_min> <x_max> <y_min> <y_max> <z_min> <z_max>
  deposit <symbol> position gaussian <x0> <y0> <sigma> velocity gaussian <v> <theta_sigma_deg> surface local <radius> offset antivel <sep>

Insert molecules from an xyz file (uniform sampling in a box)::

  deposit <file.xyz> number <N> velocity <vx_min> <vx_max> <vy_min> <vy_max> <vz_min> <vz_max> position <x_min> <x_max> <y_min> <y_max> <z_min> <z_max>

Uniform box sampling (atoms or molecules)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``position <x_min> <x_max> <y_min> <y_max> <z_min> <z_max>``: sample the insertion point (atom coordinates, or molecule center of mass) uniformly in the axis-aligned box. If ``min = max`` on an axis, that coordinate is fixed.
* ``velocity <vx_min> <vx_max> <vy_min> <vy_max> <vz_min> <vz_max>``: sample each velocity component independently and uniformly (Å/fs). If ``min = max``, that component is fixed.
* ``number <N>``: insert ``N`` independent samples in one command.

A single-atom species with these keywords is equivalent to depositing a one-atom molecule file with the same ranges.

Beam-deposition keywords (special-purpose)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These options are for a directed beam onto a surface. They are not needed for ordinary MD or for box sampling.

* ``position gaussian <x0> <y0> <sigma>``: :math:`x = x_0 + \mathcal{N}(0,\sigma)`, :math:`y = y_0 + \mathcal{N}(0,\sigma)`, then wrap XY with PBC. Requires ``surface local`` to set :math:`z`.
* ``velocity gaussian <v> <theta_sigma_deg>``: the speed :math:`|v|` is fixed (Å/fs); :math:`\theta \sim |\mathcal{N}(0,\sigma_\theta)|` in degrees; :math:`\phi \sim U(0,2\pi)`; the beam is along :math:`-z`:

  .. math::

     v_x = v\sin\theta\cos\phi,\quad
     v_y = v\sin\theta\sin\phi,\quad
     v_z = -v\cos\theta

* ``surface local <radius>``: :math:`z` is the highest atom within the XY radius (PBC) of the sampled :math:`(x,y)`; if none, the global :math:`z_{\mathrm{max}}` is used.
* ``offset antivel <sep>``: ``pos = anchor - sep * v_hat``. Needs a non-zero velocity.

This keyword only creates atoms. It does not delete atoms and does not run a connectivity search.
To drop flying or isolated atoms, issue :ref:`delete <kw_delete>` separately.

Examples
--------

Insert one carbon atom at a fixed point::

  deposit C 8.8 8.8 16.0 0.0 0.0 -0.001

Insert many F atoms uniformly in a box::

  deposit F number 100 velocity -0.001 0.001 -0.001 0.001 -0.002 -0.001 position 0 50 0 50 20 30

Insert molecules from a file::

  deposit O3.xyz number 10 velocity -0.0002 0.0002 -0.0001 -0.0001 -0.0002 0.0002 position 0 272 210 230 0 27

Beam deposition onto a local surface (special-purpose)::

  deposit C position gaussian 8.8 8.8 1.0 velocity gaussian 0.001 5 surface local 5.0 offset antivel 2.0

Repeated injection with :ref:`for <kw_for>`::

  for i range 1 500
      deposit Ge position gaussian 0 0 8.33 velocity gaussian 0.007 5 surface local 5.0 offset antivel 2.6
      run 100
  end

Caveats
-------
* The species must be allowed by the :ref:`potential <kw_potential>`.
* Changing the number of atoms while PIMD beads exist is not supported.
* :ref:`electron_stop <kw_electron_stop>` and ``add_random_force`` size some buffers when they are parsed.
  Issue those keywords after ``deposit`` in the same run block, or they will not see the new number of atoms.
