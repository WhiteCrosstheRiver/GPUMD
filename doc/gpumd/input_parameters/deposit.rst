.. _kw_deposit:
.. index::
   single: deposit (keyword in run.in)

:attr:`deposit`
===============

Define a named source, then explicitly insert particles or execute a pulse.
This replaces ``deposit grid/random/...``. Ordinary MD commands are unchanged.
Old deposition lines must be rewritten as source facets followed by ``emit``;
they are not accepted as aliases. Incident velocity now requires explicit
``angstrom/fs`` units, so convert old internal velocity numbers by dividing
by 10.18051.

Syntax and lifecycle
-------------------

::

  deposit <name> <style> <arguments...>

Each configuration line replaces only that facet of its source; it never emits
or advances time. Required facets: particle, placement, region, velocity.
Sources persist across loops/runs in this process. Commands execute in script
order, without implicit cross-command batching. Write one command per line.
The source name identifies independent configuration and random state; it is
not an atom type or grouping label.

Particle and placement
----------------------

::

  deposit beam particle atom Si
  deposit beam particle mixture Si 0.8 Ge 0.2
  deposit beam particle molecule molecule.xyz
  deposit beam placement point at 10 10
  deposit beam placement grid spacing 4 4 select random
  deposit beam placement random attempt 100
  deposit beam placement gaussian center 10 10 sigma 2 attempt 100

Mixture probabilities describe emitted atoms, not retained-film composition.
They must be nonnegative and sum to one. Molecules retain template geometry and
share translational velocity, without rotational/vibrational thermalization.
Grid selection is random (default) or sequential. Random/Gaussian retries default
to 10; samples outside region bounds consume attempts. Grid visits each site
once. Only grid accepts ``number all``.

Region
------

::

  deposit beam region plane normal 0 0 1 bounds 0 40 0 40 height 100
  deposit beam region surface normal 0 0 1 bounds 0 40 0 40 height global gap 20 scope all
  deposit beam region surface normal 0 0 1 bounds 0 40 0 40 height local radius 6 gap 3 scope all
  deposit beam region surface normal 0 0 1 bounds 0 40 0 40 height local radius 6 gap 3 scope mobile
  deposit beam region box bounds 0 40 0 40 80 100

Distances are angstrom. Plane/surface bounds are umin umax vmin vmax. The explicit
normal n defines the frame, independently of velocity. For a plane region,
optional ``basis bx by bz`` sets u by projection; otherwise use the
least-parallel Cartesian axis. Then
v = n cross u. Normal (0,0,1) defaults to u=x,v=y. Fixed height is r dot n; surface
gap is measured along n. Local height takes the maximum within the lateral
radius, falling back to the global maximum if empty. Scope mobile excludes fixed
atoms from surface-height estimation only (global or local), never from
collision/flight clearance.
There is no shared mobile/all cache between sources.

Box regions are Cartesian and support random placement only. Surface height uses
the pre-emit configuration. Exclusion also sees earlier accepted entities in the
same emit. Molecule bounds apply to its template center.
Periodic local-surface queries currently require an orthogonal box and Cartesian
normal; other periodic local-surface geometries produce an explicit error.
Exclusion itself uses an exact minimum-image scan for skewed or very small boxes,
and the existing spatial index for supported ordinary boxes.

Velocity
--------

::

  deposit beam velocity vector 0 0 -0.0078581525 units angstrom/fs
  deposit beam velocity beam speed 0.0078581525 units angstrom/fs direction 0 0 -1 spread 0
  deposit beam velocity thermal temperature 300 drift 0 0 -0.01 units angstrom/fs

Units are mandatory. Beam normalizes direction; spread is Gaussian angular
width in degrees. Thermal adds Maxwell-Boltzmann Cartesian components to drift,
using total molecular mass for molecular translation. This is not a flux-weighted
effusive distribution. Away-pointing thermal samples can fail skip validation.
Old internal speed 0.08 becomes 0.08/10.18051 angstrom/fs, about 0.0078581525.
Do not reuse old velocity numbers unchanged.

Flight policy
-------------

::

  deposit beam flight skip off
  deposit beam flight skip on clearance auto

The default is skip OFF: insert at source; subsequent MD computes flight.
Skip ON explicitly approximates ballistic time compression: advance the sampled
entity along its sampled velocity to just outside a conservative envelope,
without changing velocity. Omitted time is logged in fs, NOT added to the MD
clock. Substrate evolution and thermostat action during omitted time are not
simulated; relative arrivals can change. Internal molecular evolution and
interactions between particles in the omitted incident beam are also not
integrated. This is not exact full-flight MD or a proof of force-free trajectories.
Validate high-flux/molecular cases against explicit flight before using skipping.

The envelope considers ALL preexisting atoms, including fixed/previously emitted
atoms. Its skipping bound also includes earlier accepted entities in the same
emit, so dense pulses may gain less. Ordinary NEP uses twice the maximum
radial/angular cutoff; LJ uses its
largest pair cutoff. Other potentials require explicit finite-range capability.
DFTD3, non-energy NEP and flexible ZBL are unsupported. Geometry must be orthogonal
with Cartesian normal and nonperiodic injection axis. No interactions are
truncated to force acceleration. Already inside the envelope means zero skip.

Skip/pulse require explicit NVE or constant-T NVT, positive timestep, and no
external fields/forces, electron stopping, MC, velocity correction, moving
groups, deformation or HNEMD. NPT, temperature ramps, PIMD and wall ensembles
are unsupported in these modes. Ordinary skip-OFF emit retains broader insertion.

Exclusion and random state
--------------------------

::

  deposit beam exclusion distance 2.5
  deposit beam seed 440001

Exclusion checks existing atoms and earlier accepted entities with minimum-image
geometry, not intramolecular pairs; default is disabled. Each source owns a
persistent RNG. Repeated emits advance it; reissuing seed resets it. Replacing
another facet does not reset it. Omitted seed initializes randomly. Expressions
are resolved when configuration executes, not at a later emit.
Numeric configuration arguments can use ``v_name`` for delayed variable
evaluation. Reissue the relevant configuration line inside a loop to evaluate
it once per pulse; random or Gaussian placement samples separately for each
emitted entity. See :ref:`variable <kw_variable>`.

Source definitions and RNG state are NOT stored in restart.xyz. An XYZ restart
is not a bitwise random-stream continuation; redefine sources and record seeds.

Execution
---------

::

  deposit beam emit number 5
  run 500
  deposit beam pulse number 5 wait entry relax 200 max_wait 5000

Emit inserts only; inserted count is not retained count. A fixed run 500 still
executes 500 MD steps with skipping ON. Pulse instead waits for every new atom
to cross a fixed reference plane (pre-emit maximum projection plus clearance),
then runs relax additional steps. Entry is latched per atom; rebound does not
undo entry. This plane is a conservative reference, not actual rough-surface
contact or proof of adsorption. Max_wait counts waiting steps; expiry is a
nonzero-exit error, not success. Pulse needs no following run command.

One continuously initialized MD run uses GPU entry checks. Logs report actual
wait, relaxation, total steps and elapsed MD time. Event-dependent pulse length
supports only dump_xyz, dump_thermo and dump_restart; use emit plus run for other
measurements. Normal per-run finalization applies: reissue fix/measurements for
each pulse. With relax 0 and all atoms already entered, zero MD steps are taken
and there is no step-based dump frame. Skipping remains an approximation that
requires independent physical validation.

Example
-------

After loading a Si/Ge potential and selecting an ensemble::

  deposit alloy particle mixture Si 0.8 Ge 0.2
  deposit alloy placement grid spacing 4 4 select random
  deposit alloy region surface normal 0 0 1 bounds 0 65.3244 0 65.3244 height local radius 6 gap 3 scope all
  deposit alloy velocity beam speed 0.0078581525 units angstrom/fs direction 0 0 -1 spread 0
  deposit alloy flight skip off
  deposit alloy exclusion distance 2.5
  deposit alloy seed 440001
  deposit alloy emit number 5
  run 500

A 3-angstrom gap can already be inside the interaction envelope, so skipping
alone may save no time. Validate event-based relaxation before replacing a
production fixed-duration protocol. Deletion/connectivity cleanup is explicit
via :ref:`delete <kw_delete>`.
