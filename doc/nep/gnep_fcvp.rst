.. _gnep_fcvp:
.. index::
   single: FCVP (GNEP training)
   single: fcvp (keyword in gnep.in)
   single: loss_rmse (keyword in gnep.in)

Experimental FCVP training with GNEP
===================================

Force-consistent variable projection (FCVP) is an optional training method in
the ``gnep`` executable. It optimizes the linear output weights and per-element
output biases with the nonlinear parameters held fixed, then updates the
nonlinear parameters with Adam. Forces are differentiated from all contributing
site energies. The resulting model uses the existing NEP5 ``nep.txt`` format.
This option does not change the ``nep`` SNES executable or MD inference.

FCVP is disabled by default. Omitting ``fcvp``, ``loss_rmse``, and ``fcvp_check``
retains the legacy GNEP loss and update path. It inherits this branch's GNEP descriptor architecture,
which includes three-body angular descriptors; it does not support every model
mode available in NEP inference.

Build and run
-------------

Build with a complete CUDA toolkit and an architecture matching the GPU::

  make -C src -j2 gnep CUDA_ARCH=-arch=sm_80

For an A100, ``sm_80`` is appropriate. Run in a directory containing
``gnep.in`` and ``train.xyz``::

  CUDA_VISIBLE_DEVICES=0 /absolute/path/to/GPUMD/src/gnep

The existing GNEP ``test.xyz`` input provides validation data during training.
Keep a separate held-out test set when measuring time to a validation target.
Use ``CUDA_VISIBLE_DEVICES=0,1`` to let two GPUs cooperate on one training run.
Each global batch is partitioned into nonempty shards of complete structures.
Atoms within a structure are not split between devices. Each GPU computes its
local predictions, analytical parameter gradients, and output-basis statistics.
Small statistics and gradients are reduced through the host; the MM solve and
Adam update run once, and validation runs on device 0. Descriptor scales are
initialized from global minima and maxima and shared by every device.

The batch size and objective do not grow with the device count. At most one
device per structure in a batch is active, so a one-structure batch cannot use
multiple GPUs. The implementation accepts up to 16 visible devices; only one
and two A100 GPUs are included in the native regression and benchmark setup.
Additional device counts and heterogeneous GPUs require separate validation.
Small workloads may be slower because of synchronization and transfer overhead.

Input syntax
------------

Add the following options to an otherwise valid ``gnep.in``::

  seed 20261014
  weight_decay 0
  fcvp 1 40 1e-8 1e-8
  lambda_1 0
  lambda_2 0

The syntax is::

  fcvp <interval> <iterations> <epsilon> <tolerance>

All four arguments are required:

* ``interval`` is a positive integer number of training steps between
  projections. A step is one batch update. Projection occurs at step zero and
  subsequently at multiples of this interval.
* ``iterations`` is the positive integer maximum number of inner
  majorization-minimization (MM) iterations per projection.
* ``epsilon`` is the smoothing parameter in the loss, within the positive
  normal FP32 range.
* ``tolerance`` is the positive finite relative convergence tolerance of the
  inner solver.

``fcvp`` also enables the consistent smoothed RMSE objective. To use Adam with
the same objective and train every parameter with Adam instead, replace the
``fcvp`` line with::

  loss_rmse 1e-8

The epsilon argument is required and must be positive and finite. Without
FCVP, ``loss_rmse`` performs no output projection. There is no ``fcvp off``
syntax; omit the keyword to disable projection.

Additional controls are:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Option
     - Default
     - Meaning
   * - ``seed <integer>``
     - Initialization: ``1234567``; random batch shuffle
     - An explicit nonnegative seed controls initialization and batch shuffling.
   * - ``fcvp_check 1``
     - Omitted (disabled)
     - Run a sampled native GPU finite-difference gradient check before training.
       This also enables the consistent loss, even without ``fcvp``.
   * - ``lambda_1 <value>``
     - ``0``
     - Nonnegative normalized L1 penalty on the complete parameter vector.
       Requires the consistent loss path.
   * - ``lambda_2 <value>``
     - ``0``
     - Nonnegative smoothed normalized L2-norm penalty on the complete parameter
       vector. Requires the consistent loss path.
   * - ``weight_decay <value>``
     - ``0``
     - Decoupled Adam weight decay. It must remain zero when FCVP is enabled.

The ``fcvp_check`` keyword accepts only ``1``; omit it to disable the check.
With multiple visible GPUs, the check verifies the summed analytical gradient.

Batch behavior
--------------

At a projection, only the current training batch contributes to the objective
and Gram matrices. Set ``batch`` to at least the number of training
configurations to make this a full-dataset projection. The GNEP default batch
size is 2 configurations, so leaving it at its default does not generally give
full-batch training.

With smaller batches, batch order is shuffled each epoch and the projection
fits the batch selected for that step. It does not accumulate information from
all batches before solving. The legacy ``batch <size> 1`` flag does not implement
an aggregate FCVP solve and is rejected by the consistent loss path.

At interval one, projection precedes every nonlinear Adam update. At larger
intervals, Adam continues to update nonlinear parameters between projections,
while output weights and biases remain fixed. Adam excludes output parameters
before clipping, moment updates, and parameter updates. A finite inner iteration
budget or an interval larger than one gives inexact or periodic variable
projection; exact solution of the output subproblem is not assumed.

Objective and numerical checks
------------------------------

For fixed nonlinear parameters, energy, force, and virial predictions have the
form :math:`A\beta + o`, where :math:`\beta` contains the output parameters.
A fixed ZBL contribution belongs to the affine offset :math:`o`. The loss is

.. math::

   L = \sum_{k \in \{E,F,V\}} \lambda_k
       \sqrt{\frac{\|S_k(A_k\beta+o_k-y_k)\|^2}{N_k}+\epsilon^2}
       + \frac{\lambda_1\|p\|_1}{D}
       + \lambda_2\sqrt{\frac{\|p\|_2^2}{D}+\epsilon^2}.

Here :math:`p` is the complete parameter vector of length :math:`D`.
Energy and virial labels are per atom. :math:`N_E` counts configurations,
:math:`N_F` counts Cartesian force components, and :math:`N_V` counts six
components per configuration with virial labels. Missing virials do not become
zero targets. :math:`S_k` includes configuration weights, force type weights,
the square root of the ``force_delta`` factor for forces, and ``lambda_shear``
for shear virials.

Validation and prediction virial RMSE exclude training weights, including
``lambda_shear``. The weighted training loss and raw validation metric therefore
have different meanings when weights differ from one.

This objective uses absolute energy residuals and explicit smoothing. It does
not reproduce all SNES objective conventions, including energy centering and
per-type ranking. Use ``loss_rmse`` Adam as the objective-matched control.

The solver accumulates Gram matrices in FP64 and uses MM with Cholesky and L1
coordinate minimization. It retains the contribution of fixed nonlinear
parameters to norm penalties. A small proximal shift stabilizes singular
systems around the current output vector. The temporary basis is limited to
512 MiB per device and 512 output parameters; reduce the batch or neuron count
if this limit is exceeded. This limit does not bound total training memory.

Native GPU training sets the relative proximal shift to FP32 machine epsilon,
because the basis entries originate in FP32. The host-only FP64 reference
solver uses a smaller default shift. This damping is centered on the current
iterate, not zero, and does not introduce an extra ridge penalty into the
accepted objective. It changes finite-iteration numerical trajectories and
does not guarantee good validation error for underdetermined training data.

For multiple devices, let :math:`\widetilde A_{ki}=S_{ki}A_{ki}` and
:math:`\widetilde y_{ki}=S_{ki}(y_{ki}-o_{ki})` denote shard :math:`i` of
error group :math:`k`. All devices contribute to one globally normalized
subproblem:

.. math::

   H_k &= \frac{\sum_i \widetilde A_{ki}^{\mathsf T}\widetilde A_{ki}}{N_k},
   &g_k &= \frac{\sum_i \widetilde A_{ki}^{\mathsf T}\widetilde y_{ki}}{N_k},
   &c_k &= \frac{\sum_i \widetilde y_{ki}^{\mathsf T}\widetilde y_{ki}}{N_k},\\
   N_k &= \sum_i N_{ki},
   &\operatorname{MSE}_k(\beta)
     &= \beta^{\mathsf T}H_k\beta - 2g_k^{\mathsf T}\beta + c_k.

The FP64 local Gram matrices, right-hand sides, and target norms are summed
before the MM solve. Full basis matrices remain on their owning GPUs. For
nonlinear gradients, residual squared sums are likewise combined before
computing the global RMSE normalization factor. Local gradients then sum to
the gradient of the same global loss. Averaging per-device RMSEs would give a
different objective, especially with unequal atom counts or missing virials,
and is not used. Regularization is applied once, not once per device.

This is single-process data parallelism, not distributed multi-node training.
Host reductions, the small MM solve, validation, and synchronization impose a
serial floor; the decomposition does not imply linear speedup with device
count. Structure-count partitioning also does not guarantee balanced work when
structures have different atom counts or neighbor densities.

Every accepted inner iterate is checked for a non-increasing smoothed
objective. Rounded FP32 parameters are then checked with the native GPU
predictor; a worse objective triggers backtracking or rejection. The first
projection checks native energy, force, and virial reconstruction automatically.
``fcvp_check 1`` adds sampled gradient checks but does not exhaustively verify
every parameter, material, or supported input combination.

The ``fcvp.out`` file reports::

  step before after accepted mm_iterations inner_converged seconds basis_bytes proximal_shift

These are projection diagnostics. A non-increasing current-batch objective
does not guarantee validation improvement or monotonic full-dataset loss.
``basis_bytes`` is the sum of temporary output-basis storage across active
devices, while the allocation limit above applies separately to each device.

Measured scope
--------------

The structure-sharded build was tested on two A100-PCIE-40GB GPUs on
2026-10-15. A fixed-work experiment used identical saved initialization,
30 epochs, and a global batch of 200 configurations obtained by repeating the
25 PbTe training frames eight times. These repeated frames increase workload,
not the number of independent training samples. Three alternating-order paired
runs gave median process times of 25.4755 s on one GPU and 17.0539 s on two
GPUs: **1.494 times faster**. Median training-loop times were 22.3797 s and
13.6738 s: **1.637 times faster**. Startup, validation, and output remain in
the process-time measurement; these are not time-to-accuracy speedups.

A separate 200-epoch paired run on the original 25 training structures took
36.9021 s and 31.6910 s (1.164 times faster, one repeat only). Independent CPU
evaluation gave held-out force RMSEs of 0.06377384 and 0.06376945 eV/Angstrom
for the one- and two-GPU final models. The 29-test CPU/native suite passed,
and a two-GPU CUDA memcheck reported zero errors and zero leaked bytes.
More than two GPUs and other materials have not been benchmarked here.

The measured binary SHA-256 was
``cb6f7083c0549473f77a2380520a3d440aaa6896d2aa038192b5d5b0b1a6f005``.
The ``benchmark_scaling.py`` and ``audit_scaling.py`` tools record timings,
initialization settings, model/data hashes, and independent accuracy checks.

An exploratory test on 2026-10-14 used one NVIDIA A100-PCIE-40GB, a two-element
NEP5 model with 30 neurons, and a deduplicated public PbTe dataset with 25
training, 142 validation, and 143 held-out test configurations. ``batch 32``
gave a single full training batch. Each run had a budget of 200 epochs, and
the Adam and FCVP variants shared saved initial parameters within each seed.
These historical results used the delivered single-GPU implementation before
the structure-sharded refactor and FP32-aware damping; they are not a new
time-to-target measurement of the refactored binary.

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 45

   * - Seed
     - Energy / force target
     - Time-to-target ratio
     - Result
   * - ``20261014``
     - 0.001 eV/atom / 0.05 eV/Angstrom
     - Not established
     - Neither Adam control nor FCVP intervals 1, 5, or 10 reached the target.
   * - ``20261015``
     - 0.005 eV/atom / 0.15 eV/Angstrom
     - 25.11 (validation only)
     - Matched Adam failed the held-out force target. This is not a
       quality-gated speedup.
   * - ``20261016``
     - 0.005 eV/atom / 0.17 eV/Angstrom
     - 18.98
     - Matched Adam took 140.38 s and FCVP interval 1 took 7.40 s. Both
       checkpoints passed the held-out targets.

For the last seed, held-out energy/force RMSE was 0.00376835 eV/atom and
0.156605 eV/Angstrom for Adam, and 0.000909869 eV/atom and 0.0795113 eV/Angstrom
for FCVP. Times include process startup, projection, and validation to the
selected checkpoint. This measures time to a particular accuracy, not time
per epoch: the strict-target pilot took 252.36 s for 200 matched-Adam epochs
and 264.52 s for 200 FCVP interval-1 epochs.

The two looser targets were chosen after examining earlier exploratory runs,
then fixed before each follow-up run. This is not a preregistered multi-seed
test at one fixed target, and the three different-target seeds must not be
averaged into a general speedup. The PbTe data have no virial labels. Larger
datasets, real virial targets, tighter force accuracy, SiGe transferability,
and downstream MD stability remain separate validation tasks.

See ``tools/fcvp/README.md`` for the complete reproduction commands, CPU oracle,
benchmark runner, experiment record names, and measured binary hash. The runner
checks validation checkpoints with an independent CPU predictor and records a
quality-gated speedup only when both methods also meet held-out test targets.
