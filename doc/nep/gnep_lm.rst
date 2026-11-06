.. index::
   single: Levenberg-Marquardt (GNEP training)

Moment-Space LM Training
========================

``gnep`` provides an opt-in, single-GPU Levenberg-Marquardt (LM) optimizer.
It updates both descriptor coefficients and neural-network parameters. This
is not LM-MA-ES, an SNES replacement in ``nep``, or a change to MD inference.
Omitting ``optimizer`` retains the existing Adam/FCVP path.

Input
-----

Use a training-derived guard set that is separate from the fitting data and
the final validation set. The guard influences step selection and is therefore
not an independent test set. The minimum additions to an existing compatible
``gnep.in`` are::

  optimizer lm
  loss_rmse 1e-7
  lm_guard guard.xyz 1.1

These retain the tested LM defaults. Keep one keyword and its arguments per
line, as in other native input files. Model settings such as ``type``,
``cutoff``, ``n_max``, ``l_max``, and ``neuron`` remain ordinary GNEP inputs.
LM belongs to the ``gnep`` executable and ``gnep.in``; these options are not
accepted by the separate SNES ``nep`` executable or its ``nep.in``.

A complete Si/Ge template is provided in
``examples/nep_train/lm/gnep.in``. It uses a ten-epoch budget for the measured
benchmark configuration, not a convergence guarantee for other datasets.
``epoch`` is a fixed run budget: there is no automatic target-RMSE stopping
criterion in this implementation. A clean working directory without
``gnep.restart`` starts from random parameters; an existing restart supplies
initial parameters instead.

For explicit advanced settings, use::

  optimizer lm
  lm 16 8 0.2 0.01
  lm_guard guard.xyz 1.1
  lm_check 1
  loss_rmse 1e-7
  weight_decay 0
  lambda_1 0
  lambda_2 0
  batch 1000 0
  seed 20261015

Launch on one visible GPU::

  CUDA_VISIBLE_DEVICES=0 ./gnep

``optimizer adam`` explicitly selects the default optimizer. ``lm`` sets the
force-probe count, damping-candidate count, force-curvature EMA weight, and
initial damping. The defaults are ``16 8 0.2 0.01``. Valid ranges are 1--1024
probes, 1--64 candidates, EMA in (0, 1], and damping in [1e-9, 1e9].

``lm_guard <filename> <ratio>`` is required for LM. The file must be nonempty
and must not be ``train.xyz`` or ``test.xyz``. The finite ratio must be at least
one; 1.1 permits at most 10% above the best accepted guard objective across
the current run, not a compounded 10% increase at every step. ``test.xyz``
remains optional validation data and never participates in LM step selection.
Guard inputs must also be distinct from fixed native output files such as
``lm.out``, ``loss.out``, ``nep.txt``, and ``gnep.restart``; output-path and
existing file-alias collisions are rejected before those files are opened.

``lm_check 1`` compares the new gradient against the native analytic gradient
on the first update. ``lm_check 0`` is the default. Moment descriptors are
always checked against the native predictor when a batch cache is created.
LM-specific options with Adam, and LM with prediction mode, are rejected.

``lm_fast 1`` enables a faster row and candidate path; ``lm_fast 0`` is the
default. It caches per-atom intermediates once per update, builds each
structure's rows in GPU shared memory with 16 cooperating threads per atom, and
evaluates all training candidates before the guard. It targets the same
objective and candidate choice, but results are not bitwise identical to
``lm_fast 0``. It requires a build for ``sm_70`` or newer. Large parameter counts
that exceed shared memory fall back to cached atomic rows, and an atom cache
over the memory budget falls back to the default rows. One Si/Ge benchmark ran
3.5 times faster over a fixed 10 epochs on an A100; this is a single-seed timing,
not a time-to-accuracy result, so compare validation errors on your own data.

LM requires ``loss_rmse``, zero decoupled weight decay, zero L1 penalty, no
batch accumulation, and no FCVP projection. A nonnegative ``lambda_2`` is
supported through an IRLS norm penalty. Adam learning-rate schedules are
not applied to LM steps; damping determines the step instead.

Method
------

Geometry-only radial and angular basis moments are cached for each fitting
batch. Descriptors and their parameter derivatives are then reconstructed
from those moments. Native GNEP predictions still supply the energy, force,
and virial residuals, including any fixed ZBL contribution.

Energy rows and six strain/virial rows are explicit. The force gradient uses
the actual force residual as a directional probe, so it is not replaced by
a random gradient estimate. Rademacher force probes approximate only the
force curvature; their Gram contribution is normalized by probe count and
smoothed with an EMA. Configuration weights, type weights, ``force_delta``,
missing virials, and shear weights follow the native RMSE objective.

Channel weights are recomputed from the current smoothed RMSE objective.
The dense step solves a damped Gauss-Newton system. Curvature products use
FP32/TF32 cuBLAS; channel history, the residual-gradient reduction, and the
Cholesky solve use FP64. Thus this is an approximate-curvature method, not an
exact Hessian or an exact LM solve of every force residual.
Jacobian rows use FP32 atomic accumulation. Large training trajectories are
not guaranteed bitwise reproducible across launches, builds, or changes to
native validation allocations, even with the same random seed. A seed fixes
the initialization, batch shuffle, and probes, not GPU reduction ordering.

Several damping values are solved and evaluated sequentially. An accepted
candidate must strictly decrease the current batch objective and satisfy the
historical guard cap. Failed Cholesky solves and nonfinite candidates are
rejected. If no candidate passes, parameters remain unchanged and damping
increases. Candidate evaluation is not advertised as GPU-batched.

Outputs And Restart
-------------------

Normal GNEP model, restart, prediction, and loss files retain their formats.
``lm.out`` contains ten columns::

  step before after accepted lambda build_seconds candidate_seconds
  guard_before guard_selected guard_best

The first two objective columns refer to the current fitting batch, not a
full-dataset evaluation. ``guard_best`` is the best accepted/current guard
objective across this process. Timings separate moment/row/solve work from
candidate evaluation. Epoch reporting retains the native convention of
aggregating batches evaluated at successive parameter states.

``gnep.restart`` restores parameters only. It does not restore the curvature
EMA, damping history, or historical guard minimum. Restarted training is not
an exact continuation of the previous optimizer state.

Scope And Resources
-------------------

The supported model family is the native GNEP NEP5 single-hidden-layer model,
including three-, four-, and five-body descriptors, per-element NEP cutoffs, and
embedded fixed or flexible ZBL. This does not add NEP3/4, qNEP, dipole, or
polarizability training to GNEP. Native descriptor and model-capacity limits
still apply; training support does not enlarge the MD executable's limits.

The current optimizer requires exactly one visible GPU. No multi-GPU LM
speedup or NCCL training support has been established.
Two GPUs can run independent training jobs in separate working directories.
This improves throughput, not the time for one model. Do not share output or
restart files between concurrent jobs.

Data-parallel LM is a future implementation option: distribute complete
structures, globally sum residual squared errors and counts before computing
IRLS weights, and reduce channel curvature and gradients before a common
solve and guard decision. Averaging per-GPU RMSEs, duplicating regularization,
or maintaining independent damping histories would change the update.
The current timer groups predictions, row construction, curvature products,
and factorization in ``build_seconds``; it does not isolate each substage.
In the final external-validation run that group occupied 95.06% of the LM
loop, while candidate evaluation occupied 4.94%. Parallelizing only candidate
evaluation on two GPUs could therefore improve the loop by at most about
1.025 times even with zero communication overhead. These are measured shares
and an idealized bound, not a measured multi-GPU speedup.

For P trainable parameters, the solver's dense matrices require about
``52 * P * P`` bytes, plus linear buffers and cuSOLVER workspace. Moment,
Jacobian-row, native predictor, fitting-data, guard, and validation allocations
are additional. Cached moments remain resident for every visited fitting
batch, so their total storage grows with the fitting set, not just one batch.
The 4,026-parameter solver alone uses about 868 MB. Dense
storage and cubic factorization make very large parameter counts unsuitable;
LM is not a universal memory-saving replacement for Adam/SNES. The implementation
rejects matrix dimensions for which P squared exceeds a signed 32-bit index.

Validation And Performance Claims
---------------------------------

Accept performance claims only from random-start, wall-clock-to-accuracy
comparisons, with independent checks of exported models. A fast solver kernel,
a toy problem, or fine-tuning an already trained model is not sufficient.
The supplied design's projected timings were not treated as measurements.

Native regression tests cover moment/gradient agreement, weighted E/F/V,
higher-body descriptors, ZBL, guard selection, input errors, and opt-out
compatibility. See ``tests/nep_lm`` and ``tools/nep_lm``.

The development Si/Ge benchmark retains a historical validation split with
1,092 of 1,484 validation frames sharing geometry keys with the original
training set. Its fixed-split RMSE measures comparability to that historical
target, not geometry-disjoint generalization. GNEP NEP5 also has 4,026
parameters versus the native SNES NEP4 control's 4,025: their random initial
predictions are matched, but the output-bias parameterizations are different.
Any speed ratio refers to these two complete training pipelines, not an
isolated optimizer comparison under an identical parameterization.

On 2026-10-18, the public input interface was independently replayed from a
random start on one A100-PCIE-40GB. Ten epochs plus full CPU verification
took 233.570 seconds with external validation, yielding E = 0.06090876
eV/atom and F = 0.14929961 eV/Angstrom. A second replay also loaded the full
native validation set and enabled the gradient check: 240.581 seconds,
E = 0.06159819 and F = 0.14612691. Both meet the historical E <= 0.0785
and F <= 0.153 targets. They use the same random initialization, not a
pretrained checkpoint. Seven-epoch public replays were slightly above the
force threshold and are not counted as successful runs.

The paired SNES control had no CPU-confirmed joint target within 1,800
seconds. The conservative confirmed-target time advantage is therefore
greater than 7.48 times for these public-interface checks, not an exact SNES
convergence-time ratio. Compilation and data preparation are excluded.

Earlier prototype runs with three random seeds reached the target in
185.119, 124.806, and 124.751 seconds. These are separately versioned results,
not timings of the final public interface. The extra prototype seeds vary
both initialization and the grouped guard split and have no paired SNES
controls. Virial errors were recorded without a pass threshold. None of these
measurements establishes the same speedup on other datasets or hardware.
