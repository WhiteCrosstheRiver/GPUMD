.. _experimental_lmma:

Experimental LM-MA-ES Training
=============================

Status and scope
----------------

This is an opt-in research implementation, not a replacement for SNES and not
a demonstrated training speedup. It changes the search distribution and update
rule, while retaining the native NEP GPU energy, force, virial, and loss
evaluation. The optimizer itself currently runs on the CPU. With the environment
variable below unset, or set to ``snes``, the original SNES path is retained.

The experimental path accepts ordinary NEP4 training only: ``version 4``,
``train_mode 0``, no charge model, no fine-tuning, and no prediction mode. Its
initial mean must be finite and its initial sigmas must be equal, positive, and
finite. This implementation additionally requires ``2 * population < number of
parameters``. These are explicit prototype restrictions, not general claims
about every LM-MA-ES variant or NEP model. Descriptor and potential kernels are
unchanged; compatibility beyond the tested cases is not claimed.

Usage
-----

Build ``nep`` normally, prepare ``nep.in``, ``train.xyz``, and optional ``test.xyz``,
then opt in for that process only:

.. code-block:: bash

   GPUMD_EXPERIMENT_OPTIMIZER=lmma GPUMD_EXPERIMENT_SEED=20261015 /path/to/nep

``GPUMD_EXPERIMENT_SEED`` is a decimal unsigned 64-bit integer. Empty values,
signs, whitespace, non-digits, and overflow are rejected. It seeds the LM-MA-ES
perturbation generator, not native NEP initialization. To reproduce an initial
mean or compare optimizers, prepare an identical ``nep.restart`` for each run
and record its hash. Without that file, native initialization remains
time-seeded. Random-normal sequences are reproducible within a given C++
standard-library implementation, not guaranteed across different libraries.

The existing ``population``, ``generation``, ``batch``, loss weights, and
``save_potential`` settings remain in ``nep.in``. The prototype uses global
weighted loss to rank candidates; native SNES has per-element rankings. Both
can use the native multi-GPU candidate evaluator. This does not imply that the
CPU LM-MA-ES update has been GPU-parallelized.

To return to SNES, unset both environment variables or use
``GPUMD_EXPERIMENT_OPTIMIZER=snes``. Unknown optimizer names are errors rather
than silently falling back. Unset the optimizer variable before using
``prediction 1`` with an exported model.

Algorithm and differences
-------------------------

``src/main_nep/experimental_lmma.cuh`` independently implements Algorithm 1 of
`Loshchilov, Glasmachers, and Beyer, Limited-Memory Matrix Adaptation for
Large Scale Black-box Optimization <https://arxiv.org/abs/1705.06693>`_.
It is not the appendix variant. The population is user-selected rather than
fixed by the paper's default. The number of memory directions is
``4 + floor(3 * log(number_of_parameters))`` and the best half of the population
receives positive logarithmic rank weights.

For each candidate, standard-normal directions are progressively transformed
by the active memory directions. The mean update uses those transformed
directions. The evolution path and memory updates use the original normal
directions. The step-size rule is the paper's squared-path expression:

.. math::

   \sigma_{t+1} = \sigma_t\exp\left[
     \frac{c_\sigma}{2}\left(\frac{\|p_\sigma\|^2}{n}-1\right)\right].

This differs from the norm/expected-chi rule in the authors' public Python
implementation. Tests compare against an independent NumPy implementation of
the stated paper recurrence, not against that different Python variant.
State is double precision; candidate parameters passed to NEP are floats.
Non-finite candidate values or global fitness stop the experimental run.

Native generation zero calibrates descriptor scaling and does not evaluate
candidate fitness. LM-MA-ES skips its ask/tell update for this generation;
native SNES startup is left unchanged. Include this startup difference,
global versus element-wise ranking, and CPU versus GPU optimizer updates
when interpreting a comparison.

Models and state files
----------------------

``nep.txt`` and saved potential files remain ordinary inference models. The
reported elite model is not the optimizer mean. Native reporting may adjust
the exported energy bias; this does not modify LM-MA-ES's stored search
directions.

LM-MA-ES writes ``nep.restart.lmma`` at the native restart interval. When
restart saving is requested alongside saved potentials, their restart names
also receive a ``.lmma`` suffix. These files contain dimension, population,
iteration, mean, sigma, paths, memory directions, and random-generator state.
**Loading/resuming these files is not implemented.** They are state exports
for inspection and future recovery support, not usable restart inputs.
``nep.restart`` is deliberately not overwritten: it remains the original
initialization input. Starting the program again from it starts a new run.

Validation and benchmarking
---------------------------

``tests/lmma`` contains CPU recurrence tests, an independent NumPy checker,
and an optional GPU integration smoke test. See its README for commands.
CPU tests include the 4,025-parameter, population-50 case with all 28 memory
directions activated. The GPU test checks disabled-path parity when a baseline
binary is supplied, finite training, nonzero regularization, and input guards.
These checks establish implementation behavior, not scientific superiority.

Benchmark claims should use random initial models, identical data and model
definitions, equal GPU resources, multiple seeds, and actual elapsed time to
the same full-validation thresholds. Include validation cost, record censored
runs that miss the budget, and do not compare a pretrained warm start with a
random start. GPU busy percentage alone is not a measure of useful arithmetic
throughput or absence of redundant candidate computation.

The initial SiGe comparison is ongoing as of 2026-10-16; no speedup over SNES
is claimed by this experimental implementation.
