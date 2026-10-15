# FCVP-NEP: experimental force-consistent output projection

**Experimental, opt-in GNEP training path.** Earlier revisions passed the CPU
suite and sampled native GPU checks and were exercised on an A100. A small PbTe
experiment showed a shorter time to an exploratory accuracy target. Those
results predate the expanded model scope and row-tiled FCVP implementation;
they do not establish a general training speedup or production readiness.

The target branch already includes GNEP in `src/main_gnep/`. This implementation
uses its NEP5 analytical gradients and standard `nep.txt` serialization. It does
not change MD force kernels or introduce a second potential architecture.
The MD model loader adds explicit capacity and ZBL-coverage checks.

## What is implemented

- A cached CUDA output-basis path, using the existing radial/angular force
  kernels. Forces include derivatives of **all contributing site energies**.
  The descriptors, angular sums, and neighbors are reused across basis columns.
- A physical-row-tiled **batch-shard** basis per active GPU, with FP64 cuBLAS
  Gram/RHS accumulation. No dataset-wide force Jacobian is retained. The
  `fcvp_memory` setting budgets explicit FCVP GPU temporaries per device,
  including Grams and reconstruction buffers; there is no fixed 512-column cap.
- Host FP64 MM with Cholesky, L1 coordinate minimization, and norm-L2
  majorization. It includes fixed nonlinear parameters in the norm penalty.
  A tiny proximal term about the current iterate stabilizes null spaces without
  silently adding ridge regularization to the requested objective.
- Every accepted MM iterate is checked against its smoothed objective. The final
  FP32 parameters are checked again by the **native GPU predictor**; an increase
  causes backtracking or rejection. The first projection also checks native
  E/F/V linear reconstruction before training can proceed.
- Adam masks output parameters **before clipping and moment updates**, and also
  excludes them from parameter updates/weight decay. Only nonlinear parameters
  receive Adam steps. `weight_decay` must be zero for FCVP; use `lambda_2` for an
  explicit norm penalty.
- Separate CPU oracle (upstream NEP_CPU), independent dense NumPy/SciPy SVD-MM
  reference, numerical tests, data preparation, and timed GPU benchmark runner.
- Standard NEP5 energy models with radial two-body terms and optional three-,
  four-, and five-body angular terms under the `l_max` constraints below.
  Flexible ZBL coefficients use per-device storage sized to the model instead
  of a fixed 550-coefficient table. Universal ZBL is also supported, and radial
  neighbors cover descriptor and ZBL cutoffs.

MD inference evaluates ZBL through its angular neighbor list. Its loader
rejects universal or flexible ZBL when any effective pairwise ZBL outer cutoff
exceeds the corresponding effective angular cutoff. Native GNEP's radial
neighbor union supports broader training cutoff combinations, but those
exports are not necessarily MD-compatible.

## Supported models and prediction

`l_max L3 [L4 [L5]]` accepts `L3` from 0 through 8, `L4` of 0 or 2 no larger
than `L3`, and `L5` of 0 or 1; `L5 = 1` requires `L4 = 2`. The original
one-argument form keeps `L4 = L5 = 0`. GNEP allows values through 19 for each
`n_max` and `basis_size` entry, 200 neurons, and 220 descriptor dimensions.
This describes native GNEP, not arbitrary NEP3/4, qNEP, dipole,
polarizability, or other model families.

The MD inference loader in this GPUMD branch has lower capacities: 103 total
descriptor dimensions, 90 angular dimensions, `basis_size` at most 16,
angular `n_max` at most 16, and flexible ZBL for at most 10 element types.
GNEP can train models beyond those limits, so a successful native run alone
does not establish safe MD inference of its export. GNEP warns when an export
exceeds them, and this branch's MD loader rejects it. External GPUMD and
NEP_CPU builds may have further, different limits.

Prediction treats `nep.txt` as the authoritative model and accepts NEP5 only.
It restores embedded ZBL before invoking the prediction backend and folds the
serialized global bias into per-element biases. Malformed or unsupported models
fail explicitly. This is distinct from the training input's optional `zbl.in`
and does not imply support for arbitrary NEP text variants.

## Objective and model details that matter

With fixed nonlinear parameters, predictions are `A beta + offset`. Fixed ZBL is
an **affine offset**, not a basis column. The reference target is `y - offset`.

For energy, force, and virial groups respectively, the opt-in objective is

```
L = sum_k lambda_k sqrt(||S_k (A_k beta + offset_k - y_k)||^2 / N_k + epsilon^2)
  + lambda_1 ||p||_1 / D
  + lambda_2 sqrt(||p||_2^2 / D + epsilon^2)
```

`p` is the entire training parameter vector and `D` its length. Energy and virial
are per atom. `N_E` counts configurations, `N_F` counts Cartesian force components,
and `N_V` counts six components of configurations **with virial labels**. Missing
virials never become zero targets. Configuration weights multiply residuals;
force residuals also receive type weights and the square root of the
`force_delta` factor; shear residuals receive `lambda_shear`.

For each MM step the normalized Gram gets multiplier
`lambda_k / sqrt(MSE_k + epsilon^2)`. The L2 norm gets the analogous multiplier;
L1 is handled as a nonsmooth term, not approximated by a ridge. The solver returns
both its iteration count and a convergence flag. Finite iteration budgets and
projection intervals above one make this **inexact/periodic variable projection**;
the exact envelope-theorem optimum is not assumed to have been reached.

The inherited GNEP implementation used MSE-like gradient factors while reporting
RMSE, with different atom-count and residual-weight factors. `loss_rmse` enables
consistent factors for both the Adam control and FCVP. Legacy GNEP mode remains
available for context. The new objective is explicitly smoothed and uses absolute
energy residuals; it does **not** reproduce SNES's energy-centering procedure,
per-type ranking utilities, or all NEP training modes. Never describe this as an
unqualified exact replacement of every GPUMD SNES objective.

NEP4 with two elements and 30 neurons has 61 output parameters. The integrated
**NEP5 GNEP** path has 62 (30 weights plus one bias for **each** element). Its
serialized additional shared bias stays zero. The CPU oracle supports ordinary
NEP3/4/5 and keeps NEP5's redundant serialized shared bias fixed. Composition
constraints can make bias columns rank-deficient; solving an explicit inverse is
not appropriate. The redundant NEP5 shared bias is excluded from the training
parameter count and regularization, matching the native GNEP vector.

GNEP partitions each global batch into nonempty shards of complete structures.
It does not split the atoms of one structure, so all force interactions remain
local to the GPU holding that structure. Every GPU has its own gradient buffer
and computes its local forward, gradient, and output-basis work. Descriptor
scales use global minima and maxima across training shards, then the same scale
is broadcast to all devices. The batch size and optimizer update count remain
unchanged when more GPUs are enabled.

For group `k` and device `i`, let `B_ki = S_ki A_ki` and
`t_ki = S_ki (y_ki - offset_ki)`. The global projection uses

```
N_k = sum_i N_ki
H_k = sum_i (B_ki.T B_ki) / N_k
g_k = sum_i (B_ki.T t_ki) / N_k
c_k = sum_i (t_ki.T t_ki) / N_k
MSE_k(beta) = beta.T H_k beta - 2 g_k.T beta + c_k
```

Local FP64 Gram matrices, right-hand sides, and scalar target norms are summed
through the host; basis row tiles are not transferred between devices.
The host solves one global output subproblem. For nonlinear updates, local
residual squared sums are combined first, then each device uses the **global**
RMSE normalization factor. Partial gradients are summed before the single Adam
update on device 0. Per-device RMSEs are not averaged, which would change the
objective for uneven shards. Norm penalties are applied once to the global
parameter vector. Validation runs on device 0.

This is single-process data parallelism, not multi-node training. The MM solve,
host reduction, validation, and synchronization remain serial costs. Shards are
balanced by structure count, not estimated neighbor work, so very different
structure sizes can leave devices unevenly loaded. More GPUs do not guarantee
proportionally shorter runtime. The native regression and benchmark setup
includes one and two identical A100 GPUs; no larger-device scaling claim is made.

Two pre-existing correctness issues encountered during integration were fixed:
uninitialized Adam moment buffers and an out-of-range atom-type read in a padded
ANN launch. Constant descriptor ranges now use finite scaling instead of division
by zero. The consistent gradient path also handles isolated atoms inside a
multi-atom configuration.

## Build and use on a GPU

From the repository root, with a complete CUDA toolkit and libraries:

```bash
make -C src -j2 gnep CUDA_ARCH=-arch=sm_80
```

Use the architecture appropriate for the actual GPU. Add to `gnep.in`:

```
seed 20261014
weight_decay 0
fcvp 1 40 1e-8 1e-8
fcvp_memory 512
lambda_1 0
lambda_2 0
```

`fcvp` arguments are projection interval (training steps), maximum MM iterations,
smoothing epsilon, and convergence tolerance. Projection uses only the **current
training batch**. At interval one the output layer is reoptimized immediately
before every nonlinear update. The first projection occurs at step zero; later
projections occur when the step is divisible by the interval. A step is one batch
update, not an epoch. Output weights and per-element output biases stay fixed
between projections; Adam updates only the nonlinear parameters.

Set `batch` to at least the number of training configurations for a genuine
full-batch experiment. With smaller batches, a projection minimizes the current
batch objective; it does not accumulate Gram matrices over the full dataset.
Batch order is shuffled each epoch. The legacy `batch <size> 1` flag does not
provide an aggregate FCVP projection and is rejected by the consistent loss path.

For an Adam control with the same objective, replace `fcvp ...` with
`loss_rmse 1e-8`. Omit `fcvp`, `loss_rmse`, and `fcvp_check` to retain the legacy
GNEP loss/update path. There is no `fcvp off` keyword: omission disables it.
Without `seed`, parameter initialization retains seed 1234567 and batch shuffling
retains the legacy random seed. An explicit `seed` controls both reproducibly.
The defaults are `weight_decay 0`, zero `lambda_1` and `lambda_2`,
and no finite-difference check. All four `fcvp` arguments are required; the
interval and iteration count must be positive integers, and epsilon and
tolerance must be positive finite numbers. Epsilon must also be within the
positive normal FP32 range to keep native gradient scaling representable.
`lambda_1` and `lambda_2` are
nonnegative and require the consistent loss path. They penalize the full
parameter vector, including the nonlinear parameters held fixed during a
projection. Do not combine `fcvp` with nonzero `weight_decay`.

`fcvp_memory <MiB>` accepts a positive integer and defaults to 512 per active
GPU. It bounds the explicit FCVP GPU basis tile, three FP64 Gram matrices and
right-hand sides, and reconstruction buffers. A budget unable to hold the
Grams and one basis row fails with an input error. It does not bound total GPU
or process memory: native predictor buffers are separate, and host residual
snapshots take memory proportional to the batch row count. With `P` output
parameters, Gram storage grows as `O(P^2)` and the direct MM solve costs
`O(P^3)` time. Removing the former 512-column cap does not guarantee practical
performance for arbitrary model dimensions.

```
CUDA_VISIBLE_DEVICES=0 /absolute/path/to/GPUMD/src/gnep
```

`fcvp.out` records actual before/after objectives, acceptance, MM convergence,
projection wall time, temporary basis bytes, and numerical proximal shift. The
reported basis bytes sum all active devices; `fcvp_memory` applies separately
to each device and accounts for the explicit Gram and reconstruction buffers
as well. The first native linearity check is automatic. Add `fcvp_check 1` for a separate
finite-difference check of selected ANN, descriptor, and output derivatives. The
benchmark runner requires this gate to pass before timed runs. This sampled GPU
gate is necessary but not a replacement for a broader material-specific check.
Use `fcvp_check 2` to check every parameter; this may take much longer.
Either level selects the consistent loss even without `fcvp`; omit the keyword
to disable the check. With multiple GPUs the check uses the summed gradient
and global finite-difference objective.
Use `CUDA_VISIBLE_DEVICES=0,1` for two-GPU training. Training-set prediction
traverses all shards and preserves the configuration and atom output order;
validation uses device 0.
Between 1 and 16 visible devices are accepted; only one and two devices have
native regression coverage here. At most `min(device_count, batch_structures)`
devices participate in a batch. Each active GPU must fit its shard and working
buffers, while device 0 must also fit the validation set. A one-structure batch
cannot gain data-parallel speedup.

Training virial loss includes `lambda_shear`, while validation and prediction
virial RMSE are unweighted physical errors, including when `lambda_shear` is zero.

To measure fixed-work scaling independently of convergence, run:

```bash
python tools/fcvp/benchmark_scaling.py --gnep "$PWD/src/gnep" \
  --data local/fcvp/pbte_clean --output local/fcvp/scaling \
  --devices 0,1 --epochs 20 --repeats 3
```

This alternates single/two-GPU run order and reports process wall time and
training-loop time. `--repeat-training 8 --batch 200` can increase the workload
by repeating the 25 training frames; these are not new independent samples and
must not be presented as an accuracy benchmark.

Check the repeat-0 final models with the independent CPU predictor outside the
timed region:

```bash
python tools/fcvp/audit_scaling.py --report local/fcvp/scaling/summary.json \
  --data local/fcvp/pbte_clean --bridge local/fcvp/libfcvp.so
```

Run the native generality regression from the repository root with a built
GNEP executable and NEP_CPU bridge:

```bash
FCVP_GNEP="$PWD/src/gnep" FCVP_BRIDGE="$PWD/local/fcvp/libfcvp.so" \
  FCVP_DEVICES=0,1 python -m pytest -q tests/fcvp/test_gpu_generality.py
```

The native solver damps its MM steps at FP32 basis precision using a proximal
term about the current output vector; the FP64 reference keeps its smaller
default shift. This does not add a ridge penalty to the accepted objective.
Underdetermined unregularized fits still need validation: reducing training
loss does not guarantee good held-out predictions. Strict single/two-GPU
parameter regression uses well-conditioned regularized fixtures; separate
unregularized cases test finite gradients and non-increasing projections.
Historical exploratory convergence timings below predate this damping, the
structure-sharded refactor, and the expanded descriptor and FCVP memory paths.

## Reproduce the CPU verification

Requires `g++`, NumPy, and SciPy. No CUDA or GPU is needed.

```bash
git clone https://github.com/brucefan1983/NEP_CPU.git ../NEP_CPU
git -C ../NEP_CPU checkout af615ce0e81bbb860e77923762c505567affa8d1
python tools/fcvp/build_cpu.py --nep-cpu ../NEP_CPU
python -m unittest discover -s tests/fcvp -v
```

The bridge compiles the same C++ MM solver used by the GPU training path.
The independent reference solves weighted least squares with SVD, avoiding normal
equations. Tests cover E/F/V reconstruction, force and strain derivatives, NEP5
bias layout, fixed ZBL, model roundtrips, rank deficiency, zero-weight groups,
exact fits, invalid numbers, missing virials, data weights, and L1/L2 penalties.

Fetch a pinned real XYZ dataset and remove leakage **before** benchmarking:

```bash
python tools/fcvp/fetch_public_data.py --output local/fcvp/pbte_source
python tools/fcvp/prepare_data.py \
  --source local/fcvp/pbte_source/train.xyz local/fcvp/pbte_source/test.xyz \
  --output local/fcvp/pbte_clean --preserve-training
OPENBLAS_NUM_THREADS=1 python tools/fcvp/benchmark_cpu.py \
  --bridge local/fcvp/libfcvp.so --model examples/nep_train/nep.txt \
  --train local/fcvp/pbte_clean/train.xyz --test local/fcvp/pbte_clean/test.xyz \
  --output local/fcvp/pbte_reference
```

The source has 25 training and 300 test frames. There are 4 geometries shared
between those source files and 15 duplicate occurrences overall. Preparation
keeps 25 training frames and splits the remaining 285 unique geometries into
142 validation and 143 test frames (seed 20261014). This removes exact coordinate
leakage, not all possible trajectory or symmetry-related correlations.

## Historical CPU results, 2026-10-14

See `results/pbte_cpu_reference.json` and `results/pbte_split_manifest.json` for
complete machine-readable records and hashes.

| Check | Observed result |
| --- | --- |
| Test suite | 13 numerical + 3 benchmark process-control tests passed |
| Real PbTe output parameters | 61, fixed NEP4 representation |
| Max E/F/V linearity error | 4.9e-15 / 1.6e-15 / 2.8e-15 |
| Accepted MM steps | 8, all non-increasing |
| Training objective | 0.07651418 → 0.07554036 |
| Difference from independent SVD-MM objective | approximately 5.8e-12 |
| Test energy RMSE | 0.45450 → 0.45173 meV/atom |
| Test force RMSE | 0.0397143 → 0.0393207 eV/Å |
| Basis construction | 9.849 s on CPU |
| Gram construction plus MM | 0.00536 s on CPU |
| Gram condition estimate | 7.84e9 |

These are **frozen pretrained representation** results. They neither compare full
SNES/Adam training nor establish a training speedup. Basis construction dominates
this CPU oracle; the GPU path avoids recomputing descriptors per column, but its
real cost must still be measured. The real PbTe data have no virial labels;
virial reconstruction/strain checks and joint-group synthetic solver tests passed,
but fitting real stress labels is still an outstanding benchmark.

The initial CPU-only development environment used CUDA 12.6.85 to compile the
changed translation units. Subsequent testing linked the full executable with
`nvcc -O3 -arch=sm_80`, passed the sampled native GPU gate, and ran the controlled
experiments below on an NVIDIA A100-PCIE-40GB.

## Historical GPU results, 2026-10-14

These experiments used 25 PbTe training configurations, 142 validation
configurations, and 143 held-out test configurations after exact-coordinate
deduplication. `batch 32` therefore gave one full training batch. The model was
NEP5 with two elements, 30 neurons, cutoffs 5/4, `n_max 4 4`, `basis_size 8 8`,
`l_max 4`, `lambda_e 1`, `lambda_f 2`, `lambda_v 0`, and zero weight decay. Each
run had a maximum budget of 200 epochs. Adam and FCVP used identical saved
initial parameters within each seed, and the primary control used
`loss_rmse 1e-8`.

Energy targets below are in eV/atom and force targets in eV/Angstrom. Times
include process startup, projection, and validation up to the first accepted
validation checkpoint. The held-out set was then evaluated at that checkpoint.

| Seed | Energy / force target | Objective-matched Adam | FCVP interval 1 | Interpretation |
| --- | --- | --- | --- | --- |
| 20261014 | 0.001 / 0.05 | Not reached | Not reached | No speedup established; legacy Adam and FCVP intervals 5/10 also did not reach the target in 200 epochs |
| 20261015 | 0.005 / 0.15 | 184.55 s | 7.35 s | 25.11x validation-time ratio, but Adam failed the held-out force target; no quality-gated speedup |
| 20261016 | 0.005 / 0.17 | 140.38 s | 7.40 s | 18.98x time-to-target ratio; both checkpoints passed the held-out targets |

For seed 20261016, held-out energy/force RMSE was 0.00376835/0.156605 for Adam
and 0.000909869/0.0795113 for FCVP. FCVP reached its target at the first reported
epoch and Adam at epoch 105. The 18.98x ratio measures reaching that accuracy,
not the speed of each epoch. In the strict-target pilot, 200-epoch process times
were 252.36 s for matched Adam and 264.52 s for FCVP interval 1.

The two looser targets were selected after examining earlier exploratory runs.
Although they were fixed before their respective follow-up runs, this is not a
preregistered multi-seed evaluation of a single application target. These three
seeds cannot be pooled into a mean speedup because their targets differ. The
small PbTe set has no virial labels and is not evidence of SiGe transferability,
large-dataset performance, or convergence at tighter force accuracy.

The source experiment records are `gpu_pilot_summary.json`,
`gpu_confirm_seed20261015_summary.json`, and
`gpu_confirm_seed20261016_summary.json` in the delivery evaluation artifacts.
The tested binary SHA-256 was
`6efbfac5164c84174d413d20c5a9fe50c7033b6232cb9e144612a3878bd8df41`.
These measurements describe that binary; subsequent code changes require their
own correctness and performance checks.

## End-to-end GPU benchmark

Targets below are example thresholds. The strict-target 200-epoch pilot above
did not attain them. Choose and freeze application thresholds before a new
confirmatory benchmark; do not select them after inspecting its runs.

```bash
python tools/fcvp/benchmark_gpu.py \
  --gnep "$PWD/src/gnep" --bridge local/fcvp/libfcvp.so \
  --data local/fcvp/pbte_clean --output local/fcvp/gpu_benchmark \
  --device 0 --seeds 20261014 20261015 20261016 --intervals 1 5 10 \
  --batch 32 --epochs 2000 --energy-target 0.001 --force-target 0.05 \
  --build-notes 'CUDA toolkit/compiler/architecture flags used for these binaries'
```

Optionally pass `--upstream-gnep /path/to/GNEP/src/gnep` and
`--snes /path/to/GPUMD/src/nep`. The inspected upstream GNEP revision was
`10052f7120496497c56de8b12f6b3f8e4a1d11fd`; freeze its binary and build environment
as well. The target GPUMD base was
`efeebe198ea7aec7d8c436852873e8f8e40610d1` (`feature/deposit`).

The runner executes legacy Adam, objective-matched Adam, and FCVP from identical
saved NEP5 initial parameters per seed. It timestamps validation reports from
process launch, includes projection and validation costs, samples process GPU
memory every 250 ms, and evaluates the held-out test set only at the selected
checkpoint. An independent CPU predictor rechecks the saved checkpoint against
the validation report and thresholds before accepting its timing. Both models
must also pass the held-out thresholds for a quality-gated speedup to be recorded.
Runs that never reach the target retain a null time/speedup, not a
fake success. Tests are never copied into the training working directories;
GPUMD's `test.xyz` there contains **validation** data.

`source_commits` is null for binaries installed outside a Git checkout; provide
their source revision in `--build-notes`. The optional SNES runs vary their saved
initial parameters across seeds; this branch fixes its GPU sampling RNG at
1234567, so these are not independent SNES sampling streams.

Native SNES on this branch only supports NEP3/4, while GNEP exports NEP5. Its
optional result is historical context, not a strict architecture-matched control.
For different compositions this distinction is material. Compare FCVP primarily
with both the objective-matched Adam control and pinned upstream GNEP, then require
comparable held-out quality and stability before claiming acceleration.

## Remaining acceptance gates

1. Run the native GPU suite after each code change. It includes unequal atom
   counts, nonunit weights, missing virials, synthetic virial labels, ZBL,
   regularization, isolated atoms, and one/two-device equivalence. These small
   regression fixtures do not establish performance or accuracy on real
   virial-labelled datasets or on more than two GPUs.
2. Measure time to one independently chosen validation/test target across
   multiple seeds and larger datasets. Include runs that exhaust their budgets;
   the exploratory 18.98x result is not a universal acceleration factor.
3. Add independent Si/Ge/SiGe data and domain holdouts, EOS, strained virials,
   short MD stability and energy-drift checks. No material-specific quality or
   transferability claim is made from the small PbTe example.
4. Profile basis construction and solver costs. If projection cannot recover its
   overhead at matched quality, stop this route rather than accumulate heuristics.

CPU tests require ordinary fully periodic potential models. Charge, temperature,
dipole, polarizability, and nonperiodic oracle modes are rejected. The native
experimental path supports the standard NEP5 energy terms described above,
subject to the stated `l_max` and capacity limits; it does not cover every NEP
inference mode or guarantee MD compatibility at native GNEP's maximum sizes.
