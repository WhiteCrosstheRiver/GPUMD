# Native NEP-LM Verification

These scripts exercise `optimizer lm` through public `gnep.in` keywords. They
create small random-start fixtures with separate training, guard, and test files.
An idle CUDA GPU, Python 3, and NumPy are required. Each output directory must be
new; inputs, models, complete logs, binary hashes, and `verification.json` are
retained there.

From the repository root, run the six native generality cases:

```bash
python3 tools/nep_lm/verify_generality.py \
  --binary ./src/gnep --gpu 0 --output ./local/nep_lm/generality
```

The cases cover pure radial descriptors and isolated atoms, missing virials and
zero shear weight, four elements, 220 descriptors with basis count 20, and fixed
or flexible ZBL with typewise cutoffs and a real NEP/ZBL neighbor union. They
require native descriptor and analytical-gradient agreement, finite outputs,
decreasing accepted objectives, and the historical guard cap. Add
`--baseline /absolute/path/to/previous/gnep` to compare default Adam restart and
model files byte for byte. Use repeatable `--case NAME` arguments to select cases;
`--help` lists their names.

The weighted row and export suite additionally compares three/four/five-body
native predictions against the independent NEP_CPU bridge:

```bash
python3 tools/nep_lm/verify_rows_and_export.py \
  --root . --bridge ./local/fcvp/libfcvp.so --gpu 0 \
  --output ./local/nep_lm/rows_and_export
```

The bridge can be built using the pinned NEP_CPU checkout described in
`../fcvp/README.md`:

```bash
python3 tools/fcvp/build_cpu.py \
  --nep-cpu /absolute/path/to/NEP_CPU --output ./local/fcvp/libfcvp.so
```

All six weighted cases require the native descriptor and exact-gradient gates.
Five also compare exported energy, force, and virial predictions against NEP_CPU.
The extended typewise cutoff header is unsupported by the pinned CPU parser;
that case is explicitly recorded as native-only. Fixed and flexible embedded ZBL
exports are included. These are correctness checks, not convergence or speed
benchmarks. The native maximum-descriptor case may exceed the separate GPUMD MD
inference capacity; its gate checks the GNEP trainer.

The scripts set only device and CPU thread environment controls. Optimizer,
probe count, damping candidates, EMA, initial damping, gradient checking, and
guard selection are configured in `gnep.in`:

```text
optimizer lm
lm 4 3 0.2 0.01
lm_guard guard.xyz 1.1
lm_check 1
```
