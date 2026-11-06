# Single-GPU LM Training

This is a complete `gnep.in` template for the measured Si/Ge configuration.
It is read by `gnep`, not the separate SNES executable `nep`.
No dataset or trained checkpoint is supplied in this example directory.

The working directory needs:

| File | Purpose |
|---|---|
| `gnep.in` | Model, loss, and optimizer settings |
| `train.xyz` | Fitting structures in the native extended-XYZ format |
| `guard.xyz` | Separate, training-derived structures for LM step selection |
| `test.xyz` | Optional held-out validation, never used to select LM steps |

The benchmark reserved about 5% of the original training set as guard, keeping
identical geometry groups together. Keep final validation out of the guard.
This template does not automatically split or deduplicate data.

From this directory, after building `src/gnep`, run on one GPU:

```sh
CUDA_VISIBLE_DEVICES=0 ../../../src/gnep
```

The template uses the tested LM defaults, equivalent to `lm 16 8 0.2 0.01`.
Only three additions are needed in an otherwise compatible GNEP input:

```text
optimizer lm
loss_rmse 1e-7
lm_guard guard.xyz 1.1
```

`epoch 10` is a fixed ten-epoch budget, not an automatic stop when a target
error is reached. Different datasets may need a different budget. Model sizes,
loss weights, and cutoff choices also remain dataset-specific.

Without `gnep.restart`, training initializes random parameters. With that file,
GNEP reads initial parameters from it; this does not restore LM damping, curvature,
or guard history. `nep.txt` is an exported trained potential, not this input file.

The default output files retain their native names; `lm.out` adds accepted-step
and guard diagnostics. Guard files must not alias inputs or fixed training outputs.
See `doc/nep/gnep_lm.rst` for the limits and full options.

The current LM implementation requires one visible GPU. Independent models can
run on different GPUs in separate working directories. A shared multi-GPU update
for one model has not been implemented or benchmarked.
