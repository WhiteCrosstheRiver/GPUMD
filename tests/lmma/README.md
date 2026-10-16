# Experimental LM-MA-ES tests

The production default remains SNES. Read
[`doc/nep/experimental_lmma.rst`](../../doc/nep/experimental_lmma.rst)
before using the experimental optimizer. No faster time-to-accuracy is claimed.

## CPU tests

From the repository root, with a C++17 compiler and Python + NumPy:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -pedantic -I src/main_nep tests/lmma/test_lmma_core.cpp -o /tmp/test_lmma_core
/tmp/test_lmma_core
python3 tests/lmma/verify_lmma_reference.py --exe /tmp/test_lmma_core
```

The first command tests input guards, seeded reproducibility, ask/tell ordering,
checkpoint output, first-generation isotropy, and sphere convergence. The
independent NumPy check compares complete mean/path/memory/sigma states in
128-, 257-, and 4,025-dimensional problems. The largest case covers activation
of every memory direction. It checks Algorithm 1's squared-path sigma rule,
not the distinct norm/chi rule in the authors' public example.

On Windows, compile with `cl /std:c++17 /O2 /EHsc /I src/main_nep` and use an
explicit executable path with `--exe`. Tests write and remove a temporary
checkpoint in the current directory; run concurrent copies in separate folders.

## GPU integration

`smoke_lmma_gpu.py --help` describes the optional integration test. It requires
a built candidate `nep`, Python + NumPy, and a free GPU. Supplying an unchanged
baseline `nep` enables default-SNES parity checks. Use a new output directory;
the script must not overwrite production runs. It generates a tiny synthetic
mixed-element fixture solely for software testing, not an accuracy benchmark.

The large SiGe training dataset, checkpoints, generated outputs, credentials,
and machine-specific experiment paths are intentionally not part of this test
directory or the published branch.

## Verified on 2026-10-16

- GCC C++17 core tests and all three independent NumPy recurrence tests passed.
- A fresh CUDA build targeting `sm_80` passed.
- All 12 GPU smoke checks passed on an A100, including the unchanged baseline
  built from `e9a4b3bbdcf4e42b781bd892a2e982dc6817769f`.
- Unset/default, explicit `snes`, and baseline `loss.out`, `nep.txt`, and
  `nep.restart` were byte-for-byte identical on the synthetic fixture.
- LMMA's nonzero L1/L2 reports were `0.02405` / `0.02028`; independently
  computed values were `0.0240512807521` / `0.0202843657954`, within output
  precision. This specifically covers initialization of the GPU parameter-type
  map in the experimental path.

These are software validation results, not a convergence or speedup claim.
