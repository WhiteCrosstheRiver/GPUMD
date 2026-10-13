# Tall-slab fixed-region validation

This opt-in GPU benchmark separates frozen-atom selection from force skipping.
It builds diamond Si slabs with a nearly constant mobile surface band while the
frozen depth increases. It never modifies the executable or existing results.
Python 3 and NumPy are required. Select an idle GPU and a new output directory.

From this directory:

```sh
python3 benchmark.py --root /tmp/tall-slab-new --gpu 0
python3 verify.py --root /tmp/tall-slab-new
python3 compatibility.py --root /tmp/tall-slab-new --gpu 0 \
    --baseline /path/to/preserved/baseline/gpumd
```

The repository executable and bundled `Si_2022_NEP3_4body.txt` are defaults.
`--engine` selects another executable. `benchmark.py --potential` may select a
different compatible Si NEP model, but the model generator is specifically Si:
this fixture is not an automatic test of arbitrary species or interactions.
The compatibility controls deliberately use the bundled Si NEP and Tersoff.

A bounded smoke test uses:

```sh
python3 benchmark.py --root /tmp/tall-slab-smoke-new --gpu 0 \
    --nz 24 --steps 100 --repetitions 1
python3 verify.py --root /tmp/tall-slab-smoke-new
python3 compatibility.py --root /tmp/tall-slab-smoke-new --gpu 0 --nz 24 \
    --baseline /path/to/preserved/baseline/gpumd
```

## Protocol

- Lattice constant 5.431 A; lateral size 8 by 8 cells; x/y periodic, z free.
- Default heights: 24, 72, 216 cells. A reproducible small positional jitter
  avoids testing only an exactly zero-force equilibrium configuration.
- `fix shell ... offset 38.017 surface radius 3 mode grow` selects the frozen
  band. All comparison lanes start from the exact same exported positions,
  velocities, species and group labels.
- `full`: `fix 1` with `full_observables` before every run. Atoms remain frozen,
  but force/energy/virial evaluation takes the full path.
- `skip`: `fix 1` with the default active-region path.
- `reuse`: `fix_skip min_frozen 0 rebuild 20 skin 1 cache_alloc 1`.
- NVE, 0.5 fs; 200 warm-up steps, then 2000 timed steps. Three repetitions with
  rotated lane order; use the median internal MD time. Startup, classification,
  and final output are excluded from the timed interval, but wall times are saved.
- Separate 40-step controls compare active forces and positions. The verifier
  also checks the longer timing-run endpoints and confirms the selected paths.
- Fixed atoms must not move, and exported positions, velocities and forces must
  remain finite. Force tolerance is 1e-4 eV/A; position tolerance is 1e-6 A.
- Legacy controls compare a supplied baseline executable against the candidate
  for no fix, ordinary fix, old fix-shell syntax, and Tersoff. Old-shell testing
  starts from the original bottom-only fixed model, not the grow-frozen union.

For changed potentials or longer chaotic trajectories, an endpoint mismatch
needs investigation; it is not by itself a proof of incorrect instantaneous
forces. Do not weaken thresholds silently to obtain a passing report.

## Recorded acceptance, 2026-10-13

Engine source: `313ab8c84666750964e36f6c0556727424236581`.
Executable SHA-256:
`0fc8c7ced4c23cd8bc82c3fa6efccf596db3408bac57b9267182a06800321f85`.
Hardware: one NVIDIA A100-PCIE-40GB; lanes executed serially.

| Atoms | Mobile | Frozen | Full, s | Skip, s | Reuse, s | Full/reuse |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12,288 | 3,641 | 70.37% | 4.18687 | 3.50822 | 3.45814 | 1.21 |
| 36,864 | 3,640 | 90.13% | 8.18236 | 3.56314 | 3.51214 | 2.33 |
| 110,592 | 3,637 | 96.71% | 21.17060 | 3.70934 | 3.62791 | 5.84 |

At approximately constant mobile count, 9 times as many total atoms increased
reuse-mode time by 4.91% in this size range. All short controls and 18 longer
endpoint comparisons had zero position/active-force difference at output
precision. The four baseline compatibility controls also passed.

This is **full versus local evaluation at identical fixed masks**, not a 5.84x
version-to-version improvement. The preserved baseline already supports NEP
force skipping. Nor is it equivalence between deeply frozen and fully mobile
physical systems. Generality is protected by retaining complete-evaluation
fallbacks, not by assuming every potential admits the same local optimization.
DFTD3, all NEP parameter combinations, all ensembles, HIP, and other GPUs are
not exhaustively certified by this benchmark. Full-system thermodynamic
observables require the full path; skipped deep contributions must not be
presented as complete energies or virials.

Do not commit generated structures, trajectories, logs, or timing reports.
