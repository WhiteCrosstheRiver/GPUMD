# Named Deposition GPU Regression

This suite exercises the staged `deposit NAME ...` syntax with isolated GPUMD
jobs. It generates every `run.in`, `model.xyz`, molecule file, and copied
potential in a new output directory. It never rewrites repository fixtures.

```sh
python check.py --engine /path/to/gpumd --repo /path/to/source \
  --output /new/output/named-deposit --gpu 0
```

`--root` is an alias for `--repo`. The default potential is
`potentials/nep/C_2022_NEP3.txt`; the mixture case copies the bundled SiC
Tersoff potential. `--potential` can override the default carbon potential.
`--case NAME` runs a selected case and can be repeated. A failed case leaves
its input, complete log, and any XYZ output under the output directory.
`results.json` records each case's verdict and observations. An existing
`--output` path is rejected so previous results remain intact.

The suite checks configuration without emission, all placement styles, atom,
mixture and multi-atom molecule particles, velocity modes and units, named
isolation, facet replacement, seed progression, all/mobile surface scope,
flight skip, pulse entry/timeout, and rejected invalid or legacy commands. The
`flight_default_equals_off`, `flight_pair`, and `seed_progression` checks compare
separate jobs after all cases finish. A pulse with entry already satisfied and
`relax 0` has zero MD steps and therefore no step-based dump frame; the
`pulse_zero_step` case verifies its accounting and the clock of a following run.
