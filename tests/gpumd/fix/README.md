# `fix` force tests

`fix` freezes a group. These checks compare `dump_force` with and without `fix 1`.
`time_step 0` keeps the geometry identical.

```text
export GPUMD=../../../../src/gpumd

cd lj_active
bash ../run_compare.sh

cd ../nep_slab
bash ../run_compare.sh 2 2 2
```

`nep_slab` uses `replicate 2 2 2` so NEP takes the large-box path (influence halo).
Pass `2 2 2` into `compare_force.py` so group labels are repeated the same way as `replicate`.

Timing vs freeze fraction is in `bench_ratio/` (not CI).
