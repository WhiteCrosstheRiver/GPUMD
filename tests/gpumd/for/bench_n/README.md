# Dynamic-N mutation timing

Measure `T_rebuild / T_total` for `deposit` + `run 100` loops. No dumps.

```text
/bin/cp -f run_a.in run.in   # 1000 atoms, 200 cycles
/bin/cp -f run_b.in run.in   # 10648 atoms, 100 cycles
/bin/cp -f run_c.in run.in   # 115000 atoms, 50 cycles
CUDA_VISIBLE_DEVICES=0 ../../../../src/gpumd | tee log.txt
```

Sum the four timers:

```text
awk '
/Time used for CPU sync =/ { sync += $(NF-1) }
/Time used for deposit =/ { dep += $(NF-1) }
/Time used for atom-count rebuild =/ { reb += $(NF-1) }
/Time used for this run =/ { run += $(NF-1) }
END {
  tot = sync + dep + run
  printf "sync=%.6f deposit=%.6f rebuild=%.6f run=%.6f total=%.6f rebuild/total=%.4f\n",
    sync, dep, reb, run, tot, (tot > 0 ? reb / tot : 0)
}
' log.txt
```

`T_mutation ≈ T_rebuild`. `T_sampling ≈ T_deposit - T_rebuild`.

If rebuild/total is below 5%, leave the full GPU rebuild as-is. If it reaches ~20%, consider capacity later.
