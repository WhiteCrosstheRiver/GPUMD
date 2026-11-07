# Deposition Surface And Near Regressions

These GPU tests generate independent inputs in a new output directory. Supply a
GPUMD executable and a Si/Ge NEP potential:

```sh
python3 check.py --gpumd /absolute/path/gpumd --potential /absolute/path/nep.txt --output /new/output/directory --gpu 0
```

Seven cases use named deposition recipes and must reject overlapping placements
with the specific deposition error, not an unrelated parser error: a periodic
normal, a tilted normal, grid sites across the lateral seam, deferred grid
insertion in a sequence, a preceding command without `exclusion distance`, a
periodic axis narrower than the exclusion distance, and a triclinic periodic
image.

Four controls must finish and emit the expected atom count and launch heights:
normal global-surface insertion, global/global, global/local, and local/global
command sequences. The tiny integration step is only for XYZ readback. Results
and complete logs are retained under the chosen output directory.
