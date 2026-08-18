# Different fix-ratio timing (not CI)

Reuse [`../../deposite_O3/model.xyz`](../../deposite_O3/model.xyz) (N=36785 Si slab) and `nep.txt`.
Do not edit those originals.

The checked-in xyz only has ~6% `group=1`, so `fix 1` is not a high-freeze case.
`make_models.py` relabels by z (bottom frozen):

```text
python3 make_models.py
```

Then:

```text
export GPUMD=../../../../src/gpumd
for r in 0 50 90; do
  cp run_${r}.in run.in
  CUDA_VISIBLE_DEVICES=0 $GPUMD | tee log_${r}.txt
done
awk '/Time used for this run =/ {print FILENAME, $(NF-1)}' log_0.txt log_50.txt log_90.txt
```

NVE, 100 steps, no dump/deposit. Small-cell xyz is generated, not committed.

Million-atom check (replicate **after** which groups are labeled by **global** z, so the frozen bulk is actually thick):

```text
bash run_large.sh    # python3 make_models.py 3 3 3  ->  N = 36785*27 = 993195
```

Do not commit `model_m_*.xyz`.

A thin slab plus NEP cutoff can keep a large influence region at 90% fix.
The measured times are data, not a pass/fail gate.

Example on A100 (N=36785, NEP4 Si/O, `run 100`, 2026-08-18):

```text
fix 0%   0.226 s
fix 50%  0.261 s
fix 90%  0.217 s
```

90% is only slightly faster on the **thin** cell (z ≈ 27 Å): the NEP halo still covers most of the frozen bulk.

Same potential after `make_models.py 3 3 3` (N=993195, z ≈ 81 Å, `run 100`):

```text
fix 0%   3.446 s   2.88e7 atom*step/s
fix 50%  2.618 s   3.79e7 atom*step/s   (1.32x vs 0%)
fix 90%  1.288 s   7.71e7 atom*step/s   (2.68x vs 0%)
```

At million atoms with a thicker frozen substrate the skip path does win. It is still not 10x at 90% freeze: kernels still launch over all N and skip deep frozen threads, and the halo around the active surface remains. F5 (only launch influence cells) is the remaining lever.
