#!/bin/bash
# Forces with the influence halo rebuilt every step (ref) vs reused with a skin (reuse).
# Same seed and ensemble; early steps must agree to float round-off.
set -e
GPUMD=${GPUMD:-../../../../src/gpumd}
python3 make_model.py
rm -f force.out force_ref.out
cp run_ref.in run.in && $GPUMD > log_ref.txt && mv -f force.out force_ref.out
cp run_reuse.in run.in && $GPUMD > log_reuse.txt && mv -f force.out force_reuse.out
python3 compare.py model.xyz force_ref.out force_reuse.out
