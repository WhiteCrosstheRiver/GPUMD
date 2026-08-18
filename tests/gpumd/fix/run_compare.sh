#!/bin/bash
set -e
GPUMD=${GPUMD:-../../../../src/gpumd}

rm -f force.out force_nofix.out neighbor.out

cp run_nofix.in run.in
$GPUMD
mv -f force.out force_nofix.out

cp run_fix.in run.in
$GPUMD

python3 ../compare_force.py model.xyz force_nofix.out force.out "$@"
