#!/bin/bash
set -e
GPUMD=${GPUMD:-../../../../src/gpumd}
python3 make_models.py
for r in 0 50 90; do
  cp -f model_${r}.xyz model.xyz
  cp -f run_${r}.in run.in
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $GPUMD | tee log_${r}.txt
done
awk '/Time used for this run =/ {print FILENAME, $(NF-1)}' log_0.txt log_50.txt log_90.txt
