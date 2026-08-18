#!/bin/bash
# Million-atom timing: replicate 3 3 3 then relabel by global z.
# Do not commit the generated model_m_*.xyz files.
set -e
GPUMD=${GPUMD:-../../../../src/gpumd}
python3 make_models.py 3 3 3
for r in 0 50 90; do
  cp -f model_m_${r}.xyz model.xyz
  cp -f run_${r}.in run.in
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $GPUMD | tee log_m_${r}.txt
done
echo "---- million-atom (3x3x3) ----"
awk '/Time used for this run =/ {print FILENAME, $(NF-1)}' log_m_0.txt log_m_50.txt log_m_90.txt
awk '/Speed of this run =/ {print FILENAME, $(NF-1)}' log_m_0.txt log_m_50.txt log_m_90.txt
