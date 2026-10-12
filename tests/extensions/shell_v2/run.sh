#!/usr/bin/env bash
# CPU tests for fix shell v2. Needs only g++ (C++17); no CUDA.
# shim/model/atom.cuh replaces the CUDA Atom class; model/box.cuh is the real one.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../../..
g++ -std=c++17 -O2 -pthread -D__host__= -D__device__= \
    -I shim -I "$ROOT/src" \
    test_shell_v2.cpp -o test_shell_v2
./test_shell_v2
