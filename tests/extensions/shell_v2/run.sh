#!/usr/bin/env bash
# CPU tests for fix shell v2 and the topology cell lists. Needs only g++ (C++17).
# shim/model/atom.cuh replaces the CUDA Atom class; model/box.cuh is the real one.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../../..
FLAGS="-std=c++17 -O2 -pthread -D__host__= -D__device__= -I shim -I $ROOT -I $ROOT/src"
g++ $FLAGS test_shell_v2.cpp -o test_shell_v2
g++ $FLAGS test_neighbor_cells.cpp -o test_neighbor_cells
./test_neighbor_cells
./test_shell_v2
