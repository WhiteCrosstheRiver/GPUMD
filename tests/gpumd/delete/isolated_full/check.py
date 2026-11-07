#!/usr/bin/env python3
import sys

with open("after.xyz") as f:
    n = int(f.readline())
    f.readline()
    zs = []
    for _ in range(n):
        parts = f.readline().split()
        zs.append(float(parts[3]))

if n != 8:
    sys.exit(f"full should delete the whole chain, got {n}")
if any(z > 10.0 for z in zs):
    sys.exit(f"chain still present, z={zs}")
print("delete isolated full: PASS")
