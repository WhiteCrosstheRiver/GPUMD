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
    sys.exit(f"expected 8 atoms after deleting the flyer, got {n}")
if any(z > 10.0 for z in zs):
    sys.exit(f"flyer still present, z={zs}")
print("delete isolated: PASS")
