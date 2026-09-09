#!/usr/bin/env python3
import sys

with open("after.xyz") as f:
    n = int(f.readline())
    f.readline()
    zs = []
    for _ in range(n):
        parts = f.readline().split()
        zs.append(float(parts[3]))

if n != 1:
    sys.exit(f"expected only the fixed flyer, got {n} atoms")
if zs[0] < 10.0:
    sys.exit(f"expected flyer z~20, got {zs[0]}")
print("delete never removes fix atoms: PASS")
