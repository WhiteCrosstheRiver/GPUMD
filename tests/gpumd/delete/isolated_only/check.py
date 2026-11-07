#!/usr/bin/env python3
import sys

with open("after.xyz") as f:
    n = int(f.readline())
    f.readline()
    zs = []
    for _ in range(n):
        parts = f.readline().split()
        zs.append(float(parts[3]))

if n != 9:
    sys.exit(f"only should keep crystal+middle, got {n}")
high = [z for z in zs if z > 10.0]
if len(high) != 1 or abs(high[0] - 21.5) > 1e-6:
    sys.exit(f"expected the middle chain atom at z=21.5, got {high}")
print("delete isolated only: PASS")
