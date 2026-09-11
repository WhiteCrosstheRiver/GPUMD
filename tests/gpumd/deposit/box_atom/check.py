#!/usr/bin/env python3
import sys

with open("after.xyz") as f:
    n = int(f.readline())
    header = f.readline()
    rows = [f.readline().split() for _ in range(n)]

if n != 13:
    sys.exit(f"expected 8+5 atoms, got {n}")

# last 5 deposited atoms should lie in the requested box
for row in rows[8:]:
    x, y, z = map(float, row[1:4])
    if not (0.0 <= x <= 3.57 and 0.0 <= y <= 3.57 and 20.0 <= z <= 30.0):
        sys.exit(f"deposited atom outside box: {x} {y} {z}")
print("deposit box atom: PASS")
