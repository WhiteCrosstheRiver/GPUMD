#!/usr/bin/env python3
import sys

frames = []
with open("after.xyz") as f:
    while True:
        line = f.readline()
        if not line:
            break
        n = int(line)
        header = f.readline()
        rows = [f.readline().split() for _ in range(n)]
        frames.append((n, header, rows))

if not frames:
    sys.exit("no frames in after.xyz")
n, header, rows = frames[-1]

if n != 13:
    sys.exit(f"expected 8+5 atoms, got {n}")

for row in rows[8:]:
    x, y, z = map(float, row[1:4])
    if not (0.0 <= x <= 3.57 and 0.0 <= y <= 3.57 and abs(z - 25.0) < 1.0e-8):
        sys.exit(f"deposited atom outside region/surface: {x} {y} {z}")
print("deposit box atom: PASS")
