#!/usr/bin/env python3
import sys

frames = []
with open("after.xyz") as f:
    while True:
        line = f.readline()
        if not line:
            break
        n = int(line)
        f.readline()
        rows = [f.readline().split() for _ in range(n)]
        frames.append((n, rows))

if not frames:
    sys.exit("no frames in after.xyz")
n, rows = frames[-1]

if n != 12:
    sys.exit(f"expected 8+4 atoms, got {n}")

got = sorted((float(r[1]), float(r[2]), float(r[3])) for r in rows[8:])
want = sorted([(0.0, 0.0, 20.0), (0.0, 1.785, 20.0), (1.785, 0.0, 20.0), (1.785, 1.785, 20.0)])
for a, b in zip(got, want):
    if any(abs(x - y) > 1.0e-8 for x, y in zip(a, b)):
        sys.exit(f"grid sites {got} != {want}")
print("deposit grid: PASS")
