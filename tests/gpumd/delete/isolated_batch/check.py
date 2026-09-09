#!/usr/bin/env python3
import sys

with open("gpumd.log") as f:
    log = f.read()
if "Isolated neighbor cells built once for 2 delete commands." not in log:
    sys.exit("expected one shared cell list for the two consecutive deletes")

with open("after.xyz") as f:
    n = int(f.readline())
    f.readline()
    zs = []
    for _ in range(n):
        parts = f.readline().split()
        zs.append(float(parts[3]))

# Sequential: first coord<2 drops the chain ends; the middle then has 0 neighbors
# and the second delete must remove it. Marking both on the original graph would
# leave the middle atom (9 atoms).
if n != 8:
    sys.exit(f"expected 8 atoms after sequential isolated deletes, got {n}")
if any(z > 10.0 for z in zs):
    sys.exit(f"chain atom still present, z={zs}")
print("delete isolated batch: PASS")
