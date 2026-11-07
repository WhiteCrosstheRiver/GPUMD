#!/usr/bin/env python3
# Verifies consecutive deposit commands batch correctly:
# atom count matches, and deposited coordinates are identical to the
# per-command path (checked externally against deposit/seq_equiv).
import sys

n = None
with open("after.xyz") as f:
    while True:
        line = f.readline()
        if not line:
            break
        n = int(line)
        f.readline()
        for _ in range(n):
            f.readline()

if n != 14:
    sys.exit(f"expected 8+6 atoms, got {n}")

# deposited atoms must sit in two height bands (surface stacks per command)
zs = []
with open("after.xyz") as f:
    while True:
        line = f.readline()
        if not line:
            break
        m = int(line)
        f.readline()
        for _ in range(m):
            p = f.readline().split()
            zs.append(float(p[3]))

above = sorted(z for z in zs if z > 3.0)
if len(above) != 6:
    sys.exit(f"expected 6 deposited atoms above the slab, got {len(above)}")
# command 1 launches at H=4.6775 (clears the slab);
# commands 2,3 must stack above command 1's atoms (H=6.6775, 8.6775)
bands = sorted(set(round(z, 4) for z in above))
if bands != [4.6775, 6.6775, 8.6775]:
    sys.exit(f"expected launch bands 4.6775/6.6775/8.6775, got {bands}")

print("deposit batch equiv: PASS")
