#!/usr/bin/env python3
# Verifies 'surface local ... mobile': launch height follows the MOBILE surface
# (slab top 7.0 + gap 2.0 = 9.0), ignoring the fixed cap at z=20.
import sys

with open("after_mobile.xyz") as f:
    n = int(f.readline())
    f.readline()
    zs = [float(f.readline().split()[3]) for _ in range(n)]

near9 = [z for z in zs if 8.5 < z < 9.5]
near22 = [z for z in zs if 21.5 < z < 22.5]
if len(near9) != 1:
    sys.exit(f"expected exactly one deposited atom at z~9.0 (mobile top+gap), got {near9}")
if near22:
    sys.exit(f"deposit followed the fixed cap (z~22) instead of the mobile surface: {near22}")
print("deposit surface mobile: PASS")
