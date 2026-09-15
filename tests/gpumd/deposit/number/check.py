#!/usr/bin/env python3
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

if n != 13:
    sys.exit(f"expected 8+5 atoms, got {n}")
print("deposit number 5: PASS")
