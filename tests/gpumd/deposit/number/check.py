#!/usr/bin/env python3
import sys

with open("after.xyz") as f:
    n = int(f.readline())

if n != 13:
    sys.exit(f"expected 8+5 atoms, got {n}")
print("deposit number 5: PASS")
