#!/usr/bin/env python3
import sys

labels = []
with open("labeled.xyz") as f:
    n = int(f.readline())
    f.readline()
    for line in f:
        parts = line.split()
        if len(parts) >= 5:
            labels.append(int(parts[4]))

if len(labels) != 8:
    sys.exit(f"expected 8 atoms, got {len(labels)}")

# xyz groups 0,0,1,1,2,2,2,1 → freeze 1∪2 so only the first two stay 0
expect = [0, 0, 1, 1, 1, 1, 1, 1]
if labels != expect:
    sys.exit(f"labels {labels} != {expect}")
print("fix union 1 2: PASS")
