#!/usr/bin/env python3
"""Per-step max |dF| over unfixed atoms between two dump_force files."""
import sys

groups = [int(l.split()[-1]) for l in open(sys.argv[1]).readlines()[2:]]
n = len(groups)
def blocks(path):
    rows = [tuple(map(float, l.split()[:3])) for l in open(path) if len(l.split()) >= 3]
    return [rows[k:k + n] for k in range(0, len(rows), n)]
ref, new = blocks(sys.argv[2]), blocks(sys.argv[3])
worst_early = 0.0
for step, (a, b) in enumerate(zip(ref, new)):
    d = max(max(abs(a[i][c] - b[i][c]) for c in range(3)) for i in range(n) if groups[i] == 0)
    fixed = max(max(abs(b[i][c]) for c in range(3)) for i in range(n) if groups[i] == 1)
    if step < 10:
        worst_early = max(worst_early, d)
    print(f"step {step:3d}: max |dF| unfixed {d:.3e} eV/A, max |F| fixed {fixed:.1e}")
ok = worst_early < 1e-4
print("skip reuse:", "PASS" if ok else "FAIL", f"(first 10 steps max {worst_early:.2e} eV/A)")
sys.exit(0 if ok else 1)
