#!/usr/bin/env python3
"""Compare dump_force output: active atoms match no-fix; fixed atoms are ~0."""
import sys


def load_xyz_groups(path):
    groups = []
    with open(path) as f:
        n = int(f.readline())
        f.readline()
        for _ in range(n):
            groups.append(int(f.readline().split()[-1]))
    return groups


def load_force(path):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                rows.append((float(p[0]), float(p[1]), float(p[2])))
    return rows


def main():
    if len(sys.argv) not in (4, 7):
        print("usage: compare_force.py model.xyz force_nofix.out force_fix.out [nx ny nz]")
        return 2
    groups = load_xyz_groups(sys.argv[1])
    if len(sys.argv) == 7:
        nx, ny, nz = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
        groups = groups * (nx * ny * nz)
    f_all = load_force(sys.argv[2])
    f_fix = load_force(sys.argv[3])
    if len(f_all) != len(groups) or len(f_fix) != len(groups):
        print("atom count mismatch", len(groups), len(f_all), len(f_fix))
        return 1
    max_active = 0.0
    max_fixed = 0.0
    n_active = 0
    n_fixed = 0
    for g, a, b in zip(groups, f_all, f_fix):
        if g == 1:
            n_fixed += 1
            max_fixed = max(max_fixed, abs(b[0]), abs(b[1]), abs(b[2]))
        else:
            n_active += 1
            max_active = max(
                max_active,
                abs(a[0] - b[0]),
                abs(a[1] - b[1]),
                abs(a[2] - b[2]),
            )
    print(
        "n_active=%d n_fixed=%d max_active_force_diff=%.3e max_fixed_force=%.3e"
        % (n_active, n_fixed, max_active, max_fixed)
    )
    if max_active > 1e-6:
        print("FAIL: active forces differ from full compute")
        return 1
    if max_fixed > 1e-12:
        print("FAIL: fixed forces are not zero")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
