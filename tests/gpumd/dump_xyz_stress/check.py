#!/usr/bin/env python3
import math
import os
import re
import sys

TIME_UNIT_CONVERSION = 1.018051e1


def parse_properties(header):
    match = re.search(r"Properties=(\S+)", header)
    if not match:
        sys.exit("missing Properties= in header")
    fields = []
    offset = 0
    for token in match.group(1).split(":"):
        if not token:
            continue
        fields.append(token)
    cols = {}
    i = 0
    while i + 2 < len(fields):
        name, _kind, width = fields[i], fields[i + 1], int(fields[i + 2])
        cols[name] = (offset, width)
        offset += width
        i += 3
    return cols


def quoted_floats(header, key):
    match = re.search(r'%s="([^"]+)"' % key, header)
    if not match:
        sys.exit("missing %s= in header" % key)
    return [float(x) for x in match.group(1).split()]


def box_volume(lattice):
    ax, ay, az, bx, by, bz, cx, cy, cz = lattice
    return (
        ax * (by * cz - bz * cy)
        + ay * (bz * cx - bx * cz)
        + az * (bx * cy - by * cx)
    )


def read_extxyz(path):
    with open(path) as f:
        n = int(f.readline())
        header = f.readline()
        rows = [f.readline().split() for _ in range(n)]
    cols = parse_properties(header)
    return header, cols, rows


def col(rows, cols, name):
    start, width = cols[name]
    data = []
    for row in rows:
        data.append([float(row[start + k]) for k in range(width)])
    return data


def check_formula(path):
    header, cols, rows = read_extxyz(path)
    for name in ("mass", "vel", "virial", "volume_atom", "stress"):
        if name not in cols:
            sys.exit("%s: missing %s column" % (path, name))
    masses = [v[0] for v in col(rows, cols, "mass")]
    vels = col(rows, cols, "vel")
    virials = col(rows, cols, "virial")
    volumes = [v[0] for v in col(rows, cols, "volume_atom")]
    stresses = col(rows, cols, "stress")
    va = (0, 0, 0, 1, 1, 1, 2, 2, 2)
    vb = (0, 1, 2, 0, 1, 2, 0, 1, 2)
    max_rel = 0.0
    max_abs = 0.0
    for mass, vel, vir, vol, stress in zip(masses, vels, virials, volumes, stresses):
        vnat = [c * TIME_UNIT_CONVERSION for c in vel]
        for d in range(9):
            expected = (vir[d] + mass * vnat[va[d]] * vnat[vb[d]]) / vol
            err = abs(stress[d] - expected)
            max_abs = max(max_abs, err)
            scale = max(1e-8, abs(expected), abs(stress[d]))
            max_rel = max(max_rel, err / scale)
    if max_abs > 5e-8 and max_rel > 1e-6:
        sys.exit(
            "%s: per-atom stress formula failed, max_abs=%g max_rel=%g"
            % (path, max_abs, max_rel)
        )
    header_stress = quoted_floats(header, "stress")
    volume_box = box_volume(quoted_floats(header, "Lattice"))
    for d, hs in enumerate(header_stress):
        total = sum(vol * s[d] for vol, s in zip(volumes, stresses))
        reconstructed = total / volume_box
        if abs(reconstructed - hs) > 2e-8:
            sys.exit(
                "%s: sum(Omega sigma)/V component %d = %g != header = %g"
                % (path, d, reconstructed, hs)
            )
    return max_abs, max_rel


def check_isolated(path):
    header, cols, rows = read_extxyz(path)
    virials = col(rows, cols, "virial")
    for i, vir in enumerate(virials):
        if max(abs(x) for x in vir) > 1e-10:
            sys.exit("isolated: atom %d virial should be ~0, got %s" % (i, vir))
    max_abs, max_rel = check_formula(path)
    print("isolated: PASS formula max_abs=%.3e max_rel=%.3e" % (max_abs, max_rel))


def check_simple_cubic(path):
    max_abs, max_rel = check_formula(path)
    print("simple_cubic: PASS formula max_abs=%.3e max_rel=%.3e" % (max_abs, max_rel))


def check_stress_only(path):
    header, cols, _rows = read_extxyz(path)
    if "stress=" not in header:
        sys.exit("stress_only: missing header stress=")
    if "voronoi_method=\"ball_restricted\"" not in header:
        sys.exit("stress_only: missing voronoi_method in header")
    if "volume_atom" in cols:
        sys.exit("stress_only: volume_atom should not be written without volume")
    if "stress" not in cols or cols["stress"][1] != 9:
        sys.exit("stress_only: missing :stress:R:9")
    print("stress_only: PASS columns and header")


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    check_isolated(os.path.join(root, "isolated", "stress.xyz"))
    check_simple_cubic(os.path.join(root, "simple_cubic", "stress.xyz"))
    check_stress_only(os.path.join(root, "stress_only", "stress.xyz"))


if __name__ == "__main__":
    main()
