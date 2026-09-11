#!/usr/bin/env python3
import os
import re
import sys

TIME_UNIT_CONVERSION = 1.018051e1
PRESSURE_UNIT_CONVERSION = 1.602177e2


def parse_properties(header):
    match = re.search(r"Properties=(\S+)", header)
    if not match:
        sys.exit("missing Properties= in header")
    fields = [token for token in match.group(1).split(":") if token]
    cols = {}
    offset = 0
    i = 0
    while i + 2 < len(fields):
        name, _kind, width = fields[i], fields[i + 1], int(fields[i + 2])
        cols[name] = (offset, width)
        offset += width
        i += 3
    return cols


def header_float(header, key):
    match = re.search(r"(?:^|\s)%s=([^\s]+)" % key, header)
    if not match:
        sys.exit("missing %s= in header" % key)
    return float(match.group(1))


def quoted_floats(header, key):
    match = re.search(r'%s="([^"]+)"' % key, header)
    if not match:
        sys.exit("missing %s= in header" % key)
    return [float(x) for x in match.group(1).split()]


def box_volume(lattice):
    ax, ay, az, bx, by, bz, cx, cy, cz = lattice
    return abs(
        ax * (by * cz - bz * cy)
        + ay * (bz * cx - bx * cz)
        + az * (bx * cy - by * cx)
    )


def read_extxyz(path):
    with open(path) as f:
        n = int(f.readline())
        header = f.readline()
        rows = [f.readline().split() for _ in range(n)]
    return header, parse_properties(header), rows


def col(rows, cols, name):
    start, width = cols[name]
    return [[float(row[start + k]) for k in range(width)] for row in rows]


def atom_A(mass, vel_Afs, vir):
    vnat = [c * TIME_UNIT_CONVERSION for c in vel_Afs]
    return vir[0] + vir[4] + vir[8] + mass * (vnat[0] ** 2 + vnat[1] ** 2 + vnat[2] ** 2)


def check_formula(path):
    header, cols, rows = read_extxyz(path)
    for name in ("mass", "vel", "virial", "volume_atom", "pressure"):
        if name not in cols:
            sys.exit("%s: missing %s column" % (path, name))
    masses = [v[0] for v in col(rows, cols, "mass")]
    vels = col(rows, cols, "vel")
    virials = col(rows, cols, "virial")
    volumes = [v[0] for v in col(rows, cols, "volume_atom")]
    pressures = [v[0] for v in col(rows, cols, "pressure")]
    max_abs = 0.0
    max_rel = 0.0
    sum_omega_p = 0.0
    sum_A = 0.0
    for mass, vel, vir, vol, p in zip(masses, vels, virials, volumes, pressures):
        A = atom_A(mass, vel, vir)
        expected = PRESSURE_UNIT_CONVERSION * A / (3.0 * vol)
        err = abs(p - expected)
        max_abs = max(max_abs, err)
        scale = max(1e-8, abs(expected), abs(p))
        max_rel = max(max_rel, err / scale)
        sum_omega_p += vol * p
        sum_A += A
    if max_abs > 5e-7:
        sys.exit(
            "%s: per-atom pressure formula failed, max_abs=%g max_rel=%g"
            % (path, max_abs, max_rel)
        )
    volume_box = box_volume(quoted_floats(header, "Lattice"))
    header_p = header_float(header, "pressure")
    expected_box = PRESSURE_UNIT_CONVERSION * sum_A / (3.0 * volume_box)
    if abs(header_p - expected_box) > 2e-8:
        sys.exit(
            "%s: header pressure %g != C_P sum(A)/(3V) = %g"
            % (path, header_p, expected_box)
        )
    reconstructed = sum_omega_p / volume_box
    if abs(reconstructed - header_p) > 2e-8:
        sys.exit(
            "%s: sum(Omega p)/V = %g != header pressure = %g"
            % (path, reconstructed, header_p)
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


def check_pressure_only(path):
    header, cols, _rows = read_extxyz(path)
    if "pressure=" not in header:
        sys.exit("pressure_only: missing header pressure=")
    if "voronoi_method=\"ball_restricted\"" not in header:
        sys.exit("pressure_only: missing voronoi_method in header")
    if "volume_atom" in cols:
        sys.exit("pressure_only: volume_atom should not be written without volume")
    if "pressure" not in cols or cols["pressure"][1] != 1:
        sys.exit("pressure_only: missing :pressure:R:1")
    print("pressure_only: PASS columns and header")


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    check_isolated(os.path.join(root, "isolated", "pressure.xyz"))
    check_simple_cubic(os.path.join(root, "simple_cubic", "pressure.xyz"))
    check_pressure_only(os.path.join(root, "pressure_only", "pressure.xyz"))


if __name__ == "__main__":
    main()
