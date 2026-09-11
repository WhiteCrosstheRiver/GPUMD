#!/usr/bin/env python3
import math
import os
import re
import sys

TIME_UNIT_CONVERSION = 1.018051e1


def parse_properties(header):
    match = re.search(r"Properties=(\S+)", header)
    if not match:
        sys.exit("missing Properties=")
    fields = [t for t in match.group(1).split(":") if t]
    cols = {}
    offset = 0
    i = 0
    while i + 2 < len(fields):
        name, _kind, width = fields[i], fields[i + 1], int(fields[i + 2])
        cols[name] = (offset, width)
        offset += width
        i += 3
    return cols


def col(rows, cols, name):
    start, width = cols[name]
    return [[float(row[start + k]) for k in range(width)] for row in rows]


def main():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isolated", "dump.xyz")
    with open(path) as f:
        n = int(f.readline())
        header = f.readline()
        rows = [f.readline().split() for _ in range(n)]
    cols = parse_properties(header)
    for name, width in (
        ("vel", 3),
        ("speed", 1),
        ("forces", 3),
        ("force_norm", 1),
        ("stress", 9),
        ("stress_norm", 1),
    ):
        if name not in cols or cols[name][1] != width:
            sys.exit("missing or wrong width for %s" % name)
    vels = col(rows, cols, "vel")
    speeds = [v[0] for v in col(rows, cols, "speed")]
    forces = col(rows, cols, "forces")
    force_norms = [v[0] for v in col(rows, cols, "force_norm")]
    stresses = col(rows, cols, "stress")
    stress_norms = [v[0] for v in col(rows, cols, "stress_norm")]
    for vel, speed in zip(vels, speeds):
        expected = math.sqrt(sum(c * c for c in vel))
        if abs(speed - expected) > 1e-7:
            sys.exit("speed %g != |vel| %g" % (speed, expected))
    for force, fn in zip(forces, force_norms):
        expected = math.sqrt(sum(c * c for c in force))
        if abs(fn - expected) > 1e-7:
            sys.exit("force_norm %g != |F| %g" % (fn, expected))
    for stress, sn in zip(stresses, stress_norms):
        expected = math.sqrt(sum(c * c for c in stress))
        if abs(sn - expected) > 1e-7:
            sys.exit("stress_norm %g != ||sigma||_F %g" % (sn, expected))
    print("norms: PASS")


if __name__ == "__main__":
    main()
