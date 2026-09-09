#!/usr/bin/env python3
"""Check fix shell labels in labeled.xyz."""

import sys

atoms = []
with open("labeled.xyz") as f:
    n = int(f.readline())
    f.readline()
    for line in f:
        parts = line.split()
        if len(parts) < 5:
            continue
        atoms.append(
            {
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3]),
                "g": int(parts[4]),
            }
        )

if len(atoms) != n:
    sys.exit(f"expected {n} atoms, got {len(atoms)}")


def near(atom, x, y, z, tol=0.2):
    return abs(atom["x"] - x) < tol and abs(atom["y"] - y) < tol and abs(atom["z"] - z) < tol


def require(pred, msg):
    if not pred:
        sys.exit(msg)


flyer = [a for a in atoms if a["z"] > 20]
require(len(flyer) == 1 and flyer[0]["g"] == 0, "flying atom should be unfixed")

bottom = [a for a in atoms if a["z"] < 0.1]
require(bottom and all(a["g"] == 1 for a in bottom), "z=0 substrate should be frozen")

groove_floor = [a for a in atoms if near(a, 2.0, 2.0, 8.0)]
require(groove_floor and groove_floor[0]["g"] == 0, "groove floor should be unfixed")

mesa_top = [a for a in atoms if near(a, 0.0, 0.0, 14.0)]
require(mesa_top and mesa_top[0]["g"] == 0, "mesa top should be unfixed")

mesa_bulk = [a for a in atoms if near(a, 0.0, 0.0, 0.0)]
require(mesa_bulk and mesa_bulk[0]["g"] == 1, "mesa bulk should be frozen")

print("fix shell labels: PASS")
