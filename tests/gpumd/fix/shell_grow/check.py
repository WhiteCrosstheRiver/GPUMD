#!/usr/bin/env python3
"""Checks for fix shell mode grow (run after gpumd in this directory).

  ratchet   an atom frozen in frame k (and not in group 1 of model.xyz) is frozen in k+1
  static    frozen atoms do not move between frames
  depth     every frozen non-group-1 atom lies at least offset - tol below the highest
            atom within the surface radius laterally (positions of the previous frame)
  progress  the frozen count grows and the skip path was used (log)
"""
import math
import sys

OFFSET, RADIUS, TOL = 8.0, 3.0, 1.5


def frames(path):
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            n = int(line)
            head = f.readline()
            lat = head.split('Lattice="')[1].split('"')[0].split()
            lx, ly = float(lat[0]), float(lat[4])
            rows = [f.readline().split() for _ in range(n)]
            yield lx, ly, [(float(r[1]), float(r[2]), float(r[3]), int(r[4])) for r in rows]


def fail(msg):
    sys.exit("FAIL: " + msg)


n_fixed0 = sum(1 for line in open("model.xyz").readlines()[2:] if line.split()[4] == "1")
prev = None
history = []
for k, (lx, ly, atoms) in enumerate(frames("grow.xyz")):
    frozen = [g == 1 for (_, _, _, g) in atoms]
    history.append(sum(frozen))
    if prev is not None:
        plx, ply, patoms = prev
        for i in range(len(patoms)):
            if patoms[i][3] == 1 and i >= n_fixed0 and atoms[i][3] != 1:
                fail(f"frame {k}: atom {i} was released (ratchet broken)")
            if patoms[i][3] == 1 and atoms[i][3] == 1:
                d = math.dist(patoms[i][:3], atoms[i][:3])
                if d > 1e-6 and abs(d - lx) > 1e-6 and abs(d - ly) > 1e-6:
                    fail(f"frame {k}: frozen atom {i} moved by {d:.3e} A")
        # depth rule on atoms newly frozen in this frame, against the previous positions
        for i in range(len(patoms)):
            if i < n_fixed0 or patoms[i][3] == 1 or atoms[i][3] != 1:
                continue
            x, y, z, _ = patoms[i]
            top = z
            for (xj, yj, zj, _) in patoms:
                dx = (xj - x) - plx * round((xj - x) / plx)
                dy = (yj - y) - ply * round((yj - y) / ply)
                if dx * dx + dy * dy <= (RADIUS + 1.0) ** 2:
                    top = max(top, zj)
            if z > top - OFFSET + TOL:
                fail(f"frame {k}: atom {i} frozen only {top - z:.2f} A below its local top")
    prev = (lx, ly, atoms)

if len(history) < 10:
    fail("too few frames in grow.xyz")
if history[-1] <= history[0]:
    fail(f"frozen count did not grow ({history[0]} -> {history[-1]})")
log = open("gpumd.log").read() if len(sys.argv) < 2 else open(sys.argv[1]).read()
if "Fix shell v2 (grow, incremental" not in log:
    fail("no incremental grow call in the log")
if "Fix skip is on" not in log:
    fail("skip path never enabled")
print(f"frames {len(history)}, frozen {history[0]} -> {history[-1]}: fix shell grow PASS")
