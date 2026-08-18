#!/usr/bin/env python3
"""Relabel deposite_O3/model.xyz by z. Optional: replicate nx ny nz first.

  python3 make_models.py           # original cell
  python3 make_models.py 3 3 3     # ~1e6 atoms, prefix model_m_
"""
import os
import re
import sys

src = os.path.join(os.path.dirname(__file__), "..", "..", "deposite_O3", "model.xyz")
out_dir = os.path.dirname(__file__)
nx = ny = nz = 1
if len(sys.argv) == 4:
    nx, ny, nz = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

with open(src) as f:
    n0 = int(f.readline())
    header = f.readline()
    rows0 = [f.readline().split() for _ in range(n0)]

m = re.search(r'Lattice="([^"]+)"', header)
vals = [float(x) for x in m.group(1).split()]
ax, ay, az = vals[0], vals[1], vals[2]
bx, by, bz = vals[3], vals[4], vals[5]
cx, cy, cz = vals[6], vals[7], vals[8]

species = [r[0] for r in rows0]
x0 = [float(r[1]) for r in rows0]
y0 = [float(r[2]) for r in rows0]
z0 = [float(r[3]) for r in rows0]

xs, ys, zs = [], [], []
sp = []
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            dx = i * ax + j * bx + k * cx
            dy = i * ay + j * by + k * cy
            dz = i * az + j * bz + k * cz
            for t in range(n0):
                sp.append(species[t])
                xs.append(x0[t] + dx)
                ys.append(y0[t] + dy)
                zs.append(z0[t] + dz)

n = n0 * nx * ny * nz
order = sorted(range(n), key=lambda t: zs[t])
prefix = "model_m_" if nx * ny * nz > 1 else "model_"
header_out = 'Lattice="%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f" Properties=species:S:1:pos:R:3:group:I:1\n' % (
    ax * nx,
    ay * nx,
    az * nx,
    bx * ny,
    by * ny,
    bz * ny,
    cx * nz,
    cy * nz,
    cz * nz,
)


def write(name, frac_fixed):
    nfix = int(round(n * frac_fixed))
    fixed = set(order[:nfix])
    path = os.path.join(out_dir, name)
    with open(path, "w") as g:
        g.write("%d\n" % n)
        g.write(header_out)
        for t in range(n):
            g.write("%s %.8f %.8f %.8f %d\n" % (sp[t], xs[t], ys[t], zs[t], 1 if t in fixed else 0))
    print(name, "N=%d nfix=%d (%.1f%%) box_z=%.3f" % (n, nfix, 100.0 * nfix / n, cz * nz))


write(prefix + "0.xyz", 0.0)
write(prefix + "50.xyz", 0.50)
write(prefix + "90.xyz", 0.90)
