#!/usr/bin/env python3
"""Si(001) slab for the fix shell grow test: 6 x 6 x 4 diamond cells, bottom cell fixed (group 1)."""
import random

a = 5.43
nx, ny, nz = 6, 6, 4
basis = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0),
         (.25, .25, .25), (.25, .75, .75), (.75, .25, .75), (.75, .75, .25)]
random.seed(1)
atoms = []
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            for bx, by, bz in basis:
                x, y, z = (i + bx) * a, (j + by) * a, (k + bz) * a + 2.0
                atoms.append((x, y, z, 1 if k == 0 else 0))
lx, ly, lz = nx * a, ny * a, 200.0
with open("model.xyz", "w") as f:
    f.write(f"{len(atoms)}\n")
    f.write(f'pbc="T T F" Lattice="{lx} 0 0 0 {ly} 0 0 0 {lz}" '
            "Properties=species:S:1:pos:R:3:group:I:1\n")
    for x, y, z, g in atoms:
        f.write(f"Si {x:.6f} {y:.6f} {z:.6f} {g}\n")
print(f"wrote model.xyz with {len(atoms)} atoms, box {lx:.4f} x {ly:.4f} x {lz}")
