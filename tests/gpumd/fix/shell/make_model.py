#!/usr/bin/env python3
"""Simple-cubic C slab with a central groove and one flying atom."""

a = 2.0
nx, ny, nz = 4, 4, 8
atoms = []
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            # Groove: drop the top three layers in the central 2x2.
            if i in (1, 2) and j in (1, 2) and k >= 5:
                continue
            atoms.append((i * a, j * a, k * a))
atoms.append((1.0, 1.0, 30.0))

lx, ly, lz = nx * a, ny * a, 40.0
with open("model.xyz", "w") as f:
    f.write(f"{len(atoms)}\n")
    f.write(
        f'pbc="T T T" Lattice="{lx} 0 0 0 {ly} 0 0 0 {lz}" Properties=species:S:1:pos:R:3\n'
    )
    for x, y, z in atoms:
        f.write(f"C {x:.4f} {y:.4f} {z:.4f}\n")
print(f"wrote model.xyz with {len(atoms)} atoms")
