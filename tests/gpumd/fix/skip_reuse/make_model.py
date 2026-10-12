#!/usr/bin/env python3
"""Si(001) slab, 6 x 6 x 6 diamond cells; atoms below z = 18 A are group 1 (fixed)."""
a = 5.43
basis = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0),
         (.25, .25, .25), (.25, .75, .75), (.75, .25, .75), (.75, .75, .25)]
atoms = []
for k in range(6):
    for j in range(6):
        for i in range(6):
            for bx, by, bz in basis:
                x, y, z = (i + bx) * a, (j + by) * a, (k + bz) * a + 2.0
                atoms.append((x, y, z, 1 if z < 18.0 else 0))
with open("model.xyz", "w") as f:
    f.write(f"{len(atoms)}\n")
    f.write(f'pbc="T T F" Lattice="{6 * a} 0 0 0 {6 * a} 0 0 0 200.0" '
            "Properties=species:S:1:pos:R:3:group:I:1\n")
    for x, y, z, g in atoms:
        f.write(f"Si {x:.6f} {y:.6f} {z:.6f} {g}\n")
print(f"wrote model.xyz: {len(atoms)} atoms, {sum(g for *_, g in atoms)} fixed")
