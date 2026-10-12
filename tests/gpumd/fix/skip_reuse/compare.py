#!/usr/bin/env python3
"""Check complete fixed/unfixed forces across the skin reuse run."""
import math
from pathlib import Path
import sys


EXPECTED_STEPS = 40
TOL = 1e-4


def force_frames(path, atoms):
    lines = Path(path).read_text().splitlines()
    if len(lines) != EXPECTED_STEPS * atoms:
        raise ValueError(f'{path}: expected {EXPECTED_STEPS * atoms} force rows, got {len(lines)}')
    rows = []
    for index, line in enumerate(lines):
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(f'{path}: row {index + 1} has fewer than three components')
        values = tuple(map(float, fields[:3]))
        if not all(map(math.isfinite, values)):
            raise ValueError(f'{path}: row {index + 1} has nonfinite force')
        rows.append(values)
    return [rows[i:i + atoms] for i in range(0, len(rows), atoms)]


def main():
    model, reference, reuse = map(Path, sys.argv[1:4])
    lines = model.read_text().splitlines()
    atoms = int(lines[0])
    groups = [int(line.split()[-1]) for line in lines[2:]]
    if len(groups) != atoms or not {0, 1}.issuperset(groups) or 0 not in groups or 1 not in groups:
        raise ValueError('invalid model group labels')
    ref = force_frames(reference, atoms)
    new = force_frames(reuse, atoms)
    errors = []
    for step, (a, b) in enumerate(zip(ref, new)):
        delta = max(abs(a[i][axis] - b[i][axis]) for i in range(atoms)
                    if groups[i] == 0 for axis in range(3))
        fixed = max(abs(b[i][axis]) for i in range(atoms)
                    if groups[i] == 1 for axis in range(3))
        print(f'step {step:3d}: max |dF| unfixed {delta:.3e} eV/A, max |F| fixed {fixed:.3e}')
        if delta >= TOL or fixed >= TOL:
            errors.append({'step': step, 'mobile_delta': delta, 'fixed_force': fixed})
    if errors:
        print(f'skip reuse: FAIL ({len(errors)} steps exceed {TOL:g} eV/A): {errors[:4]}')
        return 1
    print(f'skip reuse: PASS ({EXPECTED_STEPS} complete steps, max error < {TOL:g} eV/A)')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, IndexError) as exc:
        raise SystemExit(f'skip reuse: FAIL ({exc})')
