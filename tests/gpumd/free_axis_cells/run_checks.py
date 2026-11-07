#!/usr/bin/env python3
"""Free-axis cell lists must leave neighbor lists unchanged and only change speed.

usage: run_checks.py --reference GPUMD --candidate GPUMD --nep SiGe_NEP --lj LJ_FILE
                     --work NEW_DIRECTORY --device ID

The reference is a build without free-axis cells. Neighbor lists are sorted, so forces must
match to the last bit, including for atoms outside the box along a free axis.
"""
import argparse
import math
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

A = 5.43
BASIS = [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5),
         (.25, .25, .25), (.75, .75, .25), (.75, .25, .75), (.25, .75, .75)]
MASS = {'Si': 28.085, 'Ge': 72.63, 'Ar': 39.948}


def model(path, symbols, spacing, n, temperature, pbc, tilt=0.0, stray=False, vacuum=30.0):
    rng = random.Random(2026)
    length = [n[d] * spacing + (0 if pbc[d] else vacuum) for d in range(3)]
    rows = []
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                for b in BASIS:
                    r = [(i + b[0]) * spacing, (j + b[1]) * spacing, (k + b[2]) * spacing]
                    r[0] += tilt * r[1] / length[1]
                    symbol = rng.choice(symbols)
                    sd = math.sqrt(8.617333e-5 * temperature / (MASS[symbol] * 103.6427))
                    v = [rng.gauss(0, sd) for _ in range(3)]
                    rows.append([symbol] + r + v)
    if stray:
        # Close pairs outside the box, so clamped cells must still find them.
        top = n[2] * spacing + vacuum
        for z in (top + 40.0, top + 42.3, -35.0, -37.3):
            symbol = symbols[0]
            rows.append([symbol, 3.0, 3.0, z, 0.0, 0.0, 0.0])
    with open(path, 'w') as f:
        f.write('%d\n' % len(rows))
        f.write('Lattice="%g 0 0 %g %g 0 0 0 %g" pbc="%s %s %s" Properties=species:S:1:pos:R:3:vel:R:3\n' % (
            length[0], tilt, length[1], length[2], *['T' if p else 'F' for p in pbc]))
        for r in rows:
            f.write('%s %.10f %.10f %.10f %.10e %.10e %.10e\n' % tuple(r))
    return len(rows)


def run(binary, folder, device):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
    with open(folder / 'run.log', 'w') as log:
        code = subprocess.run([binary], cwd=folder, env=env, stdout=log,
                              stderr=subprocess.STDOUT, timeout=900).returncode
    text = (folder / 'run.log').read_text()
    speed = re.findall(r'Speed of this run = (\S+)', text)
    return code, float(speed[-1]) if speed else 0.0


def read(path):
    lines = path.read_text().split('\n')
    i = 0
    out = []
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        out.append([l.split()[1:] for l in lines[i + 2:i + 2 + n]])
        i += n + 2
    return out


def setup(args, name, tag, potential, body, spec):
    folder = args.work / name / tag
    folder.mkdir(parents=True)
    shutil.copy(potential, folder / 'potential.txt')
    atoms = model(folder / 'model.xyz', **spec)
    (folder / 'run.in').write_text('potential potential.txt\n' + body)
    return folder, atoms


def exact(args, name, potential, spec):
    body = ('time_step 1\nensemble nve\n'
            'dump_xyz 20 d.xyz precision double force\nrun 200\n')
    a, atoms = setup(args, name, 'reference', potential, body, spec)
    b, _ = setup(args, name, 'candidate', potential, body, spec)
    ca, _ = run(args.reference, a, args.device)
    cb, _ = run(args.candidate, b, args.device)
    if ca or cb:
        print('FAIL %s rc %d %d' % (name, ca, cb))
        print(''.join((b / 'run.log').read_text().split('\n')[-5:]))
        return False
    fa, fb = read(a / 'd.xyz'), read(b / 'd.xyz')
    same = fa == fb and len(fa) == 10
    print('%s %s: atoms %d frames %d identical %s' % ('PASS' if same else 'FAIL', name, atoms, len(fa), same))
    return same


def speed(args, name, potential, spec):
    body = 'time_step 1\nensemble nve\nrun 200\n'
    a, atoms = setup(args, name, 'reference', potential, body, spec)
    b, _ = setup(args, name, 'candidate', potential, body, spec)
    ca, sa = run(args.reference, a, args.device)
    cb, sb = run(args.candidate, b, args.device)
    print('INFO %s: atoms %d rc %d/%d reference %.3e candidate %.3e speedup %.2f' % (
        name, atoms, ca, cb, sa, sb, sb / max(sa, 1e-30)))
    return ca == 0 and cb == 0


def main():
    parser = argparse.ArgumentParser()
    for name in ('reference', 'candidate', 'nep', 'lj'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    args.work.mkdir(parents=True)
    sige = dict(symbols=['Si', 'Ge'], spacing=A, temperature=3000)
    argon = dict(symbols=['Ar'], spacing=7.6, temperature=40)
    results = [
        exact(args, 'nep_periodic', args.nep, dict(sige, n=(8, 8, 8), pbc=(1, 1, 1))),
        exact(args, 'nep_slab', args.nep, dict(sige, n=(8, 8, 8), pbc=(1, 1, 0))),
        exact(args, 'nep_wire', args.nep, dict(sige, n=(8, 8, 8), pbc=(1, 0, 0))),
        exact(args, 'nep_cluster', args.nep, dict(sige, n=(8, 8, 8), pbc=(0, 0, 0))),
        exact(args, 'nep_slab_tilted', args.nep, dict(sige, n=(8, 8, 8), pbc=(1, 1, 0), tilt=9.0)),
        exact(args, 'nep_slab_stray', args.nep, dict(sige, n=(8, 8, 8), pbc=(1, 1, 0), stray=True)),
        exact(args, 'nep_tall_vacuum', args.nep, dict(sige, n=(8, 8, 8), pbc=(1, 1, 0), vacuum=2000000.0)),
        exact(args, 'lj_slab', args.lj, dict(argon, n=(6, 6, 6), pbc=(1, 1, 0))),
        speed(args, 'speed_nep_periodic', args.nep, dict(sige, n=(30, 30, 20), pbc=(1, 1, 1))),
        speed(args, 'speed_nep_slab_tall', args.nep, dict(sige, n=(30, 30, 80), pbc=(1, 1, 0))),
        speed(args, 'speed_nep_cluster', args.nep, dict(sige, n=(30, 30, 30), pbc=(0, 0, 0))),
        speed(args, 'speed_lj_cluster', args.lj, dict(argon, n=(24, 24, 24), pbc=(0, 0, 0)))]
    print('Summary: %d of %d passed' % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
