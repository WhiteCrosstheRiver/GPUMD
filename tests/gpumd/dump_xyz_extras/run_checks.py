#!/usr/bin/env python3
"""GPU checks for the dump_xyz extras: speed, force_norm, volume, stress, stress_norm, pressure.

usage: run_checks.py --gpumd GPUMD --nep SiGe_NEP --work NEW_DIRECTORY --device ID
"""
import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'fix_skip'))
import run_checks as skip  # noqa: E402

A = 5.43
C_P = 1.602177e2
TIME = 1.018051e1  # fs per natural time unit: natural velocity = Angstrom/fs * TIME
results = []


def report(name, ok, detail=''):
    results.append(ok)
    print('%s %s %s' % ('PASS' if ok else 'FAIL', name, detail), flush=True)


def run(args, name, dump, cells=(4, 4, 4), temperature=300, pbc_z=True, frozen=0.0):
    folder = args.work / name
    folder.mkdir(parents=True)
    shutil.copy(args.nep, folder / 'potential.txt')
    skip.write_model(folder / 'model.xyz', ['Si', 'Ge'], A, cells, temperature, pbc_z, frozen=frozen)
    (folder / 'run.in').write_text('potential potential.txt\ntime_step 1\nensemble nve\n%s\nrun 2\n' % dump)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device))
    with open(folder / 'run.log', 'w') as log:
        code = subprocess.run([args.gpumd], cwd=folder, env=env, stdout=log,
                              stderr=subprocess.STDOUT, timeout=300).returncode
    return folder, code, (folder / 'run.log').read_text()


def frame(path):
    lines = path.read_text().split('\n')
    n = int(lines[0])
    head = lines[1]
    props = re.search(r'Properties=(\S+)', head).group(1).split(':')
    offset = {}
    pos = 0
    for name, _, count in zip(props[0::3], props[1::3], props[2::3]):
        offset[name] = (pos, int(count))
        pos += int(count)
    rows = [l.split() for l in lines[2:2 + n]]
    def column(name):
        start, count = offset[name]
        return [[float(x) for x in r[start:start + count]] for r in rows]
    lattice = [float(x) for x in re.search(r'Lattice="([^"]+)"', head).group(1).split()]
    return head, column, lattice


def volume_of(lattice):
    a, b, c = lattice[0:3], lattice[3:6], lattice[6:9]
    return abs(a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
               a[2] * (b[0] * c[1] - b[1] * c[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpumd', required=True)
    parser.add_argument('--nep', required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    args.work.mkdir(parents=True)

    # A cold crystal: every Voronoi cell is a^3/8 and the cells fill the box.
    folder, code, log = run(args, 'crystal_volume', 'dump_xyz 1 d.xyz volume 3.0 256', temperature=1)
    head, column, lattice = frame(folder / 'd.xyz')
    vol = [v[0] for v in column('volume_atom')]
    total = sum(vol) / volume_of(lattice)
    spread = max(abs(v - A ** 3 / 8) for v in vol) / (A ** 3 / 8)
    report('crystal_volume', code == 0 and abs(total - 1) < 0.01 and spread < 0.02,
           'sum/box %.5f, worst cell error %.4f' % (total, spread))

    # Columns must agree with the quantities they are built from.
    dump = ('dump_xyz 1 d.xyz precision double velocity speed force force_norm virial mass '
            'volume 3.0 128 stress stress_norm pressure')
    folder, code, log = run(args, 'consistency', dump, temperature=600)
    head, column, lattice = frame(folder / 'd.xyz')
    vel, speed = column('vel'), [v[0] for v in column('speed')]
    force, fnorm = column('forces'), [v[0] for v in column('force_norm')]
    virial, mass = column('virial'), [v[0] for v in column('mass')]
    vol, stress = [v[0] for v in column('volume_atom')], column('stress')
    snorm, pres = [v[0] for v in column('stress_norm')], [v[0] for v in column('pressure')]
    e_speed = max(abs(math.sqrt(sum(x * x for x in v)) - s) for v, s in zip(vel, speed))
    e_force = max(abs(math.sqrt(sum(x * x for x in f)) - s) for f, s in zip(force, fnorm))
    worst_sigma = worst_norm = worst_p = 0.0
    total_trace = 0.0
    for v, w, m, o, s, sn, p in zip(vel, virial, mass, vol, stress, snorm, pres):
        vn = [x * TIME for x in v]  # back to natural units
        sigma = [(w[3 * a + b] + m * vn[a] * vn[b]) / o for a in range(3) for b in range(3)]
        worst_sigma = max(worst_sigma, max(abs(x - y) for x, y in zip(sigma, s)) / max(1e-12, max(abs(x) for x in sigma)))
        worst_norm = max(worst_norm, abs(math.sqrt(sum(x * x for x in s)) - sn))
        worst_p = max(worst_p, abs(C_P * (s[0] + s[4] + s[8]) / 3 - p))
        total_trace += w[0] + w[4] + w[8] + m * sum(x * x for x in vn)
    box_p = float(re.search(r' pressure=(\S+)', head).group(1))
    expected_p = C_P * total_trace / (3 * volume_of(lattice))
    report('speed_matches_velocity', code == 0 and e_speed < 1e-9, 'max error %.2e' % e_speed)
    report('force_norm_matches_force', code == 0 and e_force < 1e-9, 'max error %.2e' % e_force)
    report('stress_matches_virial_and_volume', worst_sigma < 1e-6, 'max relative error %.2e' % worst_sigma)
    report('stress_norm_matches_stress', worst_norm < 1e-9, 'max error %.2e' % worst_norm)
    report('pressure_matches_stress', worst_p < 1e-6, 'max error %.2e GPa' % worst_p)
    report('box_pressure_matches_atoms', abs(box_p - expected_p) < 1e-6 * max(1, abs(expected_p)),
           'header %.6f, from atoms %.6f GPa' % (box_p, expected_p))

    # A group dump keeps the global pressure.
    folder, code, _ = run(args, 'group_pressure',
                          'dump_xyz 1 all.xyz precision double pressure 3.0 128\n'
                          'dump_xyz 1 part.xyz precision double group 0 1 pressure 3.0 128',
                          temperature=600, frozen=0.5)
    head_all, _, _ = frame(folder / 'all.xyz')
    head_part, col_part, _ = frame(folder / 'part.xyz')
    pa = float(re.search(r' pressure=(\S+)', head_all).group(1))
    pb = float(re.search(r' pressure=(\S+)', head_part).group(1))
    report('group_dump_uses_box_pressure', code == 0 and pa == pb and len(col_part('pressure')) < 512,
           'all %.6f group %.6f' % (pa, pb))

    # A slab: free axis cells must still find every neighbor.
    folder, code, log = run(args, 'slab_volume', 'dump_xyz 1 d.xyz volume 3.0 256', temperature=1, pbc_z=False)
    head, column, lattice = frame(folder / 'd.xyz')
    vol = [v[0] for v in column('volume_atom')]
    inner = [v for v, p in zip(vol, column('pos')) if 4 < p[2] < 4 * A - 4]
    spread = max(abs(v - A ** 3 / 8) for v in inner) / (A ** 3 / 8)
    report('slab_inner_volume', code == 0 and spread < 0.02 and bool(inner), 'worst cell error %.4f' % spread)

    # Rejected inputs.
    for name, dump, message in (
            ('stress_without_R', 'dump_xyz 1 d.xyz stress', 'need R'),
            ('R_without_M', 'dump_xyz 1 d.xyz stress 3.0', 'followed by the number'),
            ('bad_directions', 'dump_xyz 1 d.xyz volume 3.0 100', '128 or 256'),
            ('conflicting_R', 'dump_xyz 1 d.xyz volume 3.0 128 stress 2.5 128', 'same R'),
            ('box_thinner_than_4R', 'dump_xyz 1 d.xyz volume 6.0 128', 'smaller than 4R')):
        _, code, log = run(args, name, dump, cells=(4, 4, 4))
        report(name, code != 0 and message in log)

    print('Summary: %d of %d passed' % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
