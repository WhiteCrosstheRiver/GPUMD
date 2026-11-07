#!/usr/bin/env python3
"""GPU checks for fix_skip: skipped runs must reproduce full runs for unfixed atoms.

usage: run_checks.py --gpumd GPUMD --nep NEP_FILE --lj LJ_FILE --work NEW_DIRECTORY --device ID

Each case runs the same frozen slab with and without fix_skip from identical positions and
velocities, then compares every dumped frame for unfixed atoms. The slab is hot enough to
rebuild the candidate neighbor list, so reused influence sets are exercised.
"""
import argparse
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

A = 5.43
BASIS = [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5),
         (.25, .25, .25), (.75, .75, .25), (.75, .25, .75), (.25, .75, .75)]
MASS = {'Si': 28.085, 'Ge': 72.63, 'Ar': 39.948}
KB = 8.617333e-5
# 1 amu * (A/fs)^2 = 103.6427 eV in GPUMD's natural units; velocities are in A/fs.
AMU_A2_FS2 = 103.6427


def write_model(path, symbols, spacing, n, temperature, pbc_z, frozen=0.6, seed=2026):
    rng = random.Random(seed)
    cell = [n[0] * spacing, n[1] * spacing, n[2] * spacing + (0 if pbc_z else 20)]
    rows = []
    height = n[2] * spacing
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                for b in BASIS:
                    x, y, z = (i + b[0]) * spacing, (j + b[1]) * spacing, (k + b[2]) * spacing
                    symbol = rng.choice(symbols)
                    fixed = z < frozen * height
                    sd = 0 if fixed else math.sqrt(KB * temperature / (MASS[symbol] * AMU_A2_FS2))
                    v = [rng.gauss(0, sd) for _ in range(3)]
                    rows.append('%s %.10f %.10f %.10f %.10e %.10e %.10e %d' % (
                        symbol, x, y, z, v[0], v[1], v[2], 1 if fixed else 0))
    flag = 'T' if pbc_z else 'F'
    with open(path, 'w') as f:
        f.write('%d\n' % len(rows))
        f.write('Lattice="%g 0 0 0 %g 0 0 0 %g" pbc="T T %s" '
                'Properties=species:S:1:pos:R:3:vel:R:3:group:I:1\n' % (*cell, flag))
        f.write('\n'.join(rows) + '\n')
    return len(rows), sum(r.endswith(' 1') for r in rows)


def run(gpumd, folder, device, timeout=900):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
    begin = time.monotonic()
    with open(folder / 'run.log', 'w') as log:
        result = subprocess.run([gpumd], cwd=folder, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    return result.returncode, time.monotonic() - begin


def frames(path):
    lines = open(path).read().split('\n')
    i = 0
    out = []
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        props = re.search(r'Properties=(\S+)', lines[i + 1]).group(1).split(':')
        offset = {}
        pos = 0
        for name, _, count in zip(props[0::3], props[1::3], props[2::3]):
            offset[name.lower()] = pos
            pos += int(count)
        out.append((offset, [l.split() for l in lines[i + 2:i + 2 + n]]))
        i += n + 2
    return out


def compare(reference, candidate, active, box):
    worst_r = worst_f = moved = 0.0
    a = frames(reference)
    b = frames(candidate)
    if len(a) != len(b) or not a:
        raise SystemExit('frame counts differ: %d %d' % (len(a), len(b)))
    start = a[0][1]
    for (oa, ra), (ob, rb) in zip(a, b):
        for n in active:
            pa = [float(x) for x in ra[n][oa['pos']:oa['pos'] + 3]]
            pb = [float(x) for x in rb[n][ob['pos']:ob['pos'] + 3]]
            fa = [float(x) for x in ra[n][oa['forces']:oa['forces'] + 3]]
            fb = [float(x) for x in rb[n][ob['forces']:ob['forces'] + 3]]
            p0 = [float(x) for x in start[n][oa['pos']:oa['pos'] + 3]]
            for d in range(3):
                worst_r = max(worst_r, abs(pa[d] - pb[d]))
                worst_f = max(worst_f, abs(fa[d] - fb[d]))
                shift = abs(pa[d] - p0[d])
                moved = max(moved, min(shift, box[d] - shift))
    return worst_r, worst_f, moved, len(a)


def case(args, name, potential, body, symbols, spacing, n, temperature, pbc_z, limit_r, limit_f):
    root = args.work / name
    folders = {}
    for mode, skip in (('full', ''), ('skip', 'fix_skip\n')):
        folder = root / mode
        folder.mkdir(parents=True)
        shutil.copy(potential, folder / 'potential.txt')
        total, frozen = write_model(folder / 'model.xyz', symbols, spacing, n, temperature, pbc_z)
        (folder / 'run.in').write_text(
            'potential potential.txt\n' + skip + body)
        folders[mode] = folder
    timing = {}
    for mode, folder in folders.items():
        rc, seconds = run(args.gpumd, folder, args.device)
        if rc:
            print(name, mode, 'FAILED rc', rc)
            print(''.join(open(folder / 'run.log').readlines()[-6:]))
            return False
        timing[mode] = seconds
    log = (folders['skip'] / 'run.log').read_text()
    engaged = 'fix_skip is on' in log
    rows = frames(folders['full'] / 'dump.xyz')[0][1]
    labels = [l.split()[-1] for l in open(folders['full'] / 'model.xyz').read().split('\n')[2:] if l]
    active = [i for i, label in enumerate(labels) if label != '1']
    box = [n[0] * spacing, n[1] * spacing, 1e9]
    r, f, moved, count = compare(folders['full'] / 'dump.xyz', folders['skip'] / 'dump.xyz',
                                 active, box)
    ok = engaged and r <= limit_r and f <= limit_f
    print('%s %s: atoms %d frozen %d frames %d engaged %s moved %.2f A max dr %.2e A max dF %.2e eV/A' % (
          'PASS' if ok else 'FAIL', name, total, frozen, count, engaged, moved, r, f))
    return ok


def fallback(args, potential):
    folder = args.work / 'npt_fallback'
    folder.mkdir()
    shutil.copy(potential, folder / 'potential.txt')
    write_model(folder / 'model.xyz', ['Si', 'Ge'], A, (8, 8, 6), 300, True)
    (folder / 'run.in').write_text(
        'potential potential.txt\nfix_skip\ntime_step 1\n'
        'ensemble npt_ber 300 300 100 0 0 0 100 100 100 1000\nfix 1\nrun 5\n')
    rc, _ = run(args.gpumd, folder, args.device)
    log = (folder / 'run.log').read_text()
    ok = rc == 0 and 'fix_skip is on' not in log
    print('%s npt_fallback: rc %d, skip stayed off: %s' % (
        'PASS' if ok else 'FAIL', rc, 'fix_skip is on' not in log))
    if rc:
        print(''.join(log.split('\n')[-8:]))
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpumd', required=True)
    parser.add_argument('--nep', required=True)
    parser.add_argument('--lj', required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    args.work.mkdir(parents=True)
    nve = ('time_step 1\nensemble nve\nfix 1\n'
           'dump_xyz 25 dump.xyz precision double force\nrun 300\n')
    ber = ('time_step 1\nensemble nvt_ber 2500 2500 100\nfix 1\n'
           'dump_xyz 25 dump.xyz precision double force\nrun 300\n')
    results = [
        case(args, 'nep_bulk_nve', args.nep, nve, ['Si', 'Ge'], A, (10, 10, 8), 3000, True, 1e-3, 1e-2),
        case(args, 'nep_slab_nve', args.nep, nve, ['Si', 'Ge'], A, (10, 10, 8), 3000, False, 1e-3, 1e-2),
        case(args, 'nep_slab_nvt', args.nep, ber, ['Si', 'Ge'], A, (10, 10, 8), 3000, False, 1e-3, 1e-2),
        case(args, 'lj_bulk_nve', args.lj, nve, ['Ar'], 7.6, (6, 6, 5), 40, True, 1e-4, 1e-3),
        fallback(args, args.nep)]
    print('Summary: %d of %d passed' % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
