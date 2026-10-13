#!/usr/bin/env python3
"""Isolated tall-slab test; never changes the engine or archived simulations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

import numpy as np


REPO = Path(__file__).resolve().parents[4]
ENGINE = REPO / 'src/gpumd'
POTENTIAL = REPO / 'potentials/nep/Si_2022_NEP3_4body.txt'
LANES = ('full', 'skip', 'reuse')
A = 5.431
OFFSET = 7 * A


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def xyz(path):
    with path.open() as stream:
        n = int(stream.readline())
        header = stream.readline()
        props = re.search(r'Properties=(\S+)', header).group(1).split(':')
        fields, offset = {}, 0
        for i in range(0, len(props), 3):
            fields[props[i]] = (offset, int(props[i + 2]))
            offset += int(props[i + 2])
        lines = [stream.readline().split() for _ in range(n)]
        if any(len(row) != offset for row in lines) or stream.read().strip():
            raise ValueError(f'Invalid single-frame XYZ: {path}')
    result = {'n': n}
    for name in ('pos', 'group', 'vel', 'force', 'forces'):
        if name in fields:
            start, count = fields[name]
            data = np.array([row[start:start + count] for row in lines], dtype=float)
            if not np.isfinite(data).all():
                raise ValueError(f'Nonfinite {name}: {path}')
            result['force' if name == 'forces' else name] = data
    return result


def model(path, nz):
    basis = np.array([(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0),
                      (.25, .25, .25), (.25, .75, .75), (.75, .25, .75), (.75, .75, .25)])
    cells = np.indices((8, 8, nz)).reshape(3, -1).T
    positions = ((cells[:, None, :] + basis[None, :, :]) * A).reshape(-1, 3)
    # Nonzero, reproducible forces exercise more than an equilibrium force-zero case.
    rng = np.random.default_rng(231013 + nz)
    positions += rng.uniform(-.005, .005, positions.shape)
    positions[:, :2] %= 8 * A
    positions[:, 2] += 5
    with path.open('w') as out:
        out.write(f'{len(positions)}\n')
        out.write(f'Lattice="{8*A} 0 0 0 {8*A} 0 0 0 {nz*A+25}" '
                  'Properties=species:S:1:pos:R:3:group:I:1 pbc="T T F"\n')
        for x, y, z in positions:
            out.write(f'Si {x:.10f} {y:.10f} {z:.10f} {int(z < 5 + A)}\n')
    return len(positions)


def execute(directory, text, gpu, engine=None):
    engine = ENGINE if engine is None else engine
    (directory / 'run.in').write_text(text)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    start = time.monotonic()
    with (directory / 'stdout.log').open('w') as out:
        result = subprocess.run([str(engine)], cwd=directory, env=env,
                                stdout=out, stderr=subprocess.STDOUT, timeout=300)
    log = (directory / 'stdout.log').read_text()
    if result.returncode:
        raise RuntimeError(f'{directory}: return={result.returncode}\n{log[-3000:]}')
    times = [float(v) for v in re.findall(r'Time used for this run = ([\d.eE+-]+) second', log)]
    return {'wall_s': time.monotonic() - start, 'md_times_s': times,
            'skip_on': 'Fix skip is on' in log, 'path': str(directory)}


def block(lane):
    return 'ensemble nve\nfix 1\n' + ('full_observables\n' if lane == 'full' else '')


def prefix(lane):
    text = f'potential {POTENTIAL}\ntime_step 0.5\n'
    if lane == 'reuse':
        text += 'fix_skip min_frozen 0 rebuild 20 skin 1.0 cache_alloc 1\n'
    return text


def main():
    global ENGINE, POTENTIAL
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--nz', type=int, nargs='+', default=[24, 72, 216])
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--engine', type=Path, default=ENGINE)
    parser.add_argument('--potential', type=Path, default=POTENTIAL)
    args = parser.parse_args()
    if args.steps <= 0 or args.repetitions <= 0 or min(args.nz) < 12:
        parser.error('steps/repetitions must be positive and nz must be at least 12')
    if len(set(args.nz)) != len(args.nz):
        parser.error('nz values must be unique')
    ENGINE = args.engine.resolve(strict=True)
    POTENTIAL = args.potential.resolve(strict=True)
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    report = {'engine': str(ENGINE), 'engine_sha256': sha(ENGINE),
              'potential': str(POTENTIAL), 'potential_sha256': sha(POTENTIAL),
              'offset_A': OFFSET, 'steps': args.steps, 'gpu': args.gpu,
              'nz': args.nz, 'repetitions': args.repetitions,
              'geometry': 'diamond Si, 8x8 lateral cells, x/y periodic, z free',
              'cases': [], 'scope': 'fixed-mask force/performance, not physical equivalence to all-mobile MD'}
    save(args.root / 'report.json', report)
    for nz in args.nz:
        root = args.root / f'nz{nz}'
        prep = root / 'grow_classify'
        prep.mkdir(parents=True)
        n = model(prep / 'model.xyz', nz)
        inp = (f'potential {POTENTIAL}\nvelocity 300 seed 231013\ntime_step 0.000001\n'
               'ensemble nve\nfix 1\n'
               f'fix shell substrate z cutoff 2.9 offset {OFFSET} surface radius 3 mode grow\n'
               'dump_xyz -1 0 1 classified.xyz group velocity force\nrun 1\n')
        classification = execute(prep, inp, args.gpu)
        initial = xyz(prep / 'classified.xyz')
        fixed = initial['group'][:, 0] == 1
        if not fixed.any() or fixed.all():
            raise AssertionError('Classification must yield both fixed and mobile atoms')
        case = {'nz': nz, 'n': n, 'nominal_height_nm': nz*A/10,
                'fixed': int(fixed.sum()), 'mobile': int((~fixed).sum()),
                'classification': classification, 'correctness': {}, 'timings': []}
        frames = {}
        for lane in LANES:
            directory = root / f'correctness_{lane}'
            directory.mkdir()
            shutil.copyfile(prep / 'classified.xyz', directory / 'model.xyz')
            run = execute(directory, prefix(lane) + block(lane)
                          + 'dump_xyz -1 0 40 final.xyz group velocity force\nrun 40\n', args.gpu)
            frame = xyz(directory / 'final.xyz')
            if frame['n'] != n or not np.array_equal(initial['group'], frame['group']):
                raise AssertionError('Atom count/group mismatch')
            displacement = np.abs(frame['pos'][fixed] - initial['pos'][fixed]).max()
            fixed_force = np.abs(frame['force'][fixed]).max()
            if displacement > 1e-8 or fixed_force > 1e-8:
                raise AssertionError(f'Fixed atoms changed: {directory}')
            case['correctness'][lane] = dict(run, fixed_displacement_A=float(displacement),
                                             fixed_force_eV_A=float(fixed_force))
            frames[lane] = frame
        for lane in ('skip', 'reuse'):
            force_delta = np.abs(frames[lane]['force'][~fixed] - frames['full']['force'][~fixed])
            pos_delta = np.abs(frames[lane]['pos'] - frames['full']['pos'])
            case['correctness'][lane].update(
                active_force_max_error_eV_A=float(force_delta.max()),
                active_force_rms_error_eV_A=float(np.sqrt(np.mean(force_delta**2))),
                position_max_error_A=float(pos_delta.max()))
            if force_delta.max() > 1e-4 or pos_delta.max() > 1e-6:
                raise AssertionError(f'Force/trajectory comparison failed: {root} {lane}')
        # Rotate order across three repetitions to reduce warm-up/order bias.
        for rep in range(args.repetitions):
            rotation = rep % len(LANES)
            for lane in LANES[rotation:] + LANES[:rotation]:
                directory = root / f'bench_r{rep}_{lane}'
                directory.mkdir()
                shutil.copyfile(prep / 'classified.xyz', directory / 'model.xyz')
                inp = (prefix(lane) + block(lane) + 'run 200\n' + block(lane)
                       + f'run {args.steps}\n' + 'time_step 0.000001\n' + block(lane)
                       + 'dump_xyz -1 0 1 final.xyz group velocity force\nrun 1\n')
                run = execute(directory, inp, args.gpu)
                if len(run['md_times_s']) != 3:
                    raise AssertionError('Missing warm-up/timed/finalization durations')
                frame = xyz(directory / 'final.xyz')
                if not np.array_equal(frame['group'], initial['group']):
                    raise AssertionError('Benchmark changed labels')
                if np.abs(frame['pos'][fixed] - initial['pos'][fixed]).max() > 1e-8:
                    raise AssertionError('Benchmark moved frozen atoms')
                case['timings'].append(dict(run, lane=lane, repetition=rep,
                                            timed_md_s=run['md_times_s'][1]))
                print(json.dumps({'nz': nz, 'lane': lane, 'rep': rep,
                                  'md_s': run['md_times_s'][1]}), flush=True)
        case['median_md_s'] = {lane: statistics.median(r['timed_md_s'] for r in case['timings']
                                                       if r['lane'] == lane) for lane in LANES}
        case['speedup_vs_full'] = {lane: case['median_md_s']['full'] / case['median_md_s'][lane]
                                  for lane in LANES}
        report['cases'].append(case)
        save(args.root / 'report.json', report)
    report['complete'] = True
    save(args.root / 'report.json', report)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
