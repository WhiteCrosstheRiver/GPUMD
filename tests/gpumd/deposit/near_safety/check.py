#!/usr/bin/env python3
"""GPU regression cases for conservative deposit-near shortcuts.

Example:
  python3 test_near_safety.py --gpumd /path/to/gpumd --potential /path/to/nep.txt \
      --output /new/results/near_safety --gpu 0

The output directory must not exist. A rejected deposition must fail with the
specific "Requested entities" message, rather than an unrelated parse/runtime
error. Inputs are generated per case and never alter the potential or engine.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def xyz(cell, pbc, positions):
    lattice = ' '.join(f'{x:.12g}' for x in cell)
    lines = [str(len(positions)),
             f'pbc="{pbc}" Lattice="{lattice}" Properties=species:S:1:pos:R:3:group:I:1']
    lines.extend(f'Si {x:.10f} {y:.10f} {z:.10f} 0' for x, y, z in positions)
    return '\n'.join(lines) + '\n'


def command(region, count, surface, near=2.5, spacing='4 4', normal='0 0 1', name='d'):
    limits = region
    if surface.startswith('fixed '):
        height = f'plane normal {normal} bounds {limits} height {surface.split()[1]}'
    else:
        height = f'surface normal {normal} bounds {limits} height {surface} scope all'
    parts = [f'deposit {name} particle atom Si',
             f'deposit {name} placement grid spacing {spacing} select random',
             f'deposit {name} region {height}',
             f'deposit {name} velocity beam speed 0.00785815248941359 units angstrom/fs direction 0 0 -1']
    if near is not None:
        parts.append(f'deposit {name} exclusion distance {near}')
    parts += [f'deposit {name} seed 19', f'deposit {name} emit number {count}']
    return '\n'.join(parts)


def cases():
    slab10 = (10, 0, 0, 0, 10, 0, 0, 0, 20)
    normal10 = (10, 0, 0, 0, 10, 0, 0, 0, 10)
    slab20 = (20, 0, 0, 0, 20, 0, 0, 0, 20)
    narrow_x = (1, 0, 0, 0, 10, 0, 0, 0, 20)
    tilted_xy = (10, 0, 0, 5, 10, 0, 0, 0, 20)
    tilted = '1 0 1'
    return [
        dict(name='periodic_normal_image', cell=normal10, pbc='T T T',
             atoms=[(5, 5, 2), (5, 5, 9)],
             commands=[command('5 5 5 5', 1, 'global gap 3')],
             reject=True),
        dict(name='tilted_normal_image', cell=slab10, pbc='T T F',
             atoms=[(0, 0, 10), (2.1213203436, 0, 2.1213203436)],
             commands=[command('0 0 -7.0710678119 -7.0710678119', 1,
                               'global gap 3', normal=tilted)],
             reject=True),
        dict(name='periodic_grid_seam', cell=slab10, pbc='T T F',
             atoms=[(5, 5, 0), (5, 5, -2)],
             commands=[command('0 8 0 0', 3, 'global gap 3')],
             reject=True),
        dict(name='narrow_periodic_axis', cell=narrow_x, pbc='T F F',
             atoms=[(0.5, 2, 0), (0.5, 2, -2)],
             commands=[command('0.5 0.5 4.2 4.2', 1, 'fixed 0')],
             reject=True),
        dict(name='triclinic_periodic_image', cell=tilted_xy, pbc='T T F',
             atoms=[(5.1, 9.8, 0), (2, 2, -2)],
             commands=[command('0.1 0.1 0.2 0.2', 1, 'fixed 0')],
             reject=True),
        dict(name='batch_deferred_near', cell=slab20, pbc='T T F',
             atoms=[(10, 10, 0), (10, 10, -2)],
             commands=[command('0 4 0 0', 2, 'fixed 3', name='first'),
                       command('0 0 0 0', 1, 'fixed 3', name='second')],
             reject=True, prior_atoms=2),
        dict(name='normal_slab_control', cell=slab10, pbc='T T F',
             atoms=[(5, 5, 0), (5, 5, -2)],
             commands=[command('0 4 0 0', 2, 'global gap 3')],
             reject=False, deposited=2, expected_z=[3, 3]),
        dict(name='global_then_global', cell=slab10, pbc='T T F',
             atoms=[(5, 5, 0), (5, 5, -2)],
             commands=[command('0 0 0 0', 1, 'global gap 3', name='first'),
                       command('0 0 0 0', 1, 'global gap 3', name='second')],
             reject=False, deposited=2, expected_z=[3, 6]),
        dict(name='global_then_local', cell=slab10, pbc='T T F',
             atoms=[(5, 5, 0), (5, 5, -2)],
             commands=[command('0 0 0 0', 1, 'global gap 3', name='first'),
                       command('0 0 0 0', 1, 'local radius 6 gap 3', name='second')],
             reject=False, deposited=2, expected_z=[3, 6]),
        dict(name='local_then_global', cell=slab10, pbc='T T F',
             atoms=[(5, 5, 0), (5, 5, -2)],
             commands=[command('0 0 0 0', 1, 'local radius 6 gap 3', name='first'),
                       command('0 0 0 0', 1, 'global gap 3', name='second')],
             reject=False, deposited=2, expected_z=[3, 6]),
        dict(name='batch_prior_without_near', cell=slab20, pbc='T T F',
             atoms=[(10, 10, 0), (10, 10, -2)],
             commands=[command('0 0 0 0', 1, 'fixed 3', near=None, name='first'),
                       command('0 0 0 0', 1, 'fixed 3', name='second')],
             reject=True, prior_atoms=1),
    ]


def evaluate(case, code, log, final_path=None):
    rejected = (code not in (None, 0) and 'Input Error' in log and
                'Deposit failed: no valid insertion after placement attempts.' in log)
    deposited = sum(int(n) for n in re.findall(
        r'Deposit recipe \S+: emitted \d+ entities \((\d+) atoms\)', log))
    if case['reject']:
        passed = rejected and deposited == case.get('prior_atoms', 0)
        return passed, {'expected': 'near_rejection', 'rejected': rejected,
                          'exit_code': code, 'deposited_before_rejection': deposited}
    final_atoms, final_z = None, None
    if final_path is not None and final_path.is_file():
        with final_path.open() as stream:
            final_atoms = int(stream.readline())
            stream.readline()
            final_z = [float(stream.readline().split()[3]) for _ in range(final_atoms)]
    count_ok = final_atoms == len(case['atoms']) + case['deposited']
    height_ok = count_ok and all(abs(a - b) < 0.01 for a, b in
                                  zip(final_z[-case['deposited']:], case['expected_z']))
    passed = (code == 0 and count_ok and height_ok and deposited == case['deposited'] and
              'Finished running GPUMD' in log)
    return passed, {'expected': 'successful_deposition', 'exit_code': code,
                    'reported_deposits': deposited, 'final_atoms': final_atoms,
                    'new_atom_z': final_z[-case['deposited']:] if final_z else None,
                    'expected_deposited': case['deposited'], 'expected_z': case['expected_z']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpumd', type=Path, required=True)
    p.add_argument('--potential', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--timeout', type=int, default=180)
    a = p.parse_args()
    engine, potential = a.gpumd.resolve(strict=True), a.potential.resolve(strict=True)
    output = a.output.resolve()
    if output == potential.parent or potential.parent in output.parents:
        p.error('output must not be inside the potential source directory')
    if a.gpu < 0 or a.timeout < 1:
        p.error('gpu must be nonnegative and timeout positive')
    output.mkdir(parents=True, exist_ok=False)
    report = {'engine': str(engine), 'potential': str(potential), 'cases': []}
    for case in cases():
        directory = output / case['name']
        directory.mkdir()
        (directory / 'model.xyz').write_text(xyz(case['cell'], case['pbc'], case['atoms']))
        shutil.copyfile(potential, directory / 'nep.txt')
        run = (['potential ./nep.txt', 'time_step 0.001', 'ensemble nve'] + case['commands'] +
               ['dump_xyz -1 0 1 final.xyz group', 'run 1'])
        (directory / 'run.in').write_text('\n'.join(run) + '\n')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
        try:
            result = subprocess.run([str(engine)], cwd=directory, env=env,
                                    capture_output=True, text=True, timeout=a.timeout, check=False)
            code = result.returncode
            log = result.stdout + '\n' + result.stderr
        except subprocess.TimeoutExpired as exc:
            code = None
            log = ((exc.stdout or b'').decode(errors='replace') if isinstance(exc.stdout, bytes) else (exc.stdout or ''))
            log += '\nTIMEOUT'
        (directory / 'gpumd.log').write_text(log)
        passed, detail = evaluate(case, code, log, directory / 'final.xyz')
        detail.update(name=case['name'], passed=passed)
        report['cases'].append(detail)
        (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"{case['name']}: {'PASS' if passed else 'FAIL'}", flush=True)
    return 0 if all(c['passed'] for c in report['cases']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
