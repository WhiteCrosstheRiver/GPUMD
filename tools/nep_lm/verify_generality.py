#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native LM integration gates with separate training, guard and test files."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np


VIRIAL = 'virial="1 .2 -.3 .2 2 .4 -.3 .4 3" '
CELL = 'Lattice="40 0 0 0 40 0 0 0 40" '
PROPERTIES = 'Properties=species:S:1:pos:R:3:force:R:3\n'


def frame(atoms, energy, weight=1.0, virial=True):
    return (str(len(atoms)) + '\n' + CELL + 'energy=%g weight=%g ' % (energy, weight)
            + (VIRIAL if virial else '') + PROPERTIES
            + ''.join('%s %g %g %g .1 -.2 .15\n' % atom for atom in atoms))


TRAIN = (
    frame([('Si', 2, 2, 2), ('Ge', 4.2, 2, 2), ('Si', 2.5, 3.6, 2.4)], -1.6, 1.3),
    frame([('Ge', 8, 8, 8), ('Si', 9.6, 8.4, 8.2), ('Ge', 8.6, 9.7, 8.4)], -1.2, .7),
)
GUARD = frame([('Ge', 16, 16, 16), ('Si', 17.7, 16.3, 16.2)], -.7, 1.1)
TEST = frame([('Si', 24, 24, 24), ('Ge', 25.8, 24.2, 24.5)], -.6, .9)
FLEXIBLE = (
    '1.2 2.4 .22 3.2 .4717 .9423 .2802 .4029 .02817 .2016\n'
    '1.2 2.4 .24 3.2 .4517 .9423 .2802 .4029 .02817 .2016\n'
    '1.2 2.4 .26 3.2 .4317 .9423 .2802 .4029 .02817 .2016\n'
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_idle(gpu):
    text = subprocess.check_output(
        ['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
         '--format=csv,noheader,nounits'], text=True, timeout=15)
    if any(row.strip().isdigit() for row in text.splitlines()):
        raise RuntimeError('Assigned GPU %d is occupied' % gpu)


def parameters(types, nmax, basis, lmax, neurons):
    angular = lmax[0] + sum(value > 0 for value in lmax[1:])
    dim = nmax[0] + 1 + (nmax[1] + 1) * angular
    count = (types * ((dim + 2) * neurons + 1) + types * types
             * ((nmax[0] + 1) * (basis[0] + 1) + (nmax[1] + 1) * (basis[1] + 1)))
    return dim, count


def configuration(spec, lm=True):
    text = ('type %d %s\n' % (len(spec['elements']), ' '.join(spec['elements']))
            + 'cutoff %s\n' % spec.get('cutoff', '5 4')
            + 'n_max %d %d\n' % spec['nmax']
            + 'basis_size %d %d\n' % spec['basis']
            + 'l_max %s\n' % ' '.join(map(str, spec['lmax']))
            + 'neuron %d\nbatch %d 0\nepoch 1\n' % (spec['neurons'], len(spec['train']))
            + 'weight_decay 0\nlambda_e .7\nlambda_f 1.2\nlambda_v .3\n'
            + 'lambda_shear %g\n' % spec.get('shear', 1.5)
            + 'type_weight %s\n' % ' '.join('%g' % (1.2 - .1 * i) for i in range(len(spec['elements'])))
            + 'force_delta .2\nlambda_1 0\nlambda_2 .03\nloss_rmse 1e-3\n'
            + 'start_lr .001\nstop_lr .001\nseed 719\n'
            + spec.get('extra', ''))
    if lm:
        text += 'optimizer lm\nlm 4 3 .2 .01\nlm_guard guard.xyz 1.1\nlm_check 1\n'
    return text


def gates(log):
    patterns = {
        'descriptor_scaled_error': r'LM moment descriptor gate max_scaled_error=([\d.eE+-]+)',
        'gradient_scaled_error': r'LM native gradient gate max_scaled_error=([\d.eE+-]+)',
        'gradient_relative_error': r'LM native gradient gate[^\n]*relative_norm_error=([\d.eE+-]+)',
    }
    found = {}
    for name, pattern in patterns.items():
        values = list(map(float, re.findall(pattern, log)))
        if not values or not np.isfinite(values).all():
            raise AssertionError('Missing or invalid native gate: ' + name)
        found[name] = max(values)
    if found['descriptor_scaled_error'] > 2e-5:
        raise AssertionError('Descriptor gate failed')
    if max(found['gradient_scaled_error'], found['gradient_relative_error']) > .002:
        raise AssertionError('Gradient gate failed')
    return found


def verify_diagnostics(folder):
    lm_rows = np.loadtxt(folder / 'lm.out', ndmin=2)
    native = np.loadtxt(folder / 'loss.out', ndmin=2)
    theta = np.loadtxt(folder / 'gnep.restart', ndmin=1)
    if (lm_rows.shape != (1, 10) or native.shape != (1, 10)
            or not np.isfinite(lm_rows).all() or not np.isfinite(native).all()
            or not np.isfinite(theta).all()):
        raise AssertionError('Invalid model or diagnostics')
    row = lm_rows[0]
    if row[3] not in (0., 1.) or row[2] > row[1]:
        raise AssertionError('LM objective or acceptance indicator is invalid')
    expected_guard = row[7]
    if row[3] == 1:
        if row[2] >= row[1] or row[8] > 1.1 * row[7] + 1e-10:
            raise AssertionError('Accepted objective or guard cap failed')
        expected_guard = min(expected_guard, row[8])
    elif abs(row[8] - row[7]) > 1e-10:
        raise AssertionError('Rejected candidate changed selected guard')
    if abs(row[9] - expected_guard) > 1e-10 * (1 + abs(expected_guard)):
        raise AssertionError('Historical guard state is inconsistent')
    return {'accepted': bool(row[3]), 'loss_before': float(row[1]),
            'loss_after': float(row[2]), 'guard_before': float(row[7]),
            'guard_selected': float(row[8]), 'guard_best': float(row[9])}


def cases():
    common = {'elements': ('Si', 'Ge'), 'nmax': (1, 1), 'basis': (2, 2),
              'lmax': (4, 2, 1), 'neurons': 4, 'train': TRAIN, 'guard': GUARD, 'test': TEST}
    isolated = frame([('Si', 3, 3, 3)], -.3, 1.1, virial=False)
    radial = dict(common, name='pure_radial_and_isolated', lmax=(0,),
                  train=(TRAIN[0], isolated))
    mixed = dict(common, name='missing_virial_and_zero_shear', shear=0.,
                 train=(TRAIN[0], frame([('Ge', 8, 8, 8), ('Si', 9.6, 8.4, 8.2)],
                                       -.9, .7, virial=False)))
    four_atoms = [('Si', 2, 2, 2), ('Ge', 3.5, 2.2, 2.3),
                  ('C', 2.5, 3.6, 2.4), ('O', 2.7, 2.5, 3.8)]
    many_types = dict(common, name='four_types', elements=('Si', 'Ge', 'C', 'O'),
                      train=(frame(four_atoms, -2.1, 1.3), frame(four_atoms[::-1], -1.9, .7)))
    wide = dict(common, name='maximum_descriptor_220', nmax=(19, 19), basis=(19, 19),
                lmax=(8, 2, 1), neurons=2, initial_scale=.02)
    # Native global cutoffs require angular >=2.5; H/H's smaller effective
    # typewise NEP cutoff (1.28) makes the 1.6-A pair a real ZBL-union case.
    union = dict(common, elements=('H', 'He'), cutoff='3 2.5',
                 train=(frame([('H', 2, 2, 2), ('H', 3.6, 2, 2), ('He', 2.7, 3, 2)], -1.6, 1.3),
                        frame([('He', 8, 8, 8), ('H', 9.3, 8.2, 8.1),
                               ('He', 8.6, 9.4, 8.2)], -1.2, .7)),
                 guard=frame([('He', 16, 16, 16), ('H', 17.3, 16.2, 16.1)], -.7, 1.1),
                 test=frame([('H', 24, 24, 24), ('He', 25.4, 24.1, 24.3)], -.6, .9))
    fixed = dict(union, name='fixed_zbl_typewise_neighbor_union',
                 extra='zbl 2.4\nuse_typewise_cutoff 1.5 1.5\nuse_typewise_cutoff_zbl 2.5\n')
    flexible = dict(union, name='flexible_zbl_typewise_neighbor_union',
                    extra='zbl 2.4\nuse_typewise_cutoff 1.5 1.5\n', flexible=True)
    return (radial, mixed, many_types, wide, fixed, flexible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', action='append', choices=[spec['name'] for spec in cases()])
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--timeout', type=float, default=35.)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1')
    result = {'passed': False, 'binary_sha256': digest(binary),
              'runner_sha256': digest(__file__), 'cases': [], 'gpu': args.gpu}
    if args.baseline:
        result['baseline_binary_sha256'] = digest(args.baseline.resolve())
    started = time.monotonic()

    def execute(folder, spec, lm=True, executable=binary):
        require_idle(args.gpu)
        folder.mkdir()
        (folder / 'train.xyz').write_text(''.join(spec['train']))
        (folder / 'guard.xyz').write_text(spec['guard'])
        (folder / 'test.xyz').write_text(spec['test'])
        (folder / 'gnep.in').write_text(configuration(spec, lm))
        dimensions, count = parameters(len(spec['elements']), spec['nmax'], spec['basis'],
                                       spec['lmax'], spec['neurons'])
        initial = np.random.default_rng(719).uniform(
            -spec.get('initial_scale', .08), spec.get('initial_scale', .08), count)
        np.savetxt(folder / 'gnep.restart', initial, fmt='%.9e')
        if spec.get('flexible'):
            (folder / 'zbl.in').write_text(FLEXIBLE)
        held_out_hash = digest(folder / 'test.xyz')
        guard_hash = digest(folder / 'guard.xyz')
        before = time.monotonic()
        process = subprocess.run([str(executable)], cwd=folder, env=environment, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.timeout)
        (folder / 'run.log').write_text(process.stdout)
        if process.returncode:
            raise AssertionError('%s failed: %s' % (folder.name, process.stdout[-7000:]))
        if digest(folder / 'test.xyz') != held_out_hash or digest(folder / 'guard.xyz') != guard_hash:
            raise AssertionError('Input guard or held-out test file changed')
        return {'dimensions': dimensions, 'parameters': count,
                'seconds': time.monotonic() - before, 'log': process.stdout,
                'test_sha256': held_out_hash, 'guard_sha256': guard_hash}

    try:
        for spec in cases():
            if args.case and spec['name'] not in args.case:
                continue
            folder = args.output / spec['name']
            observed = execute(folder, spec)
            observed.update(gates(observed.pop('log')))
            observed.update(verify_diagnostics(folder))
            observed.update(name=spec['name'], passed=True,
                            model_sha256=digest(folder / 'nep.txt'))
            if spec.get('flexible'):
                header = (folder / 'nep.txt').read_text().splitlines()
                if not header[0].startswith('nep5_zbl ') or header[1].split()[:3] != ['zbl', '0', '0']:
                    raise AssertionError('Flexible ZBL header was lost')
            result['cases'].append(observed)
            print(json.dumps(observed), flush=True)
        if args.baseline:
            spec = cases()[1]
            current = args.output / 'default_current'
            previous = args.output / 'default_previous'
            execute(current, spec, lm=False)
            execute(previous, spec, lm=False, executable=args.baseline.resolve())
            for name in ('gnep.restart', 'nep.txt'):
                if (current / name).read_bytes() != (previous / name).read_bytes():
                    raise AssertionError('Default Adam changed: ' + name)
            if (current / 'lm.out').exists() or (previous / 'lm.out').exists():
                raise AssertionError('LM activated in an opt-out run')
            result['default_adam_bitwise_equivalence'] = True
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['seconds'] = time.monotonic() - started
        (args.output / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
