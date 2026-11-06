#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Weighted native LM Jacobian gates and independent CPU E/F/V export checks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


CELL = 'Lattice="40 0 0 0 40 0 0 0 40" '
PROPERTIES = 'Properties=species:S:1:pos:R:3:force:R:3\n'
VIRIAL = 'virial="1 .2 -.3 .2 2 .4 -.3 .4 3" '
# Three training frames, a dedicated guard frame, and a separate test frame.
FRAMES = (
    '4\n' + CELL + 'energy=-2 weight=1.3 ' + VIRIAL + PROPERTIES
    + 'Si 2 2 2 .1 .2 -.1\nGe 3.7 2.2 2.3 -.2 .1 .3\n'
    + 'Si 2.5 3.8 2.4 .2 -.1 .1\nGe 3.1 2.7 4.1 -.1 -.2 -.3\n',
    '3\n' + CELL + 'energy=-1 weight=.7 ' + VIRIAL + PROPERTIES
    + 'Ge 8 8 8 .1 -.1 .2\nSi 9.5 8.7 8.3 -.1 .1 -.2\n'
    + 'Ge 8.4 9.6 8.9 .2 .1 -.1\n',
    '2\n' + CELL + 'energy=.2 weight=.9 ' + VIRIAL + PROPERTIES
    + 'Si 15 15 15 0 0 0\nGe 17 16 15.5 .1 -.1 .2\n',
    '4\n' + CELL + 'energy=-1.4 weight=1.1 ' + VIRIAL + PROPERTIES
    + 'Ge 20 20 20 -.1 .1 -.2\nSi 21.7 20.2 20.3 .2 -.1 .2\n'
    + 'Ge 20.5 21.8 20.4 -.2 .2 .1\nSi 21.1 20.7 22.1 .1 -.2 -.1\n',
    '3\n' + CELL + 'energy=-.8 weight=.8 ' + VIRIAL + PROPERTIES
    + 'Si 27 27 27 .1 -.2 .1\nGe 28.8 27.3 27.5 -.1 .1 -.2\n'
    + 'Si 27.5 28.7 27.9 .2 -.1 .1\n',
)
FLEXIBLE_ZBL = (
    '1.2 2.4 .22 3.2 .4717 .9423 .2802 .4029 .02817 .2016\n'
    '1.2 2.4 .24 3.2 .4517 .9423 .2802 .4029 .02817 .2016\n'
    '1.2 2.4 .26 3.2 .4317 .9423 .2802 .4029 .02817 .2016\n'
)
COMMON = (
    'type 2 Si Ge\ncutoff 5 4\nn_max 2 2\nbasis_size 3 3\n'
    'neuron 8\nbatch 3 0\nepoch 2\nweight_decay 0\n'
    'lambda_e .7\nlambda_f 1.2\nlambda_v .3\n'
    'lambda_shear 1.5\ntype_weight 1.2 .8\nforce_delta .2\n'
    'lambda_1 0\nlambda_2 .03\nloss_rmse 1e-3\n'
    'start_lr .001\nstop_lr .001\n'
    'optimizer lm\nlm 4 3 .2 .01\nlm_guard guard.xyz 1.1\nlm_check 1\n'
)


def verify_objective_rows(rows):
    if rows.shape[1] != 10 or not np.isfinite(rows).all():
        raise AssertionError('LM diagnostics must have ten finite columns')
    if not np.isin(rows[:, 3], [0., 1.]).all():
        raise AssertionError('Invalid LM acceptance indicator')
    if np.any(rows[:, 2] > rows[:, 1] + 1e-10 * (1 + np.abs(rows[:, 1]))):
        raise AssertionError('LM training objective increased')
    accepted = rows[:, 3] == 1
    if np.any(rows[accepted, 2] >= rows[accepted, 1]):
        raise AssertionError('A nondecreasing LM step was marked accepted')
    best = float('inf')
    maximum_ratio = 0.
    for row in rows:
        best = min(best, row[7])
        if row[3] == 1:
            tolerance = 1e-10 * (1 + abs(best))
            if row[8] > 1.1 * best + tolerance:
                raise AssertionError('Accepted guard exceeds 1.1 times historical minimum')
            if best > 0:
                maximum_ratio = max(maximum_ratio, row[8] / best)
            best = min(best, row[8])
        elif abs(row[8] - row[7]) > 1e-10 * (1 + abs(row[7])):
            raise AssertionError('Rejected step changed the guard objective')
        if abs(row[9] - best) > 1e-10 * (1 + abs(best)):
            raise AssertionError('Reported guard_best differs from accepted history')
    return maximum_ratio


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_free(gpu):
    output = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    pids = [int(item.strip()) for item in output.splitlines() if item.strip().isdigit()]
    if pids:
        raise RuntimeError('GPU is occupied: ' + str(pids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', action='append', choices=('three_body', 'four_body', 'five_body',
                        'typewise_five_body', 'fixed_zbl', 'flexible_zbl'),
                        help='Run only the named smoke case; repeatable')
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.bridge = args.bridge.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.root / 'tools/fcvp'))
    from reference import Model, Oracle, read_xyz
    binary = args.root / 'src/gnep'
    result = {'passed': False, 'binary_sha256': sha(binary), 'bridge_sha256': sha(args.bridge),
              'runner_sha256': sha(__file__), 'gpu': args.gpu, 'cases': [], 'runs': []}
    started = time.monotonic()

    def run(folder, config, initial=None, model=None, flexible=False):
        require_free(args.gpu)
        folder.mkdir()
        (folder / 'train.xyz').write_text(''.join(FRAMES[:3]))
        (folder / 'guard.xyz').write_text(FRAMES[3])
        (folder / 'test.xyz').write_text(FRAMES[4])
        (folder / 'gnep.in').write_text(config)
        if initial is not None:
            np.savetxt(folder / 'gnep.restart', initial, fmt='%.9e')
        if model is not None:
            (folder / 'nep.txt').write_bytes(model.read_bytes())
        if flexible:
            (folder / 'zbl.in').write_text(FLEXIBLE_ZBL)
        test_hash = sha(folder / 'test.xyz')
        guard_hash = sha(folder / 'guard.xyz')
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        before = time.monotonic()
        process = subprocess.run([str(binary)], cwd=folder, env=environment,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        (folder / 'run.log').write_text(process.stdout)
        result['runs'].append({'folder': folder.name, 'returncode': process.returncode,
                               'seconds': time.monotonic() - before})
        if process.returncode:
            raise AssertionError(folder.name + ' failed: ' + process.stdout[-6000:])
        if sha(folder / 'test.xyz') != test_hash or sha(folder / 'guard.xyz') != guard_hash:
            raise AssertionError(folder.name + ' changed a test or guard input')
        return process.stdout

    try:
        cases = [
            ('three_body', '4', 31, ''),
            ('four_body', '4 2 0', 136, ''),
            ('five_body', '4 2 1', 184, ''),
            ('typewise_five_body', '4 2 1', 184, 'use_typewise_cutoff 1.5 1.5\n'),
            ('fixed_zbl', '4 2 1', 184, 'zbl 2.4\n'),
            ('flexible_zbl', '4 2 1', 184, 'zbl 2.4\n'),
        ]
        for name, lmax, seed, extra in cases:
            if args.case and name not in args.case:
                continue
            fields = list(map(int, lmax.split()))
            channels = fields[0] + sum(value > 0 for value in fields[1:])
            dim = 3 + 3 * channels
            count = 2 * ((dim + 2) * 8 + 1) + 4 * (3 * 4 + 3 * 4)
            initial = np.random.default_rng(seed).integers(-100, 101, count) / 1024.
            folder = args.output / name
            log = run(folder, COMMON + 'l_max ' + lmax + '\nseed %d\n' % seed + extra,
                      initial=initial, flexible=name == 'flexible_zbl')
            moment = re.findall(r'LM moment descriptor gate max_scaled_error=([\d.eE+-]+)', log)
            gradient = re.findall(r'LM native gradient gate max_scaled_error=([\d.eE+-]+)', log)
            relative = re.findall(r'LM native gradient gate[^\n]*relative_norm_error=([\d.eE+-]+)', log)
            if not moment or not gradient:
                raise AssertionError(name + ' omitted a required native gate')
            if max(map(float, moment)) > 2e-5 or max(map(float, gradient)) > .002:
                raise AssertionError(name + ' failed a numeric native gate')
            if not relative or max(map(float, relative)) > .002:
                raise AssertionError(name + ' failed the relative gradient norm gate')
            lm_rows = np.loadtxt(folder / 'lm.out', ndmin=2)
            native_rows = np.loadtxt(folder / 'loss.out', ndmin=2)
            parameters = np.loadtxt(folder / 'gnep.restart')
            if (native_rows.shape != (2, 10)
                    or not np.isfinite(lm_rows).all() or not np.isfinite(native_rows).all()
                    or not np.isfinite(parameters).all()):
                raise AssertionError(name + ' exported malformed or nonfinite diagnostics')
            guard_ratio = verify_objective_rows(lm_rows)
            model_path = folder / 'nep.txt'
            model = Model.read(model_path)
            if model.training_parameter_count != count:
                raise AssertionError(name + ' exported unexpected parameter count')
            if 'zbl' in name and not model.header[0].startswith('nep5_zbl '):
                raise AssertionError(name + ' lost embedded ZBL')
            frames = read_xyz(folder / 'train.xyz')
            expected = None
            # This pinned NEP_CPU rejects GPUMD's extended typewise cutoff line.
            # Its parser calls exit(1), so do not feed it that unsupported format.
            if name != 'typewise_five_body':
                oracle = Oracle(args.bridge, model_path)
                try:
                    expected = [oracle.predict(frame, model.beta) for frame in frames]
                finally:
                    oracle.close()
            predicted = args.output / (name + '_predict')
            run(predicted, 'prediction 1\nbatch 3 0\n', model=model_path)
            actual = {key: np.loadtxt(predicted / (key + '_train.out'), ndmin=2)
                      for key in ('energy', 'force', 'virial')}
            wanted = {} if expected is None else {
                'energy': np.asarray([row[0] for row in expected]),
                'force': np.concatenate([row[1:1 + 3 * len(frame.species)].reshape(3, -1).T
                                         for frame, row in zip(frames, expected)]),
                'virial': np.asarray([row[-6:] for row in expected]),
            }
            observed = {'energy': actual['energy'][:, 0], 'force': actual['force'][:, :3],
                        'virial': actual['virial'][:, :6]}
            differences = {}
            if not all(np.isfinite(values).all() for values in actual.values()):
                raise AssertionError(name + ' native prediction is nonfinite')
            for quantity in wanted:
                np.testing.assert_allclose(observed[quantity], wanted[quantity], rtol=2e-3, atol=2e-3)
                differences[quantity] = float(np.max(np.abs(observed[quantity] - wanted[quantity])))
            result['cases'].append({'name': name, 'passed': True, 'parameters': count,
                                    'moment_gate_max_scaled_error': max(map(float, moment)),
                                    'gradient_gate_max_scaled_error': max(map(float, gradient)),
                                    'gradient_gate_relative_norm_error': max(map(float, relative)),
                                    'maximum_accepted_guard_over_prior_best': float(guard_ratio),
                                    'test_sha256': sha(folder / 'test.xyz'),
                                    'guard_sha256': sha(folder / 'guard.xyz'),
                                    'lm_rows': lm_rows.tolist(), 'cpu_export_max_abs_difference': differences,
                                    'cpu_export_status': 'matched' if expected is not None
                                    else 'unsupported_extended_typewise_cutoff_in_pinned_oracle',
                                    'model_sha256': sha(model_path)})
            (args.output / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['seconds'] = time.monotonic() - started
        (args.output / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
