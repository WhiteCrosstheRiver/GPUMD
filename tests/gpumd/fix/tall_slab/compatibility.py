#!/usr/bin/env python3
"""Bounded legacy-input controls, separate from the performance measurements."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from benchmark import ENGINE, POTENTIAL, REPO, execute, save, sha, xyz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-name', default='compatibility')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--engine', type=Path, default=ENGINE)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--nz', type=int, default=72)
    args = parser.parse_args()
    args.root = args.root.resolve(strict=True)
    baseline = args.baseline.resolve(strict=True)
    engine = args.engine.resolve(strict=True)
    if Path(args.output_name).name != args.output_name:
        raise ValueError('output-name must be one directory name')
    root = args.root / args.output_name
    root.mkdir(exist_ok=False)
    classified = args.root / f'nz{args.nz}/grow_classify/classified.xyz'
    cases = (
        ('nep_no_fix', POTENTIAL, ''),
        ('nep_fixed', POTENTIAL, 'fix 1\n'),
        ('nep_legacy_shell', POTENTIAL, 'fix 1\nfix shell substrate z cutoff 2.9 offset 38.017\n'),
        ('tersoff_fixed', REPO / 'potentials/tersoff/Si_Tersoff_1989.txt', 'fix 1\n'),
    )
    report = {'candidate_sha256': sha(engine), 'baseline_sha256': sha(baseline),
              'baseline_binary': str(baseline), 'cases': []}
    for name, potential, fix in cases:
        frames = {}
        runs = {}
        source = classified
        velocity = ''
        if name == 'nep_legacy_shell':
            # Exercise legacy classification itself, not a union already frozen by grow.
            source = args.root / f'nz{args.nz}/grow_classify/model.xyz'
            velocity = 'velocity 300 seed 231013\n'
        inp = (f'potential {potential}\n' + velocity + 'time_step 0.5\nensemble nve\n' + fix
               + 'dump_xyz -1 0 40 final.xyz group velocity force\nrun 40\n')
        for lane, binary in (('baseline', baseline), ('candidate', engine)):
            directory = root / f'{name}_{lane}'
            directory.mkdir()
            shutil.copyfile(source, directory / 'model.xyz')
            runs[lane] = execute(directory, inp, args.gpu, binary)
            frames[lane] = xyz(directory / 'final.xyz')
        a, b = frames['baseline'], frames['candidate']
        if not np.array_equal(a['group'], b['group']) or a['n'] != b['n']:
            raise AssertionError(f'{name}: identity mismatch')
        force = float(np.abs(a['force'] - b['force']).max())
        position = float(np.abs(a['pos'] - b['pos']).max())
        passed = force <= 1e-4 and position <= 1e-6
        report['cases'].append({'case': name, 'runs': runs, 'force_max_error_eV_A': force,
                                'position_max_error_A': position, 'passed': passed,
                                'source_model': str(source)})
        save(root / 'report.json', report)
        print(json.dumps(report['cases'][-1]), flush=True)
        if not passed:
            raise AssertionError(name)
    report['complete'] = True
    save(root / 'report.json', report)


if __name__ == '__main__':
    main()
