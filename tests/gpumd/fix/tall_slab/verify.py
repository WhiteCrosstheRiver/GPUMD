#!/usr/bin/env python3
"""Verify benchmark logs and matching warm-up/timed/final output frames."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark import save, sha, xyz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.root / 'report.json').read_text())
    assert report['complete'] and len(report['cases']) == len(report['nz'])
    assert sha(report['engine']) == report['engine_sha256']
    verification = {'cases': [], 'engine_unchanged': True}
    for case in report['cases']:
        n = case['n']
        root = args.root / f'nz{case["nz"]}'
        for run in list(case['correctness'].values()) + case['timings']:
            lane = run.get('lane', Path(run['path']).name.split('_')[-1])
            log = (Path(run['path']) / 'stdout.log').read_text()
            assert ('Fix skip is on' in log) == (lane != 'full')
            assert 'nan' not in log.lower().split()
        endpoints = []
        for rep in range(report['repetitions']):
            reference = xyz(root / f'bench_r{rep}_full/final.xyz')
            mobile = reference['group'][:, 0] == 0
            for lane in ('skip', 'reuse'):
                candidate = xyz(root / f'bench_r{rep}_{lane}/final.xyz')
                assert candidate['n'] == reference['n'] == n
                assert np.array_equal(candidate['group'], reference['group'])
                force = float(np.abs(candidate['force'][mobile] - reference['force'][mobile]).max())
                position = float(np.abs(candidate['pos'] - reference['pos']).max())
                endpoints.append({'repetition': rep, 'lane': lane,
                                  'mobile_force_max_error_eV_A': force,
                                  'position_max_error_A': position})
        verification['cases'].append({'n': n, 'endpoints': endpoints,
                                      'all_skip_modes_confirmed': True})
    verification['complete'] = True
    verification['endpoints_within_tolerance'] = all(
        item['mobile_force_max_error_eV_A'] <= 1e-4 and item['position_max_error_A'] <= 1e-6
        for case in verification['cases'] for item in case['endpoints'])
    save(args.root / 'endpoint_verification.json', verification)
    print(json.dumps(verification, indent=2))
    if not verification['endpoints_within_tolerance']:
        raise SystemExit('Endpoint trajectory comparison exceeded tolerance; inspect the saved errors')


if __name__ == '__main__':
    main()
