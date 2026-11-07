#!/usr/bin/env python3
"""Run the deposit and delete cases, each in a copy, then its check.py.

usage: run_all.py --gpumd GPUMD --repo REPO --work NEW_DIRECTORY --device ID
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--gpumd', required=True)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--work', type=Path, required=True)
parser.add_argument('--device', default='0')
args = parser.parse_args()
args.work.mkdir(parents=True)
root = args.repo / 'tests/gpumd'
cases = sorted(p.parent for name in ('deposit', 'delete') for p in (root / name).glob('*/run.in'))
passed = 0
for case in cases:
    label = '%s/%s' % (case.parent.name, case.name)
    folder = args.work / label
    shutil.copytree(case, folder)
    text = (folder / 'run.in').read_text()
    (folder / 'run.in').write_text(text.replace('../../../../potentials', str(args.repo / 'potentials')))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device))
    with open(folder / 'gpumd.log', 'w') as log:
        code = subprocess.run([args.gpumd], cwd=folder, env=env, stdout=log,
                              stderr=subprocess.STDOUT, timeout=300).returncode
    check = folder / 'check.py'
    result = subprocess.run([sys.executable, str(check)], cwd=folder, capture_output=True, text=True)
    ok = code == 0 and result.returncode == 0
    passed += ok
    detail = (result.stdout + result.stderr).strip().split('\n')[-1][:110]
    if code:
        detail = 'gpumd rc %d: %s' % (code, (folder / 'gpumd.log').read_text().strip().split('\n')[-1][:100])
    print('%s %s %s' % ('PASS' if ok else 'FAIL', label, detail), flush=True)
print('Summary: %d of %d passed' % (passed, len(cases)))
sys.exit(0 if passed == len(cases) else 1)
