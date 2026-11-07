#!/usr/bin/env python3
"""GPU checks for variable, for and if in run.in.

usage: run_checks.py --gpumd GPUMD --lj LJ_FILE --work NEW_DIRECTORY --device ID
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'fix_skip'))
import run_checks as skip  # noqa: E402

RUN = 'time_step 1\nensemble nve\ndump_xyz 1 %s\nrun 1\n'
results = []


def run(args, name, body):
    folder = args.work / name
    folder.mkdir(parents=True)
    shutil.copy(args.lj, folder / 'potential.txt')
    skip.write_model(folder / 'model.xyz', ['Ar'], 7.6, (6, 6, 5), 40, True)
    (folder / 'run.in').write_text('potential potential.txt\n' + body)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device))
    with open(folder / 'run.log', 'w') as log:
        code = subprocess.run([args.gpumd], cwd=folder, env=env, stdout=log,
                              stderr=subprocess.STDOUT, timeout=120).returncode
    return folder, code, (folder / 'run.log').read_text()


def files(args, name, body, expected):
    folder, code, log = run(args, name, body)
    found = sorted(p.name for p in folder.glob('f_*.xyz'))
    want = sorted(expected)
    runs = log.count('Speed of this run')
    ok = code == 0 and found == want and runs == len(want)
    results.append(ok)
    print('%s %s: runs %d files %s' % ('PASS' if ok else 'FAIL', name, runs, found[:6]))


def rejected(args, name, body, message):
    _, code, log = run(args, name, body)
    ok = code != 0 and message in log
    results.append(ok)
    tail = log.strip().split('\n')[-1][:90]
    print('%s %s: rc %d %s' % ('PASS' if ok else 'FAIL', name, code, '' if ok else tail))


def loop(header, inner):
    return header + '\n' + inner + 'end\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpumd', required=True)
    parser.add_argument('--lj', required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    args.work.mkdir(parents=True)

    files(args, 'range', loop('for i range 1 3', RUN % 'f_${i}.xyz'), ['f_%d.xyz' % i for i in (1, 2, 3)])
    files(args, 'range_down', loop('for i range 5 1 -2', RUN % 'f_${i}.xyz'), ['f_%d.xyz' % i for i in (1, 3, 5)])
    files(args, 'values', loop('for s values alpha 2.5', RUN % 'f_${s}.xyz'), ['f_alpha.xyz', 'f_2.5.xyz'])
    files(args, 'nested', loop('for i range 1 2', loop('for j range 1 2', RUN % 'f_${i}_${j}.xyz')),
          ['f_1_1.xyz', 'f_1_2.xyz', 'f_2_1.xyz', 'f_2_2.xyz'])
    files(args, 'if_then', 'if 1 < 2\n' + RUN % 'f_then.xyz' + 'else\n' + RUN % 'f_else.xyz' + 'end\n', ['f_then.xyz'])
    files(args, 'if_else', 'if 2 <= 1\n' + RUN % 'f_then.xyz' + 'else\n' + RUN % 'f_else.xyz' + 'end\n', ['f_else.xyz'])
    files(args, 'if_string', loop('for s values a b', 'if ${s} == b\n' + RUN % 'f_${s}.xyz' + 'end\n'), ['f_b.xyz'])
    files(args, 'if_at_runtime', loop('for i range 1 4', 'if ${i} > 2\n' + RUN % 'f_${i}.xyz' + 'end\n'), ['f_3.xyz', 'f_4.xyz'])
    files(args, 'formula', RUN % 'f_$(2+3*4).xyz', ['f_14.xyz'])
    files(args, 'box_variable', 'variable L equal lx\nif ${L} == 45.6\n' + RUN % 'f_box.xyz' + 'end\n', ['f_box.xyz'])
    files(args, 'chained', 'variable a equal 3\nvariable b equal v_a*2\n' + RUN % 'f_${b}.xyz', ['f_6.xyz'])
    files(args, 'redefine', 'variable a equal 3\nvariable a equal 4\n' + RUN % 'f_${a}.xyz', ['f_4.xyz'])
    files(args, 'loop_in_formula', loop('for i range 2 3', RUN % 'f_$(v_i*10).xyz'), ['f_20.xyz', 'f_30.xyz'])
    files(args, 'comment_in_block', loop('for i range 1 1 # one pass', '# skip me\n' + RUN % 'f_${i}.xyz'), ['f_1.xyz'])

    # random() must parse and stay inside its range.
    body = 'variable r equal random(0,100,7)\nif ${r} < 200\n' + RUN % 'f_ok.xyz' + 'end\n'
    files(args, 'random', body, ['f_ok.xyz'])

    rejected(args, 'no_end', 'for i range 1 2\n' + RUN % 'f.xyz', 'missing a matching end')
    rejected(args, 'stray_end', RUN % 'f.xyz' + 'end\n', 'end without matching')
    rejected(args, 'stray_else', 'else\n', 'else without matching')
    rejected(args, 'reused_loop_name', loop('for i range 1 2', loop('for i range 1 2', RUN % 'f.xyz')), 'cannot reuse')
    rejected(args, 'header_substitution', loop('for i range 1 ${n}', RUN % 'f.xyz'), 'not allowed in a for header')
    rejected(args, 'zero_step', loop('for i range 1 3 0', RUN % 'f.xyz'), 'step cannot be 0')
    rejected(args, 'undefined', RUN % 'f_${nope}.xyz', "Undefined variable")
    rejected(args, 'cyclic', 'variable a equal v_b\nvariable b equal v_a\n' + RUN % 'f_${a}.xyz', 'Cyclic variable')
    rejected(args, 'relational_string', 'if abc < def\nend\n', 'needs numeric')
    rejected(args, 'loop_name_as_variable', loop('for i range 1 2', 'variable i equal 2\n'), 'variable')
    rejected(args, 'ensemble_without_run', loop('for i range 1 1', 'ensemble nve\n'), 'not followed by a run')

    print('Summary: %d of %d passed' % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
