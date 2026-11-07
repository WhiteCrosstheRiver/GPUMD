#!/usr/bin/env python3
"""GPU checks for fix shell, fix type and fix group.

usage: run_checks.py --gpumd GPUMD --nep SiGe_NEP --repo REPO --work NEW_DIRECTORY --device ID
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'fix_skip'))
import run_checks as skip  # noqa: E402

SPACING = skip.A
CELLS = (6, 6, 5)
BOX = [CELLS[0] * SPACING, CELLS[1] * SPACING, 1e9]
DUMP = 'dump_xyz 10 d.xyz precision double group_labels force\n'
SHELL = 'fix shell substrate z cutoff 3.0 offset 6.0\n'
GROW = 'fix shell substrate z cutoff 3.0 offset 6.0 surface radius 3 mode grow every 1\n'
results = []


def report(name, ok, detail=''):
    results.append(ok)
    print('%s %s %s' % ('PASS' if ok else 'FAIL', name, detail))


def run(args, name, body, potential=None, model=None):
    folder = args.work / name
    folder.mkdir(parents=True)
    shutil.copy(potential or args.nep, folder / 'potential.txt')
    if model:
        shutil.copy(model, folder / 'model.xyz')
    else:
        skip.write_model(folder / 'model.xyz', ['Si', 'Ge'], SPACING, CELLS, 800, False, frozen=0.4)
    (folder / 'run.in').write_text('potential potential.txt\n' + body)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device))
    with open(folder / 'run.log', 'w') as log:
        code = subprocess.run([args.gpumd], cwd=folder, env=env, stdout=log,
                              stderr=subprocess.STDOUT, timeout=600).returncode
    return folder, code, (folder / 'run.log').read_text()


def read(folder):
    frames = skip.frames(folder / 'd.xyz')
    species = [row[0] for row in frames[0][1]]
    pos, labels, forces = [], [], []
    for offset, rows in frames:
        key = 'group_labels' if 'group_labels' in offset else 'group'
        pos.append([[float(x) for x in r[offset['pos']:offset['pos'] + 3]] for r in rows])
        labels.append([int(r[offset[key]]) for r in rows])
        forces.append([[float(x) for x in r[offset['forces']:offset['forces'] + 3]] for r in rows])
    return species, pos, labels, forces


def model_labels(folder):
    rows = (folder / 'model.xyz').read_text().split('\n')[2:]
    return [int(l.split()[-1]) for l in rows if l]


def shift(a, b, i):
    worst = 0.0
    for d in range(3):
        s = abs(a[i][d] - b[i][d])
        worst = max(worst, min(s, BOX[d] - s))
    return worst


def split_moved(p0, p1, frozen):
    """Largest displacement of frozen atoms, smallest-of-max check for free atoms."""
    still = max((shift(p0, p1, i) for i in range(len(p0)) if frozen[i]), default=0.0)
    free = [shift(p0, p1, i) for i in range(len(p0)) if not frozen[i]]
    return still, (sum(s > 1e-4 for s in free), len(free))


def diff(a, b):
    return max(abs(x - y) for f, g in zip(a, b) for r, s in zip(f, g) for x, y in zip(r, s))


def check_frozen(name, p0, p1, frozen):
    still, (moving, total) = split_moved(p0, p1, frozen)
    # gpu_apply_pbc round-trips every atom through fractional coordinates, which moves some
    # layers by one bit per step, frozen or not. Upstream does this too.
    report(name, still < 1e-11 and moving == total and sum(frozen) > 0 and total > 0,
           'frozen %d max shift %.2e, free moving %d of %d' % (sum(frozen), still, moving, total))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpumd', required=True)
    parser.add_argument('--nep', required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    args.work.mkdir(parents=True)

    # Shell geometry on the labelled test model, which has no group column.
    src = HERE.parent / 'shell'
    carbon = args.repo / 'potentials/nep/C_2024_NEP4.txt'
    body = (src / 'run.in').read_text().split('\n', 1)[1]
    folder, code, log = run(args, 'shell_labels', body + 'ensemble nve\nrun 1\n', carbon, src / 'model.xyz')
    check = subprocess.run([sys.executable, str(src / 'check.py')], cwd=folder,
                           capture_output=True, text=True)
    report('shell_labels', code == 0 and check.returncode == 0, (check.stdout + check.stderr).strip()[:80])

    # fix group 1 must be the same as upstream's fix 1.
    plain, c1, _ = run(args, 'plain', 'time_step 1\nensemble nve\nfix 1\n' + DUMP + 'run 50\n')
    named, c2, _ = run(args, 'named', 'time_step 1\nensemble nve\nfix group 1\n' + DUMP + 'run 50\n')
    _, pa, la, _ = read(plain)
    _, pb, lb, _ = read(named)
    gap = max(diff(pa[-1:], pb[-1:]), 0.0)
    report('fix_group_matches_plain', c1 == 0 and c2 == 0 and gap < 1e-9 and la[-1] == lb[-1], 'max dr %.2e' % gap)
    check_frozen('plain_control', pa[0], pa[-1], [x == 1 for x in la[0]])

    # fix type
    folder, code, _ = run(args, 'type', 'time_step 1\nensemble nve\nfix type Si\n' + DUMP + 'run 50\n')
    species, pos, labels, _ = read(folder)
    frozen = [s == 'Si' for s in species]
    report('type_labels', code == 0 and [x == 1 for x in labels[0]] == frozen)
    check_frozen('type_frozen', pos[0], pos[-1], frozen)

    # A plain fix followed by fix type is a union.
    folder, code, _ = run(args, 'union', 'time_step 1\nensemble nve\nfix 1\nfix type Ge\n' + DUMP + 'run 50\n')
    species, pos, labels, _ = read(folder)
    user = model_labels(folder)
    frozen = [u == 1 or s == 'Ge' for u, s in zip(user, species)]
    report('union_labels', code == 0 and [x == 1 for x in labels[0]] == frozen)
    check_frozen('union_frozen', pos[0], pos[-1], frozen)

    # Group 0 goes back to the model labels after the run.
    body = ('time_step 1\nensemble nve\n' + SHELL + DUMP + 'run 20\n'
            'ensemble nve\nfix 1\n' + DUMP + 'run 20\n')
    folder, code, _ = run(args, 'restore', body)
    _, pos, labels, _ = read(folder)
    user = model_labels(folder)
    half = len(pos) // 2
    shell_frozen = [x == 1 for x in labels[0]]
    report('restore_shell_differs', code == 0 and labels[0] != user, 'shell frozen %d' % sum(shell_frozen))
    report('restore_labels', labels[-1] == user)
    check_frozen('restore_block1', pos[0], pos[half - 1], shell_frozen)
    check_frozen('restore_block2', pos[half], pos[-1], [u == 1 for u in user])

    # Rejected combinations.
    _, code, log = run(args, 'plain_after', 'time_step 1\nensemble nve\nfix type Si\nfix 1\nrun 1\n')
    report('plain_after_union_rejected', code != 0 and 'Plain fix cannot follow' in log)
    _, code, log = run(args, 'two_ints', 'time_step 1\nensemble nve\nfix 1 3\nrun 1\n')
    report('fix_two_ints_is_method_group', code != 0 and 'grouping method' in log)

    # The grow ratchet, then a plain run that must drop it.
    block = 'ensemble nve\n' + GROW + DUMP + 'run 20\n'
    body = 'time_step 1\n' + block + block + 'ensemble nve\nfix 1\nrun 5\n' + block
    folder, code, log = run(args, 'grow', body)
    _, pos, labels, _ = read(folder)
    first, second = labels[1], labels[3]
    ratchet = all(b >= a for a, b in zip(first, second))
    full = log.count('grow, full')
    report('grow_ratchet', code == 0 and ratchet and sum(first) > 0, 'frozen %d then %d' % (sum(first), sum(second)))
    report('grow_full_after_drop', full == 2, 'full classifications %d' % full)
    check_frozen('grow_frozen', pos[0], pos[1], [x == 1 for x in first])

    # fix_skip with a shell-derived frozen set.
    body = 'time_step 1\nensemble nve\n' + SHELL + 'dump_xyz 10 d.xyz precision double group_labels force\nrun 50\n'
    full_run, c1, _ = run(args, 'skip_off', body)
    skip_run, c2, log = run(args, 'skip_on', 'fix_skip\n' + body)
    _, pa, la, fa = read(full_run)
    _, pb, lb, fb = read(skip_run)
    free = [i for i, x in enumerate(la[0]) if x == 0]
    dr = max(abs(a - b) for p, q in zip(pa, pb) for i in free for a, b in zip(p[i], q[i]))
    df = max(abs(a - b) for p, q in zip(fa, fb) for i in free for a, b in zip(p[i], q[i]))
    report('skip_with_shell', c1 == 0 and c2 == 0 and 'fix_skip is on' in log and la == lb and dr < 1e-3 and df < 1e-2,
           'max dr %.2e dF %.2e' % (dr, df))

    print('Summary: %d of %d passed' % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
