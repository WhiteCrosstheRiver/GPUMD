#!/usr/bin/env python3
"""Write git-status.html in the repository root.

The page shows branches, worktrees, uncommitted work, and a commit timeline with
each commit marked MAJOR or MINOR. It needs only git and the Python standard
library, and never changes the repository.

  python3 scripts/git_status_html.py                 # write the page once
  python3 scripts/git_status_html.py --install-hook  # also refresh on commit/merge

Adjust the marking rules in classify().
"""
import argparse
import html
import json
import os
import re
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path

MAJOR_LINES = 300  # source lines changed that make a commit MAJOR by size alone
FIELD = '\x1f'
RECORD = '\x1e'
CORE = ('src/', 'extensions/')
BUILD = ('src/makefile', 'src/makefile.hip', 'CMakeLists.txt')
MINOR_TYPES = {'docs', 'doc', 'style', 'test', 'tests', 'chore', 'ci', 'build', 'refactor'}
CONVENTIONAL = re.compile(r'^(?:@\s*)?([a-z]+)(?:\(([^)]*)\))?(!)?:\s', re.I)
HOOK = """#!/bin/sh
# Added by scripts/git_status_html.py; refreshes git-status.html in the background.
top=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
[ -f "$top/scripts/git_status_html.py" ] || exit 0
(python3 "$top/scripts/git_status_html.py" -C "$top" >/dev/null 2>&1 &)
exit 0
"""


def git(root, *args, check=True):
    # Hooks run with GIT_DIR, GIT_INDEX_FILE, etc. of the worktree that
    # committed. Drop them, or `git -C <other worktree>` reports that worktree.
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    result = subprocess.run(
        ['git', '-C', str(root)] + list(args),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        errors='replace', env=env)
    if check and result.returncode:
        raise SystemExit('git %s failed: %s' % (' '.join(args), result.stderr.strip()))
    return result.stdout if result.returncode == 0 else ''


def classify(subject, parents, files):
    """Return ('major' or 'minor', reason). Every commit gets a stated reason."""
    match = CONVENTIONAL.match(subject)
    kind = match.group(1).lower() if match else ''
    source = [f for f in files if f['path'].startswith(CORE)]
    churn = sum(f['added'] + f['deleted'] for f in source)
    new_source = [f for f in source if f['status'] == 'A' and not f['path'].endswith('.md')]
    if (match and match.group(3)) or 'BREAKING' in subject:
        return 'major', 'marked as a breaking change'
    if any(f['path'] in BUILD for f in files) and churn >= 20:
        return 'major', 'changes the build and %d source lines' % churn
    if kind == 'feat' and source:
        return 'major', 'new feature touching source'
    if new_source:
        return 'major', '%d new source file%s' % (len(new_source), 's'[:len(new_source) != 1])
    if churn >= MAJOR_LINES:
        return 'major', '%d source lines changed' % churn
    if kind in MINOR_TYPES:
        return 'minor', '%s commit, %d source lines' % (kind, churn)
    if not source:
        return 'minor', 'no source files touched'
    return 'minor', '%d source lines changed' % churn


def area(path):
    parts = path.split('/')
    if parts[0] == 'src' and len(parts) > 2:
        return parts[1]
    return parts[0]


def read_commits(root, limit):
    fmt = FIELD.join(['%H', '%h', '%P', '%an', '%aI', '%s', '%D'])
    raw = git(
        root, 'log', '--branches', '--remotes=origin', '--remotes=upstream',
        '--max-count=%d' % limit, '--date-order', '--diff-merges=first-parent',
        '--find-renames', '--numstat', '--raw', '--no-abbrev',
        '--format=' + RECORD + fmt)
    commits = []
    for block in raw.split(RECORD)[1:]:
        lines = block.strip('\n').split('\n')
        full, short, parents, author, date, subject, refs = (lines[0].split(FIELD) + [''] * 7)[:7]
        status = {}
        files = {}
        for line in lines[1:]:
            if line.startswith(':'):
                pieces = line.split('\t')
                status[pieces[-1]] = pieces[0].split()[-1][0]
            elif line.count('\t') >= 2:
                added, deleted, path = line.split('\t', 2)
                if '=>' in path:
                    path = re.sub(r'\{[^{}]*=> ([^{}]*)\}', r'\1', path).replace('//', '/')
                    path = path.split(' => ')[-1]
                files[path] = {
                    'path': path, 'status': status.get(path, 'M'),
                    'added': int(added) if added.isdigit() else 0,
                    'deleted': int(deleted) if deleted.isdigit() else 0}
        items = list(files.values())
        level, reason = classify(subject, parents.split(), items)
        counts = {}
        for item in items:
            name = area(item['path'])
            counts[name] = counts.get(name, 0) + item['added'] + item['deleted']
        commits.append({
            'hash': full[:9], 'author': author, 'date': date[:16].replace('T', ' '),
            'subject': subject, 'refs': refs, 'level': level, 'reason': reason,
            'merge': len(parents.split()) > 1,
            'added': sum(f['added'] for f in items), 'deleted': sum(f['deleted'] for f in items),
            'files': len(items),
            'areas': sorted(counts, key=lambda k: -counts[k])[:4]})
    return commits


def ahead_behind(root, left, right):
    out = git(root, 'rev-list', '--left-right', '--count', '%s...%s' % (left, right), check=False)
    pieces = out.split()
    return (int(pieces[0]), int(pieces[1])) if len(pieces) == 2 else (None, None)


def read_branches(root):
    base = 'upstream/master' if git(
        root, 'rev-parse', '--verify', '-q', 'upstream/master', check=False) else 'origin/master'
    fmt = '%(refname:short)' + FIELD + '%(committerdate:short)' + FIELD + '%(objectname:short)' + \
        FIELD + '%(contents:subject)'
    raw = git(root, 'for-each-ref', '--sort=-committerdate', '--format=' + fmt,
              'refs/heads', 'refs/remotes/origin')
    branches = []
    for line in raw.splitlines():
        name, date, short, subject = (line.split(FIELD) + [''] * 4)[:4]
        if name in ('origin', 'origin/HEAD'):
            continue
        ahead, behind = ahead_behind(root, name, base)
        branches.append({'name': name, 'date': date, 'hash': short, 'subject': subject,
                         'ahead': ahead, 'behind': behind})
    return base, branches


def read_worktrees(root):
    worktrees = []
    current = {}
    for line in git(root, 'worktree', 'list', '--porcelain').splitlines() + ['']:
        if line.startswith('worktree '):
            current = {'path': line[9:], 'branch': '(detached)'}
        elif line.startswith('branch '):
            current['branch'] = line[7:].replace('refs/heads/', '')
        elif not line and current:
            status = git(current['path'], 'status', '--porcelain', check=False).splitlines()
            current['changed'] = len(status)
            current['conflicts'] = sum(
                s[:2] in ('UU', 'AA', 'DD', 'AU', 'UA', 'DU', 'UD') for s in status)
            current['files'] = [s[3:] for s in status[:12]]
            merging = git(current['path'], 'rev-parse', '-q', '--verify', 'MERGE_HEAD', check=False)
            current['merging'] = bool(merging.strip())
            worktrees.append(current)
            current = {}
    return worktrees


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>Git Status</title>
<style>
:root{--bg:#f7f7f4;--panel:#fff;--ink:#1d2125;--muted:#646c74;--line:#dfe1dc;--major:#b3261e;
--major-bg:#fdeceb;--minor:#3f6b4f;--minor-bg:#eaf3ec;--warn:#8a5a00;--warn-bg:#fdf3dc;--accent:#25588f}
@media (prefers-color-scheme:dark){:root{--bg:#15181b;--panel:#1e2226;--ink:#e7e9ea;--muted:#9aa2aa;
--line:#343a40;--major:#ff8a80;--major-bg:#3a1f1d;--minor:#8fcf9f;--minor-bg:#1f3026;--warn:#f0c674;
--warn-bg:#3a2f18;--accent:#8ab8e8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 16px 56px}
h1{font-size:24px;margin:0}h2{font-size:16px;margin:32px 0 10px}
.sub{color:var(--muted);font-size:13px;margin-top:2px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-top:18px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tile b{display:block;font-size:22px;line-height:1.2}.tile span{color:var(--muted);font-size:12px}
.tile.major b{color:var(--major)}.tile.warn b{color:var(--warn)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px}tr:last-child td{border-bottom:0}
code{font:12.5px ui-monospace,Consolas,monospace}.num{text-align:right;white-space:nowrap}
.pill{display:inline-block;padding:0 7px;border-radius:999px;font-size:11px;font-weight:700;letter-spacing:.02em}
.pill.warn{background:var(--warn-bg);color:var(--warn)}
.bar{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;align-items:center}
.bar button{font:inherit;font-size:13px;padding:4px 12px;border:1px solid var(--line);border-radius:6px;
background:var(--panel);color:var(--ink);cursor:pointer}
.bar button[aria-pressed=true]{border-color:var(--accent);color:var(--accent);font-weight:600}
.bar input{font:inherit;font-size:13px;flex:1 1 200px;min-width:0;padding:4px 10px;border:1px solid var(--line);
border-radius:6px;background:var(--panel);color:var(--ink)}
.commit{display:grid;grid-template-columns:62px 1fr;gap:10px;padding:9px 12px;border-bottom:1px solid var(--line)}
.commit:last-child{border-bottom:0}.commit.major{border-left:4px solid var(--major)}
.commit.minor{border-left:4px solid var(--minor)}
.badge{align-self:start;text-align:center;font-size:10.5px;font-weight:800;letter-spacing:.04em;padding:2px 0;border-radius:4px}
.major .badge{background:var(--major-bg);color:var(--major)}.minor .badge{background:var(--minor-bg);color:var(--minor)}
.subject{overflow-wrap:anywhere}.minor .subject{color:var(--muted)}.major .subject{font-weight:600}
.meta{color:var(--muted);font-size:12px;overflow-wrap:anywhere}
.refs{color:var(--accent);font-size:12px;overflow-wrap:anywhere}.none{padding:16px;color:var(--muted)}
</style></head><body><main>
<h1>Git Status</h1>
<div class="sub">__ROOT__ &middot; generated __NOW__ &middot; page reloads every 60 s &middot; compared with __BASE__</div>
<div class="tiles">__TILES__</div>
<h2>Worktrees and uncommitted work</h2><div class="card"><table>
<tr><th>Path</th><th>Branch</th><th>State</th></tr>__WORKTREES__</table></div>
<h2>Branches</h2><div class="card"><table>
<tr><th>Branch</th><th class="num">Ahead</th><th class="num">Behind</th><th>Last commit</th><th>Date</th></tr>__BRANCHES__</table></div>
<h2>Commits</h2>
<div class="bar"><button data-f="all" aria-pressed="false">All</button>
<button data-f="major" aria-pressed="true">Major only</button>
<button data-f="minor" aria-pressed="false">Minor only</button>
<input id="q" type="search" placeholder="Filter by text, author or area" aria-label="Filter commits"></div>
<div class="card" id="list"></div>
<div class="sub" id="shown"></div>
<script>
const C=__DATA__;let f='major',q='';
const e=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function draw(){const r=C.filter(c=>(f==='all'||c.level===f)&&(!q||(c.subject+c.author+c.hash+c.areas.join(' ')+c.refs).toLowerCase().includes(q)));
document.getElementById('list').innerHTML=r.map(c=>`<div class="commit ${c.level}"><div class="badge">${c.level.toUpperCase()}</div><div>
<div class="subject">${c.merge&&!/^merge/i.test(c.subject)?'Merge: ':''}${e(c.subject)}</div>
${c.refs?`<div class="refs">${e(c.refs)}</div>`:''}
<div class="meta"><code>${c.hash}</code> &middot; ${e(c.author)} &middot; ${c.date} &middot; +${c.added} &minus;${c.deleted} in ${c.files} files${c.areas.length?' &middot; '+e(c.areas.join(', ')):''} &middot; ${e(c.reason)}</div>
</div></div>`).join('')||'<div class="none">No commits match.</div>';
document.getElementById('shown').textContent=`Showing ${r.length} of the newest ${C.length} commits.`}
document.querySelectorAll('.bar button').forEach(b=>b.onclick=()=>{f=b.dataset.f;
document.querySelectorAll('.bar button').forEach(x=>x.setAttribute('aria-pressed',x===b));draw()});
document.getElementById('q').oninput=ev=>{q=ev.target.value.toLowerCase();draw()};draw();
</script></main></body></html>
"""


def render(root, commits, base, branches, worktrees):
    esc = html.escape
    majors = sum(c['level'] == 'major' for c in commits)
    conflicts = sum(w['conflicts'] for w in worktrees)
    changed = sum(w['changed'] for w in worktrees)
    tiles = [
        ('major', majors, 'major commits of the newest %d' % len(commits)),
        ('', len(commits) - majors, 'minor commits'),
        ('', len(branches), 'branches'),
        ('warn' if changed else '', changed, 'uncommitted files'),
        ('major' if conflicts else '', conflicts, 'unresolved conflicts')]
    rows = []
    for w in worktrees:
        state = 'clean'
        if w['merging']:
            state = '<span class="pill warn">MERGE IN PROGRESS</span> '
            state += '%d conflicts, %d files' % (w['conflicts'], w['changed'])
        elif w['changed']:
            state = '<span class="pill warn">UNCOMMITTED</span> %d file%s: <code>%s</code>' % (
                w['changed'], 's'[:w['changed'] != 1], esc(', '.join(w['files'][:6])))
        rows.append('<tr><td><code>%s</code></td><td><code>%s</code></td><td>%s</td></tr>' % (
            esc(w['path']), esc(w['branch']), state))
    branch_rows = []
    for b in branches:
        branch_rows.append(
            '<tr><td><code>%s</code></td><td class="num">%s</td><td class="num">%s</td>'
            '<td>%s</td><td>%s</td></tr>' % (
                esc(b['name']), '-' if b['ahead'] is None else b['ahead'],
                '-' if b['behind'] is None else b['behind'], esc(b['subject'][:90]), b['date']))
    data = json.dumps(commits, ensure_ascii=True).replace('</', '<\\/')
    page = PAGE
    for key, value in (
        ('__ROOT__', esc(str(root))), ('__NOW__', datetime.now().strftime('%Y-%m-%d %H:%M')),
        ('__BASE__', esc(base)),
        ('__TILES__', ''.join('<div class="tile %s"><b>%d</b><span>%s</span></div>' % t for t in tiles)),
        ('__WORKTREES__', ''.join(rows)), ('__BRANCHES__', ''.join(branch_rows)), ('__DATA__', data)):
        page = page.replace(key, value)
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('-C', dest='root', default='.', help='any path inside the repository')
    parser.add_argument('--limit', type=int, default=400, help='newest commits to include')
    parser.add_argument('--output', help='default: git-status.html in the main checkout')
    parser.add_argument('--install-hook', action='store_true',
                        help='refresh after commit, merge and checkout')
    args = parser.parse_args()
    common = Path(git(args.root, 'rev-parse', '--path-format=absolute', '--git-common-dir').strip())
    root = common.parent
    if args.install_hook:
        for name in ('post-commit', 'post-merge', 'post-checkout'):
            hook = common / 'hooks' / name
            if hook.exists() and 'git_status_html.py' not in hook.read_text(errors='replace'):
                sys.exit('%s already exists; add the refresh line by hand.' % hook)
            hook.parent.mkdir(exist_ok=True)
            hook.write_text(HOOK)
            hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    commits = read_commits(root, args.limit)
    base, branches = read_branches(root)
    output = Path(args.output) if args.output else root / 'git-status.html'
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(render(root, commits, base, branches, read_worktrees(root)), encoding='utf-8')
    temporary.replace(output)
    print('%s: %d commits, %d major' % (
        output, len(commits), sum(c['level'] == 'major' for c in commits)))


if __name__ == '__main__':
    main()
