"""PyCharm entry point. Production execution is intentionally unavailable."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

from offline_parallel import WORKERS, _write_new, schedule

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'results/iid-qualification-preflight-20261103/plan.json'
EXPECTED = '919d092ac675d9c854bffda08c7e397ee9f41415902471757194d14e916f66e2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight():
    digest = sha(PLAN)
    if digest != EXPECTED:
        raise RuntimeError('historical qualification plan hash mismatch')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    versions = {}
    for name in ('numpy', 'scipy', 'torch', 'gpytorch', 'botorch', 'scikit-learn'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                        cwd=ROOT, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    source_checks = {}
    for relative, expected in plan.get('source_sha256', {}).items():
        path = ROOT / relative
        actual = sha(path) if path.is_file() else None
        source_checks[relative] = {'expected': expected, 'actual': actual, 'match': actual == expected}
    with tempfile.TemporaryDirectory(prefix='ra_lsbo_write_check_') as folder:
        Path(folder, 'probe').write_text('ok', encoding='ascii')
    return {'python': sys.executable, 'python_version': sys.version, 'platform': platform.platform(),
            'versions': versions, 'git_commit': commit, 'tracked_git_changes': dirty,
            'plan_sha256': digest, 'historical_source_checks': source_checks,
            'plan_source_commit': plan.get('source_commit'),
            'seed_history_status': plan.get('seed_history_status'), 'logical_cpus': os.cpu_count(),
            'thread_limits': {n: os.environ[n] for n in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
            'execution_ready': False, 'truth_calls': 0, 'rcaide_calls': 0,
            'blockers': ['seed history not cleared', 'production runner and phase approval gate not implemented']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'benchmark_offline', 'plan', 'execute'), default='preflight')
    parser.add_argument('--workers', type=int, choices=WORKERS, default=1)
    parser.add_argument('--max-wall-seconds', type=float, default=86400)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='read-only historical checkpoint for synthetic IO/hash load')
    parser.add_argument('--rounds', type=int, default=1000)
    args = parser.parse_args(argv)
    info = preflight()
    print(json.dumps({'mode': args.mode, 'preflight': info,
                      'budgets': {'aircraft': 0, 'rcaide': 0}, 'options': vars(args)},
                     default=str, indent=2, ensure_ascii=False), flush=True)
    if args.mode == 'execute':
        raise RuntimeError('execution disabled: production accounting and phase receipt are not implemented')
    if args.mode == 'preflight':
        return
    if args.output is None:
        raise ValueError('a new --output path is required')
    if args.mode == 'plan':
        args.output.mkdir(parents=True, exist_ok=False)
        _write_new(args.output / 'plan.json', {'kind': 'zero_truth_scheduler_rehearsal',
            'source_commit': info['git_commit'], 'historical_plan_sha256': info['plan_sha256'],
            'workers': args.workers, 'max_wall_seconds': args.max_wall_seconds,
            'aircraft_calls': 0, 'rcaide_calls': 0, 'execution_authorized': False})
        print('Rehearsal plan SHA256:', sha(args.output / 'plan.json'))
        return
    if args.checkpoint is None or not args.checkpoint.is_file() or args.rounds < 1:
        raise ValueError('benchmark requires an existing read-only --checkpoint and positive --rounds')
    checkpoint = args.checkpoint.resolve()
    tasks = [{'id': 'task%03d' % i, 'seed': i, 'checkpoint': str(checkpoint), 'rounds': args.rounds}
             for i in range(32)]
    print(json.dumps(schedule(tasks, args.output, args.workers, args.max_wall_seconds), indent=2))


if __name__ == '__main__':
    main()
