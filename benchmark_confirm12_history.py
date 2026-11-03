"""Zero-truth P5 replay benchmark from a separately verified historical worktree."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['MPLBACKEND'] = 'Agg'

import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile


def peak_rss():
    if sys.platform != 'win32':
        return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    get_info.restype = wintypes.BOOL
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if not get_info(handle, ctypes.byref(counters), counters.cb):
        raise OSError('GetProcessMemoryInfo failed')
    return counters.PeakWorkingSetSize


def worker(job):
    replay_root, archive, case, seed = job
    sys.path.insert(0, replay_root)
    from tools import h2_p5_full_path as p5
    with zipfile.ZipFile(archive) as z:
        plan = p5.p4.read_json(z, 'plan.json')
        start = time.perf_counter()
        with patch.object(p5.AircraftCase, 'evaluate', p5.forbidden), \
             patch.object(p5.CorpusRepresentation, 'fit', p5.forbidden):
            row = p5.execute_pair(z, plan, case, seed)
        return {'case': case, 'seed': seed, 'elapsed_seconds': time.perf_counter() - start,
                'gp_fit_seconds': row.get('gp_fit_seconds'), 'gp_candidate_fits': row.get('gp_candidate_fits'),
                'category': row.get('category'), 'reason': row.get('reason'), 'peak_rss_bytes': peak_rss(),
                'aircraft_calls': 0, 'rcaide_calls': 0, 'vae_training': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay-root', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, archive, output = map(lambda x: str(Path(x).resolve()),
                                (args.replay_root, args.archive, args.output))
    sys.path.insert(0, root)
    from tools import h2_p5_full_path as p5
    z, plan = p5.p4.validate_archive(archive)
    sources = p5.source_check(z, plan)
    z.close()
    before = hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for workers in (4, 8, 12):
        jobs = [(root, archive, 'aircraft_ordinary', 12078) if i % 2 == 0
                else (root, archive, 'aircraft_sparse', 12075) for i in range(12)]
        start = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            rows = list(pool.map(worker, jobs))
        elapsed = time.perf_counter() - start
        if any(r['category'] != 'identified' for r in rows):
            raise RuntimeError('historical full-path replay did not reproduce; stop')
        results.append({'workers': workers, 'elapsed_seconds': elapsed, 'tasks': len(rows),
                        'tasks_per_second': len(rows) / elapsed, 'rows': rows,
                        'sum_per_task_peak_rss_bytes_upper_bound': sum(r['peak_rss_bytes'] or 0 for r in rows)})
        (output / ('workers_%s.json' % workers)).write_text(json.dumps(results[-1], indent=2), encoding='utf-8')
        print(workers, elapsed, flush=True)
    after = hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError('input archive changed')
    (output / 'summary.json').write_text(json.dumps({'archive_sha256': before,
        'source_count': len(sources), 'results': results, 'truth_calls': 0}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
