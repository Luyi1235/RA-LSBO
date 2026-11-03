"""PyCharm launcher. Only zero-aircraft simulation is implemented."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import argparse
import ctypes
import json
import multiprocessing
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from confirm12 import simulation, ledger_simulation, call_ledger, adapters

ROOT = Path(__file__).resolve().parent
MAIN_MAX = 2 * 1024 * (12 + 2 * (18 + 1))
AUX_MAX = 2 * 256 * 5 * (18 + 1)


def hardware():
    info = {'physical_memory_bytes': None, 'available_memory_bytes': None,
            'torch': None, 'cuda_available': None, 'gpu': None}
    if sys.platform == 'win32':
        from ctypes import wintypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD),
                        ('total_physical', ctypes.c_ulonglong), ('available_physical', ctypes.c_ulonglong),
                        ('total_pagefile', ctypes.c_ulonglong), ('available_pagefile', ctypes.c_ulonglong),
                        ('total_virtual', ctypes.c_ulonglong), ('available_virtual', ctypes.c_ulonglong),
                        ('available_extended', ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            info['physical_memory_bytes'] = status.total_physical
            info['available_memory_bytes'] = status.available_physical
    try:
        import torch
        torch.set_num_threads(1)
        info['torch'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
    except ImportError:
        pass
    tool = shutil.which('nvidia-smi')
    if tool:
        try:
            info['gpu'] = subprocess.check_output([tool, '--query-gpu=name,memory.total,memory.free',
                         '--format=csv,noheader'], text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def preflight():
    try:
        from confirm12 import qualification
        review = qualification.inspect(ROOT / 'results/iid-qualification-preflight-20261103/plan.json')
    except Exception as error:
        review = {'qualification_ready': False, 'error': type(error).__name__ + ': ' + str(error)}
    try:
        from confirm12.aircraft_contract_preflight import inspect as inspect_aircraft_contracts
        aircraft_contracts = inspect_aircraft_contracts()
    except Exception as error:
        aircraft_contracts = {'task_qualified': False, 'production_authorized': False,
                              'error': type(error).__name__ + ': ' + str(error)}
    return {'mode': 'preflight', 'platform': platform.platform(), 'python': sys.executable,
            'logical_cpu': os.cpu_count(), 'disk_free_bytes': shutil.disk_usage(ROOT).free,
            'hardware': hardware(),
            'workers_default': 12, 'main_maximum': MAIN_MAX, 'aux_maximum': AUX_MAX,
            'total_maximum': MAIN_MAX + AUX_MAX, 'new_aircraft_calls': 0, 'new_rcaide_calls': 0,
            'confirmation_wall_time_limit_seconds': None,
            'algorithm_qualification': adapters.QUALIFICATION,
            'qualification_review': review,
            'aircraft_contract_preflight': aircraft_contracts,
            'execution_ready': False, 'blockers': [
                'no frozen independent 1024-seed plan or 256-seed subset',
                'historical seed clearance and task qualification incomplete',
                'seven actual algorithm adapters and CBOB qualification incomplete',
                'synthetic ledger has not been certified for real truth calls',
                'seven-arm 12-worker memory, isolation and long-run stability not qualified; no 24-hour requirement',
                'phase-specific signed-off execution receipt not implemented',
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'qualification', 'simulate', 'status', 'proposed_plan', 'legacy_probe', 'offline_actual_seven', 'execute'), default='preflight')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--max-wall-seconds', type=float, default=86400,
                        help='deadline for the optional simulation mode only; not a confirmation limit')
    args = parser.parse_args()
    print(json.dumps(preflight(), indent=2), flush=True)
    if args.mode == 'proposed_plan':
        from confirm12.nested_seed_plan import write_candidate_plan
        if args.output is None:
            parser.error('proposed_plan requires a new --output directory')
        args.output.mkdir(parents=True, exist_ok=False)
        name = args.output / 'proposed_nested_confirmation_plan.json'
        plan_hash = write_candidate_plan(name)
        print(json.dumps({'plan_sha256': plan_hash, 'path': str(name),
                          'execution_authorized': False,
                          'note': 'Candidate roster requires historical seed, methods, task and budget approval'}, indent=2))
        return
    if args.mode == 'legacy_probe':
        from dataclasses import asdict
        from unittest.mock import patch
        from ra_lsbo.experiments.core.problems import AircraftCase
        from ra_lsbo.structure_corpus import CorpusRepresentation
        from confirm12.real_legacy_adapters import LEGACY_METHODS, build_historical_engine
        if args.output is None:
            parser.error('legacy_probe requires a new --output directory')
        args.output.mkdir(parents=True, exist_ok=False)
        def forbidden(*unused_args, **unused_kwargs):
            raise RuntimeError('No model evaluation or encoder training allowed in legacy_probe')
        receipts = []
        with patch.object(AircraftCase, 'evaluate', forbidden), patch.object(
                CorpusRepresentation, 'fit', forbidden):
            for case, seed in (('aircraft_ordinary', 390001),
                               ('aircraft_sparse', 390002)):
                for method in sorted(LEGACY_METHODS):
                    _, _, receipt = build_historical_engine(
                        case, seed, method, args.output / case / method)
                    receipts.append(asdict(receipt))
        result = {'status': 'LEGACY_INTERFACE_ONLY', 'new_truth_calls': 0,
                  'execution_authorized': False, 'legacy_engines': receipts}
        with (args.output / 'legacy_adapter_probe.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result, indent=2))
        return
    if args.mode == 'offline_actual_seven':
        # The seven *real* BO codes run against fabricated evidence only.
        # Deliberately separate from any physical model execution route.
        if args.output is None:
            parser.error('offline_actual_seven requires a new --output directory')
        from confirm12.offline_actual_seven_spawn import run as offline_workload
        output = offline_workload(args.output, workers=args.workers)
        print(json.dumps(output, indent=2, default=str, ensure_ascii=False))
        return
    if args.mode == 'execute':
        # This is the *actual* user-initiated reduced-aircraft campaign.
        # No synthetic evaluator is ever used on this route.
        from datetime import datetime, timezone
        from confirm12.real_aircraft_campaign import run as true_campaign
        if args.output is None:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            args.output = ROOT / 'results' / ('true_aircraft_12workers_' + stamp)
        print('STARTING GENUINE AircraftCase.evaluate(), 12 workers,'
              ' max 151040 calls; scientific status PROVISIONAL',
              flush=True)
        result = true_campaign(args.output, workers=args.workers)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.mode == 'qualification':
        from confirm12 import qualification
        print(json.dumps(qualification.inspect(ROOT / 'results/iid-qualification-preflight-20261103/plan.json'), indent=2))
    if args.mode == 'simulate':
        if args.output is None:
            parser.error('simulation requires a new --output directory')
        print(json.dumps(ledger_simulation.run(args.output, workers=args.workers,
                                                max_wall_seconds=args.max_wall_seconds), indent=2))
    if args.mode == 'status':
        if args.output is None:
            parser.error('status requires --output')
        plan = json.loads((args.output / 'plan.json').read_text(encoding='utf-8'))
        if (args.output / 'calls.sqlite').exists():
            result = ledger_simulation.audit(args.output)
        else:
            expected = len(plan['cases']) * len(plan['seeds']) * (12 + len(plan['arms']) * 19)
            result = simulation.audit_simulation(args.output, plan, expected)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
