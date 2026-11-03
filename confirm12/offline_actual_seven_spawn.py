"""One-step seven-BO-engine Windows spawn rehearsal with structured fake records.

This performs actual GP/VAE/SLSQP algorithm computations but **no** physical
evaluations. The candidate observations are deliberately fabricated in memory
and stored only in a write-once OFFLINE test directory. It is not science data,
not a baseline-qualification campaign and not an approval to run 151,040 calls.
"""
import concurrent.futures
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

from . import structured_record_ledger as ledger
from .real_candidate_engines import ALL_REQUESTED

SCHEMA="offline_actual_seven_policy_one_step_v1"
CASES=("aircraft_ordinary","aircraft_sparse")
TEST_SEEDS={"aircraft_ordinary":918001,"aircraft_sparse":918002}


def _forbidden(*args,**kwargs):
    raise RuntimeError("OFFLINE only: aircraft evaluator must never run")

def _process_peak_rss_bytes():
    """Best available per-process peak, never a system-simultaneous memory claim."""
    import sys
    if sys.platform == "win32":
        try:
            import ctypes
            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb",ctypes.c_ulong), ("PageFaultCount",ctypes.c_ulong),
                    ("PeakWorkingSetSize",ctypes.c_size_t),
                    ("WorkingSetSize",ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage",ctypes.c_size_t),
                    ("QuotaPagedPoolUsage",ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage",ctypes.c_size_t),
                    ("PagefileUsage",ctypes.c_size_t),
                    ("PeakPagefileUsage",ctypes.c_size_t),
                ]
            counter=Counters()
            counter.cb=ctypes.sizeof(Counters)
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counter), counter.cb
            ):
                return int(counter.PeakWorkingSetSize)
        except (OSError,AttributeError):
            return None
    else:
        try:
            import resource
            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux ru_maxrss reports KiB; macOS reports bytes.
            return int(kb*(1 if sys.platform=="darwin" else 1024))
        except (OSError,ImportError):
            pass
    return None



def _patch_physics():
    # Called independently inside each spawned worker, before construction.
    from ra_lsbo.experiments.core.problems import AircraftCase
    from ra_lsbo.partial_bo.problems import SimplifiedElectricAircraft
    AircraftCase.evaluate = _forbidden
    SimplifiedElectricAircraft.evaluate = _forbidden


def _problem(case,step=20.):
    from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
    return problem_for(case,step)


def _synthetic_record(problem,candidate):
    # This is a *synthetic* training record, NOT a physics-result surrogate.
    screen=dict(problem.precheck(candidate))
    screen.update({name:-.15 for name in problem.required_constraints
                   if name not in problem.analytical_constraints})
    x=candidate["continuous"]
    objective=30000.+(float(x["wing_area"])-14.)**2*250.
    return problem.record(
        candidate, objective, screen,
        {"offline_synthetic_training_only":True,
         "never_from_aircraft_evaluate":True})


def _make_plan():
    cases={}
    for case in CASES:
        seed=TEST_SEEDS[case]
        cases[case]={
            "seeds":[seed],
            "contracts":{"20s":_problem(case).contract_id,
                         "5s":_problem(case,5.).contract_id},
            "arms_by_seed":{str(seed):list(ALL_REQUESTED)},
        }
    return {
        "schema":ledger.SCHEMA,
        "scope":ledger.SCOPE,
        "execution_authorized":False,
        "phase":"confirmation",
        "initial":12,"online":18,"verification":1,
        "max_calls":2*(12+len(ALL_REQUESTED)),
        "cases":cases,
        "note":SCHEMA,
    }


def _worker(job):
    case,seed,method,number,output,db=job
    # One BLAS thread in each Windows spawned process, before model imports.
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[k]="1"
    _patch_physics()
    from .offline_ledger_trial import OfflineLedgerTrial
    from .real_candidate_engines import create_engine
    from ra_lsbo.partial_bo.problems import candidate_pool

    started=time.perf_counter()
    output=Path(output)
    problem=_problem(case)
    trial=OfflineLedgerTrial(
        db,_make_plan(),case,seed,method,output/case/str(seed)/method,
        engine_builder=create_engine)
    assert len(trial.policy._history)==12
    before=trial.policy.history_digest
    pool,rejected=candidate_pool(
        problem,9300000+seed*11+number,128,
        {row.observation.design_id for row in trial.policy._history})
    candidate,meta=trial.propose(pool)
    # Never evaluate() the selected design; feed synthetic shape explicitly.
    trial.accept(_synthetic_record(problem,candidate))
    result={
        "method":method,"case":case,"seed":seed,
        "pid":os.getpid(),
        "source_kind":"OFFLINE_SYNTHETIC_NO_PHYSICS",
        "candidate_sha256":hashlib.sha256(
            json.dumps(candidate,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "initial_history_digest":before,
        "post_history_digest":trial.policy.history_digest,
        "algorithm_origin":asdict(trial.policy.probe)["origin_h1_method"],
        "metadata_method":meta.get("method"),
        "is_actual_candidate_policy":True,
        "flight_simulations":0,
        "run_seconds":time.perf_counter()-started,
        "worker_peak_rss_bytes":_process_peak_rss_bytes(),
        "pool_rejected_geometry":rejected,
    }
    path=output/case/str(seed)/method
    path.mkdir(parents=True,exist_ok=True)
    with (path/"offline_result.json").open("x",encoding="utf8") as f:
        json.dump(result,f,indent=2,allow_nan=False)
    return result


def run(output,workers=12):
    """Run 14 actual BO one-step policies under Windows spawn; no truth labels."""
    if type(workers) is not int or not 1 <= workers <= 12:
        raise ValueError("Workers must be 1..12, Windows spawn")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    _patch_physics()
    from ra_lsbo.partial_bo.problems import candidate_pool

    plan=_make_plan()
    with (output/"offline_plan.json").open("x",encoding="utf8") as f:
        json.dump(plan,f,indent=2,sort_keys=True,allow_nan=False)
    database=output/"offline_evidence.sqlite"
    ledger.create(database,plan)
    for case in CASES:
        problem=_problem(case)
        seed=TEST_SEEDS[case]
        shared,_=candidate_pool(problem,9300000+seed,12)
        for i,candidate in enumerate(shared):
            ticket=ledger.start(
                database,case,seed,"initial","",i,candidate,problem)
            ledger.settle(database,ticket,_synthetic_record(problem,candidate))

    started=time.perf_counter()
    jobs=[(case,TEST_SEEDS[case],method,i,str(output),str(database))
          for case in CASES for i,method in enumerate(ALL_REQUESTED)]
    results=[]
    errors=[]
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn")) as pool:
        futures={pool.submit(_worker,job):(job[0],job[2]) for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                result=future.result()
                results.append(result)
                print("OFFLINE policy complete:",result["case"],
                      result["method"],flush=True)
            except BaseException as error:
                ledger.stop(database)
                errors.append({
                    "case":futures[future][0],"method":futures[future][1],
                    "error":type(error).__name__+": "+str(error)})
                print("OFFLINE FAILURE",errors[-1],flush=True)
    audit=ledger.audit(database,plan)
    report={
        "schema":SCHEMA,
        "scope":ledger.SCOPE,
        "seven_distinct_policy_names":list(ALL_REQUESTED),
        "workers":workers,
        "total_wall_seconds":time.perf_counter()-started,
        "completed_policy_steps":len(results),
        "expected_policy_steps":len(jobs),
        "total_fake_records_charged":audit["calls"],
        "worker_processes_seen":len({row["pid"] for row in results}),
        "max_reported_single_worker_peak_rss_bytes":max(
            (row["worker_peak_rss_bytes"] or 0) for row in results
        ) if results else 0,
        "peak_rss_not_simultaneous_system_sample": True,
        "atomic_ledger_contract_passed":audit["ledger_contract_passed"],
        "errors":errors,
        "audit_errors":audit["errors"],
        "aircraft_calls":0,"rcaide_calls":0,
        "scientific_method_qualification":False,
        "production_approval":False,
        "results":sorted(results,key=lambda x:(x["case"],x["method"])),
    }
    with (output/"offline_summary.json").open("x",encoding="utf8") as f:
        json.dump(report,f,indent=2,allow_nan=False)
    if errors or not audit["ledger_contract_passed"] or len(results)!=len(jobs):
        raise RuntimeError("Offline policy workload incomplete; inspect preserved evidence")
    return report
