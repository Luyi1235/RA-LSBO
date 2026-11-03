"""User-initiated 12-worker seven-method TRUE reduced-aircraft experiment.

All 151,040 potential evaluations are from AircraftCase.evaluate, not fake data.
A new dataset is marked PROVISIONAL until the independent task and baseline
qualifications are actually established; this runner never invents that claim.

The operator requests an explicit IDE --mode execute. This file is not an
authorization system: it makes the evaluated dataset reproducible/auditable,
with no resume, implicit rerun, duplicated shared initial, or auto-retry.
"""
import concurrent.futures
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from . import real_call_ledger as ledger
from .nested_seed_plan import candidate_plan
from .real_candidate_engines import ALL_REQUESTED
from . import design
from ra_lsbo.budget import digest

TASKS=("aircraft_ordinary","aircraft_sparse")
SHARED_INITIAL=12
ONLINE=18
WORKERS=12
MAX_CALLS=151040
SCOPE="REAL_AIRCRAFT_USER_INITIATED"
SCIENCE_STATUS="PROVISIONAL_UNQUALIFIED_SEVEN_METHOD_REDUCED_AIRCRAFT"


def _json_new(path, value):
    from ra_lsbo.experiments.core.runner import plain
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as stream:
        json.dump(plain(value),stream,sort_keys=True,indent=2,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _json_append(path,value):
    from ra_lsbo.experiments.core.runner import plain
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf8") as stream:
        stream.write(json.dumps(plain(value),sort_keys=True,allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git_sha():
    try:
        return subprocess.check_output(
            ["git","rev-parse","HEAD"],cwd=Path(__file__).resolve().parents[1],
            text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        return "source_commit_unavailable"


def build_plan():
    """Freeze the registered full 1024/256 nested task/arm/seed roster.

    This intentionally records unresolved historical qualification as a
    scientific limitation, not a falsely successful qualification receipt.
    """
    prospective=candidate_plan()
    design.validate(prospective)
    if (prospective["budget"]["max_calls"]!=MAX_CALLS or
            tuple(prospective["main_arms"]+prospective["auxiliary_arms"])!=
            ALL_REQUESTED[:2]+(
                "full_onehot","reverse_representation",
                "interval_as_missing","random","cbob_adapted")):
        raise ValueError("The requested seven arms or exact budget changed")
    from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
    cases={}
    for case in TASKS:
        ids=list(prospective["seeds_by_case"][case])
        nested=set(prospective["auxiliary_by_case"][case])
        arms_by_seed={
            str(seed):list(prospective["main_arms"])+(
                list(prospective["auxiliary_arms"]) if seed in nested else [])
            for seed in ids
        }
        cases[case]={
            "seeds":ids,
            "contracts":{
                "20s":problem_for(case,20.).contract_id,
                "5s":problem_for(case,5.).contract_id},
            "arms_by_seed":arms_by_seed,
        }
    return {
        "schema":ledger.SCHEMA,"scope":ledger.SCOPE,
        "phase":"full_seven_arm_experiment",
        "execution_authorized":True,
        "authorization_basis":"explicit_user_ide_execute",
        "scientific_status":SCIENCE_STATUS,
        "source_commit":_git_sha(),
        "python":sys.version,"platform":platform.platform(),
        "online":ONLINE,"initial":SHARED_INITIAL,
        "verification":1,"max_calls":MAX_CALLS,
        "workers":WORKERS,"cases":cases,
        "nested_plan_sha256":digest(prospective),
        "method_qualification_independent":False,
        "qA_task_qualification_independent":False,
        "historical_seed_exclusion_complete":False,
        "CBOB_official_equivalent":False,
        "RCAIDE_calls_max":0,
    }


def _problem(case,step=20.):
    from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
    return problem_for(case,step)


def _ensure_trusted(record):
    if (record.observation.status in ("unknown","infrastructure_error")
            or record.diagnostics.get("numerical_failure")):
        raise RuntimeError("Unknown or numerical/infrastructure outcome; incomplete, no retry")


def _real_evaluate(db,problem,case,seed,role,arm,index,candidate):
    """Reserve first, call genuine physical evaluator once, settle then accept."""
    from ra_lsbo.partial_bo.records import DesignRecord
    ticket=ledger.start(db,case,seed,role,arm,index,candidate,problem)
    tic=time.perf_counter()
    try:
        actual=problem.evaluate(candidate)
        if not isinstance(actual,DesignRecord):
            raise TypeError("AircraftCase.evaluate did not return DesignRecord")
        record=replace(actual,diagnostics={
            **actual.diagnostics,
            "confirm12_real_call":{
                "call_id":ticket,
                "evaluator":"AircraftCase.evaluate",
                "elapsed_seconds":time.perf_counter()-tic,
                "worker_pid":os.getpid(),
                "case":case,"seed":seed,"role":role,
                "arm":arm,"index":index,
            },
        })
        state=ledger.settle(db,ticket,record)
        if state!="settled":
            raise RuntimeError("Unknown evaluator outcome; global ledger marked incomplete")
        _ensure_trusted(record)
        return record
    except BaseException:
        # It is unsafe to automatically retry a truth call after a process
        # failure; the charged STARTED ticket remains durable for inspection.
        ledger.stop(db)
        raise


def _case_seed_worker(job):
    """One spawn worker handles a full task-seed, globally shared initials."""
    case,seed,allowed,root,db,expected_plan_sha=job
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name]="1"
    import torch
    torch.set_num_threads(1)
    from ra_lsbo.experiments.iid_qualification import sample
    from ra_lsbo.partial_bo.problems import candidate_pool
    from .offline_candidate_trial import OfflineCandidateTrial

    root=Path(root)
    db=str(db)
    case_root=root/"trials"/case/str(seed)
    case_root.mkdir(parents=True,exist_ok=False)
    problem=_problem(case)
    if expected_plan_sha!=digest(json.loads((root/"plan.json").read_text("utf8"))):
        raise RuntimeError("Plan changed during execution")

    initial=[];consumed=0;results=[]
    try:
        # IID q_A: exactly 12 with-replacement analytic-screen designs;
        # every accepted index gets precisely ONE actual truth evaluation.
        draws=sample(seed,SHARED_INITIAL)
        initial_designs=[entry["candidate"] for entry in draws["records"]]
        if len(initial_designs)!=SHARED_INITIAL:
            raise AssertionError("q_A returned incomplete initial set")
        for i,candidate in enumerate(initial_designs):
            row=_real_evaluate(db,problem,case,seed,"initial","",i,candidate)
            initial.append(row);consumed+=1
            _json_append(case_root/"initial_observations.jsonl",row.to_dict())
        _json_new(case_root/"initial_draws.json",{
            "seed":seed,"sampler":"iid_qualification.sample",
            "candidate_sha256":[digest(x) for x in initial_designs],
            "total_attempts":draws["attempts"],
            "duplicate_charged_as_distinct_index":True,
            "physical_model":"AircraftCase.evaluate",
        })

        for method in allowed:
            method_dir=case_root/method
            # Must not pre-create method_dir; the frozen H1 adapter checks it.
            trial=OfflineCandidateTrial(
                case,seed,method,method_dir,initial)
            method_dir.mkdir(parents=True,exist_ok=False)
            begin=time.perf_counter()
            for i in range(ONLINE):
                if expected_plan_sha!=digest(json.loads((root/"plan.json").read_text("utf8"))):
                    raise RuntimeError("Plan/source freeze check failed")
                history=trial._history
                pool,rejected=candidate_pool(
                    problem,440000000+seed*10000+TASKS.index(case)*100000000+i,
                    128,{r.observation.design_id for r in history})
                prefix=trial.history_digest
                candidate,metadata=trial.propose(pool)
                _json_append(method_dir/"proposals.jsonl",{
                    "iteration":i,"candidate":candidate,
                    "candidate_sha256":digest(candidate),
                    "prefix_digest":prefix,
                    "metadata":metadata,
                    "geometry_rejections":rejected,
                })
                row=_real_evaluate(
                    db,problem,case,seed,"online",method,i,candidate)
                _json_append(method_dir/"observations.jsonl",row.to_dict())
                trial.accept(row)
                consumed+=1
            # H1 shared nominal history includes its 12 initials and 18 online.
            if len(trial._history)!=SHARED_INITIAL+ONLINE:
                raise RuntimeError("Trial stopped before the complete online horizon")
            available=[r for r in trial._history if r.usable_feasible]
            best=min(available,key=lambda x:x.observation.objective.value,
                     default=None)
            verification=None
            if best is not None:
                fine=_problem(case,5.)
                verification=_real_evaluate(db,fine,case,seed,"verification",
                                             method,0,best.candidate)
                consumed+=1
                _json_new(method_dir/"verification.json",verification.to_dict())
            summary={
                "case":case,"seed":seed,"method":method,
                "shared_initial_sha256":digest([r.to_dict() for r in initial]),
                "online_calls":ONLINE,
                "nominal_usable_feasible":bool(best),
                "nominal_best_energy_Wh":(
                    best.observation.objective.value if best else None),
                "same_design_5sec_verification":verification is not None,
                "verification_usable_feasible":(
                    verification.usable_feasible if verification else None),
                "wall_seconds":time.perf_counter()-begin,
                "scientific_status":SCIENCE_STATUS,
            }
            _json_new(method_dir/"result.json",summary)
            results.append(summary)
        _json_new(case_root/"complete.json",{
            "case":case,"seed":seed,"methods":allowed,
            "charged_true_calls":consumed,"experiment":SCIENCE_STATUS,
            "status":"COMPLETE_CASE_SEED",
        })
    except BaseException:
        ledger.stop(db)
        with (case_root/"error.txt").open("x",encoding="utf8") as f:
            f.write(traceback.format_exc())
        raise
    return {"case":case,"seed":seed,"methods":len(results),
            "true_calls":consumed,
            "results":results}


def _audit_all_groups(db,plan):
    import sqlite3
    errors=[]
    with sqlite3.connect(str(db)) as conn:
        rows=conn.execute(
            "SELECT case_name,seed,role,arm,state,count(*) FROM calls "
            "GROUP BY case_name,seed,role,arm,state").fetchall()
    groups={(case,seed,role,arm,state):n
            for case,seed,role,arm,state,n in rows}
    for case,spec in plan["cases"].items():
        for seed in spec["seeds"]:
            if groups.get((case,seed,"initial","","settled"),0)!=SHARED_INITIAL:
                errors.append(f"{case}/{seed}: shared initials incomplete")
            for arm in spec["arms_by_seed"][str(seed)]:
                if groups.get((case,seed,"online",arm,"settled"),0)!=ONLINE:
                    errors.append(f"{case}/{seed}/{arm}: online incomplete")
                if groups.get((case,seed,"verification",arm,"settled"),0)>1:
                    errors.append(f"{case}/{seed}/{arm}: duplicate verification")
    return errors


def run(root,*,workers=WORKERS):
    """Full campaign; only reached by the user choosing the IDE execute mode."""
    if type(workers) is not int or workers!=WORKERS:
        raise ValueError("Full campaign fixes exactly 12 Windows spawn workers")
    root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    plan=build_plan()
    _json_new(root/"plan.json",plan)
    _json_new(root/"launch.json",{
        "source_sha":plan["source_commit"],"workers":WORKERS,
        "expected_max_calls":MAX_CALLS,
        "operator_action":"explicit --mode execute",
        "scientific_status":SCIENCE_STATUS,
        "qA_unqualified":True,"historical_seed_exclusion_unfinished":True,
    })
    db=root/"real_calls.sqlite"
    ledger.create(db,plan)
    jobs=[
        (case,seed,spec["arms_by_seed"][str(seed)],
         str(root),str(db),digest(plan))
        for case,spec in plan["cases"].items()
        for seed in spec["seeds"]
    ]
    done=[];failures=[]
    # Bounded in-flight futures: stop dispatch when any truth call fails;
    # do not keep submitting thousands of new calls after a fatal worker error.
    remaining=iter(jobs)
    pending={}
    exhausted=False
    try:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=WORKERS,
                mp_context=multiprocessing.get_context("spawn")) as pool:
            while pending or not exhausted:
                while not exhausted and not failures and len(pending)<WORKERS:
                    try:
                        job=next(remaining)
                    except StopIteration:
                        exhausted=True
                        break
                    pending[pool.submit(_case_seed_worker,job)]=(job[0],job[1])
                if not pending:
                    break
                completed,_=concurrent.futures.wait(
                    pending,return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in completed:
                    case,seed=pending.pop(fut)
                    try:
                        result=fut.result()
                        done.append(result)
                        _json_append(root/"progress.jsonl",{
                            "case":case,"seed":seed,
                            "methods":result["methods"],
                            "charged_true_calls":result["true_calls"],
                            "completed_groups":len(done),
                            "total_groups":len(jobs),
                        })
                        print(
                            "TRUE aircraft COMPLETE",case,seed,
                            f"groups={len(done)}/{len(jobs)}",
                            f"calls_in_group={result['true_calls']}",flush=True)
                    except BaseException:
                        ledger.stop(db)
                        failures.append({
                            "case":case,"seed":seed,
                            "exception":traceback.format_exc()})
                if failures:
                    for fut in pending:
                        fut.cancel()
                    break
    except BaseException:
        ledger.stop(db)
        failures.append({"scheduler":traceback.format_exc()})
        raise
    finally:
        # Never mark an incomplete history as a successful campaign.
        if failures:
            ledger.stop(db)
        status={
            "status":"INCOMPLETE" if failures else "FINISHED_WORKERS",
            "completed_groups":len(done),
            "total_groups":len(jobs),
            "elapsed_seconds":time.perf_counter()-started,
            "failures":failures,
        }
        _json_new(root/"execution_status.json",status)
    report=ledger.audit(db,plan)
    group_errors=_audit_all_groups(db,plan)
    report["group_errors"]=group_errors
    report["groups_complete"]=len(done)==len(jobs)
    report["scientific_status"]=SCIENCE_STATUS
    report["wall_seconds"]=time.perf_counter()-started
    _json_new(root/"audit.json",report)
    if (failures or group_errors or len(done)!=len(jobs) or
            not report["ledger_contract_passed"]):
        raise RuntimeError("True experiment INCOMPLETE: inspect audit and source journals")
    _json_new(root/"completed.json",{
        "status":"ALL_REAL_EVALUATIONS_COMPLETED",
        "charged_calls":report["calls"],
        "upper_bound":MAX_CALLS,
        "no_RCAIDE_calls":True,
        "method_qualification_proven":False,
        "scientific_status":SCIENCE_STATUS,
    })
    return {"status":"COMPLETE_PROVISIONAL", "charged_calls":report["calls"],
            "maximum_calls":MAX_CALLS,"groups":len(done),
            "wall_seconds":report["wall_seconds"],
            "scientific_qualification":False}
