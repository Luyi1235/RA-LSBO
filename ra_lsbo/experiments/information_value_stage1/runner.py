"""Real GP/VAE diagnostic campaign, then isolated paid candidate truth audit.

All online traces finish and are sealed BEFORE the first oracle audit call.
No audit labels are accepted by an optimizer. Heavy imports are lazy, so the
finite mechanism and integrity tools need only the Python standard library.
"""
from contextlib import ExitStack
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

from .audit import fingerprint, snapshot, summarize
from .io import CASES, append, manifest, tree_hashes, write_new
from .mechanism import run_mechanisms

SCHEMA = "ra_lsbo_information_value_stage1_v1"


def plan(profile):
    if profile not in ("smoke", "full"):
        raise ValueError("Use the frozen smoke/full protocol")
    full = profile == "full"
    seeds = list(range(998601, 998605)) if full else [998591]
    steps, checkpoints, size = (6, [1, 4], 512) if full else (2, [1], 160)
    traces = len(CASES) * len(seeds)
    online = traces * steps
    initial = traces * 12
    oracle = traces * len(checkpoints) * size
    return {"schema": SCHEMA, "profile": profile, "cases": list(CASES),
            "seeds": seeds, "initial": 12, "online_steps": steps,
            "snapshot_iterations_zero_based": checkpoints,
            "original_pool_size": 128, "audit_pool_size": size,
            "n_onpolicy_traces": traces, "initial_calls": initial,
            "online_calls": online, "postrun_audit_calls": oracle,
            "total_mathematical_call_ceiling": initial + online + oracle,
            "policy": "exact PR63 soft_gp, old 12-step internal horizon retained",
            "hypothesis": "Separate generation, support/forecast and decision opportunity loss",
            "new_algorithm_performance_test": False,
            "diagnostic_labels_never_fed_back": True,
            "extra_pool_draw_base": 862000000,
            "aircraft_calls": 0, "rcaide_calls": 0}


class NativeAdapter:
    """Unmodified PR63 software on the workstation's existing model environment."""
    def __init__(self):
        from ..risk_budget63_runner import _case, _candidates
        from ..risk_budget63_local_value_engine import LocalCredibilityValueBO
        from ..risk_budget63_protocol import INIT_DRAW_BASE
        from ..core.protocol import BASE
        from ..core.problems import AircraftCase
        from ...partial_bo.problems import candidate_pool, SimplifiedElectricAircraft
        if not Path(BASE).exists():
            raise FileNotFoundError("Frozen original LSBO base is missing: %s" % BASE)
        self.case_factory, self.pool_factory = _case, _candidates
        self.engine_class, self.base = LocalCredibilityValueBO, BASE
        self.draw, self.initial_draw = candidate_pool, INIT_DRAW_BASE
        self.forbidden_classes = (AircraftCase, SimplifiedElectricAircraft)

    def scope(self):
        stack = ExitStack()
        def forbidden(*args, **kwargs):
            raise RuntimeError("Stage 1 forbids every aircraft/RCAIDE evaluation")
        for cls in self.forbidden_classes:
            stack.enter_context(patch.object(cls, "evaluate", forbidden))
        return stack

    def problem(self, case):
        return self.case_factory(case)

    def initial(self, problem, case, seed):
        return self.draw(problem, self.initial_draw + 1000000 * CASES.index(case) + seed, 12)[0]

    def pool(self, problem, case, seed, iteration, history):
        return self.pool_factory(problem, case, seed, iteration, history)

    def engine(self, problem, seed, output):
        return self.engine_class(problem, self.base, seed, output,
                                 variant="soft_gp", initial_count=12, failure_cap=6)

    def extend(self, problem, case, seed, iteration, history, original, size):
        result = list(original)
        seen = {fingerprint(r.candidate) for r in history} | {fingerprint(c) for c in result}
        for attempt in range(4):
            draw = 862000000 + 10000000 * CASES.index(case) + 100 * seed + iteration + attempt * 1000000000
            extras, _ = self.draw(problem, draw, size + len(history) + 128)
            for candidate in extras:
                token = fingerprint(candidate)
                if token not in seen:
                    result.append(candidate)
                    seen.add(token)
                    if len(result) == size:
                        return result
        raise RuntimeError("Could not construct the registered distinct audit pool")


class ChargedEvaluator:
    """Count starts even when an exception prevents a trusted settled result."""
    def __init__(self, path):
        self.path = Path(path)
        self.started = self.settled = 0

    def call(self, problem, candidate, purpose):
        ticket = self.started
        append(self.path, {"ticket": ticket, "event": "started", "purpose": purpose,
                           "candidate": candidate, "candidate_id": fingerprint(candidate)})
        self.started += 1
        t0 = time.perf_counter()
        try:
            rec = problem.evaluate(candidate)
            if (rec.candidate != candidate or rec.observation.status != "completed" or
                    rec.physical_feasible is None or rec.diagnostics.get("numerical_failure")):
                raise ValueError("Unknown/infrastructure/numerical output is not physical infeasibility")
            if rec.usable_feasible and (rec.observation.objective.kind != "exact" or
                                       not math.isfinite(rec.observation.objective.value)):
                raise ValueError("A feasible design must have a finite exact objective")
        except BaseException as exc:
            append(self.path, {"ticket": ticket, "event": "error", "purpose": purpose,
                               "error_type": type(exc).__name__, "message": str(exc),
                               "elapsed_seconds": time.perf_counter() - t0})
            raise
        append(self.path, {"ticket": ticket, "event": "settled", "purpose": purpose,
                           "candidate_id": fingerprint(candidate), "record": rec.to_dict(),
                           "elapsed_seconds": time.perf_counter() - t0})
        self.settled += 1
        return rec


def _best(history):
    return min((r.observation.objective.value for r in history if r.usable_feasible), default=None)


def _legal(problem, pool, history, count):
    ids = [fingerprint(c) for c in pool]
    if (len(pool) != count or len(set(ids)) != count or
            set(ids) & {fingerprint(r.candidate) for r in history} or
            any(c != problem.canonical(c) for c in pool)):
        raise ValueError("Wrong pool size, noncanonical candidate or repeated evaluation")


def execute(output, spec, api, source):
    """Injection seam is for tests; production CLI always supplies NativeAdapter."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/"plan.json", spec)
    write_new(out/"source_receipt.json", source)
    write_new(out/"mechanism_report.json", run_mechanisms())
    trace_eval = ChargedEvaluator(out/"trace"/"evaluations.jsonl")
    audit_eval = ChargedEvaluator(out/"audit"/"evaluations.jsonl")
    snapshots, trials, diagnostics = [], [], []
    started = time.perf_counter()
    try:
        with api.scope():
            # Phase A: actual sequential GP/VAE trajectories, no oracle queries.
            for case in spec["cases"]:
                for seed in spec["seeds"]:
                    problem = api.problem(case)
                    folder = out/"trace"/case/str(seed)
                    folder.mkdir(parents=True, exist_ok=False)
                    raw = api.initial(problem, case, seed)
                    _legal(problem, raw, [], spec["initial"])
                    history = [trace_eval.call(problem, c, "initial") for c in raw]
                    write_new(folder/"initial.json", [r.to_dict() for r in history])
                    engine = api.engine(problem, seed, folder)
                    initial_best = _best(history)
                    failures, modes, curve = 0, [], [initial_best]
                    for iteration in range(spec["online_steps"]):
                        pool = api.pool(problem, case, seed, iteration, history)
                        _legal(problem, pool, history, spec["original_pool_size"])
                        before = fingerprint([r.to_dict() for r in history])
                        candidate, meta = engine.propose(history, pool, iteration)
                        if (candidate not in pool or
                                before != fingerprint([r.to_dict() for r in history])):
                            raise ValueError("Policy changed its history or left the frozen candidate pool")
                        append(folder/"proposals.jsonl", {"iteration": iteration,
                            "candidate": candidate, "history_sha256": before,
                            "original_pool_sha256": fingerprint(pool), "metadata": meta,
                            "future_result_unseen": True})
                        if iteration in spec["snapshot_iterations_zero_based"]:
                            expanded = api.extend(problem, case, seed, iteration, history,
                                                  pool, spec["audit_pool_size"])
                            _legal(problem, expanded, history, spec["audit_pool_size"])
                            snap = snapshot(case, seed, iteration, history, pool, expanded,
                                            candidate, meta, _best(history))
                            path = folder/("snapshot_%02d.json" % iteration)
                            write_new(path, snap)  # BEFORE the selected physical call
                            snapshots.append(path)
                        rec = trace_eval.call(problem, candidate, "online")
                        engine.accept(rec)
                        history.append(rec)
                        failures += int(not rec.usable_feasible)
                        modes.append(meta.get("mode", "unknown"))
                        curve.append(_best(history))
                        append(folder/"observations.jsonl", {"iteration": iteration,
                            "record": rec.to_dict(), "best_feasible": curve[-1]})
                    row = {"case": case, "seed": seed, "online_calls": spec["online_steps"],
                           "initial_best": initial_best, "final_best": curve[-1],
                           "best_feasible_by_call": curve, "online_nonfeasible": failures,
                           "modes": modes, "new_policy_test": False}
                    write_new(folder/"result.json", row)
                    trials.append(row)
                    del engine, history, problem
                    print("TRACE", case, seed, "complete", flush=True)
            expected_trace = spec["initial_calls"] + spec["online_calls"]
            if trace_eval.started != expected_trace or trace_eval.settled != expected_trace:
                raise RuntimeError("Actual trace cost differs from frozen plan")
            seal = tree_hashes(out/"trace")
            write_new(out/"trace_seal.json", {"files": seal, "all_traces_complete": True,
                "trace_evaluation_calls": trace_eval.settled, "audit_calls_so_far": audit_eval.started})
            # Phase B: fresh evaluator instances; no optimizer instance or accept call.
            for path in snapshots:
                if tree_hashes(out/"trace") != seal:
                    raise ValueError("A sealed online trace changed before audit")
                snap = json.loads(path.read_text(encoding="utf-8"))
                problem = api.problem(snap["case"])
                truth = []
                for candidate in snap["audit_pool"]:
                    rec = audit_eval.call(problem, candidate, "postrun_opportunity_audit")
                    truth.append({"id": fingerprint(candidate), "feasible": bool(rec.usable_feasible),
                                  "feasible_objective": rec.observation.objective.value
                                      if rec.usable_feasible else None})
                row = summarize(snap, truth)
                folder = out/"audit"/snap["case"]/str(snap["seed"])/str(snap["iteration"])
                write_new(folder/"truth.json", truth)
                write_new(folder/"summary.json", row)
                diagnostics.append(row)
                print("AUDIT", snap["case"], snap["seed"], snap["iteration"], "complete", flush=True)
            if tree_hashes(out/"trace") != seal:
                raise ValueError("Audit modified sealed optimization evidence")
            if audit_eval.started != spec["postrun_audit_calls"] or audit_eval.started != audit_eval.settled:
                raise RuntimeError("Audit cost does not reconcile")
        report = {"schema": SCHEMA, "status": "COMPLETE_STAGE1_DIAGNOSTIC_NOT_NEW_BO",
                  "plan": spec, "trials": trials, "snapshots": diagnostics,
                  "actual_initial_and_online_calls": trace_eval.settled,
                  "actual_postrun_audit_calls": audit_eval.settled,
                  "actual_total_mathematical_calls": trace_eval.settled + audit_eval.settled,
                  "aircraft_calls": 0, "rcaide_calls": 0,
                  "elapsed_seconds": time.perf_counter() - started,
                  "all_traces_sealed_before_any_audit": True,
                  "audit_labels_never_available_to_policy": True,
                  "limitations": ["Only the PR63 soft_gp policy generated these states.",
                    "Four seeds per task in full; diagnosis, not superiority inference.",
                    "Pool truth uses additional paid mathematical calls and cannot guide this run.",
                    "512-pool has no model forecasts; it is a generation opportunity probe only.",
                    "Finite mechanism is not integrated into the GP/VAE policy yet.",
                    "Old internal horizon 12 is preserved for this baseline-only diagnosis."]}
        if report["actual_total_mathematical_calls"] != spec["total_mathematical_call_ceiling"]:
            raise RuntimeError("Total evaluator bill mismatch")
        write_new(out/"report.json", report)
        manifest(out)
        return report
    except BaseException as exc:
        write_new(out/"failure.json", {"status": "FAILED_RETAIN_EVIDENCE_NO_RETRY_IN_PLACE",
            "error_type": type(exc).__name__, "message": str(exc),
            "trace_started": trace_eval.started, "trace_settled": trace_eval.settled,
            "audit_started": audit_eval.started, "audit_settled": audit_eval.settled})
        raise
