"""Actual paired histories, started/settled ledger, write-once evidence and audit."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

from ..information_value_stage1.io import write_new, append, manifest, verify as verify_manifest, source_receipt
from ..information_value_stage1.audit import fingerprint
from ..information_value_stage1.runner import ChargedEvaluator, _legal, _best
from .protocol import PARENT, ARMS, plan, summarize
from ...budget import digest


def locked_source(root, expected_sha):
    if not isinstance(expected_sha, str) or len(expected_sha) != 40 or any(c not in '0123456789abcdef' for c in expected_sha):
        raise ValueError("An exact expected 40-digit code SHA is mandatory")
    receipt = source_receipt(root)
    if receipt["head_sha"] != expected_sha:
        raise ValueError("Worktree is not at the requested fixed SHA")
    def entries(ref):
        raw = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-rz", ref])
        return dict(row.decode("utf-8").split("\t", 1)[::-1] for row in raw.split(b"\0") if row)
    old, new = entries(PARENT), entries(expected_sha)
    changed = [p for p, v in old.items() if new.get(p) != v]
    if changed:
        raise RuntimeError("PR67 frozen files changed: " + repr(changed[:5]))
    receipt.update(stage3_parent=PARENT, all_pr67_blobs_unchanged=True)
    return receipt


class NativeAdapter:
    def __init__(self):
        from ..information_value_stage1.runner import NativeAdapter as ParentAdapter
        from ..risk_budget63_local_value_engine import LocalCredibilityValueBO
        from .native import FantasyValueBO
        self.parent = ParentAdapter()
        self.legacy_class, self.new_class = LocalCredibilityValueBO, FantasyValueBO

    def scope(self):
        return self.parent.scope()

    def problem(self, case):
        return self.parent.problem(case)

    def initial(self, problem, case, seed):
        return self.parent.initial(problem, case, seed)

    def pool(self, problem, case, seed, iteration, history):
        return self.parent.pool(problem, case, seed, iteration, history)

    def engine(self, problem, seed, folder, arm, steps):
        if arm == "soft_gp_eligible":
            return self.legacy_class(problem, self.parent.base, seed, folder,
                                     variant="soft_gp", initial_count=12, failure_cap=6)
        return self.new_class(problem, self.parent.base, seed, folder, arm=arm, horizon=steps)


def _decision(meta, candidate):
    iv = meta.get("information_value_decision")
    old = meta.get("credibility_decision")
    receipt = iv if iv is not None else old
    if receipt is not None:
        recorded = receipt.get("selected_candidate_hash", receipt.get("selected_candidate_sha256"))
        if recorded != digest(candidate):
            raise ValueError("Prediction receipt is not for the dispatched design")
    stats = iv.get("fantasy_fits", {}) if iv else {}
    return {"mode": meta.get("mode"), "forecast_available": receipt is not None,
            "predicted_failure": receipt.get("selected_predicted_failure") if receipt else None,
            "fantasy_head_fits_started": stats.get("head_fits_started", 0),
            "fantasy_head_fits_completed": stats.get("head_fits_completed", 0),
            "fantasy_seconds": stats.get("head_fit_seconds", 0.),
            "fantasy_fallback": bool(iv and iv.get("fallback")),
            "depth": iv.get("depth") if iv else None}


def execute(output, spec, api_factory, source):
    """Test injection seam only; CLI always supplies the real NativeAdapter."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/"plan.json", spec)
    write_new(out/"source_receipt.json", source)
    ledger = ChargedEvaluator(out/"evaluations.jsonl")
    trials = []
    start = time.perf_counter()
    try:
        api = api_factory()
        with api.scope():
            for case in spec["cases"]:
                for seed in spec["seeds"]:
                    problem = api.problem(case)
                    initial_designs = api.initial(problem, case, seed)
                    _legal(problem, initial_designs, [], spec["initial"])
                    initial, tickets = [], []
                    for j, c in enumerate(initial_designs):
                        tickets.append(ledger.started)
                        initial.append(ledger.call(problem, c, {"phase": "initial", "case": case, "seed": seed, "index": j}))
                    initial_hash = fingerprint([r.to_dict() for r in initial])
                    write_new(out/"shared_initial"/case/(str(seed)+".json"),
                              {"records": [r.to_dict() for r in initial], "tickets": tickets, "hash": initial_hash})
                    for arm in spec["arms"]:
                        folder = out/"trials"/case/str(seed)/arm
                        folder.mkdir(parents=True, exist_ok=False)
                        engine = api.engine(problem, seed, folder, arm, spec["steps"])
                        history, path, curve, decisions = list(initial), [], [_best(initial)], []
                        proposal_seconds = 0.
                        for iteration in range(spec["steps"]):
                            pool = api.pool(problem, case, seed, iteration, history)
                            _legal(problem, pool, history, spec["pool"])
                            prefix = fingerprint([r.to_dict() for r in history])
                            t0 = time.perf_counter()
                            c, meta = engine.propose(history, pool, iteration)
                            elapsed = time.perf_counter()-t0
                            proposal_seconds += elapsed
                            if c not in pool or prefix != fingerprint([r.to_dict() for r in history]):
                                raise ValueError("Proposal left the pool or mutated settled history")
                            decision = _decision(meta, c)
                            append(folder/"proposals.jsonl", {"iteration": iteration, "candidate": c,
                                   "candidate_hash": fingerprint(c), "history_hash": prefix,
                                   "pool_hash": fingerprint(pool), "metadata": meta,
                                   "proposal_seconds": elapsed, "outcome_unseen": True})
                            ticket = ledger.started
                            rec = ledger.call(problem, c, {"phase": "online", "case": case, "seed": seed, "arm": arm, "iteration": iteration})
                            append(folder/"observations.jsonl", {"iteration": iteration, "ticket": ticket,
                                   "candidate_hash": fingerprint(c), "record": rec.to_dict()})
                            engine.accept(rec)
                            history.append(rec)
                            decision.update(iteration=iteration, candidate_hash=fingerprint(c), nonfeasible=int(not rec.usable_feasible))
                            decisions.append(decision)
                            path.append(fingerprint(c))
                            curve.append(_best(history))
                        trial = {"case": case, "seed": seed, "arm": arm, "online_calls": spec["steps"],
                                 "initial_hash": initial_hash, "initial_best": curve[0], "final_best": curve[-1],
                                 "curve": curve, "path_hash": fingerprint(path), "decisions": decisions,
                                 "nonfeasible": sum(r["nonfeasible"] for r in decisions),
                                 "proposal_seconds": proposal_seconds,
                                 "fantasy_head_fits_started": sum(r["fantasy_head_fits_started"] for r in decisions),
                                 "fantasy_head_fits_completed": sum(r["fantasy_head_fits_completed"] for r in decisions),
                                 "fantasy_fallbacks": sum(r["fantasy_fallback"] for r in decisions),
                                 "model_free_calls": sum(not r["forecast_available"] for r in decisions)}
                        write_new(folder/"result.json", trial)
                        trials.append(trial)
                        append(out/"progress.jsonl", {k: trial[k] for k in ("case", "seed", "arm", "online_calls", "nonfeasible", "final_best")})
                        print("PR68", case, seed, arm, "calls", spec["steps"], "failed", trial["nonfeasible"], "best", trial["final_best"], flush=True)
        if ledger.started != spec["total_calls"] or ledger.settled != ledger.started:
            raise RuntimeError("Actual evaluation bill did not reconcile")
        report = {"schema": spec["schema"], "code_sha": source["head_sha"],
                  "status": "COMPLETE_GP_FANTASY_PILOT" if spec["profile"] == "full" else "COMPLETE_"+spec["profile"].upper()+"_TECHNICAL_ONLY",
                  "profile": spec["profile"], "initial_calls": spec["initial_calls"], "online_calls": spec["online_calls"],
                  "actual_calls_started": ledger.started, "actual_calls_settled": ledger.settled,
                  "aircraft_calls": 0, "rcaide_calls": 0, "summary": summarize(trials, spec),
                  "trials": trials, "elapsed_seconds": time.perf_counter()-start,
                  "interpretation": "Four-seed/task exploratory matched-cost pilot; fixed-kernel EP fantasies with 4-node/branch quadrature, no calibrated safety or stopping guarantee"}
        write_new(out/"report.json", report)
        manifest(out)
        return report
    except BaseException as error:
        write_new(out/"failure.json", {"code_sha": source["head_sha"], "error_type": type(error).__name__,
                  "message": str(error), "actual_calls_started": ledger.started, "actual_calls_settled": ledger.settled,
                  "completed_trials": len(trials), "unknown_started_calls": ledger.started-ledger.settled})
        manifest(out)
        raise


def run(output, profile, expected_sha):
    root = Path(__file__).resolve().parents[3]
    out = Path(output).resolve()
    if out == root or root in out.parents:
        raise ValueError("Results must be outside the fixed source worktree")
    source = locked_source(root, expected_sha)
    return execute(out, plan(profile), NativeAdapter, source)


def verify(output):
    """Read-only reconciliation; no GP imports, fits, or evaluator calls."""
    from ...partial_bo.records import DesignRecord
    out = Path(output)
    hashes = verify_manifest(out)
    def read(p):
        return json.loads((out/p).read_text(encoding="utf-8"))
    def lines(p):
        return [json.loads(s) for s in (out/p).read_text(encoding="utf-8").splitlines() if s.strip()]
    spec = read("plan.json")
    if spec != plan(spec["profile"]):
        raise ValueError("Unregistered experiment plan")
    report = read("report.json")
    if report["code_sha"] != read("source_receipt.json")["head_sha"]:
        raise ValueError("Source identity mismatch")
    events = lines("evaluations.jsonl")
    count = spec["total_calls"]
    if len(events) != 2*count:
        raise ValueError("Incomplete or unexpected evaluation events")
    settled = {}
    for i in range(count):
        a, b = events[2*i:2*i+2]
        if a["ticket"] != i or b["ticket"] != i or a["event"] != "started" or b["event"] != "settled":
            raise ValueError("Broken started/settled sequence")
        rec = DesignRecord.from_dict(b["record"])
        if a["candidate"] != rec.candidate or a["candidate_id"] != b["candidate_id"] or b["candidate_id"] != fingerprint(rec.candidate) or a["purpose"] != b["purpose"]:
            raise ValueError("Ledger binding mismatch")
        if rec.observation.status != "completed" or rec.physical_feasible is None or rec.diagnostics.get("numerical_failure"):
            raise ValueError("Untrusted evaluation in completed campaign")
        settled[i] = b
    used, rows = set(), []
    for case in spec["cases"]:
        for seed in spec["seeds"]:
            init = read(Path("shared_initial")/case/(str(seed)+".json"))
            if len(init["records"]) != spec["initial"] or len(init["tickets"]) != spec["initial"] or fingerprint(init["records"]) != init["hash"]:
                raise ValueError("Initial count/hash mismatch")
            for j, (ticket, rec) in enumerate(zip(init["tickets"], init["records"])):
                if ticket in used or settled[ticket]["record"] != rec or settled[ticket]["purpose"] != {"phase": "initial", "case": case, "seed": seed, "index": j}:
                    raise ValueError("Initial ledger mismatch")
                used.add(ticket)
            for arm in ARMS:
                folder = Path("trials")/case/str(seed)/arm
                props, obs = lines(folder/"proposals.jsonl"), lines(folder/"observations.jsonl")
                result = read(folder/"result.json")
                if len(props) != spec["steps"] or len(obs) != spec["steps"]:
                    raise ValueError("Wrong online call count")
                history = [DesignRecord.from_dict(r) for r in init["records"]]
                curve, path, decisions = [_best(history)], [], []
                for i, (p, o) in enumerate(zip(props, obs)):
                    ticket = o["ticket"]
                    rec = DesignRecord.from_dict(o["record"])
                    if ticket in used or settled[ticket]["record"] != o["record"] or settled[ticket]["purpose"] != {"phase": "online", "case": case, "seed": seed, "arm": arm, "iteration": i}:
                        raise ValueError("Online ledger mismatch")
                    used.add(ticket)
                    if p["iteration"] != i or o["iteration"] != i or p["candidate"] != rec.candidate or p["candidate_hash"] != o["candidate_hash"] or o["candidate_hash"] != fingerprint(rec.candidate) or p["history_hash"] != fingerprint([r.to_dict() for r in history]):
                        raise ValueError("Proposal/observation/prefix mismatch")
                    if any(r.candidate == rec.candidate for r in history):
                        raise ValueError("Repeated actual evaluation")
                    d = _decision(p["metadata"], rec.candidate)
                    d.update(iteration=i, candidate_hash=fingerprint(rec.candidate), nonfeasible=int(not rec.usable_feasible))
                    decisions.append(d)
                    history.append(rec)
                    curve.append(_best(history))
                    path.append(fingerprint(rec.candidate))
                if (result["curve"] != curve or result["path_hash"] != fingerprint(path) or
                    result["initial_hash"] != init["hash"] or result["decisions"] != decisions or
                    result["final_best"] != curve[-1] or result["initial_best"] != curve[0] or
                    result["nonfeasible"] != sum(d["nonfeasible"] for d in decisions) or
                    result["online_calls"] != spec["steps"]):
                    raise ValueError("Trial summary mismatch")
                for name in ("fantasy_head_fits_started", "fantasy_head_fits_completed"):
                    if result[name] != sum(d[name] for d in decisions):
                        raise ValueError("Fantasy fit count mismatch")
                rows.append(result)
    if used != set(range(count)) or report["trials"] != rows or report["summary"] != summarize(rows, spec):
        raise ValueError("Campaign summary or complete charge coverage mismatch")
    if report["actual_calls_started"] != count or report["actual_calls_settled"] != count:
        raise ValueError("Report charge mismatch")
    return {**hashes, "trials": len(rows), "actual_calls_started": count, "actual_calls_settled": count,
            "initial_calls": spec["initial_calls"], "online_calls": spec["online_calls"],
            "gp_fits_in_this_verification": 0, "new_evaluations_in_this_verification": 0}
