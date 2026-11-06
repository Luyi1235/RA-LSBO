"""PR70 actual online pool-consistency pilot; lazy imports and read-only verifier."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from pr70_protocol import PARENT, ARMS, plan, summarize, verify, check_transition, check_receipt
from pr69_evidence_verify import fingerprint, native_digest as digest, require
from ra_lsbo.experiments.information_value_stage1.io import write_new, append, manifest
from ra_lsbo.experiments.information_value_stage3.runner import locked_source


def run(output, profile, expected_sha):
    root = Path(__file__).resolve().parent
    out = Path(output).resolve()
    require(out != root and root not in out.parents and not out.exists(), "NEW external output required")
    source = locked_source(root, expected_sha)
    changes = subprocess.check_output(["git", "-C", str(root), "diff", "--name-status", PARENT, expected_sha], text=True).splitlines()
    require(all(x.startswith("A\t") for x in changes), "PR69 parent files changed")
    source.update(pr70_parent=PARENT, all_pr69_blobs_unchanged=True)
    spec = plan(profile)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/"plan.json", spec)
    write_new(out/"source_receipt.json", source)
    from ra_lsbo.experiments.information_value_stage1.runner import ChargedEvaluator, _legal, _best
    ledger = ChargedEvaluator(out/"evaluations.jsonl")
    trials = []
    started = time.perf_counter()
    try:
        from ra_lsbo.experiments.information_value_stage1.runner import NativeAdapter
        from ra_lsbo.experiments.information_value_stage3.native import FantasyValueBO
        from ra_lsbo.experiments.discovery_stages.policies import filter_near
        from pr70_future_pool import NextPoolBO
        api = NativeAdapter()
        with api.scope():
            for case in spec["cases"]:
                for seed in spec["seeds"]:
                    problem = api.problem(case)
                    designs = api.initial(problem, case, seed)
                    _legal(problem, designs, [], 12)
                    initial, tickets = [], []
                    for j, c in enumerate(designs):
                        tickets.append(ledger.started)
                        initial.append(ledger.call(problem, c, {"phase": "initial", "case": case, "seed": seed, "index": j}))
                    initial_hash = fingerprint([r.to_dict() for r in initial])
                    write_new(out/"shared_initial"/case/(str(seed)+".json"),
                              {"records": [r.to_dict() for r in initial], "tickets": tickets, "hash": initial_hash})
                    for arm in ARMS:
                        folder = out/"trials"/case/str(seed)/arm
                        folder.mkdir(parents=True, exist_ok=False)
                        engine = (NextPoolBO(problem, api.base, seed, folder, case=case, horizon=spec["steps"])
                                  if arm == ARMS[2] else FantasyValueBO(problem, api.base, seed, folder, arm=arm, horizon=spec["steps"]))
                        history, curve, path, decisions = list(initial), [_best(initial)], [], []
                        previous, checks, proposal_seconds = None, 0, 0.
                        for i in range(spec["steps"]):
                            raw = api.pool(problem, case, seed, i, history)
                            _legal(problem, raw, history, 128)
                            protected = filter_near(problem, raw, history)[0]
                            checks += check_transition(previous, raw, protected, i)
                            prefix = fingerprint([r.to_dict() for r in history])
                            t0 = time.perf_counter()
                            c, meta = engine.propose(history, raw, i)
                            elapsed = time.perf_counter()-t0
                            proposal_seconds += elapsed
                            require(c in raw and prefix == fingerprint([r.to_dict() for r in history]), "Invalid proposal/history mutation")
                            require(meta["pool"] == protected, "Actual protected pool differs from registered pool")
                            decision = check_receipt(meta, c, i, spec["steps"], arm)
                            append(folder/"proposals.jsonl", {"iteration": i, "candidate": c,
                                "candidate_hash": fingerprint(c), "history_hash": prefix,
                                "raw_pool": raw, "protected_pool": protected, "pool_hash": fingerprint(raw),
                                "metadata": meta, "proposal_seconds": elapsed, "outcome_unseen": True})
                            previous = meta.get("information_value_decision", {}).get("selected_future_support")
                            ticket = ledger.started
                            rec = ledger.call(problem, c, {"phase": "online", "case": case, "seed": seed, "arm": arm, "iteration": i})
                            append(folder/"observations.jsonl", {"iteration": i, "ticket": ticket,
                                   "candidate_hash": fingerprint(c), "record": rec.to_dict()})
                            engine.accept(rec)
                            history.append(rec)
                            path.append(fingerprint(c))
                            curve.append(_best(history))
                            decision.update(iteration=i, candidate_hash=fingerprint(c), nonfeasible=int(not rec.usable_feasible))
                            decisions.append(decision)
                        trial = {"case": case, "seed": seed, "arm": arm, "online_calls": spec["steps"],
                            "initial_hash": initial_hash, "initial_best": curve[0], "final_best": curve[-1],
                            "curve": curve, "path_hash": fingerprint(path), "decisions": decisions,
                            "nonfeasible": sum(d["nonfeasible"] for d in decisions),
                            "proposal_seconds": proposal_seconds, "future_pool_checks": checks,
                            "fantasy_head_fits_started": sum(d["fantasy_head_fits_started"] for d in decisions),
                            "fantasy_head_fits_completed": sum(d["fantasy_head_fits_completed"] for d in decisions),
                            "fantasy_fallbacks": sum(d["fantasy_fallback"] for d in decisions),
                            "model_free_calls": sum(not d["forecast_available"] for d in decisions)}
                        write_new(folder/"result.json", trial)
                        trials.append(trial)
                        append(out/"progress.jsonl", {k: trial[k] for k in ("case", "seed", "arm", "online_calls", "nonfeasible", "final_best", "future_pool_checks")})
                        print("PR70", case, seed, arm, "failed", trial["nonfeasible"], "best", trial["final_best"], "pool checks", checks, flush=True)
        require(ledger.started == ledger.settled == spec["total_calls"], "Actual bill mismatch")
        report = {"schema": spec["schema"], "profile": profile, "code_sha": expected_sha,
            "status": "COMPLETE_NEXT_POOL_PILOT" if profile == "full" else "COMPLETE_"+profile.upper()+"_TECHNICAL_ONLY",
            "actual_calls_started": ledger.started, "actual_calls_settled": ledger.settled,
            "initial_calls": spec["initial_calls"], "online_calls": spec["online_calls"],
            "aircraft_calls": 0, "rcaide_calls": 0, "trials": trials, "summary": summarize(trials, spec),
            "elapsed_seconds": time.perf_counter()-started,
            "interpretation": "Isolates continuation support with unchanged four-node approximation; exploratory, not safety or novelty proof"}
        write_new(out/"report.json", report)
        manifest(out)
        return report
    except BaseException as error:
        write_new(out/"failure.json", {"code_sha": expected_sha, "error_type": type(error).__name__, "message": str(error),
            "actual_calls_started": ledger.started, "actual_calls_settled": ledger.settled,
            "unknown_started_calls": ledger.started-ledger.settled, "completed_trials": len(trials)})
        manifest(out)
        raise


def main():
    p = argparse.ArgumentParser(description="PR70 exact-next-pool continuation isolation")
    p.add_argument("--task", choices=("run", "verify"), default="run")
    p.add_argument("--profile", choices=("ci", "smoke", "full"), default="smoke")
    p.add_argument("--expected-sha")
    p.add_argument("--output", type=Path)
    p.add_argument("--input", type=Path)
    a = p.parse_args()
    if a.task == "verify":
        if a.input is None:
            p.error("--input required")
        result = verify(a.input)
    else:
        if a.output is None or a.expected_sha is None:
            p.error("--output and --expected-sha required")
        r = run(a.output, a.profile, a.expected_sha)
        result = {k: r[k] for k in ("status", "code_sha", "actual_calls_started", "actual_calls_settled", "aircraft_calls", "rcaide_calls")}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
