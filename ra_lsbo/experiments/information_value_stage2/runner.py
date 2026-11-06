"""PR67 matched actual on-policy comparison: eligible-only vs all-support soft GP."""
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import time

from ...budget import digest
from ...partial_bo.problems import SimplifiedElectricAircraft
from ..core.problems import AircraftCase
from ..core.protocol import BASE
from ..risk_budget63_runner import (
    _case, _shared_initial, _candidates, _status, _write, _append, _sha, clean_commit,
)
from ..risk_budget63_local_value_engine import LocalCredibilityValueBO
from ..risk_budget54_policy import observed_failures
from .engine import AllSupportSoftGPBO
from .protocol import ARMS, CASES, INITIAL, SCHEMA, feasible_first, plan


def _engine(problem, base, seed, folder, arm):
    if arm == "soft_gp_eligible":
        return LocalCredibilityValueBO(problem, base, seed, folder,
            variant="soft_gp", initial_count=INITIAL, failure_cap=6)
    if arm == "soft_gp_all_support":
        return AllSupportSoftGPBO(problem, base, seed, folder,
            initial_count=INITIAL, failure_cap=6)
    raise ValueError("Unregistered PR67 arm")


def _trial(problem, base, case, seed, arm, initial, folder, steps):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    engine = _engine(problem, base, seed, folder, arm)
    history = list(initial)
    receipts, modes, path_ids, curve = [], {}, [], [_status(history)]
    started = time.perf_counter()
    for iteration in range(steps):
        failures_before = observed_failures(history, INITIAL, iteration)
        if failures_before >= 6:
            raise RuntimeError("PR67 matched-call contract cannot reach the next registered call")
        pool = _candidates(problem, case, seed, iteration, history)
        if len(pool) != 128:
            raise RuntimeError("PR67 requires the frozen 128-candidate pool")
        prefix = digest([r.to_dict() for r in history])
        candidate, meta = engine.propose(history, pool, iteration)
        if candidate not in pool or digest([r.to_dict() for r in history]) != prefix:
            raise RuntimeError("PR67 proposal is not a legal pre-outcome candidate")
        candidate_id = digest(candidate)
        receipt = meta.get("credibility_decision")
        if receipt is not None:
            if receipt.get("selected_candidate_sha256") != candidate_id:
                raise RuntimeError("PR67 utility receipt does not bind to the actual selected candidate")
            if receipt.get("no_hard_risk_gate") is not True:
                raise RuntimeError("PR67 must not add a probability hard gate")
        _append(folder / "proposals.jsonl", {
            "iteration": iteration, "candidate": candidate,
            "candidate_sha256": candidate_id,
            "history_prefix_sha256": prefix,
            "candidate_pool_sha256": digest(pool),
            "metadata": meta, "future_result_unseen": True,
        })
        record = problem.evaluate(candidate)
        if (record.candidate != candidate or record.observation.status != "completed" or
                record.physical_feasible is None or record.diagnostics.get("numerical_failure")):
            raise RuntimeError("PR67 evaluator returned untrusted or numerical output")
        packet = record.to_dict()
        _append(folder / "observations.jsonl", {
            "iteration": iteration, "candidate_sha256": candidate_id,
            "record": packet, "record_sha256": digest(packet),
        })
        engine.accept(record)
        history.append(record)
        ablation = meta.get("support_ablation")
        eligible_count = (ablation.get("original_eligible_count") if ablation
                          else sum(meta.get("eligible", [])))
        all_count = (ablation.get("all_candidate_count") if ablation
                     else len(meta.get("eligible", [])))
        receipts.append({
            "iteration": iteration,
            "candidate_sha256": candidate_id,
            "mode": meta.get("mode", "unknown"),
            "forecast_available": receipt is not None,
            "predicted_failure": receipt.get("selected_predicted_failure") if receipt else None,
            "actual_nonfeasible": int(not record.usable_feasible),
            "support_ablation": ablation,
            "original_eligible_count": eligible_count,
            "all_candidate_count": all_count,
            "observed_failures_before": failures_before,
        })
        mode = meta.get("mode", "unknown")
        modes[mode] = modes.get(mode, 0) + 1
        path_ids.append(candidate_id)
        curve.append(_status(history))
    failed = observed_failures(history, INITIAL, steps)
    result = {
        "schema": SCHEMA, "case": case, "seed": seed, "arm": arm,
        "online_calls": steps, "observed_nonfeasible": failed,
        "initial_best": _status(initial), "final_best": _status(history),
        "best_feasible_at_each_call": curve,
        "initial_prefix_sha256": digest([r.to_dict() for r in initial]),
        "selected_path_sha256": digest(path_ids),
        "receipts": receipts, "mode_counts": modes,
        "all_calls_are_actual": True,
        "probability_guarantee": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write(folder / "result.json", result)
    return result


def _summarize(rows, profile):
    expected_seeds = [998701] if profile == "smoke" else list(range(998701, 998713))
    keyed = {}
    for row in rows:
        key = (row["case"], row["seed"], row["arm"])
        if key in keyed:
            raise ValueError("Duplicate PR67 trial")
        keyed[key] = row
    expected = {(c, s, a) for c in CASES for s in expected_seeds for a in ARMS}
    if set(keyed) != expected:
        raise ValueError("PR67 task/seed/arm roster mismatch")
    per_case, paired = {}, {}
    for case in CASES:
        for seed in expected_seeds:
            a = keyed[case, seed, ARMS[0]]
            b = keyed[case, seed, ARMS[1]]
            if a["initial_prefix_sha256"] != b["initial_prefix_sha256"] or a["initial_best"] != b["initial_best"]:
                raise ValueError("PR67 arms do not share identical initial evidence")
        for arm in ARMS:
            group = [keyed[case, s, arm] for s in expected_seeds]
            scores = [x["final_best"] for x in group if x["final_best"] is not None]
            receipts = [r for x in group for r in x["receipts"]]
            per_case[case + "/" + arm] = {
                "n_seeds": len(group),
                "final_feasible": len(scores),
                "cold_start_discoveries": sum(x["initial_best"] is None and x["final_best"] is not None for x in group),
                "mean_objective_conditional_on_feasible": sum(scores) / len(scores) if scores else None,
                "mean_online_nonfeasible": sum(x["observed_nonfeasible"] for x in group) / len(group),
                "online_calls_per_seed": group[0]["online_calls"],
                "forecast_coverage": sum(r["forecast_available"] for r in receipts),
                "support_collapse_decisions": sum((r.get("original_eligible_count") or 999) <= 1 for r in receipts),
            }
        wl = [0, 0, 0]
        both_feasible_delta = []
        for seed in expected_seeds:
            a, b = keyed[case, seed, ARMS[0]], keyed[case, seed, ARMS[1]]
            outcome = feasible_first(b, a)  # all-support vs eligible-only
            wl[0 if outcome == 1 else 1 if outcome == -1 else 2] += 1
            if a["final_best"] is not None and b["final_best"] is not None:
                both_feasible_delta.append(b["final_best"] - a["final_best"])
        paired[case] = {
            "all_support_vs_eligible_wins_losses_ties": wl,
            "both_feasible_pairs": len(both_feasible_delta),
            "objective_delta_all_support_minus_eligible_mean": (sum(both_feasible_delta) / len(both_feasible_delta) if both_feasible_delta else None),
        }
    return per_case, paired


def run(output, profile="smoke"):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("PR67 requires a fresh write-once output directory")
    source_sha = clean_commit()
    here = Path(__file__).resolve()
    # The original PR54 policy and engine must remain byte-identical.
    from ..risk_budget63_protocol import PR54_POLICY_SHA256, PR54_ENGINE_SHA256
    if (_sha(here.parents[1] / "risk_budget54_policy.py") != PR54_POLICY_SHA256 or
            _sha(here.parents[1] / "risk_budget54_engine.py") != PR54_ENGINE_SHA256):
        raise ValueError("PR67 refuses modified frozen PR54 control source")
    spec = plan(profile, source_sha)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "plan.json", spec)
    _write(output / "source_receipt.json", {
        "code_sha": source_sha,
        "parent_pr63_sha": "18b48e25dda6ff7d17296087981d0c4dc3ce99b2",
        "source_policy": "PR63 soft_gp unchanged",
        "new_arm": "PR63 soft_gp utility over all legal GP-predicted candidates",
        "frozen_pr54_policy_sha256": PR54_POLICY_SHA256,
        "frozen_pr54_engine_sha256": PR54_ENGINE_SHA256,
        "no_historical_result_reads": True,
        "new_aircraft_calls": 0, "new_rcaide_calls": 0,
    })
    rows, calls = [], 0
    with ExitStack() as stack:
        def forbidden(*_args, **_kwargs):
            raise RuntimeError("PR67 mathematical-only: aircraft evaluator forbidden")
        stack.enter_context(patch.object(AircraftCase, "evaluate", forbidden))
        stack.enter_context(patch.object(SimplifiedElectricAircraft, "evaluate", forbidden))
        for case in CASES:
            problem = _case(case)
            for seed in spec["seeds"]:
                initial, rejected = _shared_initial(problem, case, seed)
                calls += len(initial)
                _write(output / "shared_initial" / case / (str(seed) + ".json"), {
                    "case": case, "seed": seed, "records": [r.to_dict() for r in initial],
                    "shared_once_across_arms": True,
                    "initial_prefix_sha256": digest([r.to_dict() for r in initial]),
                    "precheck_rejections": rejected,
                })
                for arm in ARMS:
                    steps = spec["online_calls_per_arm"]
                    folder = output / "trials" / case / str(seed) / arm
                    result = _trial(problem, BASE, case, seed, arm, initial, folder, steps)
                    rows.append(result)
                    calls += steps
                    _append(output / "progress.jsonl", {
                        "case": case, "seed": seed, "arm": arm,
                        "online_calls": steps, "observed_nonfeasible": result["observed_nonfeasible"],
                        "final_best": result["final_best"],
                    })
    if calls != spec["total_mathematical_calls"]:
        raise RuntimeError("PR67 actual evaluation count does not reconcile")
    per_case, paired = _summarize(rows, profile)
    report = {
        "schema": SCHEMA,
        "status": "COMPLETE_FRESH_FIXED_SIX_ON_POLICY" if profile == "full" else "SMOKE_TWO_CALLS_NOT_FULL_STUDY",
        "code_sha": source_sha,
        "actual_mathematical_calls": calls,
        "initial_calls_shared_once": spec["initial_calls"],
        "actual_online_calls": spec["online_calls"],
        "new_aircraft_calls": 0, "new_rcaide_calls": 0,
        "same_actual_online_effort": True,
        "result_summary_by_case_arm": per_case,
        "paired_comparisons": paired,
        "trial_details": rows,
        "claim_limits": [
            "The only policy change is removing the optimistic-violation eligible mask; this is not a calibrated risk or safety method.",
            "Both arms retain PR63 soft-GP risk/gain utility and its uncalibrated marginal GP feasibility proxy.",
            "The experiment is exploratory and uses custom mathematical tasks, not aircraft/RCAIDE.",
            "A no-gate arm can evaluate high predicted-risk points; actual failures and final feasible-first outcomes must both be reported.",
            "This stage is a myopic matched control, not the GP two-step lookahead algorithm.",
        ],
    }
    _write(output / "report.json", report)
    return report
