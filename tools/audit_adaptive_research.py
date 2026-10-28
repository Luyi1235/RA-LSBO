"""Independent accounting, isolation and matched-design checks for adaptive study."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.adaptive_system import ADAPTIVE_METHODS
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import RiskLedger, digest
from ra_lsbo.features import history_identity
from ra_lsbo.selection import incumbent
from ra_lsbo.system_experiment import save_new
from tools.run_adaptive_research import lines, problem_for, validate_plan


def audit(folder):
    torch.set_num_threads(1)
    plan = load_json((folder/"plan.json").read_bytes())
    validate_plan(plan)
    analysis = load_json((folder/"analysis.json").read_bytes())
    if analysis["status"] != "completed":
        raise ValueError("Every scheduled trial must terminate without model errors")
    settings, summaries, matched, curves = plan["settings"], [], [], []
    input_hashes = {str(folder/"plan.json"): hashlib.sha256((folder/"plan.json").read_bytes()).hexdigest()}
    forecasts, calls, truth_replays = 0, 0, 0
    for case in plan["cases"]:
        problem = problem_for(case, plan["base_project"])
        for seed in plan["seeds"]:
            initial_path, test_path = folder/case/("initial_"+str(seed)+".json"), folder/case/("test_"+str(seed)+".json")
            initial, test = load_json(initial_path.read_bytes()), load_json(test_path.read_bytes())
            initial_keys = {problem.space.key(problem.candidate_class(**r["candidate"])) for r in initial}
            test_keys = {problem.space.key(problem.candidate_class(**r["candidate"])) for r in test}
            if len(initial_keys) != len(initial) or initial_keys & test_keys:
                raise ValueError("Shared initial/test design duplication")
            before = []
            for method in plan["methods"]:
                output = folder/case/(method+"_"+str(seed))
                manifest = load_json((output/"manifest.json").read_bytes())
                if manifest["history_id"] != history_identity(initial):
                    raise ValueError("Methods did not share initial history")
                proposed, observed = lines(output/"proposals.jsonl"), lines(output/"observations.jsonl")
                fits = lines(output/"model_fits.jsonl")
                summary = load_json((output/"summary.json").read_bytes())
                ledger = RiskLedger(output/"budget.jsonl").summary()
                if (ledger != summary["ledger"] or len(observed) != len(proposed)
                        or ledger["charged_calls"] != len(observed) or ledger["reserved_calls"]
                        or ledger["unresolved_started_calls"] or len(observed) > settings["calls"]
                        or ledger["observed_nonfeasible"] > settings["failure_cap"]):
                    raise ValueError("Call accounting or hard stop differs")
                rows, expected_risk, information_calls = list(initial), 0., 0
                trajectory, seen = [incumbent(initial)], set(initial_keys)
                for i, (proposal, row) in enumerate(zip(proposed, observed)):
                    if (row["evaluation"] != len(initial)+i+1 or proposal["evaluation"] != row["evaluation"]
                            or proposal["history_id"] != history_identity(rows)):
                        raise ValueError("Proposal was not bound to its available history")
                    key = problem.space.key(problem.candidate_class(**row["candidate"]))
                    if key in seen:
                        raise ValueError("Repeated evaluated design")
                    seen.add(key)
                    if (proposal["proposal"]["candidate"] != row["candidate"]
                            or row["risk"] != problem.evaluate(row["candidate"])):
                        raise ValueError("Controlled truth mismatch")
                    truth_replays += 1
                    if method in ADAPTIVE_METHODS:
                        selection = proposal["selection"]
                        if selection["history_id"] != history_identity(rows):
                            raise ValueError("Selected forecast history differs")
                        if not np.isclose(selection["estimated_risk_before"], expected_risk):
                            raise ValueError("Soft expected-risk ledger differs")
                        p = (selection["raw_events"] if method == "adaptive_no_calibration"
                             else selection["transformed_events"])[0]
                        if (not np.isclose(selection["used_final"], p)
                                or not np.isclose(selection["estimated_risk_debit"], 1.-p)):
                            raise ValueError("Scored probability or debit differs")
                        expected_risk += 1.-p
                        information_calls += int(selection["mode"] == "information")
                        if information_calls > settings["boundary_cap"]:
                            raise ValueError("Information call cap exceeded")
                        fit = next(f for f in fits if f["fit_number"] == proposal["fit_number"])
                        if (row["evaluation"] in fit["training_indices"]
                                or row["evaluation"] in fit["reference_indices"]
                                or fit["before_evaluation"] > row["evaluation"]):
                            raise ValueError("Outcome leaked into its preceding forecast")
                        forecasts += 1
                    rows.append(row)
                    trajectory.append(incumbent(rows))
                if method in ADAPTIVE_METHODS:
                    if (ledger["risk_charge"] != len(observed) or not np.isclose(
                            summary["expected_risk"], expected_risk)
                            or summary["information_calls"] != information_calls):
                        raise ValueError("Hard versus soft ledger mismatch")
                    if method == "adaptive_no_information" and information_calls != 0:
                        raise ValueError("Information ablation still explores")
                    first = load_json((output/"static_forecasts.json").read_bytes())
                    before.append((method, fits[0], first))
                while len(trajectory) < settings["calls"]+1:
                    trajectory.append(trajectory[-1])
                initial_best = trajectory[0]
                gains = [100*(initial_best-value)/abs(initial_best) for value in trajectory[1:]]
                curves.append({"case": case, "seed": seed, "method": method,
                               "best_by_call": trajectory, "gain_percent_by_call": gains,
                               "mean_gain_across_call_horizon": float(np.mean(gains))})
                summaries.append({"case": case, "seed": seed, "method": method,
                                  "status": summary["status"], "calls": len(observed)})
                calls += len(observed)
                for path in [output/"manifest.json", output/"summary.json", output/"budget.jsonl",
                             output/"proposals.jsonl", output/"observations.jsonl", output/"model_fits.jsonl"]:
                    if path.exists():
                        input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            full = next(x for x in before if x[0] == "adaptive_full")
            no_cal = next(x for x in before if x[0] == "adaptive_no_calibration")
            if (full[1]["model_id"] != no_cal[1]["model_id"]
                    or full[1]["calibration_id"] != no_cal[1]["calibration_id"]
                    or full[2]["raw"] != no_cal[2]["raw"]):
                raise ValueError("Calibration ablation changed the initial raw model")
            for method, fit, forecast in before:
                if fit["training_indices"] != full[1]["training_indices"]:
                    raise ValueError("Ablations have different initial data")
                if fit["representation_snapshot"] != full[1]["representation_snapshot"]:
                    raise ValueError("Ablations have different initial VAE")
            matched.append({"case": case, "seed": seed, "common_initial_history": True,
                            "calibration_ablation_same_raw_model": True, "all_ablations_same_vae": True})
    wins = []
    for case in plan["cases"]:
        for method in plan["methods"]:
            if method == "adaptive_full":
                continue
            paired = []
            for seed in plan["seeds"]:
                a = next(c for c in curves if c["case"] == case and c["seed"] == seed
                         and c["method"] == "adaptive_full")
                b = next(c for c in curves if c["case"] == case and c["seed"] == seed and c["method"] == method)
                paired.append(a["gain_percent_by_call"][-1]-b["gain_percent_by_call"][-1])
            wins.append({"case": case, "comparator": method,
                         "full_endpoint_gain_difference_percentage_points": paired,
                         "wins": sum(v > 1e-10 for v in paired), "ties": sum(abs(v) <= 1e-10 for v in paired),
                         "losses": sum(v < -1e-10 for v in paired)})
    result = {
        "status": "passed", "additional_real_mda_calls": 0, "trials": len(summaries),
        "online_controlled_calls": calls, "independent_truth_replays": truth_replays,
        "selected_forecast_bindings": forecasts,
        "checkpoint_forecasts_replayed_by_primary_analysis": analysis["selected_risk_forecasts_replayed"],
        "preparatory_controlled_labels": len(plan["cases"])*len(plan["seeds"])*(
            plan["initial_count"]+plan["static_test_count"]),
        "matched_designs": matched, "paired_endpoint_comparisons": wins, "curves": curves,
        "trial_statuses": dict(Counter(t["status"] for t in summaries)), "input_sha256": input_hashes,
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": "Checks outcomes, history, risk forecasts and budgets; not full GP acquisition replay"}
    verify_sources(plan["source_sha256"])
    save_new(folder/"independent_audit.json", result)
    print("Independent audit passed", result["trials"], "trials;", calls, "calls;", forecasts, "forecasts", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    audit(args.folder.resolve())
