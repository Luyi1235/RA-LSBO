"""Recompute audited study metrics without performing any aircraft evaluation."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.calibration import observed_target
from ra_lsbo.mechanism_system import FiveModeProblem, METHODS, mechanism_support
from ra_lsbo.metrics import binary_metrics
from ra_lsbo.records import HIDDEN_MODES
from ra_lsbo.system_experiment import save_new
from tools.run_system_validation import read_lines


def same_value(left, right):
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(same_value(left[k], right[k]) for k in left))
    if isinstance(left, list):
        return (isinstance(right, list) and len(left) == len(right)
                and all(same_value(a, b) for a, b in zip(left, right)))
    if isinstance(left, float):
        return right is not None and np.isclose(left, right, rtol=1e-11, atol=1e-12)
    return left == right


def best_value(risk):
    if risk["feasible_label"] is True and risk["objective_observed"]:
        return risk["objective"]
    return None


def trial_metrics(run):
    folder = Path(run["directory"])
    rows = read_lines(folder/"observations.jsonl")
    if len(rows) != run["evaluations"]:
        raise ValueError("Evaluation count differs: " + str(folder))
    best, first_feasible, first_improvement = run["initial_best"], None, None
    curve = [{"calls": 0, "best": best}]
    for call, row in enumerate(rows, 1):
        value = best_value(row["risk"])
        if value is not None:
            if first_feasible is None:
                first_feasible = call
            if best is None or value < best:
                best = value
                if first_improvement is None:
                    first_improvement = call
        curve.append({"calls": call, "best": best})
    if best != run["best_feasible_objective"]:
        raise ValueError("Incumbent differs: " + str(folder))
    counts = dict(Counter(r["risk"]["state"] for r in rows))
    if counts != run["states"] or run["ledger"]["charged_calls"] != len(rows):
        raise ValueError("Outcome/ledger count differs: " + str(folder))
    constraints = Counter(
        name for r in rows for name, value in r["risk"]["constraints"].items()
        if r["risk"]["constraint_observed"].get(name) is True
        and value is not None and value > 0)
    proposals = read_lines(folder/"proposals.jsonl")
    forecasts = {r["evaluation"]: r["selection"].get("forecast") for r in proposals}
    scored = {"raw": ([], []), "calibrated": ([], [])}
    for row in rows:
        forecast = forecasts.get(row["evaluation"])
        label = observed_target(row["risk"], "final_feasibility")
        if forecast is None or label is None:
            continue
        for key, probability in (("raw", forecast["raw"]["final_feasibility"]),
                                 ("calibrated", forecast["calibrated_final"])):
            if probability is not None:
                scored[key][0].append(probability)
                scored[key][1].append(label)
    modes = Counter(r.get("phase", "unknown") for r in rows)
    fits = read_lines(folder/"model_fits.jsonl")
    return {
        **{name: run[name] for name in
           ("method", "seed", "status", "stop", "evaluations", "initial_best",
            "best_feasible_objective", "feasible_evaluations", "ledger", "wall_seconds")},
        "improvement": None if run["initial_best"] is None or best is None
        else run["initial_best"]-best,
        "improvement_percent": None if run["initial_best"] in (None, 0) or best is None
        else 100*(run["initial_best"]-best)/abs(run["initial_best"]),
        "first_new_feasible_call": first_feasible,
        "first_improvement_call": first_improvement, "curve": curve,
        "states": counts, "explicit_constraint_violations": dict(constraints),
        "constraint_counts_are_multilabel": True,
        "mechanism_support": mechanism_support(rows),
        "selection_modes": dict(modes),
        "proposal_sources": dict(Counter(r.get("proposal", {}).get("source", "unknown") for r in rows)),
        "model_fits": [{"training_count": len(f["training_indices"]),
                        "calibration_count": len(f["calibration_indices"]),
                        "active_mechanisms": f["active_mechanisms"],
                        "final_calibration_status": None if f["calibration"] is None
                        else f["calibration"]["final_feasibility"]["status"]}
                       for f in fits],
        "selected_point_diagnostics": {k: binary_metrics(*v) for k, v in scored.items()},
        "selected_point_diagnostics_scope":
            "Adaptive method-specific selections; unsuitable for cross-method calibration ranking",
    }


def compare_trials(trials):
    paired = []
    for seed in sorted({t["seed"] for t in trials}):
        by_method = {t["method"]: t for t in trials if t["seed"] == seed}
        full = by_method["ra_full"]
        for method in METHODS[1:]:
            other = by_method[method]
            common = min(full["evaluations"], other["evaluations"])
            a, b = full["curve"][common]["best"], other["curve"][common]["best"]
            paired.append({
                "seed": seed, "reference": "ra_full", "other": method,
                "common_actual_calls": common, "full_best": a, "other_best": b,
                "other_minus_full": None if a is None or b is None else b-a,
                "positive_favors_full": True,
                "has_new_call_evidence": common > 0,
            })
    aggregate = []
    for method in METHODS:
        group = [t for t in trials if t["method"] == method]
        total = sum(t["evaluations"] for t in group)
        feasible = sum(t["feasible_evaluations"] for t in group)
        available = [t["improvement_percent"] for t in group if t["improvement_percent"] is not None]
        aggregate.append({
            "method": method, "runs": len(group), "calls": total,
            "feasible_calls": feasible, "feasible_fraction": feasible/total if total else None,
            "mean_improvement_percent": float(np.mean(available)) if available else None,
            "improved_runs": sum(t["first_improvement_call"] is not None for t in group),
            "statuses": dict(Counter(t["status"] for t in group)),
            "states": dict(sum((Counter(t["states"]) for t in group), Counter())),
            "all_runs_retained": True,
        })
    return {"trials": trials, "aggregate": aggregate, "paired_common_actual_calls": paired,
            "inference": "Three random repetitions of one task; no significance or general dominance claim"}


def physical_acquisition(folder):
    initial = load_json((folder/"initial_history.json").read_bytes())
    best = min((best_value(r["risk"]) for r in initial if best_value(r["risk"]) is not None),
               default=None)
    stages, cumulative = [], 0
    for name, filename in (("mechanism_probes", "probe_results.jsonl"),
                           ("fresh_calibration", "calibration_results.jsonl")):
        values = [r for r in read_lines(folder/filename) if r["risk"] is not None]
        before = best
        curve = [{"calls": 0, "best": best}]
        for call, row in enumerate(values, 1):
            value = best_value(row["risk"])
            if value is not None and (best is None or value < best):
                best = value
            curve.append({"calls": call, "best": best})
        calls = sum(r["ledger"]["charged_calls"] for r in values)
        cumulative += calls
        stages.append({
            "stage": name, "mda_calls": calls, "cumulative_new_mda_calls": cumulative,
            "initial_best": before, "end_best": best, "curve": curve,
            "states": dict(Counter(r["risk"]["state"] for r in values)),
            "mechanism_support": mechanism_support(values),
            "attribution": "Shared data acquisition; not an optimizer-specific improvement"})
    random_history = initial[:300]
    return {"historical_mda_calls": 420, "stages": stages,
            "optimization_initial_best": best,
            "historical_random_states": dict(Counter(r["risk"]["state"] for r in random_history)),
            "historical_random_feasible_fraction": sum(
                r["risk"]["feasible_label"] is True for r in random_history)/len(random_history),
            "sampling_scope": "Conditional on cheap screening; not a full-domain feasibility-volume estimate",
            "sparse_feasible_domain_demonstrated": False,
            "shared_acquisition_cost_not_free": True}


def static_metrics(folder, base):
    report = load_json((folder/"controlled_static_report.json").read_bytes())
    rows = load_json((folder/"controlled_static_history.json").read_bytes())
    if len(rows) != 1500:
        raise ValueError("Controlled sample size differs")
    problem = FiveModeProblem(base)
    for row in rows:
        if problem.evaluate(row["candidate"]) != row["risk"]:
            raise ValueError("Controlled truth differs")
    blocks = {"training": rows[:600], "calibration": rows[600:900], "test": rows[900:]}
    support = {name: mechanism_support(block) for name, block in blocks.items()}
    if support != report["support"]:
        raise ValueError("Controlled stage censoring/support differs")
    truth = {r["evaluation"]: r for r in rows[900:]}
    forecasts = read_lines(folder/"controlled_forecasts.jsonl")
    scored = read_lines(folder/"controlled_scored.jsonl")
    if len(forecasts) != 1800 or len(scored) != 1800:
        raise ValueError("Controlled forecast count differs")
    seen = set()
    for prediction, score in zip(forecasts, scored):
        key = prediction["method"], prediction["evaluation"]
        if key in seen or key[1] not in truth:
            raise ValueError("Controlled prediction duplicate/test index differs")
        seen.add(key)
        if ({k: v for k, v in score.items() if k != "observed"} != prediction
                or score["observed"] != truth[key[1]]["risk"]
                or prediction["candidate"] != truth[key[1]]["candidate"]):
            raise ValueError("Controlled forecast/scoring pairing differs")
    result = []
    for model in report["models"]:
        method = model["method"]
        predictions = [r for r in scored if r["method"] == method]
        if {r["evaluation"] for r in predictions} != set(truth):
            raise ValueError("Controlled held-out set differs")
        metadata = model["metadata"]
        if metadata["training_indices"] != list(range(1, 601)):
            raise ValueError("Controlled training leaked test labels")
        if metadata["calibration_indices"] != list(range(601, 901)):
            raise ValueError("Controlled calibration split differs")
        scores = {}
        for name in ("final_feasibility", "hidden_success", *HIDDEN_MODES):
            known = [r for r in predictions if observed_target(r["observed"], name) is not None]
            labels = [observed_target(r["observed"], name) for r in known]
            values = [r["calibrated"][name] for r in known]
            raw = [r["raw"][name] for r in known]
            scores[name] = {
                "raw": binary_metrics(raw, labels),
                "calibrated": binary_metrics([], []) if any(v is None for v in values)
                else binary_metrics(values, labels),
                "unknown_labels": 600-len(known)}
            if not same_value(scores[name], model["scores"][name]):
                raise ValueError("Controlled metric differs: " + method + "/" + name)
            diagnostic = model["calibration"][name]
            if diagnostic["calibrator_fitted"]:
                parameters = diagnostic["parameters"]
                clipped = np.clip(raw, diagnostic["policy"]["probability_clip"],
                                  1-diagnostic["policy"]["probability_clip"])
                logits = np.log(clipped)-np.log1p(-clipped)
                expected = 1/(1+np.exp(-(parameters["slope"]*logits+parameters["intercept"])))
                if not np.allclose(expected, values, rtol=1e-11, atol=1e-12):
                    raise ValueError("Controlled calibrated predictions differ")
        result.append({"method": method, "active_mechanisms": metadata["active_mechanisms"],
                       "scores": scores})
    by_method = {m: {r["evaluation"]: r for r in scored if r["method"] == m}
                 for m in ("ra_full", "ra_no_mechanisms", "ra_onehot_risk")}
    full_scores = [by_method["ra_full"][i]["raw"]["final_feasibility"] for i in sorted(truth)]
    removed_scores = [by_method["ra_no_mechanisms"][i]["raw"]["final_feasibility"] for i in sorted(truth)]
    differences = np.asarray(removed_scores)-np.asarray(full_scores)
    contribution = {
        "interpretation": "Descriptive post-hoc mechanism contribution, not an additional tuning criterion",
        "shared_test_points": len(truth),
        "raw_final_score_changed": int(np.sum(np.abs(differences) > 1e-12)),
        "raw_final_score_reduced_by_mechanisms": int(np.sum(differences > 1e-12)),
        "maximum_raw_score_reduction": float(np.max(differences)),
        "mean_raw_score_reduction": float(np.mean(differences)),
        "controlled_closed_loop_candidate_matches": [],
    }
    for seed in (31, 32, 33):
        root = folder/"controlled"/("seed_"+str(seed))
        full_rows = read_lines(root/"ra_full"/"observations.jsonl")
        removed_rows = read_lines(root/"ra_no_mechanisms"/"observations.jsonl")
        common = min(len(full_rows), len(removed_rows))
        contribution["controlled_closed_loop_candidate_matches"].append({
            "seed": seed, "common_call_positions": common,
            "identical_candidates_at_same_position": sum(
                a["candidate"] == b["candidate"]
                for a, b in zip(full_rows[:common], removed_rows[:common])),
            "full_selection_modes": dict(Counter(r["phase"] for r in full_rows)),
            "removed_selection_modes": dict(Counter(r["phase"] for r in removed_rows))})
    return {"support": support, "models": result, "truth_recomputed": 1500,
            "forecast_pairings_checked": 1800, "sample_split": [600, 300, 600],
            "mechanism_contribution": contribution,
            "scope": "Controlled serial mechanism process; not aircraft observations"}


def figures(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#136F63", "#D1495B", "#3068B2", "#9467BD",
              "#D08017", "#495057", "#2589A6", "#7C8134"]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for kind in ("controlled", "physical"):
        if kind not in result:
            continue
        trials = result[kind]["trials"]
        seeds = sorted({t["seed"] for t in trials})
        fig, axes = plt.subplots(1, len(seeds), figsize=(13, 4.6), squeeze=False)
        for ax, seed in zip(axes[0], seeds):
            for color, method in zip(colors, METHODS):
                trial = next(t for t in trials if t["method"] == method and t["seed"] == seed)
                ax.step([p["calls"] for p in trial["curve"]], [p["best"] for p in trial["curve"]],
                        where="post", color=color, label=method, linewidth=1.6,
                        linestyle="--" if method == "ra_no_mechanisms" else "-")
                ax.scatter([trial["evaluations"]], [trial["best_feasible_objective"]],
                           s=13, color=color)
            ax.set(title="Seed "+str(seed), xlabel="Actual new evaluations")
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid(alpha=.2)
        axes[0, 0].set_ylabel("Best feasible objective"+(" (Wh)" if kind == "physical" else ""))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.suptitle(kind.capitalize()+" closed loop: all runs, endpoints retained")
        fig.tight_layout(rect=(0, .14, 1, .94))
        fig.savefig(output/(kind+"_incumbent.png"), dpi=170)
        plt.close(fig)
    static = result["controlled_static"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    names = ["final_feasibility", *HIDDEN_MODES]
    labels = ["Final", "Geometry", "Network", "Motor", "Mission", "Energy"]
    x, width = np.arange(len(names)), .24
    for i, model in enumerate(static["models"]):
        scores = [model["scores"][name]["calibrated"]["brier"] for name in names]
        axes[0].bar(x+(i-1)*width, scores, width, color=colors[i], label=model["method"])
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Calibrated Brier score (lower is better)",
                title="Independent controlled test set")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(frameon=False, fontsize=8)
    support = static["support"]["training"]
    counts = [support[n]["positive"] for n in HIDDEN_MODES]
    axes[1].bar(labels[1:], counts, color=colors[0])
    axes[1].axhline(20, linestyle="--", color="#666666", linewidth=1)
    axes[1].set(ylabel="Observed training failure positives",
                title="Controlled training support (600 points)")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output/"controlled_diagnostics.png", dpi=170)
    plt.close(fig)


def markdown(result):
    lines = ["# System validation analysis", "",
             "All early stops and zero-improvement runs are retained. Lower objective is better.", ""]
    if "physical" in result:
        account = result["accounting"]
        lines += [f"Complete new MDA calls: {account['complete_new_mda_calls']} / 480; "
                  f"carried: {account['carried_mda_calls']}; "
                  f"new current version: {account['new_current_mda_calls']}.", "",
                  "Shared acquisition costs and gains (not credited to an individual optimizer):", "",
                  "| Stage | MDA calls | Initial best Wh | End best Wh |",
                  "| --- | ---: | ---: | ---: |"]
        for stage in result["physical_acquisition"]["stages"]:
            lines.append(f"| {stage['stage']} | {stage['mda_calls']} | "
                         f"{stage['initial_best']} | {stage['end_best']} |")
        lines += ["",
                  "The original 300-point screened random history has feasible fraction "
                  f"{result['physical_acquisition']['historical_random_feasible_fraction']:.3f}. "
                  "This task has not demonstrated a sparse feasible domain.", "",
                  "Physical mechanism training support:", "",
                  "| Mechanism | Positive | Negative | Unknown |", "| --- | ---: | ---: | ---: |"]
        for name, counts in result["mechanism_data"]["training_support"].items():
            lines.append(f"| {name} | {counts['positive']} | {counts['negative']} | {counts['unknown']} |")
        lines += ["", "The 20-positive/20-negative flag is a diagnostic threshold, not a generalization guarantee.", ""]
    for kind in ("controlled", "physical"):
        if kind not in result:
            continue
        lines += ["## "+kind.capitalize(), "",
                  "| Method | Calls | Feasible | Mean improvement % | Improved runs | Status counts |",
                  "| --- | ---: | ---: | ---: | ---: | --- |"]
        for row in result[kind]["aggregate"]:
            improvement = row["mean_improvement_percent"]
            text = "NA" if improvement is None else f"{improvement:.5f}"
            lines.append(f"| {row['method']} | {row['calls']} | {row['feasible_calls']} | "
                         f"{text} | {row['improved_runs']}/3 | {json.dumps(row['statuses'])} |")
        lines += ["", "Paired comparisons at each pair's smaller actual call count:", "",
                  "| Other method | Seed | Common calls | Other minus full objective |",
                  "| --- | ---: | ---: | ---: |"]
        for row in result[kind]["paired_common_actual_calls"]:
            difference = row["other_minus_full"]
            text = "NA" if difference is None else f"{difference:.8f}"
            lines.append(f"| {row['other']} | {row['seed']} | {row['common_actual_calls']} | {text} |")
        lines += ["", "Positive differences favor full RA; zero common calls supply no new-call evidence.", "",
                  "Three seeds are random repetitions of the same task, not independent physical tasks.", ""]
    lines += ["## Controlled Held-out Diagnostics", "",
              "| Model | Final calibrated Brier | Hidden-success calibrated Brier |",
              "| --- | ---: | ---: |"]
    for model in result["controlled_static"]["models"]:
        score = model["scores"]
        lines.append(f"| {model['method']} | {score['final_feasibility']['calibrated']['brier']:.8f} | "
                     f"{score['hidden_success']['calibrated']['brier']:.8f} |")
    lines += ["", "Controlled labels never enter aircraft training. Historical incumbents are not new gains.",
              "Selected-point Brier scores are adaptive diagnostics and do not compare calibration across methods.",
              "The no-mechanisms ablation also removes the aggregate hidden-success contribution; "
              "it does not isolate five separate mechanism heads from an otherwise identical aggregate-hidden model.",
              "Random selects the first admissible point in the common candidate generator; after a global "
              "point is rejected, this may be an elite-local proposal. It is not an independent uniform baseline.",
              "Full per-run outcomes, stop reasons, curves, support and source hashes are in analysis.json.", ""]
    return "\n".join(lines)


def analyze(folder, output, controlled_only=False):
    folder, output = folder.resolve(), output.resolve()
    report = load_json((folder/"report.json").read_bytes())
    audit = load_json((folder/"independent_audit.json").read_bytes())
    if report["status"] != "terminal" or audit["status"] != "passed":
        raise ValueError("Require terminal independently audited study")
    verify_sources(audit["input_sha256"])
    controlled_folder = (folder if controlled_only else Path(report["controlled_source"]))
    if not controlled_only:
        inherited = load_json((controlled_folder/"independent_audit.json").read_bytes())
        verify_sources(inherited["input_sha256"])
    plan = load_json((folder/"plan.json").read_bytes())
    verify_sources(plan["source_sha256"])
    static = static_metrics(controlled_folder, plan["physical"]["base_project"])
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "controlled_static": static,
        "controlled": compare_trials([trial_metrics(r) for r in report["controlled"]]),
        "source_folder": str(folder), "inherited_controlled_folder": str(controlled_folder),
        "analyst_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "Three seeds of one task and short warm-start runs do not establish general stable dominance.",
            "RA reserves a chronological calibration window; CBO/LSBO can train on all known observations.",
            "No-mechanisms removes aggregate hidden-success as well as separate mechanism contributions.",
            "Directed neighborhood counts are correlated support, not independent failure regions or prevalence.",
            "Empirical calibration and expected-risk accounting provide no safety confidence bound.",
            "Parallel wall-clock measurements share hardware with other runs and existing user applications.",
        ],
        "analysis_scope": "Predeclared descriptive metrics; no tuning or additional evaluations"}
    if not controlled_only:
        result["physical"] = compare_trials([trial_metrics(r) for r in report["physical"]])
        result["physical_acquisition"] = physical_acquisition(folder)
        result["mechanism_data"] = load_json((folder/"mechanism_data_report.json").read_bytes())
        result["accounting"] = {name: report[name] for name in
                                ("complete_new_mda_calls", "carried_mda_calls",
                                 "probe_calls_including_carried", "fresh_calibration_calls")}
        result["accounting"]["new_current_mda_calls"] = report.get(
            "new_current_mda_calls", report.get("new_v2_mda_calls"))
        if "call_cost_by_version" in report:
            result["accounting"]["call_cost_by_version"] = report["call_cost_by_version"]
    input_paths = [folder/"report.json", folder/"independent_audit.json",
                   controlled_folder/"report.json", controlled_folder/"independent_audit.json",
                   controlled_folder/"controlled_static_report.json",
                   controlled_folder/"controlled_static_history.json",
                   controlled_folder/"controlled_forecasts.jsonl", controlled_folder/"controlled_scored.jsonl"]
    result["input_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    output.mkdir(parents=True, exist_ok=False)
    save_new(output/"analysis.json", result)
    (output/"RESULTS.md").write_text(markdown(result), encoding="utf-8")
    figures(result, output)
    print("Analysis written:", output)
    for kind in ("controlled", "physical"):
        if kind in result:
            for row in result[kind]["aggregate"]:
                print(kind, row["method"], "calls", row["calls"], "feasible", row["feasible_calls"],
                      "mean improvement %", row["mean_improvement_percent"], row["statuses"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--controlled-only", action="store_true")
    args = parser.parse_args()
    analyze(args.folder, args.output, args.controlled_only)
