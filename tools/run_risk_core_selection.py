"""A whole 488-label development/held-out-seed study, with no physical MDA."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ra_lsbo.budget import digest
from ra_lsbo.controlled_sparse import SparseInteriorFiveModeProblem
from ra_lsbo.mechanism_system import FiveModeProblem, SystemSettings
from ra_lsbo.research_statistics import holm, ordered_endpoint, paired_effect, run_metrics
from ra_lsbo.risk_core.adapter import global_pool, observable_row
from ra_lsbo.risk_core.optimizer import CoreSelector, FAMILIES
from ra_lsbo.risk_core.online_calibration import OnlineResidualCalibrator
from ra_lsbo.system_experiment import OriginalLSBOWarmStart, save_new

BASE = (ROOT/"runs/compact_mda_20261021T081006Z_20eb608d/"
        "qualification_recovered_20261025/frozen_base")
TASKS = ("complete", "partial_sparse")
SCREEN = tuple(range(3101, 3105))
HELDOUT = tuple(range(4101, 4105))
METHODS = (*FAMILIES, "lsbo")
INITIAL = 4
POOL = 192
MAXIMUM = 488


def effective_seed(task, seed):
    return seed+(1000000 if task == "partial_sparse" else 0)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            raise ValueError("NaN cannot be serialized as valid experiment evidence")
        return {"__extended_real__": "positive_infinity" if value > 0 else "negative_infinity"}
    return value


def restore_extended(value):
    if isinstance(value, dict):
        if set(value) == {"__extended_real__"}:
            if value["__extended_real__"] == "positive_infinity":
                return float("inf")
            if value["__extended_real__"] == "negative_infinity":
                return -float("inf")
            raise ValueError("Unsupported extended real encoding")
        return {k: restore_extended(v) for k, v in value.items()}
    if isinstance(value, list):
        return [restore_extended(v) for v in value]
    return value


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [] if not path.exists() else [
        json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(clean(value), sort_keys=True, allow_nan=False)+"\n")
        stream.flush()


def problem_for(task):
    return FiveModeProblem(BASE) if task == "complete" else SparseInteriorFiveModeProblem(BASE)


def initial_candidates(problem, seed):
    return global_pool(problem, seed, INITIAL, set())


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=False)
    paths = [*sorted((ROOT/"ra_lsbo").rglob("*.py")), Path(__file__).resolve()]
    paths += sorted(BASE.rglob("*.py"))
    snapshots = folder/"source_snapshot"
    snapshots.mkdir()
    sources = {}
    for path in paths:
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        sources[str(path)] = sha
        target = snapshots/(sha+".source")
        if not target.exists():
            target.write_bytes(data)
    candidates = {}
    contracts = {}
    for task in TASKS:
        problem = problem_for(task)
        contracts[task] = problem.contract_id
        candidates[task] = {str(seed): initial_candidates(problem, effective_seed(task, seed))
                            for seed in (*SCREEN, *HELDOUT)}
    save_new(folder/"plan.json", {
        "schema": "risk_core_selection_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "exploratory method screening and held-out seeds on two known controlled equations",
        "maximum_new_truth_labels": MAXIMUM, "maximum_new_physical_mda_calls": 0,
        "accounting": "all shared initial labels charged once; all online labels charged per method",
        "screen_formula": "2 tasks * 4 seeds * (4 initial + 4 methods * 8 online) = 288",
        "heldout_formula": "2 tasks * 4 seeds * (4 initial + 3 methods * 7 online) = 200",
        "tasks": list(TASKS), "screen_seeds": list(SCREEN), "heldout_seeds": list(HELDOUT),
        "effective_seeds": {
            phase: {task: [effective_seed(task, seed) for seed in seeds] for task in TASKS}
            for phase, seeds in (("screen", SCREEN), ("heldout", HELDOUT))},
        "screen_methods": list(METHODS), "heldout_methods": ["selected_core", "selected_integrated", "lsbo"],
        "initial_count": INITIAL, "screen_horizon": 8, "heldout_horizon": 7, "pool_size": POOL,
        "contract_ids": contracts, "initial_candidates": candidates, "base_project": str(BASE),
        "observation_regimes": {
            "complete": "exact observed outputs, serial unobserved stages remain missing",
            "partial_sparse": "positive expensive margins replaced by verified g>0; truth sealed separately"},
        "common_information": "same observable record incl bounds; native LSBO consumes its existing supported fields",
        "controlled_sensor_note": "sign censoring is a constructed feedback regime, not new physical evidence",
        "cold_start": "balanced feasibility; BE-CBO upper-band discovery adaptation, not original reference",
        "screen_selection": [
            "minimum sum of within-task, within-seed feasible-first ordered endpoint losses vs all other methods",
            "minimum total restricted first-feasible cost", "minimum hidden failures",
            "fixed parsimony order mixed_eicb, calibrated_eicb, ensemble_boundary"],
        "novelty_note": "selection cannot prove novelty; closest-prior-work review and component effects remain necessary",
        "integration": {
            "selected_core": "selected family, hierarchical onehot, aggregate completion",
            "selected_integrated": "same selected family, structured latent representation, conditional stage survival",
            "ablation_limit": "integrated-vs-core tests the combined representation/stage extension, not each independently",
            "encoder": {"latent_dim": 4, "initial_epochs": 120, "update_epochs": 20},
        },
        "primary_heldout": "selected_integrated vs native LSBO, feasible-first ordered endpoint",
        "secondary_heldout": "selected_integrated vs selected_core, same ordered endpoint",
        "multiplicity": "Holm family of those two comparisons; two tasks fixed, no task-population inference",
        "promotion_gate": "both ordinal mean effects <=0; integrated discovery cost and hidden failures no worse than LSBO",
        "interpretation": "a pass is provisional development selection, not journal-level effectiveness or safety proof",
        "no_post_result_tuning_seed_extension_or_task_selection": True,
        "native_baseline": "unmodified frozen LSBO proposal/update; same labels and calls, original search defaults",
        "runtime_fairness": "no equal CPU-time claim; library and training effort differ",
        "native_early_stop": "preserve the original consecutive missing-objective stop; unused calls are not reassigned",
        "native_stop_estimand": "policy termination is absorbing: retain best observed design to B; undiscovered T=infinity; no invented evaluations",
        "python": sys.executable, "python_version": platform.python_version(),
        "packages": {n: importlib.metadata.version(n)
                     for n in ("numpy", "scipy", "torch", "gpytorch", "botorch", "scikit-learn")},
        "source_sha256": sources,
    })
    print(folder/"plan.json", flush=True)


def verify(folder):
    plan = read(folder/"plan.json")
    if (plan["schema"] != "risk_core_selection_v1"
            or plan["maximum_new_truth_labels"] != MAXIMUM
            or plan["maximum_new_physical_mda_calls"] != 0
            or plan["screen_seeds"] != list(SCREEN) or plan["heldout_seeds"] != list(HELDOUT)
            or plan["screen_methods"] != list(METHODS)):
        raise ValueError("Study scope changed")
    for name, sha in plan["source_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
            raise ValueError("Frozen source changed: "+name)
    if plan["python"] != sys.executable or plan["python_version"] != platform.python_version():
        raise ValueError("Frozen interpreter differs")
    for name, version in plan["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Frozen dependency differs")
    if (folder/"execution_started.json").exists():
        if (read(folder/"execution_started.json")["plan_sha256"]
                != hashlib.sha256((folder/"plan.json").read_bytes()).hexdigest()):
            raise ValueError("The plan changed after execution started")
    return plan


def truth_call(folder, problem, candidate, tag, evaluation):
    journal = lines(folder/"calls.jsonl")
    started = [e for e in journal if e["event"] == "started"]
    settled = {e["call_id"] for e in journal if e["event"] == "settled"}
    if len(started) >= MAXIMUM or any(e["call_id"] not in settled for e in started):
        raise RuntimeError("Whole study cap reached or unresolved call exists")
    call_id = len(started)+1
    append(folder/"calls.jsonl", {
        "event": "started", "call_id": call_id, "tag": tag,
        "candidate": candidate, "candidate_key": digest(candidate)})
    risk = problem.evaluate(candidate)
    if risk["feasible_label"] is None:
        raise RuntimeError("Unknown truth outcome; no retry or imputation")
    row = {"evaluation": evaluation, "candidate": candidate, "risk": risk,
           "evaluator_contract_sha256": problem.contract_id}
    append(folder/"calls.jsonl", {"event": "settled", "call_id": call_id,
                                 "tag": tag, "truth_digest": digest(row), "row": row})
    return row


def native_accept(native, observed, metadata):
    try:
        native.accept(observed, metadata)
    except RuntimeError as error:
        if str(error) != ("Repeated missing/failed objective returns; "
                          "stopped before wasting remaining budget"):
            raise
        return True
    return False


def study_metrics(initial, online, horizon, native_stopped=False):
    metrics = run_metrics(initial, online, horizon)
    if native_stopped:
        if not metrics["all_labels_resolved"]:
            raise ValueError("Unknown observations cannot be treated as policy termination")
        metrics["endpoint_best"] = metrics["observed_best"]
        if not metrics["first_feasible_event"]:
            metrics["restricted_first_feasible_cost"] = horizon
        metrics["policy_termination_absorbing"] = True
    else:
        metrics["policy_termination_absorbing"] = False
    return metrics


def run_trial(folder, plan, phase, task, seed, method, family, initial_truth):
    problem, output = problem_for(task), folder/phase/task/(method+"_"+str(seed))
    output.mkdir(parents=True, exist_ok=False)
    horizon = plan["screen_horizon" if phase == "screen" else "heldout_horizon"]
    censor = task == "partial_sparse"
    rows = [observable_row(r, censor) for r in initial_truth]
    online = []
    settings = SystemSettings(calls=horizon, failure_cap=horizon, gp_steps=25)
    random_seed = effective_seed(task, seed)
    native = OriginalLSBOWarmStart(
        problem, BASE, rows, random_seed, settings, output) if method == "lsbo" else None
    integrated = method == "selected_integrated"
    selector = None if native is not None else CoreSelector(
        family, problem, BASE, random_seed, latent=integrated, stages=integrated)
    if native is not None:
        save_new(output/"native_settings.json", native.settings)
    save_new(output/"manifest.json", {
        "phase": phase, "task": task, "seed": seed, "effective_seed": random_seed,
        "method": method, "family": family,
        "observable_initial_digest": digest(rows), "truth_initial_digest": digest(initial_truth)})
    stopped = False
    for iteration in range(horizon):
        pool = []
        if native is not None:
            proposal, metadata = native.propose()
            candidate = proposal["candidate"]
        else:
            pool = global_pool(problem, random_seed*10000+iteration, POOL,
                               {digest(r["candidate"]) for r in rows})
            candidate, metadata = selector.propose(rows, pool, iteration)
            proposal = {"candidate": candidate}
        append(output/"proposals.jsonl", {
            "iteration": iteration, "pool": pool, "proposal": proposal,
            "metadata": metadata, "observable_history_digest": digest(rows)})
        truth = truth_call(folder, problem, candidate,
                           [phase, task, seed, method, iteration], len(rows)+1)
        observed = observable_row(truth, censor)
        append(output/"observations.jsonl", {"truth": truth, "observed": observed})
        rows.append(observed)
        online.append(truth)
        if native is not None:
            stopped = native_accept(native, observed, metadata)
            if stopped:
                break
        else:
            selector.accept(observed)
    summary = {"phase": phase, "task": task, "seed": seed, "effective_seed": random_seed,
               "method": method, "family": family,
               "native_early_stop": stopped, "planned_horizon": horizon,
               **study_metrics(initial_truth, online, horizon, stopped)}
    save_new(output/"summary.json", summary)
    append(folder/"completed.jsonl", summary)
    print(phase, task, seed, method, "feasible", summary["online_feasible_count"],
          "first", summary["first_feasible_online_call"], flush=True)


def select_family(summaries):
    lookup = {(r["task"], r["seed"], r["method"]): r for r in summaries}
    scored = []
    for method in FAMILIES:
        loss, discovery, failures = 0, 0, 0
        for task in TASKS:
            for seed in SCREEN:
                row = lookup[task, seed, method]
                loss += sum(ordered_endpoint(row["endpoint_best"],
                                            lookup[task, seed, other]["endpoint_best"])
                            for other in METHODS if other != method)
                discovery += row["restricted_first_feasible_cost"]
                failures += row["hidden_failure_count"]
        priority = ("mixed_eicb", "calibrated_eicb", "ensemble_boundary").index(method)
        scored.append({"family": method, "ordered_loss": loss,
                       "restricted_cost_sum": discovery, "hidden_failures": failures,
                       "parsimony_priority": priority})
    scored.sort(key=lambda r: (r["ordered_loss"], r["restricted_cost_sum"],
                              r["hidden_failures"], r["parsimony_priority"]))
    return {"selected_family": scored[0]["family"], "scores": scored,
            "scope": "frozen internal development selection; not evidence of superiority"}


def execute(folder):
    torch.set_num_threads(1)
    plan = verify(folder)
    save_new(folder/"execution_started.json", {
        "utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": hashlib.sha256((folder/"plan.json").read_bytes()).hexdigest()})
    for phase, seeds, methods in (
            ("screen", SCREEN, METHODS),
            ("heldout", HELDOUT, ("selected_core", "selected_integrated", "lsbo"))):
        selected = None
        if phase == "heldout":
            selection = select_family(lines(folder/"completed.jsonl"))
            save_new(folder/"selection.json", selection)
            selected = selection["selected_family"]
        for task in TASKS:
            problem = problem_for(task)
            if problem.contract_id != plan["contract_ids"][task]:
                raise ValueError("Controlled task contract changed")
            for seed in seeds:
                verify(folder)
                initial_truth = [
                    truth_call(folder, problem, c, [phase, task, seed, "shared_initial"], i+1)
                    for i, c in enumerate(plan["initial_candidates"][task][str(seed)])]
                initial_folder = folder/phase/task/("initial_"+str(seed))
                initial_folder.mkdir(parents=True)
                save_new(initial_folder/"truth.json", initial_truth)
                for method in methods:
                    verify(folder)
                    family = method if phase == "screen" else selected
                    run_trial(folder, plan, phase, task, seed, method, family, initial_truth)
    save_new(folder/"execution_result.json", {
        "status": "complete",
        "new_truth_labels": sum(r["event"] == "settled" for r in lines(folder/"calls.jsonl")),
        "maximum_new_truth_labels": MAXIMUM, "new_physical_mda_calls": 0})


def complete_summaries(folder):
    summaries = lines(folder/"completed.jsonl")
    expected_runs = {
        (phase, task, seed, method)
        for phase, seeds, methods in (
            ("screen", SCREEN, METHODS),
            ("heldout", HELDOUT, ("selected_core", "selected_integrated", "lsbo")))
        for task in TASKS for seed in seeds for method in methods}
    actual_runs = [(r["phase"], r["task"], r["seed"], r["method"]) for r in summaries]
    if len(actual_runs) != len(expected_runs) or set(actual_runs) != expected_runs:
        raise ValueError("Planned runs missing, duplicated, or added")
    if len(summaries) != 56:
        raise ValueError("All 32 screen and 24 held-out runs must be retained")
    return summaries


def report(folder):
    verify(folder)
    summaries = complete_summaries(folder)
    audit_record = read(folder/"audit.json")
    actual_count = read(folder/"execution_result.json")["new_truth_labels"]
    linked_truth_index(lines(folder/"calls.jsonl"), actual_count)
    if (audit_record["status"] != "passed"
            or audit_record["charged_and_settled_labels"] != actual_count):
        raise ValueError("A complete trajectory audit is required before reporting")
    for relative, sha in audit_record["evidence_sha256"].items():
        if hashlib.sha256((folder/relative).read_bytes()).hexdigest() != sha:
            raise ValueError("Audited experiment evidence changed: "+relative)
    heldout = [r for r in summaries if r["phase"] == "heldout"]
    lookup = {(r["task"], r["seed"], r["method"]): r for r in heldout}
    comparisons = []
    for reference in ("lsbo", "selected_core"):
        effects = [
            ordered_endpoint(lookup[task, seed, "selected_integrated"]["endpoint_best"],
                             lookup[task, seed, reference]["endpoint_best"])
            for task in TASKS for seed in HELDOUT]
        effect = paired_effect(effects)
        rng = np.random.default_rng(20261025)
        strata = np.asarray(effects).reshape(len(TASKS), len(HELDOUT))
        bootstrap = np.mean(np.stack([
            values[rng.integers(len(values), size=(20000, len(values)))].mean(1)
            for values in strata]), axis=0)
        effect["paired_bootstrap_percentile_95"] = np.quantile(bootstrap, [.025, .975]).tolist()
        effect["bootstrap_unit"] = "independent paired effective seed, stratified within each fixed task"
        comparisons.append({"reference": reference,
                            "effect": effect,
                            "effect_note": "positive means integrated worse; two fixed tasks, paired seeds"})
    for item, p in zip(comparisons, holm([
            r["effect"]["exact_two_sided_sign_flip_p"] for r in comparisons])):
        item["holm_p"] = p
    aggregate = []
    for phase, methods, seeds in (
            ("screen", METHODS, SCREEN),
            ("heldout", ("selected_core", "selected_integrated", "lsbo"), HELDOUT)):
        for task in TASKS:
            for method in methods:
                rows = [r for r in summaries
                        if (r["phase"], r["task"], r["method"]) == (phase, task, method)]
                aggregate.append({
                    "phase": phase, "task": task, "method": method, "n_seeds": len(rows),
                    "feasible_calls": sum(r["online_feasible_count"] for r in rows),
                    "hidden_failures": sum(r["hidden_failure_count"] for r in rows),
                    "discovered": sum(r["first_feasible_event"] for r in rows),
                    "restricted_cost_mean": float(np.mean([
                        r["restricted_first_feasible_cost"] for r in rows])),
                    "endpoint_values": [r["endpoint_best"] for r in rows]})
    integrated = [r for r in heldout if r["method"] == "selected_integrated"]
    native = [r for r in heldout if r["method"] == "lsbo"]
    gate = (all(c["effect"]["mean_candidate_minus_reference"] <= 0 for c in comparisons)
            and sum(r["restricted_first_feasible_cost"] for r in integrated)
            <= sum(r["restricted_first_feasible_cost"] for r in native)
            and sum(r["hidden_failure_count"] for r in integrated)
            <= sum(r["hidden_failure_count"] for r in native))
    result = {"selection": read(folder/"selection.json"), "aggregate": aggregate,
              "heldout_comparisons": comparisons, "provisional_integration_gate_passed": gate,
              "publication_ready": False, "new_truth_labels": actual_count,
              "new_physical_mda_calls": 0,
              "limitations": [
                  "three literature-inspired adaptations, not all original published implementations",
                  "four paired seeds per fixed task and short horizon",
                  "held-out seeds, not independent engineering tasks",
                  "representation and stage extension evaluated jointly",
                  "numeric residual calibration on observed subsets has no marginal safety guarantee",
                  "native LSBO cannot fit interval evidence through its original numeric API",
                  "no equal CPU-time claim or inherited convergence/safety theorem"]}
    save_new(folder/"result.json", result)
    text = ["# Independent risk-core method selection", "",
            "Exploratory development and held-out seeds; not publication confirmation.", "",
            "Selected family: "+result["selection"]["selected_family"],
            "Provisional integration gate: "+str(gate),
            str(actual_count)+" new controlled labels; 0 new physical MDA calls.", "",
            "| Phase | Task | Method | Feasible | Hidden failures | Discovery runs | Restricted cost |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in aggregate:
        text.append("| {phase} | {task} | {method} | {feasible_calls} | {hidden_failures} | "
                    "{discovered} | {restricted_cost_mean:.3f} |".format(**row))
    text.extend(["", "## Limitations", ""]+["- "+s for s in result["limitations"]])
    (folder/"report.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    print("Selected", result["selection"]["selected_family"], "integration gate", gate, flush=True)


def linked_truth_index(events, expected_count):
    if len(events) != 2*expected_count:
        raise ValueError("Every charged truth call needs one start and one settlement")
    index = {}
    for i in range(expected_count):
        start, settled = events[2*i:2*i+2]
        if (start["event"] != "started" or settled["event"] != "settled"
                or start["call_id"] != i+1 or settled["call_id"] != i+1
                or start["tag"] != settled["tag"]
                or start["candidate"] != settled["row"]["candidate"]
                or digest(settled["row"]) != settled["truth_digest"]):
            raise ValueError("Charged truth event identity/order mismatch")
        key = tuple(settled["tag"]), settled["row"]["evaluation"]
        if key in index:
            raise ValueError("Duplicate charged truth identity")
        index[key] = settled["row"]
    return index


def assert_linked_truth(index, tag, row):
    if index.get((tuple(tag), row["evaluation"])) != row:
        raise ValueError("Saved outcome differs from its charged settled truth")


def audit(folder):
    torch.set_num_threads(1)
    plan = verify(folder)
    events = lines(folder/"calls.jsonl")
    actual_count = read(folder/"execution_result.json")["new_truth_labels"]
    if not 0 < actual_count <= MAXIMUM:
        raise ValueError("Invalid actual study accounting")
    truth_index = linked_truth_index(events, actual_count)
    summaries = complete_summaries(folder)
    expected_selection = select_family([r for r in summaries if r["phase"] == "screen"])
    if expected_selection != read(folder/"selection.json"):
        raise ValueError("Selected method differs from the frozen screen-only rule")
    checked, native_checked, interval_records = 0, 0, 0
    evidence_paths = {folder/name for name in (
        "plan.json", "calls.jsonl", "completed.jsonl", "selection.json", "execution_result.json")}
    for summary in summaries:
        phase, task, seed, method = [summary[n] for n in ("phase", "task", "seed", "method")]
        problem = problem_for(task)
        output = folder/phase/task/(method+"_"+str(seed))
        if read(output/"summary.json") != summary:
            raise ValueError("Run summary differs from the completion journal")
        evidence_paths.update(output/name for name in (
            "summary.json", "proposals.jsonl", "observations.jsonl", "manifest.json"))
        evidence_paths.add(folder/phase/task/("initial_"+str(seed))/"truth.json")
        initial_truth = read(folder/phase/task/("initial_"+str(seed))/"truth.json")
        if [r["candidate"] for r in initial_truth] != plan["initial_candidates"][task][str(seed)]:
            raise ValueError("Shared initial designs differ from the frozen plan")
        horizon = plan["screen_horizon" if phase == "screen" else "heldout_horizon"]
        if (summary["planned_horizon"] != horizon
                or not 0 < summary["online_calls"] <= horizon
                or (summary["online_calls"] < horizon and not summary["native_early_stop"])
                or (summary["native_early_stop"] and method != "lsbo")):
            raise ValueError("Method horizon differs from the frozen plan")
        for row in initial_truth:
            assert_linked_truth(truth_index, [phase, task, seed, "shared_initial"], row)
        rows = [observable_row(r, task == "partial_sparse") for r in initial_truth]
        settings = SystemSettings(calls=horizon, failure_cap=horizon, gp_steps=25)
        random_seed = effective_seed(task, seed)
        if method == "lsbo":
            native = OriginalLSBOWarmStart(
                problem, BASE, rows, random_seed, settings, output/"replay")
            selector = None
        else:
            native = None
            selector = CoreSelector(
                summary["family"], problem, BASE, random_seed,
                latent=method == "selected_integrated", stages=method == "selected_integrated")
        outcomes, proposals = lines(output/"observations.jsonl"), lines(output/"proposals.jsonl")
        if len(outcomes) != summary["online_calls"] or len(proposals) != len(outcomes):
            raise ValueError("Missing run trajectory")
        for event, proposal in zip(outcomes, proposals):
            assert_linked_truth(
                truth_index, [phase, task, seed, method, proposal["iteration"]], event["truth"])
            if digest(rows) != proposal["observable_history_digest"]:
                raise ValueError("Proposal history includes future or modified data")
            if observable_row(event["truth"], task == "partial_sparse") != event["observed"]:
                raise ValueError("Observation censoring differs")
            for state in proposal["metadata"].get("calibration_state_before", {}).values():
                OnlineResidualCalibrator.from_state_dict(restore_extended(state))
            interval_records += len(event["observed"]["risk"]["evidence"]["verified_constraint_bounds"])
            if native is not None:
                candidate, metadata = native.propose()
                if candidate != proposal["proposal"] or clean(metadata) != proposal["metadata"]:
                    raise ValueError("Native proposal replay differs")
                stopped = native_accept(native, event["observed"], metadata)
                if stopped != (summary["native_early_stop"] and event is outcomes[-1]):
                    raise ValueError("Native stopping replay differs")
                native_checked += 1
            else:
                expected_pool = global_pool(
                    problem, random_seed*10000+proposal["iteration"], POOL,
                    {digest(r["candidate"]) for r in rows})
                if expected_pool != proposal["pool"]:
                    raise ValueError("Saved pool differs from frozen global sampling")
                candidate, metadata = selector.propose(rows, proposal["pool"], proposal["iteration"])
                if candidate != proposal["proposal"]["candidate"] or clean(metadata) != proposal["metadata"]:
                    raise ValueError("Independent core full proposal replay differs")
                selector.accept(event["observed"])
                checked += 1
            rows.append(event["observed"])
        metrics = study_metrics(
            initial_truth, [e["truth"] for e in outcomes], horizon, summary["native_early_stop"])
        if any(summary[key] != value for key, value in metrics.items()):
            raise ValueError("Saved summary differs from the charged truth trajectory")
    save_new(folder/"audit.json", {
        "status": "passed", "charged_and_settled_labels": actual_count,
        "full_core_proposal_replays": checked, "native_proposal_replays": native_checked,
        "interval_output_records": interval_records, "additional_truth_evaluations": 0,
        "additional_physical_mda_calls": 0, "frozen_source_files": len(plan["source_sha256"]),
        "evidence_sha256": {
            str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(evidence_paths)}})
    print("Audit passed", checked, native_checked, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "report", "audit"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    {"prepare": prepare, "execute": execute, "report": report, "audit": audit}[
        args.command](args.output.resolve())
