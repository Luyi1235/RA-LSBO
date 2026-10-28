"""Small representation-by-core screen; no physical evaluator may run."""

import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ra_lsbo.budget import digest
from ra_lsbo.controlled_sparse import SparseInteriorFiveModeProblem
from ra_lsbo.mechanism_system import SystemSettings
from ra_lsbo.research_statistics import ordered_endpoint, paired_effect
from ra_lsbo.risk_core.adapter import global_pool, observable_row
from ra_lsbo.risk_core.optimizer import CoreSelector, FAMILIES
from ra_lsbo.system_experiment import OriginalLSBOWarmStart, save_new
from ra_lsbo.x57_live import X57LiveProblem
from tools.run_risk_core_selection import (
    BASE, append, clean, lines, native_accept, read, study_metrics)

SCREEN_SEEDS = (6201, 6202)
HELDOUT_SEEDS = (7201, 7202)
INITIAL, POOL, MAXIMUM = 16, 192, 376
SCREEN_HORIZON, HELDOUT_HORIZON = 12, 24
ARMS = tuple(family+suffix for family in FAMILIES
             for suffix in ("__onehot", "__latent"))+("lsbo",)
PARSIMONY = ("mixed_eicb", "calibrated_eicb", "ensemble_boundary")
PHYSICAL = ROOT/"runs/compact_mda_20261021T081006Z_20eb608d"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_parts(method):
    if method == "lsbo":
        return None, False
    family, representation = method.rsplit("__", 1)
    if family not in FAMILIES or representation not in ("onehot", "latent"):
        raise ValueError("Unknown fixed screening arm")
    return family, representation == "latent"


def select_family(summaries):
    lookup = {(row["seed"], row["method"]): row for row in summaries}
    scores = []
    for family in FAMILIES:
        latent_loss, gain_loss, discovery, failures = 0, 0, 0, 0
        for seed in SCREEN_SEEDS:
            latent = lookup[seed, family+"__latent"]
            control = lookup[seed, family+"__onehot"]
            latent_loss += sum(ordered_endpoint(
                latent["endpoint_best"], lookup[seed, other+"__latent"]["endpoint_best"])
                for other in FAMILIES if other != family)
            gain_loss += ordered_endpoint(latent["endpoint_best"], control["endpoint_best"])
            discovery += latent["restricted_first_feasible_cost"]
            failures += latent["hidden_failure_count"]
        scores.append({"family": family, "latent_ordered_loss": latent_loss,
                       "latent_vs_own_onehot_loss": gain_loss,
                       "restricted_cost_sum": discovery, "hidden_failures": failures,
                       "parsimony_priority": PARSIMONY.index(family)})
    scores.sort(key=lambda row: tuple(row[name] for name in (
        "latent_ordered_loss", "latent_vs_own_onehot_loss", "restricted_cost_sum",
        "hidden_failures", "parsimony_priority")))
    return {"selected_family": scores[0]["family"], "scores": scores,
            "basis": "latent combinations first; screen seeds only; no stage-product change"}


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=False)
    problem = SparseInteriorFiveModeProblem(BASE)
    files = [*sorted((ROOT/"ra_lsbo").rglob("*.py")), *sorted(BASE.rglob("*.py")),
             Path(__file__).resolve(), ROOT/"tools/run_risk_core_selection.py",
             ROOT/"docs/AIRCRAFT_MAINLINE_20261025.md", PHYSICAL/"config_55.json",
             PHYSICAL/"qualification_recovered_20261025/observations.jsonl"]
    source_hashes = {str(path): sha(path) for path in files}
    snapshot = folder/"source_snapshot"
    snapshot.mkdir()
    for path, identity in source_hashes.items():
        target = snapshot/(identity+".source")
        if not target.exists():
            target.write_bytes(Path(path).read_bytes())
    save_new(folder/"plan.json", {
        "schema": "latent_core_screen_v1", "maximum_new_truth_labels": MAXIMUM,
        "maximum_new_physical_mda_calls": 0, "physical_workers": 0,
        "screen_seeds": list(SCREEN_SEEDS), "heldout_seeds": list(HELDOUT_SEEDS),
        "screen_arms": list(ARMS), "initial_count": INITIAL, "candidate_pool": POOL,
        "screen_horizon": SCREEN_HORIZON, "heldout_horizon": HELDOUT_HORIZON,
        "budget_formula": "2*(16+7*12)+2*(16+3*24)=376; no reallocation after native stop",
        "initial_candidates": {str(seed): global_pool(problem, seed, INITIAL, set())
                               for seed in (*SCREEN_SEEDS, *HELDOUT_SEEDS)},
        "task_contract": problem.contract_id, "conditional_stage_product": False,
        "core_hyperparameters": "existing frozen CoreSelector defaults for every family",
        "latent_settings": {"latent_dim": 4, "initial_epochs": 120},
        "feedback": "existing controlled positive-margin sign sensor; identical for all methods",
        "selection": "latent ordinal loss, latent gain, discovery, failures, fixed parsimony",
        "confirmation": "selected latent vs same core onehot vs original native LSBO",
        "interpretation": "small development screen; neither aircraft optimization nor final superiority",
        "source_sha256": source_hashes, "python": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "torch", "gpytorch", "botorch", "scikit-learn")}})
    print(folder/"plan.json", flush=True)


def verify(folder):
    plan = read(folder/"plan.json")
    if (plan["schema"] != "latent_core_screen_v1"
            or plan["maximum_new_truth_labels"] != MAXIMUM
            or plan["maximum_new_physical_mda_calls"] != 0
            or plan["screen_arms"] != list(ARMS)
            or plan["screen_seeds"] != list(SCREEN_SEEDS)
            or plan["heldout_seeds"] != list(HELDOUT_SEEDS)):
        raise ValueError("Frozen screening scope changed")
    for path, identity in plan["source_sha256"].items():
        if sha(path) != identity:
            raise ValueError("Frozen source changed: "+path)
    if plan["python"] != sys.executable:
        raise ValueError("Frozen interpreter differs")
    for name, version in plan["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Frozen dependency differs: "+name)
    started = folder/"execution_started.json"
    if started.exists() and read(started)["plan_sha256"] != sha(folder/"plan.json"):
        raise ValueError("Plan changed after execution")
    return plan


def charge(folder, problem, candidate, tag, evaluation):
    events = lines(folder/"calls.jsonl")
    starts = [event for event in events if event["event"] == "started"]
    settled = {event["call_id"] for event in events if event["event"] == "settled"}
    if len(starts) >= MAXIMUM or any(event["call_id"] not in settled for event in starts):
        raise RuntimeError("Study cap or unresolved truth evaluation")
    call_id = len(starts)+1
    append(folder/"calls.jsonl", {"event": "started", "call_id": call_id, "tag": tag,
                                 "candidate": candidate})
    risk = problem.evaluate(candidate)
    if risk["feasible_label"] is None:
        raise RuntimeError("Unknown controlled label; no automatic retry")
    row = {"evaluation": evaluation, "candidate": candidate, "risk": risk,
           "evaluator_contract_sha256": problem.contract_id}
    append(folder/"calls.jsonl", {"event": "settled", "call_id": call_id, "tag": tag,
                                 "row": row, "truth_digest": digest(row)})
    return row


def run_trial(folder, phase, seed, method, initial):
    problem = SparseInteriorFiveModeProblem(BASE)
    output = folder/phase/(method+"_"+str(seed))
    output.mkdir(parents=True, exist_ok=False)
    horizon = SCREEN_HORIZON if phase == "screen" else HELDOUT_HORIZON
    family, latent = arm_parts(method)
    rows = [observable_row(row, True) for row in initial]
    settings = SystemSettings(calls=horizon, failure_cap=horizon, gp_steps=25)
    native = (OriginalLSBOWarmStart(problem, BASE, rows, seed, settings, output)
              if method == "lsbo" else None)
    selector = (None if native is not None else
                CoreSelector(family, problem, BASE, seed, latent=latent, stages=False))
    online, stopped = [], False
    for iteration in range(horizon):
        if native is None:
            pool = global_pool(problem, seed*10000+iteration, POOL,
                               {digest(row["candidate"]) for row in rows})
            candidate, metadata = selector.propose(rows, pool, iteration)
            proposal = {"candidate": candidate}
        else:
            pool = []
            proposal, metadata = native.propose()
            candidate = proposal["candidate"]
        append(output/"proposals.jsonl", {
            "iteration": iteration, "pool": pool, "proposal": proposal,
            "metadata": metadata, "observable_history_digest": digest(rows)})
        truth = charge(folder, problem, candidate, [phase, seed, method, iteration], len(rows)+1)
        observed = observable_row(truth, True)
        append(output/"observations.jsonl", {"truth": truth, "observed": observed})
        online.append(truth)
        rows.append(observed)
        if native is not None:
            stopped = native_accept(native, observed, metadata)
            if stopped:
                break
        else:
            selector.accept(observed)
    summary = {"phase": phase, "seed": seed, "method": method, "family": family,
               "latent": latent, "stages": False, "native_early_stop": stopped,
               "planned_horizon": horizon, **study_metrics(initial, online, horizon, stopped)}
    save_new(output/"summary.json", summary)
    append(folder/"completed.jsonl", summary)
    print(phase, seed, method, "feasible", summary["online_feasible_count"],
          "failed", summary["hidden_failure_count"], flush=True)


def aircraft_diagnostic(folder):
    source = PHYSICAL/"qualification_recovered_20261025/observations.jsonl"
    rows = [row for row in lines(source) if row["distance_nmi"] == 55]
    if len(rows) != 30 or any(row["risk"]["feasible_label"] is None for row in rows):
        raise ValueError("Expected 30 settled aircraft records, no unknown labels")
    problem = X57LiveProblem(BASE, PHYSICAL/"config_55.json", folder/"unused_workers", False)
    if any(row["evaluator_contract_sha256"] != problem.contract_id for row in rows):
        raise ValueError("Aircraft physical identity differs")
    training, queries = rows[:12], rows[12:]
    results = []
    for family in FAMILIES:
        for latent in (False, True):
            selector = CoreSelector(family, problem, BASE, 8201, latent=latent, stages=False)
            _, metadata = selector.propose(training, [row["candidate"] for row in queries], 0)
            labels = np.asarray([row["risk"]["hidden_success"] for row in queries], float)
            prediction = np.asarray(metadata["raw_completion"])
            feasible = np.asarray([row["risk"]["feasible_label"] for row in queries], float)
            results.append({"family": family, "latent": latent,
                            "completion_brier": float(np.mean((prediction-labels)**2)),
                            "feasibility_brier": float(np.mean((
                                np.asarray(metadata["raw_feasibility"])-feasible)**2)),
                            "completion_probability": prediction.tolist(),
                            "metadata": metadata})
    save_new(folder/"aircraft_diagnostic.json", clean({
        "new_physical_mda_calls": 0, "train": 12, "test": 18,
        "source_sha256": sha(source), "contract": problem.contract_id,
        "training_digest": digest(training), "query_digest": digest(queries),
        "results": results, "used_in_selection": False,
        "limitation": "already inspected biased development panel; all 30 infeasible; no optimization proof"}))


def execute(folder):
    torch.set_num_threads(1)
    plan = verify(folder)
    save_new(folder/"execution_started.json", {"plan_sha256": sha(folder/"plan.json")})
    problem = SparseInteriorFiveModeProblem(BASE)
    for phase, seeds in (("screen", SCREEN_SEEDS), ("heldout", HELDOUT_SEEDS)):
        methods = ARMS
        if phase == "heldout":
            selection = select_family(lines(folder/"completed.jsonl"))
            save_new(folder/"selection.json", selection)
            save_new(folder/"selection_frozen.json", {
                "selection_sha256": sha(folder/"selection.json"),
                "preceding_settled_labels": sum(event["event"] == "settled"
                    for event in lines(folder/"calls.jsonl")),
                "heldout_calls_at_selection": sum(event["tag"][0] == "heldout"
                    for event in lines(folder/"calls.jsonl"))})
            selected = selection["selected_family"]
            methods = (selected+"__onehot", selected+"__latent", "lsbo")
            print("Frozen selection", selected, flush=True)
        for seed in seeds:
            initial = [charge(folder, problem, candidate, [phase, seed, "initial", index], index+1)
                       for index, candidate in enumerate(plan["initial_candidates"][str(seed)])]
            destination = folder/phase/("initial_"+str(seed)+".json")
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_new(destination, initial)
            for method in methods:
                run_trial(folder, phase, seed, method, initial)
    aircraft_diagnostic(folder)
    verify(folder)
    actual = sum(event["event"] == "started" for event in lines(folder/"calls.jsonl"))
    save_new(folder/"execution_result.json", {"status": "complete", "new_truth_labels": actual,
                                            "maximum_new_truth_labels": MAXIMUM,
                                            "new_physical_mda_calls": 0})
    report(folder)


def report(folder):
    verify(folder)
    summaries = lines(folder/"completed.jsonl")
    selection = read(folder/"selection.json")
    if selection != select_family([row for row in summaries if row["phase"] == "screen"]):
        raise ValueError("Saved selection differs from screen-only ranking")
    family = selection["selected_family"]
    heldout = {(row["seed"], row["method"]): row for row in summaries if row["phase"] == "heldout"}
    effects = {}
    for reference in (family+"__onehot", "lsbo"):
        effects[reference] = paired_effect([ordered_endpoint(
            heldout[seed, family+"__latent"]["endpoint_best"],
            heldout[seed, reference]["endpoint_best"]) for seed in HELDOUT_SEEDS])
    latent_rows = [heldout[seed, family+"__latent"] for seed in HELDOUT_SEEDS]
    baseline = [heldout[seed, "lsbo"] for seed in HELDOUT_SEEDS]
    gate = (any(row["endpoint_best"] is not None for row in latent_rows)
            and all(effect["mean_candidate_minus_reference"] <= 0 for effect in effects.values())
            and sum(row["restricted_first_feasible_cost"] for row in latent_rows)
            <= sum(row["restricted_first_feasible_cost"] for row in baseline)
            and sum(row["hidden_failure_count"] for row in latent_rows)
            <= sum(row["hidden_failure_count"] for row in baseline))
    aggregates = []
    for phase in ("screen", "heldout"):
        for method in sorted({row["method"] for row in summaries if row["phase"] == phase}):
            group = [row for row in summaries if row["phase"] == phase and row["method"] == method]
            aggregates.append({"phase": phase, "method": method, "runs": len(group),
                               "online_calls": sum(row["online_calls"] for row in group),
                               "feasible": sum(row["online_feasible_count"] for row in group),
                               "hidden_failures": sum(row["hidden_failure_count"] for row in group),
                               "final_best": [row["endpoint_best"] for row in group],
                               "initial_feasible": [row["initial_feasible"] for row in group]})
    save_new(folder/"result.json", {"selection": selection, "development_gate_passed": gate,
                                   "heldout_ordinal_effects": effects, "aggregates": aggregates,
                                   "new_physical_mda_calls": 0,
                                   "interpretation": "provisional candidate only; two heldout seeds; one controlled task"})
    print("Development gate", gate, flush=True)


def audit(folder):
    torch.set_num_threads(1)
    plan = verify(folder)
    events = lines(folder/"calls.jsonl")
    starts = {event["call_id"]: event for event in events if event["event"] == "started"}
    settles = {event["call_id"]: event for event in events if event["event"] == "settled"}
    if len(starts) != len(settles) or set(starts) != set(settles) or len(starts) > MAXIMUM:
        raise ValueError("Unsettled or over-budget calls")
    index = {}
    for call_id, event in settles.items():
        prior = starts[call_id]
        if (prior["tag"] != event["tag"] or prior["candidate"] != event["row"]["candidate"]
                or event["truth_digest"] != digest(event["row"])):
            raise ValueError("Broken charged truth linkage")
        if tuple(event["tag"]) in index:
            raise ValueError("Duplicate charged tag")
        index[tuple(event["tag"])] = event["row"]
    problem = SparseInteriorFiveModeProblem(BASE)
    checked, native_checked = 0, 0
    summaries = lines(folder/"completed.jsonl")
    if len(summaries) != 20:
        raise ValueError("Missing planned runs")
    for summary in summaries:
        phase, seed, method = (summary[name] for name in ("phase", "seed", "method"))
        output = folder/phase/(method+"_"+str(seed))
        initial = read(folder/phase/("initial_"+str(seed)+".json"))
        if [row["candidate"] for row in initial] != plan["initial_candidates"][str(seed)]:
            raise ValueError("Shared initial designs changed")
        for initial_index, row in enumerate(initial):
            if row != index[phase, seed, "initial", initial_index]:
                raise ValueError("Uncharged initial row")
        rows = [observable_row(row, True) for row in initial]
        family, latent = arm_parts(method)
        native = (OriginalLSBOWarmStart(
            problem, BASE, rows, seed, SystemSettings(calls=summary["planned_horizon"],
            failure_cap=summary["planned_horizon"], gp_steps=25), output/"audit_native")
            if method == "lsbo" else None)
        selector = None if native is not None else CoreSelector(
            family, problem, BASE, seed, latent=latent, stages=False)
        proposals, outcomes = lines(output/"proposals.jsonl"), lines(output/"observations.jsonl")
        if len(proposals) != len(outcomes) or len(outcomes) != summary["online_calls"]:
            raise ValueError("Missing trajectory")
        for proposal, outcome in zip(proposals, outcomes):
            iteration = proposal["iteration"]
            if (digest(rows) != proposal["observable_history_digest"]
                    or outcome["truth"] != index[phase, seed, method, iteration]
                    or outcome["observed"] != observable_row(outcome["truth"], True)):
                raise ValueError("Future leakage or changed observation")
            if native is not None:
                predicted, metadata = native.propose()
                if predicted != proposal["proposal"] or clean(metadata) != proposal["metadata"]:
                    raise ValueError("Native full proposal replay differs")
                stopped = native_accept(native, outcome["observed"], metadata)
                if stopped != (summary["native_early_stop"] and iteration == len(outcomes)-1):
                    raise ValueError("Native termination differs")
                native_checked += 1
            else:
                pool = global_pool(problem, seed*10000+iteration, POOL,
                                   {digest(row["candidate"]) for row in rows})
                candidate, metadata = selector.propose(rows, pool, iteration)
                if (pool != proposal["pool"] or candidate != proposal["proposal"]["candidate"]
                        or clean(metadata) != proposal["metadata"]):
                    raise ValueError("Core full proposal replay differs")
                selector.accept(outcome["observed"])
                checked += 1
            rows.append(outcome["observed"])
        expected = study_metrics(initial, [outcome["truth"] for outcome in outcomes],
                                 summary["planned_horizon"], summary["native_early_stop"])
        if any(summary[name] != value for name, value in expected.items()):
            raise ValueError("Summary differs from trajectory")
        print("Audited", phase, seed, method, flush=True)
    if read(folder/"selection.json") != select_family(
            [row for row in summaries if row["phase"] == "screen"]):
        raise ValueError("Selection differs")
    frozen_selection = read(folder/"selection_frozen.json")
    if (frozen_selection["selection_sha256"] != sha(folder/"selection.json")
            or frozen_selection["heldout_calls_at_selection"] != 0
            or frozen_selection["preceding_settled_labels"] != sum(
                event["tag"][0] == "screen" for event in settles.values())):
        raise ValueError("Selection was not frozen before confirmation")
    save_new(folder/"audit.json", {"status": "passed", "settled_labels": len(settles),
                                  "full_core_proposal_replays": checked,
                                  "full_native_proposal_replays": native_checked,
                                  "new_truth_evaluations": 0, "new_physical_mda_calls": 0,
                                  "frozen_source_files": len(plan["source_sha256"])})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "audit"))
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    {"prepare": prepare, "execute": execute, "audit": audit}[
        arguments.command](arguments.output.resolve())
