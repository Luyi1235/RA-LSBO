"""Prospective prediction-only experiment on the repository's reduced aircraft."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
from random import Random
import subprocess
import sys
import time
import traceback

from aircraft_risk_energy import energy_pairs, energy_receipt, energy_scores
from aircraft_risk_local_search import check_augmentation, local_anchor, local_candidates
from aircraft_risk_protocol import ARMS, LOCAL_ARMS, OPT_ARMS, PHASE_ARMS, PROFILES, plan
from ra_lsbo.budget import digest
from risk_learning.core import binary_metrics

ROOT = Path(__file__).resolve().parent
ENERGY_ARMS = {**OPT_ARMS, **LOCAL_ARMS}
ARM_RULES = {**ARMS, **PHASE_ARMS, **ENERGY_ARMS}


def source_commit(expected, *, clean=False):
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if sha != expected or len(expected) != 40:
        raise ValueError("The full expected SHA must match the checked-out code")
    if clean and subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("Run from a clean research worktree with external evidence output")
    return sha


def read(path):
    return json.loads(path.read_bytes())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_label(record, head="joint"):
    status = record["observation"]["status"]
    diagnostics = record["diagnostics"]
    if status in ("unknown", "infrastructure_error") or diagnostics.get("numerical_failure"):
        return None

    def sign(value):
        if value["kind"] == "exact":
            return value["value"] <= 0
        if value["kind"] == "interval":
            if value["upper"] is not None and value["upper"] <= 0:
                return True
            if value["lower"] is not None and (
                value["lower"] > 0 or value["lower"] == 0 and not value["lower_inclusive"]
            ):
                return False
        return None

    constraints = dict(record["observation"]["constraints"])
    if head != "joint":
        return sign(constraints[head])
    labels = [sign(value) for value in constraints.values()]
    if False in labels or diagnostics.get("certified_physical_infeasible") is True:
        return False
    return True if labels and all(label is True for label in labels) else None


def select(candidates, scores):
    if not candidates or len(candidates) != len(scores):
        raise ValueError("Candidate and score counts differ")
    return max(range(len(candidates)), key=lambda i: (scores[i], digest(candidates[i])))


def run(profile, expected, output):
    import numpy as np
    from scipy.special import ndtr
    import torch

    from aircraft_risk_acquisition import brier_scores
    from ra_lsbo.experiments.core.protocol import BASE
    from ra_lsbo.experiments.core.problems import AircraftCase
    from ra_lsbo.experiments.core.runner import Ledger, append, plain, save
    from ra_lsbo.experiments.simplified_aircraft.corpus_optimizer import encoder_settings
    from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import StableGroupedHead
    from ra_lsbo.features import load_base
    from ra_lsbo.partial_bo.models import PrefixFeatures
    from ra_lsbo.partial_bo.problems import candidate_pool
    from ra_lsbo.risk_core.observations import OutputObservation
    from ra_lsbo.risk_core.optimizer import expected_improvement
    from ra_lsbo.structure_corpus import CorpusRepresentation, feature_snapshot, native_observations

    sha = source_commit(expected, clean=True)
    if ROOT == output or ROOT in output.parents:
        raise ValueError("Use a new evidence directory outside the code worktree")
    spec = plan(profile)
    output.mkdir(parents=True)
    spec["source_sha"] = sha
    spec["runtime"] = {"python": sys.version, "platform": platform.platform(),
                       **{p: version(p) for p in ("numpy", "scipy", "torch", "gpytorch", "botorch")}}
    save(output/"plan.json", spec)
    ledger = Ledger(output, sum(spec["budget"].values()))
    start = time.perf_counter()
    try:
        torch.set_num_threads(1)
        problem = AircraftCase(BASE, "sparse", integration_step=spec["step_seconds"])
        heads = tuple(name for name in problem.required_constraints
                      if name not in problem.analytical_constraints)
        save(output/"contract.json", {"id": problem.contract_id, "settings": problem.settings,
                                      "heads": heads, "joint": "product_of_marginal_probabilities"})
        optimizing = "objective_update_every" in spec
        for seed in spec["seeds"]:
            unit = output/str(seed)
            seen = set()
            roster = {}
            for role in ("initial", "pool", "reference", "heldout"):
                rows, rejected = (candidate_pool(problem, seed+len(roster)+1, spec[role], seen)
                                  if spec[role] else ([], 0))
                roster[role] = {"designs": rows, "screen_rejections": rejected}
                seen.update(digest(row) for row in rows)
            save(unit/"roster.json", roster)
            initial, pool, reference, heldout = (roster[role]["designs"]
                                                  for role in ("initial", "pool", "reference", "heldout"))
            anchors = [ledger.evaluate(problem, row, f"{seed}/initial/{i}", "initial")
                       for i, row in enumerate(initial)]
            explicit = PrefixFeatures(problem, BASE, anchors, seed, latent=False)
            feature_ids = {"explicit": explicit.identity, "anchor_count": len(anchors)}
            if not optimizing and any(ARM_RULES[arm][1] for arm in spec["arms"]):
                modules, _ = load_base(BASE)
                session = CorpusRepresentation(explicit.space, seed, count=2000, settings=encoder_settings())
                session.fit(native_observations(anchors, modules, explicit.space), incremental=False)
                latent = feature_snapshot(problem, BASE, anchors, session)
                feature_ids.update(latent=latent.identity, encoder_state=latent.state_hash)
            save(unit/"features.json", feature_ids)
            switch = spec.get("coverage_switch")
            shared_arm = spec.get("shared_coverage_arm")
            branch_arms = PHASE_ARMS
            if "local_switch" in spec:
                switch, shared_arm = spec["local_switch"], spec["shared_search_arm"]
                branch_arms = LOCAL_ARMS
            shared_history, shared_remaining, shared_rows = [], [], []
            shared_objective_features = shared_session = None
            global_addition = None
            for arm in spec["arms"]:
                rule, uses_latent = ARM_RULES[arm]
                features = latent if uses_latent and not optimizing else explicit
                if optimizing and uses_latent and arm not in LOCAL_ARMS:
                    modules, _ = load_base(BASE)
                    session = CorpusRepresentation(explicit.space, seed, count=2000, settings=encoder_settings())
                objective_features = explicit
                journal = unit/f"{arm}.jsonl"
                history, remaining = list(anchors), list(pool)
                ref_x = features.transform(reference) if reference else None
                heldout_x = features.transform(heldout)
                first_n = spec["initial"]
                if arm in branch_arms:
                    if len(shared_history) != switch:
                        raise RuntimeError("Shared coverage prefix did not reach the switch")
                    history, remaining = list(shared_history), list(shared_remaining)
                    first_n = switch
                    if arm in LOCAL_ARMS:
                        objective_features = shared_objective_features
                        session = shared_session
                    for row in shared_rows:
                        append(journal, row)
                random = Random(seed+11)
                for n in range(first_n, spec["total"]+1):
                    if n == spec.get("local_switch"):
                        anchor = local_anchor(plain([r.to_dict() for r in history]), checked_label)
                        local = arm in LOCAL_ARMS and anchor is not None
                        if global_addition is None:
                            global_addition, _ = candidate_pool(problem, seed+6, spec["augmentation"], seen)
                        extra = (local_candidates(problem, anchor, seed+7, spec["augmentation"], seen)
                                 if local else global_addition)
                        append(journal, {"event": "augmentation", "n": n,
                                        "mode": "local" if local else "global",
                                        "anchor": digest(anchor) if local else None, "candidates": extra})
                        remaining.extend(extra)
                    past_x = features.transform([row.candidate for row in history])
                    fitted = {}
                    for name in heads:
                        observations = []
                        for row in history:
                            observation = row.observation.constraint(name)
                            if (row.observation.status in ("unknown", "infrastructure_error")
                                    or row.diagnostics.get("numerical_failure")):
                                observation = OutputObservation("missing", observation.evidence_id,
                                                                reason="untrusted_evaluation")
                            observations.append(observation)
                        fitted[name] = StableGroupedHead(past_x, observations, seed=seed,
                            continuous_count=len(problem.space.continuous), name=name)
                    model_ids = {name: head.identity for name, head in fitted.items()}

                    def probabilities(query):
                        values = {}
                        for name, head in fitted.items():
                            forecast = head.predict(query)
                            values[name] = ndtr(-forecast["mean"] /
                                                np.sqrt(np.maximum(forecast["variance"], 1e-14)))
                        values["joint"] = np.prod(list(values.values()), axis=0)
                        return values

                    if n in spec["checkpoints"]:
                        prediction = {"event": "prediction", "n": n, "models": model_ids,
                                      "probabilities": plain(probabilities(heldout_x))}
                        append(journal, prediction)
                        if arm == shared_arm and n < switch:
                            shared_rows.append(prediction)
                    if n == spec["total"]:
                        break
                    query_x = features.transform(remaining)
                    energy = None
                    if optimizing:
                        offset = n-spec["initial"]
                        updated = uses_latent and offset % spec["objective_update_every"] == 0
                        if updated:
                            session.fit(native_observations(history, modules, explicit.space),
                                        incremental=offset > 0)
                            objective_features = feature_snapshot(problem, BASE, history, session)
                        exact = [row for row in history if row.observation.status == "completed"
                                 and not row.diagnostics.get("numerical_failure")
                                 and row.observation.objective.kind == "exact"]
                        best = min((row.observation.objective.value for row in exact if row.usable_feasible),
                                   default=None)
                        improvement, probability, coverage, model_id = None, None, None, None
                        if best is not None and len(exact) >= 2:
                            objective = StableGroupedHead(
                                objective_features.transform([row.candidate for row in exact]),
                                [row.observation.objective for row in exact], objective=True,
                                reference_scale=problem.objective_reference_scale, seed=seed,
                                continuous_count=len(problem.space.continuous), name="energy")
                            prediction = objective.predict(objective_features.transform(remaining))
                            improvement = expected_improvement(
                                prediction["mean"], prediction["variance"], best).tolist()
                            probability = probabilities(query_x)["joint"].tolist()
                            model_id = objective.identity
                        else:
                            coverage = np.min(np.linalg.norm(query_x[:, None, :]-past_x[None, :, :],
                                                             axis=2), axis=1).tolist()
                        scores = energy_scores(rule, improvement, probability, coverage)
                        energy = {"best": best, "exact_count": len(exact), "objective_model": model_id,
                                  "representation": objective_features.identity,
                                  "representation_updated": updated, "improvement": improvement,
                                  "probability": probability, "coverage": coverage}
                    elif rule == "iid":
                        scores = [random.random() for _ in remaining]
                    elif rule == "coverage":
                        scores = np.min(np.linalg.norm(query_x[:, None, :]-past_x[None, :, :],
                                                       axis=2), axis=1).tolist()
                    elif rule == "boundary":
                        p = probabilities(query_x)["joint"]
                        scores = (p*(1-p)).tolist()
                    else:
                        scores = brier_scores(fitted, query_x, ref_x).tolist()
                    selected = select(remaining, scores)
                    candidate = remaining.pop(selected)
                    proposal = {"event": "proposal", "n": n, "models": model_ids,
                                "prefix": digest(plain([r.to_dict() for r in history])),
                                "index": selected, "candidate": digest(candidate), "scores": scores}
                    if energy is not None:
                        proposal["energy"] = energy
                    append(journal, proposal)
                    history.append(ledger.evaluate(problem, candidate, f"{seed}/{arm}/{n}", "online"))
                    if arm == shared_arm and n < switch:
                        shared_rows.append(proposal)
                        if n+1 == switch:
                            shared_history, shared_remaining = list(history), list(remaining)
                            if "local_switch" in spec:
                                shared_session = deepcopy(session)
                                shared_objective_features = objective_features
                print(f"{seed} {arm} complete", flush=True)
            for i, row in enumerate(heldout):
                ledger.evaluate(problem, row, f"{seed}/heldout/{i}", "heldout")
        save(output/"report.json", {"status": "COMPLETE_AIRCRAFT_RISK_COLD_START",
             "calls": ledger.count, "roles": dict(ledger.roles), "seconds": time.perf_counter()-start,
             "source_sha": sha, "rcaide_mda_calls": 0})
        files = sorted(path for path in output.rglob("*") if path.is_file())
        save(output/"manifest.json", {path.relative_to(output).as_posix(): file_hash(path) for path in files})
    except Exception:
        save(output/"failure.json", {"started": ledger.count, "settled": sum(ledger.roles.values()),
             "pending": ledger.pending, "source_sha": sha, "traceback": traceback.format_exc()})
        raise


def verify(output, expected):
    source_commit(expected)
    return verify_evidence(output, expected)


def verify_evidence(output, expected):
    """Check a sealed run against its recorded source and experiment protocol."""
    manifest = read(output/"manifest.json")
    paths = {path.relative_to(output).as_posix(): file_hash(path)
             for path in output.rglob("*") if path.is_file() and path != output/"manifest.json"}
    if manifest != paths:
        raise ValueError("Evidence manifest or file inventory mismatch")
    spec = read(output/"plan.json")
    frozen = {k: v for k, v in spec.items() if k not in {"source_sha", "runtime"}}
    if frozen != plan(spec["profile"]) or spec["source_sha"] != expected:
        raise ValueError("Evidence does not match the frozen experiment")
    report = read(output/"report.json")
    if (report["status"] != "COMPLETE_AIRCRAFT_RISK_COLD_START" or report["source_sha"] != expected
            or report["roles"] != spec["budget"] or report["calls"] != sum(spec["budget"].values())
            or report["rcaide_mda_calls"] != 0):
        raise ValueError("Run summary does not match the budget or source")
    contract = read(output/"contract.json")
    if set(contract["heads"]) != {"stall", "root_bending", "soc", "current", "voltage", "power"}:
        raise ValueError("Wrong engineering constraint roster")
    calls, roles = {}, Counter()
    with (output/"calls.jsonl").open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            started = json.loads(line)
            ended = json.loads(next(stream))
            payload = {k: v for k, v in started.items() if k not in {"event", "call_id"}}
            tag, record = payload["tag"], ended["record"]
            if (started["event"] != "started" or ended["event"] != "settled"
                    or number != payload["number"] or tag in calls or ended["tag"] != tag
                    or digest(payload) != started["call_id"] or ended["call_id"] != started["call_id"]
                    or payload["contract"] != contract["id"] or payload["new_rcaide_mda"] is not False
                    or record["observation"]["contract_id"] != contract["id"]
                    or record["candidate"] != payload["candidate"]
                    or record["observation"]["design_id"] != digest(payload["candidate"])
                    or digest(record) != ended["record_digest"]):
                raise ValueError(f"Call ledger mismatch at {tag}")
            calls[tag] = (number, record, payload["role"])
            roles[payload["role"]] += 1
    if dict(roles) != spec["budget"]:
        raise ValueError("Started/settled budget mismatch")
    used = set()
    results, trajectories = [], []
    for seed in spec["seeds"]:
        unit = output/str(seed)
        roster = read(unit/"roster.json")
        initial, pool, reference, heldout = (roster[role]["designs"]
                                             for role in ("initial", "pool", "reference", "heldout"))
        all_rows = initial+pool+reference+heldout
        if ([len(rows) for rows in (initial, pool, reference, heldout)]
                != [spec[role] for role in ("initial", "pool", "reference", "heldout")]
                or len(set(map(digest, all_rows))) != len(all_rows)):
            raise ValueError(f"Candidate overlap or missing design at {seed}")

        def record_at(tag, candidate, role, *, shared=False):
            number, record, actual_role = calls[tag]
            if (tag in used) != shared or record["candidate"] != candidate or actual_role != role:
                raise ValueError(f"Call does not match frozen roster at {tag}")
            used.add(tag)
            return number, record

        anchors = [record_at(f"{seed}/initial/{i}", row, "initial") for i, row in enumerate(initial)]
        tests = [record_at(f"{seed}/heldout/{i}", row, "heldout") for i, row in enumerate(heldout)]
        first_test = min(number for number, _ in tests)
        switch, shared_arm = spec.get("coverage_switch"), spec.get("shared_coverage_arm")
        branch_arms = PHASE_ARMS
        global_addition = None
        if "local_switch" in spec:
            switch, shared_arm = spec["local_switch"], spec["shared_search_arm"]
            branch_arms = LOCAL_ARMS
        shared_journal = ({(row["event"], row["n"]): row for row in
                           map(json.loads, (unit/f"{shared_arm}.jsonl").read_bytes().splitlines())
                           if row["n"] <= switch} if shared_arm else {})
        for arm in spec["arms"]:
            prefix, remaining = [record for _, record in anchors], list(pool)
            last_call = max(number for number, _ in anchors)
            if last_call >= first_test:
                raise ValueError("Initial observations follow the independent test panel")
            models_at_checkpoint = None
            representation = None
            fallback_calls = 0
            augmentation_ids = set()
            local_selection_calls = 0
            new_candidate_calls = 0
            switch_best = None
            augmentation_mode = None
            with (unit/f"{arm}.jsonl").open(encoding="utf-8") as stream:
                for n in range(spec["initial"], spec["total"]+1):
                    if n == spec.get("local_switch"):
                        addition = json.loads(next(stream))
                        switch_design = local_anchor(prefix, checked_label)
                        switch_best = next((r["observation"]["objective"]["value"] for r in prefix
                                            if switch_design is not None and r["candidate"] == switch_design), None)
                        anchor = switch_design if arm in LOCAL_ARMS else None
                        augmentation_mode = addition["mode"]
                        check_augmentation(addition, n, spec["augmentation"], "local" if anchor else "global",
                                           anchor, set(map(digest, all_rows)))
                        if arm == shared_arm:
                            global_addition = addition["candidates"]
                        elif anchor is None and addition["candidates"] != global_addition:
                            raise ValueError("No-incumbent arm did not reuse the global augmentation")
                        augmentation_ids = set(map(digest, addition["candidates"]))
                        remaining.extend(addition["candidates"])
                    if n in spec["checkpoints"]:
                        row = json.loads(next(stream))
                        if row["event"] != "prediction" or row["n"] != n:
                            raise ValueError("Checkpoint order mismatch")
                        if arm in branch_arms and n <= switch and row != shared_journal["prediction", n]:
                            raise ValueError("Shared checkpoint differs between matched arms")
                        models_at_checkpoint = row["models"]
                        if set(models_at_checkpoint) != set(contract["heads"]):
                            raise ValueError("Wrong checkpoint model roster")
                        values = row["probabilities"]
                        if set(values) != {*contract["heads"], "joint"} or any(
                                len(arr) != len(heldout) or not all(0 <= value <= 1 for value in arr)
                                for arr in values.values()):
                            raise ValueError("Invalid checkpoint head probabilities or roster size")
                        if any(not math.isclose(values["joint"][i], math.prod(
                                values[head][i] for head in contract["heads"]),
                                rel_tol=1e-12, abs_tol=1e-15) for i in range(len(heldout))):
                            raise ValueError("Joint probabilities do not match the registered product")
                        for head, probabilities in values.items():
                            labels = [checked_label(item, head) for _, item in tests]
                            known = [(y, p) for y, p in zip(labels, probabilities) if y is not None]
                            scores = binary_metrics([y for y, _ in known], [p for _, p in known])
                            anchor_labels = [checked_label(item, head) for _, item in anchors]
                            anchor_known = [label for label in anchor_labels if label is not None]
                            baseline = (sum(anchor_known)+1)/(len(anchor_known)+2)
                            baseline_brier = (sum((y-baseline)**2 for y, _ in known)/len(known)
                                              if known else None)
                            ece = (sum(bin_["n"]*abs(bin_["mean_probability"]-bin_["observed_fraction"])
                                       for bin_ in scores["calibration_bins"] if bin_["n"])/len(known)
                                   if known else None)
                            results.append({**scores, "seed": seed, "arm": arm, "n": n,
                                "head": head, "labeled_test_count": scores["n"], "unknown": len(heldout)-len(known), "common_anchor_probability": baseline,
                                "baseline_brier": baseline_brier, "ece_5_bins": ece,
                                "feasible_recall": (None if scores["false_infeasible_rate"] is None
                                                    else 1-scores["false_infeasible_rate"]),
                                "brier_skill": (1-scores["brier"]/baseline_brier if baseline_brier else None)})
                    if n == spec["total"]:
                        break
                    row = json.loads(next(stream))
                    if not all(math.isfinite(value) for value in row["scores"]):
                        raise ValueError("Nonfinite proposal scores")
                    i = select(remaining, row["scores"])
                    candidate = remaining.pop(i)
                    if (row["event"] != "proposal" or row["n"] != n or row["prefix"] != digest(prefix)
                            or set(row["models"]) != set(contract["heads"])
                            or (models_at_checkpoint is not None and models_at_checkpoint != row["models"])
                            or row["index"] != i or row["candidate"] != digest(candidate)):
                        raise ValueError("Proposal does not reproduce from the sealed prefix and scores")
                    if arm in ENERGY_ARMS:
                        rule, latent = ENERGY_ARMS[arm]
                        updated = latent and (n-spec["initial"]) % spec["objective_update_every"] == 0
                        energy_receipt(rule, row["energy"], row["scores"], prefix, updated, checked_label)
                        fallback_calls += row["energy"]["improvement"] is None
                        identity = row["energy"]["representation"]
                        if representation is not None and not updated and representation != identity:
                            raise ValueError("Objective representation changed outside its schedule")
                        representation = identity
                    shared = arm in branch_arms and n < switch
                    if shared and row != shared_journal["proposal", n]:
                        raise ValueError("Coverage proposal differs between matched arms")
                    charge_arm = shared_arm if shared else arm
                    number, outcome = record_at(f"{seed}/{charge_arm}/{n}", candidate, "online", shared=shared)
                    if not last_call < number < first_test:
                        raise ValueError("Online evaluation is out of order or after held-out outcomes")
                    last_call = number
                    prefix.append(outcome)
                    selected_augmentation = row["candidate"] in augmentation_ids
                    new_candidate_calls += selected_augmentation
                    local_selection_calls += selected_augmentation and augmentation_mode == "local"
                    models_at_checkpoint = None
                if stream.read():
                    raise ValueError("Extra decision evidence")
            labels = [checked_label(record) for record in prefix]
            usable = [label is True and record["observation"]["status"] == "completed"
                      and record["observation"]["objective"]["kind"] == "exact"
                      for record, label in zip(prefix, labels)]
            online_labels = labels[spec["initial"]:]
            terminations = Counter(record["diagnostics"].get("termination_reason", "unreported")
                                   for record in prefix[spec["initial"]:])
            energies = [record["observation"]["objective"]["value"] for record, valid in zip(prefix, usable)
                        if valid]
            trajectories.append({"seed": seed, "arm": arm,
                                 "initial_feasible": sum(label is True for label in labels[:spec["initial"]]),
                                 "first_usable_feasible_call": next((i for i, valid in enumerate(usable, 1)
                                                                     if valid), None),
                                 "online_feasible": sum(label is True for label in online_labels),
                                 "online_infeasible": sum(label is False for label in online_labels),
                                 "online_unknown": sum(label is None for label in online_labels),
                                 "best_feasible_Wh": min(energies, default=None),
                                 "online_terminations": dict(terminations)})
            if arm in ENERGY_ARMS:
                trajectories[-1]["shared_rule_fallback_calls"] = fallback_calls
                trajectories[-1]["objective_acquisition_calls"] = spec["total"]-spec["initial"]-fallback_calls
            if "local_switch" in spec:
                trajectories[-1].update(local_augmentation_selected=local_selection_calls,
                                        augmentation_selected=new_candidate_calls,
                                        augmentation_mode=augmentation_mode,
                                        switch_best_feasible_Wh=switch_best)
    if set(calls) != used:
        raise ValueError("Unexpected ledger calls")
    return report, results, trajectories


def analyze(source, expected, output):
    report, results, trajectories = verify(source, expected)
    output.mkdir(parents=True)
    flat = [{k: v for k, v in row.items() if k != "calibration_bins"} for row in results]
    with (output/"metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in flat for key in row}))
        writer.writeheader()
        writer.writerows(flat)
    joint_rows = [row for row in results if row["head"] == "joint"]
    joint = {(row["seed"], row["arm"], row["n"]): row for row in joint_rows}
    if len(joint) != len(joint_rows):
        raise ValueError("Duplicate joint checkpoint")
    comparisons = (
        ("brier_explicit", "boundary_explicit"),
        ("brier_explicit", "iid_explicit"),
        ("brier_explicit", "coverage_explicit"),
        ("brier_latent", "boundary_latent"),
        ("brier_latent", "brier_explicit"),
        ("boundary_latent", "boundary_explicit"),
    )
    arms = {arm for _, arm, _ in joint}
    if "coverage_brier" in arms:
        comparisons += (
            ("coverage_brier", "coverage_boundary"),
            ("coverage_brier", "brier_explicit"),
            ("coverage_boundary", "boundary_explicit"),
            ("coverage_brier", "coverage_explicit"),
            ("coverage_boundary", "coverage_explicit"),
        )
    comparisons = tuple((arm, base) for arm, base in comparisons if {arm, base} <= arms)
    paired_metrics = ("brier", "feasible_recall", "false_feasible_rate") if "coverage_brier" in arms else ("brier",)
    pairs = []
    for seed, n in sorted({(seed, n) for seed, _, n in joint}):
        for arm, baseline in comparisons:
            left, right = joint[seed, arm, n], joint[seed, baseline, n]
            row = {"seed": seed, "n": n, "arm": arm, "baseline": baseline}
            for metric in paired_metrics:
                a, b = left[metric], right[metric]
                row[f"{metric}_delta"] = a-b if a is not None and b is not None else None
            pairs.append(row)
    with (output/"joint_pairs.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]) if pairs else
                                ["seed", "n", "arm", "baseline", "brier_delta"])
        writer.writeheader()
        writer.writerows(pairs)
    with (output/"analysis.json").open("x", encoding="utf-8") as stream:
        packet = {"status": "COMPLETE_AIRCRAFT_RISK_ANALYSIS", "verified_run": report,
                  "metrics": results, "pairs": pairs, "trajectories": trajectories}
        if arms & ENERGY_ARMS.keys():
            packet["energy_pairs"] = energy_pairs(trajectories)
            with (output/"energy_pairs.csv").open("x", encoding="utf-8", newline="") as energy_stream:
                writer = csv.DictWriter(energy_stream, fieldnames=list(packet["energy_pairs"][0]))
                writer.writeheader()
                writer.writerows(packet["energy_pairs"])
        json.dump(packet, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plan", "run", "verify", "analyze"), default="plan")
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--expected-sha")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.task in ("plan", "run") and not args.profile:
        parser.error("--profile is required")
    if args.task == "plan":
        print(json.dumps(plan(args.profile), indent=2))
        return
    if not args.expected_sha:
        parser.error("--expected-sha is required")
    if args.task in ("run", "analyze") and args.output is None:
        parser.error("--output is required")
    if args.task in ("verify", "analyze") and args.input is None:
        parser.error("--input is required")
    if args.task == "run":
        run(args.profile, args.expected_sha, args.output.resolve())
    elif args.task == "verify":
        report, _, _ = verify(args.input, args.expected_sha)
        print(json.dumps({"status": "VERIFIED_AIRCRAFT_RISK_COLD_START", "calls": report["calls"]}))
    else:
        analyze(args.input, args.expected_sha, args.output.resolve())
        print("COMPLETE_AIRCRAFT_RISK_ANALYSIS")


if __name__ == "__main__":
    main()
