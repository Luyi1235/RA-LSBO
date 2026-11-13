"""Prospective prediction-only experiment on the repository's reduced aircraft."""
import argparse
from collections import Counter
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

from aircraft_risk_protocol import ARMS, PROFILES, plan
from ra_lsbo.budget import digest
from risk_learning.core import binary_metrics

ROOT = Path(__file__).resolve().parent


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
        for seed in spec["seeds"]:
            unit = output/str(seed)
            seen = set()
            roster = {}
            for role in ("initial", "pool", "reference", "heldout"):
                rows, rejected = candidate_pool(problem, seed+len(roster)+1, spec[role], seen)
                roster[role] = {"designs": rows, "screen_rejections": rejected}
                seen.update(digest(row) for row in rows)
            save(unit/"roster.json", roster)
            initial, pool, reference, heldout = (roster[role]["designs"]
                                                  for role in ("initial", "pool", "reference", "heldout"))
            anchors = [ledger.evaluate(problem, row, f"{seed}/initial/{i}", "initial")
                       for i, row in enumerate(initial)]
            explicit = PrefixFeatures(problem, BASE, anchors, seed, latent=False)
            modules, _ = load_base(BASE)
            session = CorpusRepresentation(explicit.space, seed, count=2000, settings=encoder_settings())
            session.fit(native_observations(anchors, modules, explicit.space), incremental=False)
            latent = feature_snapshot(problem, BASE, anchors, session)
            save(unit/"features.json", {"explicit": explicit.identity, "latent": latent.identity,
                                       "encoder_state": latent.state_hash, "anchor_count": len(anchors)})
            for arm, (rule, uses_latent) in ARMS.items():
                features = latent if uses_latent else explicit
                journal = unit/f"{arm}.jsonl"
                history, remaining = list(anchors), list(pool)
                ref_x, heldout_x = features.transform(reference), features.transform(heldout)
                random = Random(seed+11)
                for n in range(spec["initial"], spec["total"]+1):
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
                        append(journal, {"event": "prediction", "n": n, "models": model_ids,
                                         "probabilities": probabilities(heldout_x)})
                    if n == spec["total"]:
                        break
                    query_x = features.transform(remaining)
                    if rule == "iid":
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
                    append(journal, {"event": "proposal", "n": n, "models": model_ids,
                                     "prefix": digest(plain([r.to_dict() for r in history])),
                                     "index": selected, "candidate": digest(candidate), "scores": scores})
                    history.append(ledger.evaluate(problem, candidate, f"{seed}/{arm}/{n}", "online"))
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

        def record_at(tag, candidate, role):
            number, record, actual_role = calls[tag]
            if tag in used or record["candidate"] != candidate or actual_role != role:
                raise ValueError(f"Call does not match frozen roster at {tag}")
            used.add(tag)
            return number, record

        anchors = [record_at(f"{seed}/initial/{i}", row, "initial") for i, row in enumerate(initial)]
        tests = [record_at(f"{seed}/heldout/{i}", row, "heldout") for i, row in enumerate(heldout)]
        first_test = min(number for number, _ in tests)
        for arm in spec["arms"]:
            prefix, remaining = [record for _, record in anchors], list(pool)
            last_call = max(number for number, _ in anchors)
            if last_call >= first_test:
                raise ValueError("Initial observations follow the independent test panel")
            models_at_checkpoint = None
            with (unit/f"{arm}.jsonl").open(encoding="utf-8") as stream:
                for n in range(spec["initial"], spec["total"]+1):
                    if n in spec["checkpoints"]:
                        row = json.loads(next(stream))
                        if row["event"] != "prediction" or row["n"] != n:
                            raise ValueError("Checkpoint order mismatch")
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
                    number, outcome = record_at(f"{seed}/{arm}/{n}", candidate, "online")
                    if not last_call < number < first_test:
                        raise ValueError("Online evaluation is out of order or after held-out outcomes")
                    last_call = number
                    prefix.append(outcome)
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
    pairs = []
    for seed, n in sorted({(seed, n) for seed, _, n in joint}):
        for arm, baseline in comparisons:
            left, right = joint[seed, arm, n]["brier"], joint[seed, baseline, n]["brier"]
            pairs.append({"seed": seed, "n": n, "arm": arm, "baseline": baseline,
                          "brier_delta": left-right if left is not None and right is not None else None})
    with (output/"joint_pairs.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    with (output/"analysis.json").open("x", encoding="utf-8") as stream:
        json.dump({"status": "COMPLETE_AIRCRAFT_RISK_ANALYSIS", "verified_run": report,
                   "metrics": results, "pairs": pairs, "trajectories": trajectories}, stream, indent=2, allow_nan=False)


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
