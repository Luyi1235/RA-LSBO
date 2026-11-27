"""Replay the first online X-57 proposals and seal predictions before joining feedback."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
ALGORITHM = "ae5dd445c9656ffa31c9e340b1364bc2c746eef7"
ARCHIVE = "db33f2fe8da4323480064b06fb1e5ebb7031ab8b"
DATA = ROOT / "results/x57-47p5-five-arm-checkpoint16"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "BLIS_NUM_THREADS")
ARMS = ("lsbo_full_budget", "ra_lsbo_energy_sign", "ra_lsbo_energy_interval")
SEEDS = (643118, 705241, 767363, 829486, 891608, 953731)
SCORE_TOL = 1e-8


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(data):
    return (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_new(path, payload):
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def inspect_archive():
    """Read shared outcomes and first online dispatches only."""
    initial = {seed: [] for seed in SEEDS}
    expected = {seed: {} for seed in SEEDS}
    with (DATA / "calls.jsonl").open("rb") as stream:
        for line in stream:
            # The archive begins each line with event, seed, arm, and evaluation.
            # Never deserialize an online settlement during the prediction phase.
            header = line.split(b'"proposal":', 1)[0]
            is_shared = b'"arm": "shared"' in header
            if line.startswith(b'{"event": "settled",') and is_shared:
                row = json.loads(line)
                row["source_id"] = row["task_id"]
                del row["event"]
                initial[row["seed"]].append(row)
            elif (line.startswith(b'{"event": "dispatch_intent",')
                  and b'"evaluation": 9,' in header and not is_shared):
                row = json.loads(line)
                if row["arm"] in ARMS:
                    seed_rows = expected[row["seed"]]
                    if row["arm"] in seed_rows:
                        raise ValueError("Duplicate first online dispatch")
                    seed_rows[row["arm"]] = row
    for seed in SEEDS:
        initial[seed].sort(key=lambda row: row["evaluation"])
        if ([row["evaluation"] for row in initial[seed]] != list(range(1, 9))
                or set(expected[seed]) != set(ARMS)):
            raise ValueError(f"Incomplete archive for seed {seed}")
    return initial, expected


def verify_sources(git, plan, base):
    for name in ("run_plan.json", "distance_47.5.json", "calls.jsonl"):
        path = DATA / name
        ref = f"{ARCHIVE}:{path.relative_to(ROOT).as_posix()}"
        if path.read_bytes() != subprocess.check_output([git, "show", ref], cwd=ROOT):
            raise ValueError(f"Archive input changed: {name}")
    if (plan["code_sha"] != ALGORITHM or plan["distance_nmi"] != 47.5
            or plan["initial_count"] != 8 or tuple(plan["seeds"]) != SEEDS):
        raise ValueError("Archive plan does not match this diagnostic")
    frozen = subprocess.check_output([git, "diff", "--name-only", ALGORITHM,
                                      "--", "ra_lsbo"], cwd=ROOT)
    untracked = subprocess.check_output([git, "ls-files", "--others", "--exclude-standard",
                                         "--", "ra_lsbo"], cwd=ROOT)
    if frozen or untracked:
        raise ValueError("Frozen algorithm source differs from its commit")
    for name, digest in plan["environment"]["lsbo_sha256"].items():
        if sha256((base / "lsbo" / name).read_bytes()) != digest:
            raise ValueError(f"Frozen native optimizer source changed: {name}")


def capture_ask(adapter):
    """Capture the fitted objects and exact final candidate arrays on return."""
    from ra_lsbo import x57_cold_risk

    traces = {}

    def profile(frame, event, result):
        if event != "return" or result is None or frame.f_code.co_name != "propose":
            return
        local = frame.f_locals
        if frame.f_code is x57_cold_risk.propose.__code__ and "scores" in local:
            traces["completion"] = local
        elif (frame.f_globals.get("__name__") == "lsbo.optimizer"
              and "features" in local and "index" in local):
            traces["performance"] = local

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        proposal = adapter.ask()
    finally:
        sys.setprofile(previous)
    return proposal, traces


def scalar(value):
    value = float(value)
    return value if math.isfinite(value) else None


def compare_scores(adapter, trace, stage, lsbo_point, ra_point):
    import numpy as np
    import torch
    from ra_lsbo.x57_energy_surrogate import engineering_log_probability

    space = adapter.problem.space
    points = [lsbo_point, ra_point]
    keys = [space.key(point) for point in points]
    pool = trace["pool"] if stage == "completion" else trace["candidates"]
    pool_keys = [space.key(point) for point in pool]
    if len(pool_keys) != len(set(pool_keys)) or pool_keys[trace["index"]] != keys[1]:
        raise ValueError("Captured pool does not match the original final selection")
    scores = np.asarray(trace["scores"]).reshape(-1)
    if len(scores) != len(pool):
        raise ValueError("Final pool and score lengths differ")

    # Prediction only. These are the original fitted objects, after the original
    # selection has returned. All GP calls run in evaluation mode with no gradients.
    optimizer = adapter.optimizer
    data = json_bytes(adapter.history)
    observations = tuple(map(id, optimizer.observations))
    optimizer_rng = json_bytes(optimizer.rng.bit_generator.state)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state()
    rep = trace["rep"]
    with torch.no_grad():
        features = rep.features(points)
        if stage == "completion":
            completion = None if "completion_fallback" in trace["info"] else trace.get("diagnostic_completion")
            if completion is None and "completion_fallback" not in trace["info"]:
                raise ValueError("Completion posterior was not captured")
            energy, models, bounds = trace["head"], trace["models"], trace["bounds"]
            log_complete = (np.zeros(len(points)) if completion is None
                            else completion.log_probabilities(points))
            joint = log_complete + engineering_log_probability(
                features, energy, trace["names"], models, bounds).numpy()
            improvement = [None] * len(points)
        else:
            surrogate = trace["surrogate"]
            completion = getattr(surrogate, "completion", None)
            engineering = surrogate.native if completion is not None else surrogate
            energy = getattr(engineering, "energy", None)
            raw = engineering.native if energy is not None else engineering
            models, bounds = raw.constraints, (raw.offset, raw.scale)
            log_complete = None if completion is None else completion.log_probability(features).numpy()
            normalized = (features - raw.offset) / raw.scale
            improvement = (raw.improvement(normalized.unsqueeze(-2)).reshape(-1).tolist()
                           if raw.improvement is not None else [None] * len(points))
            joint = surrogate.score(features)
        energy_probability = (None if energy is None else energy.log_probability(features).exp().tolist())
        normalized = (features - bounds[0]) / bounds[1]
        other = {name: [None] * len(points) for name in adapter.problem.required_constraints
                 if name != "usable_energy"}
        for name, (model, threshold) in models.items():
            if model.training:
                raise ValueError("Constraint GP unexpectedly in training mode")
            posterior = model.posterior(normalized.unsqueeze(-2))
            z = (threshold - posterior.mean) / posterior.variance.clamp_min(1e-12).sqrt()
            other[name] = torch.special.ndtr(z).reshape(-1).tolist()
    if (data != json_bytes(adapter.history) or observations != tuple(map(id, optimizer.observations))
            or optimizer_rng != json_bytes(optimizer.rng.bit_generator.state)
            or python_rng != random.getstate() or not torch.equal(torch_rng, torch.random.get_rng_state())
            or any(not np.array_equal(a, b) for a, b in zip(numpy_rng, np.random.get_state()))):
        raise ValueError("Additional scoring changed observations or random state")

    joint = np.asarray(joint).reshape(-1)
    for key, score in zip(keys, joint):
        if key in pool_keys and not np.isclose(score, scores[pool_keys.index(key)], atol=SCORE_TOL, rtol=0):
            raise ValueError(f"Batch cross-score differs from original final score: {key}")
    if not np.isfinite(joint).all():
        raise ValueError("Nonfinite cross-score")

    # Use the actual scores for points already in the pool. The extra LSBO
    # score is placed alongside them for an explicitly counterfactual rank.
    ranked = {key: float(score) for key, score in zip(pool_keys, scores) if np.isfinite(score)}
    ranked.setdefault(keys[0], float(joint[0]))

    def rank(score, others):
        return 1 + sum(bool(value > score + SCORE_TOL) for value in others)

    rows = {}
    for i, label in enumerate(("lsbo", "ra")):
        key = keys[i]
        rows[label] = {
            "p_complete": scalar(np.exp(log_complete[i])) if completion is not None else None,
            "p_usable_energy": None if energy_probability is None else scalar(energy_probability[i]),
            "other_constraint_probabilities": {name: scalar(values[i]) if values[i] is not None else None
                                               for name, values in other.items()},
            "log_expected_improvement": None if improvement[i] is None else scalar(improvement[i]),
            "joint_log_score": scalar(joint[i]),
            "pair_rank": rank(ranked[key], (ranked[k] for k in set(keys))),
            "final_pool_rank": rank(ranked[key], scores) if key in pool_keys else None,
            "augmented_pool_rank": rank(ranked[key], ranked.values()),
        }
    gap = ranked[keys[1]] - ranked[keys[0]]
    present = keys[0] in pool_keys
    if present:
        diagnosis = "present_ranked_lower" if gap > SCORE_TOL else "present_tied_or_higher"
    else:
        diagnosis = "missing_and_counterfactually_ranked_lower" if gap > SCORE_TOL else "missing_only"
    return {"lsbo_in_final_pool": present,
            "lsbo_in_raw_pool": keys[0] in pool_keys[:trace.get("raw_pool_count", len(pool))],
            "final_pool_count": len(pool),
            "final_pool": [{"id": key, "joint_log_score": scalar(score)}
                           for key, score in zip(pool_keys, scores)],
            "ra_minus_lsbo_log_score": gap,
            "diagnosis": diagnosis, **rows}


def predict_seed(seed, initial, expected, base, plan, scratch):
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    from ra_lsbo import x57_cold_risk
    from ra_lsbo.x57_formal_native import FormalNative
    from ra_lsbo.x57_live import X57LiveProblem

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)

    def forbidden(*args, **kwargs):
        raise RuntimeError("Physical evaluation attempted during proposal replay")

    with threadpool_limits(limits=1), TemporaryDirectory(dir=scratch) as temp:
        local = Path(temp)
        problem = X57LiveProblem(base, DATA / "distance_47.5.json", local / "unused", allow_mda=False)
        if (problem.contract_id != plan["contract_id"]
                or problem.source_hashes != plan["environment"]["evaluator_sha256"]):
            raise ValueError("Physical contract or Python environment changed")
        problem.evaluate = problem.problem.evaluate = forbidden
        roster = plan["rosters"][str(seed)]["records"]
        if [r["candidate_id"] for r in initial] != [r["candidate_id"] for r in roster]:
            raise ValueError(f"Shared initial observations differ from the frozen roster: {seed}")

        replayed, pairs = {}, {}
        points = {}
        captured_completion = []
        original_log_probs = x57_cold_risk.CompletionPosterior.log_probabilities

        def observe_completion(posterior, candidates):
            value = original_log_probs(posterior, candidates)
            captured_completion.append(posterior)
            return value

        x57_cold_risk.CompletionPosterior.log_probabilities = observe_completion
        try:
            for arm in ARMS:
                captured_completion.clear()
                adapter = FormalNative(problem, arm, seed, initial, local / arm)
                (point, phase, info), traces = capture_ask(adapter)
                candidate_id = problem.space.key(point)
                archived = expected[arm]
                if ((candidate_id, phase, point.to_dict(), info.get("source"), info.get("risk_stage"))
                        != (archived["candidate_id"], archived["phase"], archived["candidate"],
                            archived["proposal"].get("source"), archived["proposal"].get("risk_stage"))):
                    raise ValueError(f"Proposal replay mismatch: {seed}/{arm} ({candidate_id})")
                points[arm] = point
                replayed[arm] = {"candidate_id": candidate_id, "candidate": point.to_dict(),
                                 "source": info["source"], "phase": phase,
                                 "risk_stage": info.get("risk_stage"), "archive_match": True,
                                 "risk_training_rows": info.get("risk_training_rows"),
                                 "energy_fit": info.get("engineering_energy_fit")}
                if arm == ARMS[0]:
                    continue
                if len(traces) != 1:
                    raise ValueError(f"Expected exactly one final ranking: {seed}/{arm}")
                stage, trace = next(iter(traces.items()))
                if stage != info["risk_stage"]:
                    raise ValueError("Captured ranking stage differs from the proposal")
                if stage == "completion":
                    if len(captured_completion) > 1:
                        raise ValueError("Multiple completion posterior evaluations before selection")
                    trace["diagnostic_completion"] = next(iter(captured_completion), None)
                pairs[arm] = compare_scores(adapter, trace, stage, points[ARMS[0]], point)
                count = pairs[arm]["final_pool_count"]
                if count != archived["proposal"]["candidate_count"]:
                    raise ValueError(f"Archived candidate count differs: {seed}/{arm}")
                if stage == "completion":
                    from ra_lsbo.budget import digest
                    if digest([p.to_dict() for p in trace["pool"]]) != archived["proposal"]["candidate_list_sha256"]:
                        raise ValueError(f"Archived ordered pool hash differs: {seed}/{arm}")
                print(f"{seed} {arm} verified ({stage}, {count} candidates)", flush=True)
        finally:
            x57_cold_risk.CompletionPosterior.log_probabilities = original_log_probs
        libraries = sorted({(row["internal_api"], row["num_threads"]) for row in threadpool_info()})
        if (any(count != 1 for _, count in libraries)
                or torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1):
            raise ValueError(f"Parallel math library detected: {libraries}")
        return {"seed": seed, "initial_usable": sum(row["feedback"]["usable"] for row in initial),
                "initial_ids": [row["candidate_id"] for row in initial],
                "threads": libraries, "proposals": replayed, "comparisons": pairs}


def join_truth(output):
    sealed = (output / "predictions.json").read_bytes()
    if (output / "predictions.sha256").read_text().strip() != sha256(sealed):
        raise ValueError("Prediction seal does not match")
    prediction = json.loads(sealed)
    truth = {}
    with (DATA / "calls.jsonl").open("rb") as stream:
        for line in stream:
            header = line.split(b'"proposal":', 1)[0]
            if line.startswith(b'{"event": "settled",') and b'"evaluation": 9,' in header:
                row = json.loads(line)
                lookup = (row["seed"], row["arm"], row["candidate_id"])
                if lookup in truth or not row["charged"]:
                    raise ValueError(f"Invalid first online settlement: {lookup}")
                truth[lookup] = row["feedback"]

    table = []
    for record in prediction["seeds"]:
        seed = record["seed"]
        proposals = record["proposals"]
        for arm, pair in record["comparisons"].items():
            result = {"seed": seed, "ra_arm": arm, "initial_usable": record["initial_usable"],
                      **{key: value for key, value in pair.items()
                         if key not in ("lsbo", "ra", "final_pool")}}
            for label, origin in (("lsbo", ARMS[0]), ("ra", arm)):
                proposal = proposals[origin]
                candidate_id = proposal["candidate_id"]
                feedback = truth[(seed, origin, candidate_id)]
                facts = {"candidate_id": candidate_id, "candidate": proposal["candidate"],
                         "configuration": proposal["candidate"]["discrete"],
                         "source": proposal["source"], "phase": proposal["phase"],
                         "risk_stage": proposal["risk_stage"], **pair[label],
                         **{f"true_{key}": feedback[key] for key in
                            ("outcome", "completion", "usable", "objective_Wh", "constraints")}}
                for key in ("other_constraint_probabilities", "true_constraints"):
                    facts.update({f"{key}_{name}": value for name, value in facts.pop(key).items()})
                result.update({f"{label}_{key}": value for key, value in facts.items()})
            table.append(result)

    def csv_value(value):
        if value is None:
            return "NA"
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        return value

    fields = list(dict.fromkeys(key for row in table for key in row))
    with (output / "pairwise.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, restval="NA", lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: csv_value(value) for key, value in row.items()} for row in table)
    print("Verified sealed prediction:", sha256(sealed))
    print("Diagnoses:", dict(Counter(row["diagnosis"] for row in table)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Frozen physical-base directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--git", default="git")
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=6)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    for name in THREADS:
        os.environ[name] = "1"
    plan = json.loads((DATA / "run_plan.json").read_bytes())
    verify_sources(args.git, plan, args.base)
    initial, expected = inspect_archive()
    args.output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(predict_seed, seed, initial[seed], expected[seed],
                                   args.base, plan, args.output) for seed in SEEDS]
        seeds = sorted((future.result() for future in as_completed(futures)), key=lambda row: row["seed"])
    payload = json_bytes({"archive_commit": ARCHIVE, "algorithm_commit": ALGORITHM,
                          "rcaide_calls": 0, "score_tie_tolerance": SCORE_TOL,
                          "workers": args.workers, "seeds": seeds})
    write_new(args.output / "predictions.json", payload)
    write_new(args.output / "predictions.sha256", (sha256(payload) + "\n").encode())
    join_truth(args.output)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
