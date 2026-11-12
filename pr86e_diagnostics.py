"""Read-only PR86-D mechanism diagnosis. Standard library only; no model/evaluator imports."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import subprocess

SOURCE_SHA = "6e5cbd0bafc25cf4e36f9a5675401d6991165416"
RESULT_SHA = "311ef606d173d6fe0bb3663ca18f9560c1ca5586"
METHODS = ("iid_random", "boundary", "marginal_entropy", "constrained_ei", "objective_weighted_boundary")
HEADS = ("region", "resource", "joint")
CEI, WEIGHTED = METHODS[-2:]
TASKS = ("legacy_branin_sparse", "legacy_rosenbrock_sparse") + tuple("tradeoff_s%d" % i for i in range(8))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(x):
    require(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x), "Invalid numeric evidence")
    return float(x)


def close(a, b):
    return (a is None and b is None) or (a is not None and b is not None and math.isclose(finite(a), finite(b), rel_tol=1e-9, abs_tol=1e-11))


def entropy(p):
    p = finite(p)
    require(0 <= p <= 1, "Invalid probability")
    return 0.0 if p in (0, 1) else -p * math.log(p) - (1-p) * math.log1p(-p)


def ranks(values):
    n = len(values)
    require(n > 0, "Empty score vector")
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: finite(values[i]))
    result = [0.0] * n
    pos = 0
    while pos < n:
        end = pos + 1
        while end < n and values[order[end]] == values[order[pos]]:
            end += 1
        for k in range(pos, end):
            result[order[k]] = (pos + end - 1) / (2.0 * (n-1))
        pos = end
    return result


def ei(mean, variance, incumbent):
    gap = finite(incumbent) - finite(mean)
    v = finite(variance)
    require(v >= 0, "Negative objective variance")
    if v == 0:
        return max(gap, 0.0)
    sd = math.sqrt(v)
    z = gap / sd
    # Match frozen operation order: cancellation in tiny EI can alter ranks.
    cdf = 0.5 * math.erfc(-z/math.sqrt(2.0))
    pdf = math.exp(-0.5*z*z)/math.sqrt(2.0*math.pi)
    return max(0.0, gap*cdf + sd*pdf)


def rule_vectors(probabilities, objective, incumbent):
    """Independent reproduction of the frozen 0.65/0.35 rule, NOT a new policy."""
    p = probabilities
    n = len(p["joint"])
    require(n > 0 and all(len(p[h]) == n for h in HEADS), "Probability length mismatch")
    for a, b, j in zip(*(p[h] for h in HEADS)):
        for v in (a, b, j):
            require(0 <= finite(v) <= 1, "Probability out of bounds")
        require(abs(j-a*b) <= 1e-10, "Joint product mismatch")
    if objective is not None:
        require(set(objective) == {"mean", "variance"}, "Malformed objective posterior")
        require(all(len(objective[k]) == n for k in objective), "Objective length mismatch")
        for m, v in zip(objective["mean"], objective["variance"]):
            finite(m)
            require(finite(v) >= 0, "Negative objective variance")
    cold = objective is None or incumbent is None
    improvement = [1.0]*n if cold else [ei(m, v, incumbent) for m, v in zip(objective["mean"], objective["variance"])]
    cei = [q*x for q, x in zip(p["joint"], improvement)]
    info = [entropy(q)*x for q, x in zip(p["joint"], improvement)]
    weighted = [0.65*a+0.35*b for a, b in zip(ranks(cei), ranks(info))]
    return {CEI: cei, WEIGHTED: weighted,
            "boundary": [q*(1-q) for q in p["joint"]],
            "marginal_entropy": [entropy(a)+entropy(b) for a, b in zip(p["region"], p["resource"]) ]}


def argmax(values, candidate_ids):
    require(len(values) == len(candidate_ids) > 0, "Score/ID length mismatch")
    return max(range(len(values)), key=lambda i: (finite(values[i]), candidate_ids[i]))


def observable(record):
    """Only the actually returned observation packet is exposed to diagnostics."""
    obs = record["observed"]
    cid = digest(record["candidate"])
    require(record["candidate_id"] == obs["candidate_id"] == cid, "Observable candidate mismatch")
    require(obs["candidate"] == record["candidate"], "Observable design mismatch")
    region, resource, objective = (obs[k] for k in ("region", "resource", "objective"))
    if region["kind"] == "exact":
        region_ok = finite(region["value"]) <= 0
        require(region_ok, "Native region exact label must satisfy constraint")
    else:
        require(region["kind"] == "interval" and region.get("lower") == 0 and region.get("lower_inclusive") is False, "Unsupported region censoring")
        region_ok = False
    require(resource["kind"] == "exact", "Resource observation is not exact")
    feasible = region_ok and finite(resource["value"]) <= 0
    require(objective["kind"] == ("exact" if region_ok else "missing"), "Objective observability contract mismatch")
    value = finite(objective["value"]) if region_ok else None
    return obs, value, feasible


def diagnose_unit(data):
    """Return compact rows. Never read tests, hidden objective truth or unused outcomes."""
    task, repeat = data["task"], data["repeat"]
    expected = set(itertools.product((12, 24), METHODS))
    require(len(data["trials"]) == 10 and {(t["anchor_size"], t["method"]) for t in data["trials"]} == expected, "Missing/duplicate trajectory")
    require(len(data["anchors"]) == 24, "Expected 24 nested initial observations")
    decisions, endpoints = [], []
    for trial in data["trials"]:
        n0, method = trial["anchor_size"], trial["method"]
        require((trial["task"], trial["repeat"]) == (task, repeat), "Trajectory identity mismatch")
        history, feasible_values, exact_pairs = [], {}, []
        initial_ids = set()
        for rec in data["anchors"][:n0]:
            obs, value, feasible = observable(rec)
            history.append(obs)
            initial_ids.add(rec["candidate_id"])
            if value is not None:
                exact_pairs.append((rec["candidate_id"], obs["objective"]))
            if feasible:
                feasible_values[rec["candidate_id"]] = value
        require(trial["anchor_ids"] == [o["candidate_id"] for o in history], "Initial prefix mismatch")
        require(len(trial["proposals"]) == len(trial["online_observations"]) == 48-n0, "Online budget mismatch")
        available = list(data["pool"])
        for step, (proposal, event) in enumerate(zip(trial["proposals"], trial["online_observations"])):
            ids = [digest(c) for c in available]
            incumbent = min(feasible_values.values()) if feasible_values else None
            require(proposal["step"] == event["step"] == step and proposal["method"] == method and proposal["anchor_size"] == n0, "Decision identity mismatch")
            require(proposal["history_observed_sha256"] == digest(history) and proposal["available_sha256"] == digest(available), "Decision prefix/pool mismatch")
            require(close(proposal["incumbent"], incumbent), "Unobserved incumbent")
            forecast = proposal["candidate_forecast"]
            require(proposal["forecast_sha256"] == digest(forecast), "Forecast hash mismatch")
            objective = proposal["objective_forecast"]
            receipt = proposal["method_fit_objective"]
            require(receipt["exact_training_rows"] == len(exact_pairs) and receipt["training_input_sha256"] == digest(exact_pairs), "Objective fit identity mismatch")
            require(receipt["query_ids"] == ids and receipt["training_rows"] == len(history), "Objective query/prefix mismatch")
            require((objective is not None) == (len(exact_pairs) >= 2) == (receipt["status"] == "OK"), "Objective availability mismatch")
            vectors = rule_vectors(forecast["probabilities"], objective, incumbent)
            if method == "iid_random":
                rng = random.Random(trial["seed"] + 104729*step)
                replay = [rng.random() for _ in ids]
            else:
                replay = vectors[method]
            recorded = proposal["scores"]
            require(digest(recorded) == proposal["score_digest"] and len(replay) == len(recorded), "Score evidence mismatch")
            require(all(close(a, b) for a, b in zip(replay, recorded)), "Source policy score replay failed")
            chosen = proposal["selected_index"]
            require(argmax(recorded, ids) == chosen and ids[chosen] == proposal["selected_candidate_sha256"], "Source selected identity mismatch")
            require(proposal["selected_candidate"] == available[chosen], "Source selected design mismatch")
            i, j = (argmax(vectors[m], ids) for m in (CEI, WEIGHTED))
            cei_max = vectors[CEI][i]
            phase = "no_objective_gp" if objective is None else ("no_feasible_incumbent" if incumbent is None else "objective_ready")
            # Alternative actions receive PREDICTIONS only; no outcomes are looked up.
            p = forecast["probabilities"]
            row = dict(task=task, repeat=repeat, anchor_size=n0, source_method=method,
                       step=step, n_before=len(history), phase=phase,
                       exact_labels_before=len(exact_pairs), selected_id=ids[chosen],
                       cei_choice_id=ids[i], weighted_choice_id=ids[j],
                       rules_disagree=int(i != j), source_replay_choice_diff=int(argmax(replay, ids) != chosen),
                       cei_proxy_at_cei=cei_max, cei_proxy_at_weighted=vectors[CEI][j],
                       cei_proxy_fraction_lost=(1-vectors[CEI][j]/cei_max) if cei_max > 0 else None,
                       cei_proxy_zero=int(cei_max == 0),
                       predicted_region_at_cei=p["region"][i], predicted_region_at_weighted=p["region"][j],
                       predicted_joint_at_cei=p["joint"][i], predicted_joint_at_weighted=p["joint"][j])
            rec = event["record"]
            require(rec["candidate_id"] == event["candidate_id"] == ids[chosen] and rec["candidate"] == available[chosen], "Actual feedback binding failed")
            obs, value, feasible = observable(rec)
            row.update(exact_objective_returned=int(value is not None), feasible_returned=int(feasible),
                       selected_region_brier=(p["region"][chosen]-int(value is not None))**2,
                       selected_joint_brier=(p["joint"][chosen]-int(feasible))**2,
                       first_feasible_discovery=int(incumbent is None and feasible),
                       observed_improvement=max(0.0, incumbent-value) if incumbent is not None and feasible else None,
                       objective_error=None, objective_squared_error=None, objective_z=None,
                       objective_zero_variance=None)
            if objective is not None and value is not None:
                error = value-objective["mean"][chosen]
                variance = objective["variance"][chosen]
                row.update(objective_error=error, objective_squared_error=error**2,
                           objective_z=error/math.sqrt(variance) if variance > 0 else None,
                           objective_zero_variance=int(variance == 0))
            decisions.append(row)
            history.append(obs)
            if value is not None:
                exact_pairs.append((ids[chosen], obs["objective"]))
            if feasible:
                feasible_values[ids[chosen]] = value
            available.pop(chosen)
        best = min(feasible_values.values()) if feasible_values else None
        require(close(best, trial["final_outcome"]["best_exact_feasible"]), "Final incumbent mismatch")
        winners = sorted(cid for cid, value in feasible_values.items() if abs(value-best) <= 1e-12)
        endpoints.append(dict(task=task, repeat=repeat, anchor_size=n0, method=method,
                              best_exact_feasible=best, incumbent_ids=winners,
                              initial_incumbent_ids=sorted(initial_ids.intersection(winners))))
    pairs = []
    for n0 in (12, 24):
        by_method = {e["method"]: e for e in endpoints if e["anchor_size"] == n0}
        for comparator in ("iid_random", "boundary", "marginal_entropy", CEI):
            left, right = by_method[WEIGHTED], by_method[comparator]
            a, b = left["best_exact_feasible"], right["best_exact_feasible"]
            shared = sorted(set(left["incumbent_ids"]) & set(right["incumbent_ids"]))
            shared_initial = sorted(set(left["initial_incumbent_ids"]) & set(right["initial_incumbent_ids"]))
            category = ("neither_feasible" if a is None and b is None else
                        "one_infeasible" if a is None or b is None else
                        "same_observed_candidate" if shared else
                        "different_candidates_equal_value" if abs(a-b) <= 1e-12 else "different_value")
            pairs.append(dict(task=task, repeat=repeat, anchor_size=n0, comparator=comparator,
                              category=category, shared_incumbent_ids=shared,
                              shared_initial_incumbent_ids=shared_initial))
    return decisions, endpoints, pairs


def summarize(rows):
    groups = {}
    for r in rows:
        key = (r["task"], r["anchor_size"], r["source_method"], r["phase"])
        groups.setdefault(key, []).append(r)
    fields = ("rules_disagree", "cei_proxy_fraction_lost", "exact_objective_returned",
              "feasible_returned", "selected_region_brier", "selected_joint_brier", "objective_error")
    result = []
    for (task, n0, method, phase), values in sorted(groups.items()):
        item = dict(task=task, anchor_size=n0, source_method=method, phase=phase,
                    decisions=len(values), repeats=len({v["repeat"] for v in values}))
        for field in fields:
            valid = [v[field] for v in values if v[field] is not None]
            item[field] = dict(n=len(valid), mean=math.fsum(valid)/len(valid) if valid else None)
        errors = [v["objective_squared_error"] for v in values if v["objective_squared_error"] is not None]
        item["objective_rmse"] = dict(n=len(errors), value=math.sqrt(math.fsum(errors)/len(errors)) if errors else None)
        result.append(item)
    return result


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def diagnose(root, output, analyzer_sha):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists() and output != root and root not in output.parents and output not in root.parents, "Output must be new and outside input")
    require(not (root/"failure.json").exists(), "Input is a failed run")
    manifest = read_json(root/"manifest.json")
    require(manifest["schema"] == "sha256_manifest_v1", "Unsupported manifest")
    checked = {}
    def load(rel):
        path = root / rel
        require(not path.is_symlink() and root in path.resolve().parents, "Input path escape")
        h = file_hash(path)
        require(h == manifest["files"].get(rel), "Input hash mismatch: " + rel)
        checked[rel] = h
        return read_json(path)
    plan, report, source = (load(p) for p in ("plan.json", "report.json", "source_receipt.json"))
    require(source["code_sha"] == report["code_sha"] == SOURCE_SHA, "Wrong experiment SHA")
    require(plan["profile"] == "full" and report["status"] == "COMPLETE_PR86D_INITIAL_SAMPLE_ABLATION", "Expected completed D Full")
    require(len(plan["units"]) == 30 and report["trials"] == 300, "Wrong Full roster")
    require({(u["task"], u["repeat"]) for u in plan["units"]} == set(itertools.product(TASKS, range(3))), "Wrong/duplicate task-repeat roster")
    rows, endpoints, pairs = [], [], []
    for unit in plan["units"]:
        rel = "units/%s/r%d/unit.json" % (unit["task"], unit["repeat"])
        data = load(rel)
        require((data["task"], data["repeat"]) == (unit["task"], unit["repeat"]), "Unit identity mismatch")
        d, e, p = diagnose_unit(data)
        rows.extend(d); endpoints.extend(e); pairs.extend(p)
        del data
    require(len(rows) == 9000 and len(endpoints) == 300 and len(pairs) == 240, "Diagnostic roster mismatch")
    output.mkdir(parents=True)
    summary = dict(status="COMPLETE_PR86E_READONLY_DIAGNOSTICS", analyzer_sha=analyzer_sha,
        experiment_sha=SOURCE_SHA, result_commit=RESULT_SHA,
        input_manifest_sha256=file_hash(root/"manifest.json"), verified_input_files=checked,
        decisions=len(rows), trajectories=len(endpoints), identity_pairs=len(pairs),
        new_evaluator_calls=0, new_gp_fits=0, task_stratified=summarize(rows),
        limits=["Exploratory diagnosis on already disclosed data, not prospective evidence.",
                "Same-state alternative choices have predictions only, not unobserved outcomes.",
                "CEI proxy loss is NOT actual objective regret or expected multistep value.",
                "Selected-point residuals are on-policy and conditional on objective observability.",
                "GP z residuals use stored latent variance, not an observation-noise predictive interval.",
                "Decisions within trajectories are dependent; do not count 9000 independent replications.",
                "Only consumed manifest entries were hashed; progress/evaluation journals are not re-audited.",
                "Objective errors are task-stratified; do not pool raw objective scales."])
    write_json(output/"diagnostic_summary.json", summary)
    write_json(output/"incumbent_identity.json", dict(endpoints=endpoints, pairs=pairs))
    with (output/"decision_diagnostics.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    files = {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}
    write_json(output/"manifest.json", dict(schema="sha256_manifest_v1", files=files))
    return {k: summary[k] for k in ("status", "decisions", "trajectories", "identity_pairs", "new_evaluator_calls", "new_gp_fits")}


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Frozen D Full directory in result worktree")
    p.add_argument("--output", type=Path, required=True, help="New directory outside input and analyzer worktree")
    p.add_argument("--expected-analyzer-sha", required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parent
    require(git(repo, "rev-parse", "HEAD") == a.expected_analyzer_sha and not git(repo, "status", "--porcelain"), "Wrong/dirty analyzer checkout")
    require(git(a.input, "rev-parse", "HEAD") == RESULT_SHA and not git(a.input, "status", "--porcelain"), "Wrong/dirty frozen result checkout")
    require(repo not in a.output.resolve().parents, "Output must be outside analyzer checkout")
    print(json.dumps(diagnose(a.input, a.output, a.expected_analyzer_sha), indent=2))


if __name__ == "__main__":
    main()
