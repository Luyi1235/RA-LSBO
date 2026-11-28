"""Check one-step-ahead probabilities on the selected designs of the original interval arm."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/x57-recovery-pool-pilot-1ecea8d6"
ARM = "ra_lsbo_energy_interval"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def label(row, name):
    from ra_lsbo.x57_energy_stop import verified_stop
    fb = row["feedback"]
    kind = fb["constraint_kinds"][name]
    if kind == "exact" and fb["constraints"][name] is not None:
        return int(fb["constraints"][name] <= 0)
    if (name == "usable_energy" and kind == "interval" and name in fb.get("known_constraint_violations", [])
            and verified_stop(row["worker_result"])):
        return 0
    return None


def replay(job):
    seed, base, initial, online = job
    sys.path.insert(0, str(ROOT))
    from tools.x57_checkpoint9_selection import capture_ask
    from ra_lsbo.x57_formal_native import FormalNative
    from ra_lsbo.x57_live import X57LiveProblem
    import torch
    torch.set_num_threads(1)
    stats = {"stall_speed": [], "usable_energy": []}
    def learn(row):
        for name, history in stats.items():
            y = label(row, name)
            if y is not None:
                history.append(y)
    for row in initial:
        learn(row)
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        problem = X57LiveProblem(base, DATA / "distance_47.5.json", work / "unused", allow_mda=False)
        def forbidden(*args, **kwargs):
            raise RuntimeError("Physical calls are disabled for risk replay")
        problem.evaluate = problem.problem.evaluate = forbidden
        optimizer = FormalNative(problem, ARM, seed, initial, work / "optimizer")
        scores = []
        for index, row in enumerate(online, 9):
            if index < 17:
                point, phase, info = optimizer.ask()
            else:
                (point, phase, info), traces = capture_ask(optimizer)
                trace = traces["performance"]
                if problem.space.key(trace["candidates"][trace["index"]]) != problem.space.key(point):
                    raise ValueError(f"Final ranking mismatch for {seed}/{index}")
                ranking = trace["surrogate"]
                engineering = ranking.native if hasattr(ranking, "completion") else ranking
                energy = getattr(engineering, "energy", None)
                raw = engineering.native if energy is not None else engineering
                stall = raw.constraints.get("stall_speed")
                with torch.no_grad():
                    features = trace["rep"].features([point])
                    predictions = {"usable_energy": energy.log_probability(features).exp().item()
                                   if energy is not None else None, "stall_speed": None}
                    if stall is not None:
                        model, threshold = stall
                        post = model.posterior(((features - raw.offset) / raw.scale).unsqueeze(-2))
                        z = (threshold - post.mean) / post.variance.clamp_min(1e-12).sqrt()
                        predictions["stall_speed"] = torch.special.ndtr(z).item()
            if ((point.to_dict(), phase, problem.space.key(point), index) !=
                    (row["candidate"], row["phase"], row["candidate_id"], row["evaluation"])):
                raise ValueError(f"Replay changed {seed}/{index}")
            if index >= 17:
                for name, p in predictions.items():
                    h = stats[name]
                    scores.append(dict(seed=seed, evaluation=index, candidate_id=row["candidate_id"],
                                       outcome=row["feedback"]["outcome"], constraint=name, probability=p,
                                       prior_probability=(1+sum(h))/(2+len(h)), observed_satisfied=label(row, name)))
            learn(row)
            optimizer.tell(row)
        return scores


def main(workspace):
    os.environ.update(dict.fromkeys(THREADS, "1"))
    base = workspace / "x57-qualification-v1-final/physical-base"
    data = (DATA / "calls.jsonl").read_bytes()
    if hashlib.sha256(data).hexdigest() != "22e5658d4c4dffaea7b1c18e6ba32bdd501d187315333a5ebf8197051fd4911c":
        raise ValueError("Archive ledger mismatch")
    seeds = range(2026112801, 2026112807)
    groups = defaultdict(list)
    for r in map(json.loads, data.splitlines()):
        if r["event"] == "settled" and r["arm"] in ("shared", ARM):
            r["source_id"] = r.pop("task_id")
            del r["event"]
            groups[r["seed"], r["arm"]].append(r)
    jobs = [(s, base, groups[s, "shared"], groups[s, ARM]) for s in seeds]
    if any(len(initial) != 8 or len(online) != 24 for _, _, initial, online in jobs):
        raise ValueError("Incomplete frozen trajectories")
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for seed, records in zip(seeds, pool.map(replay, jobs)):
            rows.extend(records)
            print(f"[replay] {seed}, verified to 32", flush=True)
    output = workspace / "x57-recovery-pilot-late-risk-audit"
    output.mkdir(exist_ok=True)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for name in ("stall_speed", "usable_energy"):
        group = [r for r in rows if r["constraint"] == name]
        known = [r for r in group if r["observed_satisfied"] is not None and r["probability"] is not None]
        summary[name] = {"selected": len(group), "scored_with_label": len(known),
                         "mean_probability": mean(r["probability"] for r in known) if known else None,
                         "observed_fraction": mean(r["observed_satisfied"] for r in known) if known else None,
                         "brier": mean((r["probability"]-r["observed_satisfied"])**2 for r in known) if known else None,
                         "history_prior_brier": mean((r["prior_probability"]-r["observed_satisfied"])**2
                                                     for r in known) if known else None}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Verified late risk predictions, new RCAIDE calls 0: {output}", flush=True)
