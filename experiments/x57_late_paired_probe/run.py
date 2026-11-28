"""Replay RA histories and score matched LSBO proposals as external probes, with no RCAIDE calls."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SHORT = RESULTS / "x57-recovery-pool-pilot-1ecea8d6"
LONG = RESULTS / "x57-frozen-long-pair-c71c32c1-64"
CONFIG = RESULTS / "x57-47p5-five-arm-checkpoint16/distance_47.5.json"
RA = "ra_lsbo_energy_interval"
LSBO = "lsbo_full_budget"
SEEDS = list(range(2026112801, 2026112807))


def load_jobs(base):
    inputs = [(SHORT, "22e5658d4c4dffaea7b1c18e6ba32bdd501d187315333a5ebf8197051fd4911c"),
              (LONG, "acaffd5a96e6235630ea604cad00d5bb66b8f2c99b57eaef240784e272772d89")]
    groups = defaultdict(list)
    for folder, checksum in inputs:
        data = (folder / "calls.jsonl").read_bytes()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError(f"Unexpected ledger: {folder}")
        for r in map(json.loads, data.splitlines()):
            arm = r["arm"]
            if r["event"] != "settled" or arm not in ("shared", RA, LSBO):
                continue
            if folder == SHORT and arm == LSBO:
                continue
            r["source_id"] = r.pop("task_id")
            del r["event"]
            groups[r["seed"], arm].append(r)
    jobs = []
    for seed in SEEDS:
        init, ra, ls = [groups[seed, arm] for arm in ("shared", RA, LSBO)]
        for rows, start, end in ((init, 1, 8), (ra, 9, 64), (ls, 33, 64)):
            if [r["evaluation"] for r in rows] != list(range(start, end + 1)):
                raise ValueError(f"Invalid ledger order: {seed}")
        # The replay process receives the LSBO point only, with no LSBO outcome.
        jobs.append((seed, base, init, ra, [(r["candidate_id"], r["candidate"]) for r in ls]))
    return jobs, groups


def replay(job):
    seed, base, init, online, probes = job
    sys.path.insert(0, str(ROOT))
    import torch
    from tools.x57_checkpoint9_selection import capture_ask, compare_scores
    from ra_lsbo.x57_formal_native import FormalNative
    from ra_lsbo.x57_live import X57LiveProblem
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        p = X57LiveProblem(base, CONFIG, work / "unused", allow_mda=False)
        def forbidden(*args, **kwargs):
            raise RuntimeError("Physical evaluations disabled for this audit")
        p.evaluate = p.problem.evaluate = forbidden
        model = FormalNative(p, RA, seed, init, work / "ra")
        seen = {r["candidate_id"] for r in init}
        records = []
        for i, row in enumerate(online, 9):
            if i < 33:
                point, phase, _ = model.ask()
            else:
                (point, phase, _), traces = capture_ask(model)
                key, candidate = probes[i-33]
                challenger = p.candidate_class(**candidate)
                if p.space.key(challenger) != key:
                    raise ValueError("LSBO candidate identity differs")
                scores = compare_scores(model, traces["performance"], "performance", challenger, point)
                record = dict(seed=seed, evaluation=i, ra_id=p.space.key(point), probe_id=key,
                              novel=key not in seen, in_pool=scores["lsbo_in_final_pool"],
                              rank=scores["lsbo"]["augmented_pool_rank"],
                              ra_minus_probe_log_score=scores["ra_minus_lsbo_log_score"])
                for name, values in (("probe", scores["lsbo"]), ("ra", scores["ra"])):
                    for k in ("log_expected_improvement", "p_complete", "p_usable_energy"):
                        record[f"{name}_{k}"] = values[k]
                    record[f"{name}_p_stall"] = values["other_constraint_probabilities"].get("stall_speed")
                ei_ra = record["ra_log_expected_improvement"]
                ei_probe = record["probe_log_expected_improvement"]
                if ei_ra is not None and ei_probe is not None:
                    ei_gap = ei_ra - ei_probe
                else:
                    ei_gap = None
                record["ra_minus_probe_log_ei"] = ei_gap
                record["remaining_log_score_gap"] = (
                    None if ei_gap is None else record["ra_minus_probe_log_score"] - ei_gap
                )
                records.append(record)
            if (p.space.key(point), phase, i) != (row["candidate_id"], row["phase"], row["evaluation"]):
                raise ValueError(f"RA replay differs: {seed}/{i}")
            model.tell(row)
            seen.add(row["candidate_id"])
            if i % 8 == 0:
                print(f"[seed {seed}] replayed {i}/64", flush=True)
    return records


def gain(row, incumbent):
    f = row["feedback"]
    return max(0., incumbent - f["objective_Wh"]) if f["usable"] else 0.


def join(rows, groups):
    for r in rows:
        seed, i = r["seed"], r["evaluation"]
        hist = groups[seed, "shared"] + groups[seed, RA][:i-9]
        best = min(h["feedback"]["objective_Wh"] for h in hist if h["feedback"]["usable"])
        probe, selected = groups[seed, LSBO][i-33], groups[seed, RA][i-9]
        if (probe["candidate_id"], selected["candidate_id"]) != (r["probe_id"], r["ra_id"]):
            raise ValueError("Join identity mismatch")
        r.update(incumbent_Wh=best, probe_gain_Wh=gain(probe, best), ra_gain_Wh=gain(selected, best))
        if r["probe_gain_Wh"] <= r["ra_gain_Wh"]:
            r["diagnosis"] = "no_missed_improvement"
        elif not r["novel"]:
            r["diagnosis"] = "previously_evaluated"
        elif r["ra_minus_probe_log_score"] < -1e-8:
            r["diagnosis"] = "probe_would_win"
        elif abs(r["ra_minus_probe_log_score"]) <= 1e-8:
            r["diagnosis"] = "tied_score"
        else:
            r["diagnosis"] = "probe_scored_lower"
    return dict(Counter(r["diagnosis"] for r in rows))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("D:/RA-LSBO"))
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    if hashlib.sha256(CONFIG.read_bytes()).hexdigest() != "9902197262e862b13a3b3ac4a75c6f2c7892c91d8a69d6ce1af8aa884f6e6c3d":
        raise ValueError("Unexpected physical config")
    jobs, groups = load_jobs(args.workspace / "x57-qualification-v1-final/physical-base")
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        tasks = {pool.submit(replay, job): job[0] for job in jobs}
        for future in as_completed(tasks):
            rows.extend(future.result())
            print(f"[replay] seed {tasks[future]} verified to 64 ({len(rows)}/192)", flush=True)
    rows.sort(key=lambda r:(r["seed"],r["evaluation"]))
    out = args.workspace / "x57-late-paired-probe-64"
    out.mkdir(exist_ok=True)
    write_csv(out / "predictions.csv", rows)
    counts = join(rows, groups)
    write_csv(out / "paired_probe_audit.csv", rows)
    (out / "summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(f"{counts}; RCAIDE calls 0; {out}", flush=True)


if __name__ == "__main__":
    main()
