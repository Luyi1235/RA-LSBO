"""Matched X-57 ablation of the performance GP mean, on frozen RA histories."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.x57_late_paired_probe import run as probe

AUDIT = ROOT / "results/x57-late-paired-probe-64/paired_probe_audit.csv"


def flat_fit(base, train, targets, points, best, seed, steps, location, scale):
    sys.path.insert(0, str(base))
    import torch
    from botorch.acquisition.analytic import LogExpectedImprovement
    from lsbo.surrogate import fit_gp
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    def to_tensor(value):
        return torch.as_tensor(value, dtype=torch.double)
    model = fit_gp(to_tensor(train), to_tensor(targets).reshape(-1, 1), steps)
    with torch.no_grad():
        x = to_tensor(points).unsqueeze(-2)
        post = model.posterior(x)
        log_ei = LogExpectedImprovement(model, best)(x).reshape(-1)
        return log_ei.tolist(), (-(post.mean.reshape(-1) * scale + location)).tolist(), (
            scale * post.variance.reshape(-1).sqrt()).tolist()


def replay_seed(job):
    import numpy as np
    import torch
    seed, base = job[:2]
    pending = []
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as trainee:
        def capture(adapter, trace, selected, external, record):
            head = trace["surrogate"]
            engineering = getattr(head, "native", head) if getattr(head, "completion", None) is not None else head
            raw = engineering.native if getattr(engineering, "energy", None) is not None else engineering
            if raw.improvement is None:
                raise ValueError("Performance stage has no EI model")
            candidates = list(trace["candidates"]) + [external]
            idx = trace["index"]
            with torch.no_grad():
                x = (trace["rep"].features(candidates) - raw.offset) / raw.scale
                q = x.unsqueeze(-2)
                post = raw.model.posterior(q)
                original = raw.improvement(q).reshape(-1).numpy()
                # Keep exactly the original score for every in-pool candidate.
                joint = np.append(np.asarray(trace["scores"]).reshape(-1),
                                  float(np.asarray(trace["scores"]).reshape(-1)[idx]) - record["ra_minus_probe_log_score"])
                for i, who in ((idx, "ra"), (-1, "probe")):
                    if abs(original[i] - record[who + "_log_expected_improvement"]) > 1e-7:
                        raise ValueError("Original acquisition did not reproduce")
                    record[who + "_original_mean_Wh"] = -(float(post.mean.reshape(-1)[i])*raw.std + raw.mean)
                    record[who + "_original_sd_Wh"] = float(post.variance.reshape(-1)[i].sqrt()) * raw.std
                record["native_mean_type"] = type(raw.model.mean_module).__name__
                future = trainee.submit(flat_fit, base, raw.model.train_inputs[0].detach().numpy(),
                                        raw.model.train_targets.detach().numpy(), x.detach().numpy(),
                                        float(raw.improvement.best_f), seed + len(adapter.history),
                                        adapter.settings.gp_steps, raw.mean, raw.std)
                keys = [adapter.problem.space.key(p) for p in candidates]
                pending.append((future, record, joint-original, idx, keys))

        records = probe.replay(job, score_hook=capture)
        for future, rec, remainder, idx, keys in pending:
            flat, means, sd = future.result()
            score = remainder + flat
            if not np.isfinite(score).all():
                raise ValueError("Nonfinite composite score")
            for i, who in ((idx,"ra"), (-1,"probe")):
                rec[who + "_flat_mean_Wh"] = means[i]
                rec[who + "_flat_sd_Wh"] = sd[i]
            rec["flat_ra_minus_probe_score"] = float(score[idx]-score[-1])
            rec["flat_probe_beats_original_pool"] = bool(score[-1] > max(score[:-1]) + 1e-8)
            rec["flat_best_in_original_pool"] = keys[int(np.argmax(score[:-1]))]
    return records


def evaluate(rows, groups):
    probe.join(rows, groups)
    with AUDIT.open(encoding="utf-8-sig", newline="") as f:
        frozen = {(int(r["seed"]), int(r["evaluation"])):r for r in csv.DictReader(f)}
    if len(rows) != len(frozen) or len(rows) != 192:
        raise ValueError("Unexpected number of rows")
    for r in rows:
        ref = frozen[r["seed"], r["evaluation"]]
        if any(ref[k] != r[k] for k in ("probe_id", "ra_id", "diagnosis")) or abs(
                float(ref["ra_minus_probe_log_score"]) - r["ra_minus_probe_log_score"]) > 1e-7:
            raise ValueError("Frozen cross-score mismatch")
        for who, arm, offset in (("ra",probe.RA,9), ("probe",probe.LSBO,33)):
            f = groups[r["seed"], arm][r["evaluation"]-offset]["feedback"]
            r[who+"_observed_Wh"] = f["objective_Wh"] if f["completion"] is True else None


def summarize(rows):
    from collections import Counter
    decided = [r for r in rows if r["novel"] and r["flat_ra_minus_probe_score"] < -1e-8]
    out = dict(n=len(rows), native_means=dict(Counter(r["native_mean_type"] for r in rows)),
               recovered=sum(r["probe_gain_Wh"] > r["ra_gain_Wh"] for r in decided),
               sacrificed=sum(r["probe_gain_Wh"] < r["ra_gain_Wh"] for r in decided),
               recovered_first_in_augmented_pool=sum(r["probe_gain_Wh"] > r["ra_gain_Wh"] and
                                                      r["flat_probe_beats_original_pool"] for r in decided),
               mean_paired_realized_gain_change_Wh=sum(r["probe_gain_Wh"]-r["ra_gain_Wh"] for r in decided)/len(rows),
               original_pool_winner_changed=sum(r["flat_best_in_original_pool"] != r["ra_id"] for r in rows))
    for who in ("ra", "probe"):
        exact = [r for r in rows if r[who+"_observed_Wh"] is not None]
        out[who+"_exact_n"] = len(exact)
        for model in ("original", "flat"):
            out[who+"_"+model+"_mae_Wh"] = statistics.mean(abs(
                r[who+"_"+model+"_mean_Wh"]-r[who+"_observed_Wh"]) for r in exact) if exact else None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("D:/RA-LSBO"))
    root = parser.parse_args().workspace.resolve()
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","NUMBA_NUM_THREADS"):
        os.environ[key] = "1"
    import hashlib
    if hashlib.sha256(probe.CONFIG.read_bytes()).hexdigest() != "9902197262e862b13a3b3ac4a75c6f2c7892c91d8a69d6ce1af8aa884f6e6c3d":
        raise ValueError("Changed physical config")
    jobs, groups = probe.load_jobs(root/"x57-qualification-v1-final/physical-base")
    rows = []
    with ProcessPoolExecutor(6, mp_context=mp.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(replay_seed, job) for job in jobs]):
            rows.extend(future.result())
            print(f"[verified] {len(rows)}/192", flush=True)
    rows.sort(key=lambda r:(r["seed"], r["evaluation"]))
    dest = root/"x57-performance-mean-ablation-64"
    dest.mkdir(exist_ok=True)
    probe.write_csv(dest/"predictions.csv", rows)
    evaluate(rows, groups)
    probe.write_csv(dest/"paired.csv", rows)
    summary = {"all": summarize(rows), "seeds": {s:summarize([r for r in rows if r["seed"]==s]) for s in probe.SEEDS},
               "new_rcaide_calls": 0}
    (dest/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary["all"], indent=2), flush=True)


if __name__ == "__main__":
    main()
