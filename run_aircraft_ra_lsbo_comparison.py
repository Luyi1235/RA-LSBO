"""Cold-start aircraft comparison with the native LSBO and frozen risk-aware baselines."""
import argparse
from collections import Counter
from dataclasses import replace
import math
from pathlib import Path
import time

import numpy as np

from aircraft_risk_active import configuration_brier
from ra_lsbo.budget import digest
from ra_lsbo.experiments.comparison.methods import ExternalOptimizer
from ra_lsbo.experiments.core.metrics import trajectory_metrics
from ra_lsbo.experiments.core.problems import AircraftCase, FlightScenario
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.experiments.core.runner import Ledger, NATIVE_STOP, append, lines, plain, read, save
from ra_lsbo.experiments.discovery_stages.methods import NativeLSBO
from ra_lsbo.experiments.simplified_aircraft.dualview_protocol import model_settings
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import (
    HeterogeneousRiskPosterior, ModelNumericalFailure, StableCorpusDiscoveryBO, StableOnehotDiscoveryBO,
)
from ra_lsbo.partial_bo.models import PrefixFeatures
from ra_lsbo.partial_bo.problems import candidate_pool
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.soc_revision.acquisition import quantities
from ra_lsbo.structure_corpus import feature_snapshot
from run_aircraft_risk_cold_start import file_hash, source_commit


ARMS = ("ra_separate", "original_lsbo", "ra_joint_kernel", "ra_explicit", "logcei", "scbo_adapted")
COMPARATORS = ARMS[1:]
RA_MODELS = ("ra_separate", "ra_joint_kernel")
PROFILES = {
    "comparison-smoke": (4., 0., 1, 3, 32, 12, 28469900),
    "comparison-headwind": (4., 0., 24, 20, 128, 32, 28470000),
    "comparison-payload": (0., 24., 24, 20, 128, 32, 28470000),
}


def plan(name):
    wind, payload, runs, horizon, pool, heldout, seed = PROFILES[name]
    return {"name": name, "headwind_m_s": wind, "payload_delta_kg": payload, "horizon": horizon,
            "initial": 4, "pool": pool, "heldout": heldout, "arms": list(ARMS),
            "seeds": [seed+100*i for i in range(runs)], "encoder_epochs": 30, "native_gp_steps": 45,
            "checkpoints": [4, 7] if name == "comparison-smoke" else [4, 8, 12, 16],
            "budget": {"initial": runs*4, "online": runs*horizon*len(ARMS), "heldout": runs*heldout},
            "calibration_path": "logcei", "rcaide_mda_calls": 0}


class BenchmarkAircraft(AircraftCase):
    """Use mission completion semantics consistently in every existing engine."""

    def __init__(self, spec):
        scenario = FlightScenario(headwind_m_s=spec["headwind_m_s"], payload_delta_kg=spec["payload_delta_kg"])
        super().__init__(BASE, "sparse", scenario=scenario, integration_step=20.)
        self.contract_id = digest([self.contract_id, "paired-ra-lsbo-mission-completion-v1"])

    def evaluate(self, raw):
        record = super().evaluate(raw)
        observation, diagnostics = record.observation, record.diagnostics
        if observation.status == "completed" and diagnostics.get("mission_completed") is False:
            observation = replace(observation, status="hidden_failure",
                                  failure_cause=diagnostics["termination_reason"])
            return DesignRecord(record.candidate, observation, diagnostics)
        return record


class JointKernelRA(StableCorpusDiscoveryBO):
    """Joint marginal-likelihood fit of one grouped bandwidth for all heads."""

    def _build_posterior(self, records):
        latent = feature_snapshot(self.problem, self.base, records, self.session)
        explicit = PrefixFeatures(self.problem, self.base, records, self.seed, latent=False)
        return HeterogeneousRiskPosterior(self.problem, records, self.seed, self.settings,
                                          latent, explicit, view_policy=self.allocation, shared_kernel=True)


def ra_settings(spec, *, latent=True):
    return replace(model_settings("hybrid_onehot" if latent else "onehot_stable", spec), local=False)


def engine_for(problem, seed, arm, spec, output):
    if arm == "original_lsbo":
        return NativeLSBO(problem, BASE, seed, spec, output)
    if arm in ("logcei", "scbo_adapted"):
        return ExternalOptimizer(problem, seed, arm, spec)
    if arm == "ra_explicit":
        return StableOnehotDiscoveryBO(problem, BASE, seed, ra_settings(spec, latent=False),
                                       cold_global=True, protect=True, feedback="interval", coverage_scope="cold")
    engine = JointKernelRA if arm == "ra_joint_kernel" else StableCorpusDiscoveryBO
    return engine(problem, BASE, seed, ra_settings(spec), output, allocation="onehot_risk")


def calibration_forecasts(problem, spec, seed, prefix, heldout, folder):
    """Fit both training schemes on the same independently generated path."""
    engine = engine_for(problem, seed, "ra_separate", spec, folder)
    rows = []
    for n in spec["checkpoints"]:
        history = prefix[:n]
        engine._update_representation(history, n-spec["initial"])
        latent = feature_snapshot(problem, BASE, history, engine.session)
        explicit = PrefixFeatures(problem, BASE, history, seed, latent=False)
        for arm in RA_MODELS:
            receipt = {"n": n, "arm": arm, "prefix": digest([r.to_dict() for r in history])}
            try:
                posterior = HeterogeneousRiskPosterior(problem, history, seed, engine.settings, latent, explicit,
                                                        shared_kernel=arm == "ra_joint_kernel")
                values = quantities(posterior.predict(heldout), None, 1., .5)
                receipt.update(usable=values["joint"].tolist(), completion=values["success"].tolist())
            except ModelNumericalFailure as error:
                receipt["fit_failure"] = {"head": error.head, "grid": error.failures}
            rows.append(receipt)
    save(folder/"predictions.json", rows)


def run(spec, sha, output):
    import torch

    source_commit(sha, clean=True)
    if output.resolve().is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("Use a new evidence directory outside the source worktree")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    problem = BenchmarkAircraft(spec)
    save(output/"plan.json", {"source_sha": sha, "experiment": spec, "contract": problem.contract_id})
    ledger = Ledger(output, sum(spec["budget"].values()))
    for seed in spec["seeds"]:
        unit = output/str(seed)
        seen, roster = set(), {}
        for offset, role in enumerate(("initial", "pool", "heldout"), 1):
            roster[role], _ = candidate_pool(problem, seed+offset, spec[role], seen)
            seen.update(map(digest, roster[role]))
        save(unit/"roster.json", roster)
        initial = [ledger.evaluate(problem, row, f"{seed}/initial/{i}", "initial")
                   for i, row in enumerate(roster["initial"])]
        for arm in spec["arms"]:
            folder = unit/arm
            folder.mkdir()
            optimizer = engine_for(problem, seed, arm, spec, folder)
            history, remaining = list(initial), list(roster["pool"])
            started = time.monotonic()
            for i in range(spec["horizon"]):
                candidate, info = optimizer.propose(history, remaining, i)
                if arm != "original_lsbo":
                    remaining.remove(candidate)
                if digest(candidate) in {row.observation.design_id for row in history}:
                    raise ValueError("Method proposed a repeated design")
                append(folder/"proposals.jsonl", {
                    "candidate": candidate, "prefix": digest([row.to_dict() for row in history]),
                    "mode": info.get("mode", info.get("phase", info.get("original_proposal", {}).get("source"))),
                    "selected_kernel": info.get("shared_kernel_index"),
                    "coverage_reason": info.get("coverage_reason"),
                })
                result = ledger.evaluate(problem, candidate, f"{seed}/{arm}/{i}", "online")
                optimizer.accept(result)
                history.append(result)
                if getattr(optimizer, "stop_reason", None):
                    break
            save(folder/"run.json", {"online_calls": len(history)-len(initial),
                                    "stop_reason": getattr(optimizer, "stop_reason", None),
                                    "elapsed_s": time.monotonic()-started,
                                    **({"native_settings": optimizer.engine.settings} if arm == "original_lsbo" else {})})
            if arm == spec["calibration_path"]:
                calibration_forecasts(problem, spec, seed, history, roster["heldout"], unit/"calibration")
            print(f"{seed} {arm} {len(history)-len(initial)} online calls", flush=True)
        for i, row in enumerate(roster["heldout"]):
            ledger.evaluate(problem, row, f"{seed}/heldout/{i}", "heldout")
    save(output/"seal.json", evidence_hashes(output))


def evidence_hashes(folder):
    return {p.relative_to(folder).as_posix(): file_hash(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and p != folder/"seal.json"}


def read_ledger(folder, contract):
    calls = {}
    stream = iter(lines(folder/"calls.jsonl"))
    for index, start in enumerate(stream, 1):
        end = next(stream, None)
        payload = {k: v for k, v in start.items() if k not in {"event", "call_id"}}
        if (start["event"] != "started" or start["number"] != index or start["contract"] != contract
                or start["new_rcaide_mda"] or start["call_id"] != digest(payload)
                or end is None or end["event"] != "settled" or end["call_id"] != start["call_id"]
                or end["tag"] != start["tag"] or start["tag"] in calls
                or digest(end["record"]) != end["record_digest"]):
            raise ValueError("Invalid started/settled ledger call")
        row = DesignRecord.from_dict(end["record"])
        if row.candidate != start["candidate"] or row.observation.contract_id != contract:
            raise ValueError("Ledger outcome has the wrong design or contract")
        calls[start["tag"]] = (index, row, start["role"])
    return calls


def paired_interval(values):
    if len(values) < 2:
        return None
    a = np.asarray(values, dtype=float)
    random = np.random.default_rng(20261116)
    draws = a[random.integers(len(a), size=(4000, len(a)))].mean(axis=1)
    return np.quantile(draws, [.025, .975]).tolist()


def comparisons(trajectories, spec):
    reference = {r["seed"]: r for r in trajectories if r["method"] == "ra_separate"}
    output = []
    for baseline in COMPARATORS:
        paired = [(reference[row["seed"]], row) for row in trajectories if row["method"] == baseline]
        for cohort in ("cold", "all"):
            group = [(a, b) for a, b in paired if cohort == "all" or a["initial_kind"] == cohort]
            rows = []
            for a, b in group:
                first, second = (r["first_feasible_online_call"] for r in (a, b))
                xa, xb = a["best_feasible"], b["best_feasible"]
                if xa is None or xb is None:
                    win = int(xa is not None)-int(xb is not None)
                else:
                    win = 0 if math.isclose(xa, xb, rel_tol=1e-10) else (1 if xa < xb else -1)
                limit = spec["horizon"]+1
                rows.append({"seed": a["seed"], "ra_first": first, "baseline_first": second,
                             "capped_first_call_delta": (limit if first is None else first)
                                                       -(limit if second is None else second),
                             "discovery_delta": int(first is not None)-int(second is not None),
                             "discovery_by_8_delta": int(first is not None and first <= 4)
                                                     -int(second is not None and second <= 4),
                             "terminal_feasibility_energy_win": win,
                             "energy_delta_Wh": xa-xb if xa is not None and xb is not None else None})
            result = {"comparator": baseline, "cohort": cohort, "pairs": rows, "n": len(rows)}
            for metric in ("capped_first_call_delta", "discovery_delta", "discovery_by_8_delta",
                           "terminal_feasibility_energy_win", "energy_delta_Wh"):
                values = [r[metric] for r in rows if r[metric] is not None]
                result[metric] = {"n": len(values), "mean": sum(values)/len(values) if values else None,
                                  "paired_95": paired_interval(values)}
            output.append(result)
    return output


def verify(folder, sha):
    source_commit(sha)
    if read(folder/"seal.json") != evidence_hashes(folder):
        raise ValueError("Evidence inventory or checksum differs")
    header = read(folder/"plan.json")
    spec = plan(header["experiment"]["name"])
    if header["source_sha"] != sha or header["experiment"] != spec:
        raise ValueError("Evidence belongs to another source or protocol")
    calls = read_ledger(folder, header["contract"])
    roles = Counter(r for _, _, r in calls.values())
    if (roles["initial"] != spec["budget"]["initial"] or roles["heldout"] != spec["budget"]["heldout"]
            or roles["online"] > spec["budget"]["online"]):
        raise ValueError("Charged calls differ from the frozen budget")
    used, trajectories, calibration = set(), [], []

    def use(tag, role, candidate):
        index, row, actual_role = calls[tag]
        if tag in used or actual_role != role or row.candidate != candidate:
            raise ValueError("Ledger calls do not match the frozen panel or proposal")
        used.add(tag)
        return index, row

    for seed in spec["seeds"]:
        unit = folder/str(seed)
        roster = read(unit/"roster.json")
        designs = [row for role in ("initial", "pool", "heldout") for row in roster[role]]
        if (any(len(roster[role]) != spec[role] for role in roster)
                or len(set(map(digest, designs))) != len(designs)):
            raise ValueError("Frozen panels must be sized and disjoint")
        initial = [use(f"{seed}/initial/{i}", "initial", c) for i, c in enumerate(roster["initial"])]
        heldout = [use(f"{seed}/heldout/{i}", "heldout", c) for i, c in enumerate(roster["heldout"])]
        for arm in spec["arms"]:
            folder_arm = unit/arm
            receipt = read(folder_arm/"run.json")
            proposals = lines(folder_arm/"proposals.jsonl")
            stop = receipt["stop_reason"]
            if (receipt["online_calls"] != len(proposals) or not 0 < len(proposals) <= spec["horizon"]
                    or (stop is not None and (arm != "original_lsbo" or stop != NATIVE_STOP))
                    or (stop is None and len(proposals) != spec["horizon"])):
                raise ValueError("Only a recorded native LSBO stopping rule may leave budget unused")
            anchors = [row for _, row in initial]
            history, previous = list(anchors), initial[-1][0]
            seen = {row.observation.design_id for row in anchors}
            available = set(map(digest, roster["pool"]))
            for i, p in enumerate(proposals):
                candidate = p["candidate"]
                key = digest(candidate)
                if key in seen or p["prefix"] != digest([row.to_dict() for row in history]):
                    raise ValueError("Proposal uses a different observed prefix")
                if arm != "original_lsbo":
                    available.remove(key)
                seen.add(key)
                index, row = use(f"{seed}/{arm}/{i}", "online", candidate)
                if not previous < index < heldout[0][0]:
                    raise ValueError("Online evaluation violates time or holdout order")
                previous = index
                history.append(row)
            if arm == "original_lsbo":
                native = receipt["native_settings"]
                if (native["initial"] != spec["initial"] or native["budget"] != spec["initial"]+spec["horizon"]
                        or native["gp_steps"] != spec["native_gp_steps"] or native["max_consecutive_solver_failures"] != 5):
                    raise ValueError("Original LSBO settings differ from the native frozen configuration")
                if stop is not None and (len(proposals) < 5 or any(row.observation.objective.kind == "exact"
                                                                  for row in history[-5:])):
                    raise ValueError("Native stop is not supported by the final five missing objectives")
            if arm == spec["calibration_path"]:
                calibration.extend(score_calibration(spec, seed, history, heldout, unit))
            trajectories.append({"seed": seed, "method": arm, "stop_reason": stop, "elapsed_s": receipt["elapsed_s"],
                                 **trajectory_metrics(anchors, history[len(anchors):], spec["horizon"], bool(stop))})
    if used != set(calls):
        raise ValueError("Unexpected charged calls in ledger")
    return {"status": "VERIFIED_RA_LSBO_COMPARISON", "source_sha": sha, "roles": dict(roles),
            "total_calls": len(calls), "max_calls": sum(spec["budget"].values()), "rcaide_mda_calls": 0,
            "trajectories": trajectories, "comparisons": comparisons(trajectories, spec),
            "calibration": calibration, "calibration_comparison": compare_calibration(calibration)}


def score_calibration(spec, seed, history, heldout, unit):
    rows = []
    seen = set()
    panel = [row for _, row in heldout]
    labels = {"completion": [r.observation.completion_label() for r in panel],
              "usable": [None if r.observation.completion_label() is None else r.usable_feasible for r in panel]}
    for row in read(unit/"calibration"/"predictions.json"):
        key = row["n"], row["arm"]
        if (key in seen or row["prefix"] != digest([r.to_dict() for r in history[:row["n"]]])):
            raise ValueError("Invalid shared calibration prefix")
        seen.add(key)
        scores = {"seed": seed, "n": row["n"], "method": row["arm"], "fit_failure": row.get("fit_failure")}
        for head, truth in labels.items():
            if scores["fit_failure"] is not None:
                scores[head] = None
                continue
            probabilities = row[head]
            if len(probabilities) != len(panel) or not all(0. <= p <= 1. for p in probabilities):
                raise ValueError("Invalid held-out probability forecast")
            scores[head], _ = configuration_brier(zip([r.candidate for r in panel], truth, probabilities))
        rows.append(scores)
    if seen != {(n, arm) for n in spec["checkpoints"] for arm in RA_MODELS}:
        raise ValueError("Missing calibration comparison")
    return rows


def compare_calibration(rows):
    index = {(r["seed"], r["n"], r["method"]): r for r in rows}
    output = {}
    for head in ("completion", "usable"):
        pair_means = []
        by_seed = {}
        for (seed, n, arm), left in index.items():
            if arm != "ra_separate":
                continue
            right = index[seed, n, "ra_joint_kernel"]
            by_seed.setdefault(seed, []).append(None if left[head] is None or right[head] is None
                                                 else left[head]-right[head])
        for seed, values in sorted(by_seed.items()):
            if None not in values:
                pair_means.append([seed, sum(values)/len(values)])
        deltas = [delta for _, delta in pair_means]
        output[head] = {"seed_deltas": pair_means, "incomplete_seeds": len(by_seed)-len(pair_means),
                        "mean_delta": sum(deltas)/len(deltas) if deltas else None,
                        "paired_95": paired_interval(deltas)}
    return output


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--task", choices=("plan", "run", "verify"), required=True)
    cli.add_argument("--profile", choices=PROFILES)
    cli.add_argument("--expected-sha")
    cli.add_argument("--input", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.task in ("plan", "run") and not args.profile:
        cli.error("--profile is required")
    if args.task == "plan":
        import json
        print(json.dumps(plan(args.profile), indent=2))
    elif args.task == "run":
        if not args.expected_sha or args.output is None:
            cli.error("--expected-sha and --output are required")
        run(plan(args.profile), args.expected_sha, args.output)
    else:
        if not args.expected_sha or args.input is None or args.output is None:
            cli.error("--expected-sha, --input and --output are required")
        result = verify(args.input, args.expected_sha)
        save(args.output, plain(result))
        print(result["status"], result["total_calls"])


if __name__ == "__main__":
    main()
