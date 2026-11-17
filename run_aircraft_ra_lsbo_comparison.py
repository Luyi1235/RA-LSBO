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
from ra_lsbo.experiments.discovery_stages.policies import COVERAGE_PERIOD
from ra_lsbo.experiments.simplified_aircraft.dualview_protocol import model_settings
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import (
    C_GRID, D_GRID, HeterogeneousRiskPosterior, ModelNumericalFailure, StableCorpusDiscoveryBO, StableOnehotDiscoveryBO,
)
from ra_lsbo.partial_bo.models import PrefixFeatures
from ra_lsbo.partial_bo.problems import candidate_pool
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.risk_core.observations import OutputObservation
from ra_lsbo.soc_revision.acquisition import quantities
from ra_lsbo.structure_corpus import feature_snapshot
from run_aircraft_risk_cold_start import file_hash, source_commit


ARMS = ("ra_separate", "original_lsbo", "ra_joint_kernel", "ra_explicit", "logcei", "scbo_adapted")
HANDOFF_FROM = {
    "ra_to_lsbo": "ra_separate",
    "ra_joint_to_lsbo": "ra_joint_kernel",
    "ra_feasibility_to_lsbo": "ra_feasibility",
    "ra_stop_status_to_lsbo": "ra_stop_status",
    "logcei_to_lsbo": "logcei",
}
FULL_NATIVE = "lsbo_full_budget"
NATIVE_ARMS = {"original_lsbo", FULL_NATIVE}
HANDOFF_ARMS = ("ra_to_lsbo", FULL_NATIVE, "original_lsbo", "ra_separate", "logcei_to_lsbo", "logcei")
TRAINING_ARMS = ("ra_to_lsbo", "ra_joint_to_lsbo", "logcei_to_lsbo")
TRAINING_SMOKE_SEEDS = (28470100, 28475000)
BOUNDARY_SMOKE_SEEDS = (28470100, 28480900)
BOUNDARY_ARMS = ("ra_to_lsbo", "ra_feasibility_to_lsbo")
BOUNDARY_INFORMATION = "budgeted_boundary_information"
STOP_EVIDENCE_ARMS = ("ra_to_lsbo", "ra_stop_status_to_lsbo")
STOP_EVIDENCE_SMOKE_SEEDS = (28470100, 28480900)
RA_MODELS = ("ra_separate", "ra_joint_kernel")
PROFILES = {
    "stop-evidence-smoke": (4., 0., len(STOP_EVIDENCE_SMOKE_SEEDS), 6, 128, 0, STOP_EVIDENCE_SMOKE_SEEDS[0]),
    "stop-evidence-headwind": (4., 0., 40, 20, 128, 0, 28490000),
    "stop-evidence-payload": (0., 24., 40, 20, 128, 0, 28490000),
    "boundary-smoke": (4., 0., len(BOUNDARY_SMOKE_SEEDS), 6, 128, 0, BOUNDARY_SMOKE_SEEDS[0]),
    "boundary-headwind": (4., 0., 40, 20, 128, 0, 28485000),
    "boundary-payload": (0., 24., 40, 20, 128, 0, 28485000),
    "training-smoke": (4., 0., len(TRAINING_SMOKE_SEEDS), 6, 128, 0, TRAINING_SMOKE_SEEDS[0]),
    "training-headwind": (4., 0., 40, 20, 128, 0, 28480000),
    "training-payload": (0., 24., 40, 20, 128, 0, 28480000),
    "handoff-smoke": (4., 0., 2, 6, 128, 0, 28470100),
    "handoff-headwind": (4., 0., 24, 20, 128, 0, 28475000),
    "handoff-payload": (0., 24., 24, 20, 128, 0, 28475000),
    "comparison-smoke": (4., 0., 1, 3, 32, 12, 28469900),
    "comparison-headwind": (4., 0., 24, 20, 128, 32, 28470000),
    "comparison-payload": (0., 24., 24, 20, 128, 32, 28470000),
}


def plan(name):
    wind, payload, runs, horizon, pool, heldout, seed = PROFILES[name]
    training = name.startswith("training-")
    boundary = name.startswith("boundary-")
    stop_evidence = name.startswith("stop-evidence-")
    handoff = training or boundary or stop_evidence or name.startswith("handoff-")
    arms = ARMS
    if stop_evidence:
        arms = STOP_EVIDENCE_ARMS
    elif boundary:
        arms = BOUNDARY_ARMS
    elif training:
        arms = TRAINING_ARMS
    elif handoff:
        arms = HANDOFF_ARMS
    seeds = [seed+100*i for i in range(runs)]
    if name == "handoff-smoke":
        seeds = [28470100, 28470800]
    elif name == "training-smoke":
        seeds = list(TRAINING_SMOKE_SEEDS)
    elif name == "boundary-smoke":
        seeds = list(BOUNDARY_SMOKE_SEEDS)
    elif name == "stop-evidence-smoke":
        seeds = list(STOP_EVIDENCE_SMOKE_SEEDS)
    comparator = arms[1] if training or boundary or stop_evidence else None
    return {"name": name, "headwind_m_s": wind, "payload_delta_kg": payload, "horizon": horizon,
            "initial": 4, "pool": pool, "heldout": heldout, "arms": list(arms),
            "seeds": seeds,
            "encoder_epochs": 30, "native_gp_steps": 45,
            "checkpoints": [] if handoff else [4, 7] if name == "comparison-smoke" else [4, 8, 12, 16],
            "budget": {"initial": runs*4, "online": runs*horizon*len(arms), "heldout": runs*heldout},
            "calibration_path": None if handoff else "logcei", "rcaide_mda_calls": 0,
            **({"comparison_reference": "ra_to_lsbo", "handoff": "first_observed_usable_design"}
               if handoff else {}),
            **({"primary": {"reference": "ra_to_lsbo", "comparator": comparator,
                           "cohort": "cold", "metric": "capped_first_call_delta"}}
               if comparator else {}),
            **({"stop_evidence_ablation": "mask_stopped_nonanalytical_constraint_targets"}
               if stop_evidence else {}),
            **({"boundary_ablation": {"information_fraction": .2, "feasibility_fraction": 0.,
                                     "coverage_period": COVERAGE_PERIOD}} if boundary else {})}


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


def mask_stopped_constraints(records, analytical_constraints):
    """Mask only observable constraint targets of known incomplete evaluations."""
    view = []
    for row in records:
        packet = row.observation
        if packet.completion_label() == 0:
            constraints = tuple(
                (name, OutputObservation("missing", value.evidence_id, reason="stop_evidence_ablation")
                 if name not in analytical_constraints and value.kind != "missing" else value)
                for name, value in packet.constraints
            )
            if constraints != packet.constraints:
                row = replace(row, observation=replace(packet, constraints=constraints))
        view.append(row)
    return view


class StopStatusRA(StableCorpusDiscoveryBO):
    def _posterior_records(self, records):
        return mask_stopped_constraints(records, self.problem.analytical_constraints)


def ra_settings(spec, *, latent=True):
    return replace(model_settings("hybrid_onehot" if latent else "onehot_stable", spec), local=False)


def active_backend(arm, history):
    if arm in HANDOFF_FROM:
        return FULL_NATIVE if any(row.usable_feasible for row in history) else HANDOFF_FROM[arm]
    return arm


class FullBudgetLSBO(NativeLSBO):
    """Use the original proposals and updates, lifting only the failure-stop cap."""

    def accept(self, record):
        self.engine.engine.resume_failure_limit = self.profile["horizon"]+1
        super().accept(record)


class FirstUsableHandoff:
    def __init__(self, problem, seed, arm, spec, output):
        self.problem, self.seed, self.arm = problem, seed, arm
        self.spec, self.output = spec, output
        self.optimizer = engine_for(problem, seed, HANDOFF_FROM[arm], spec, output/"cold")
        self.offset, self.native = None, None

    def propose(self, history, pool, iteration):
        if self.native is None and active_backend(self.arm, history) == FULL_NATIVE:
            self.offset = iteration
            self.native = FullBudgetLSBO(self.problem, BASE, self.seed,
                                         {**self.spec, "horizon": self.spec["horizon"]-iteration},
                                         self.output/"refinement")
            self.optimizer = self.native
        return self.optimizer.propose(history, pool, iteration-(self.offset or 0))

    def accept(self, record):
        self.optimizer.accept(record)


def native_receipt(optimizer):
    if isinstance(optimizer, FirstUsableHandoff):
        return native_receipt(optimizer.native)
    if isinstance(optimizer, NativeLSBO):
        return {"native_settings": optimizer.engine.settings}
    return {}


def engine_for(problem, seed, arm, spec, output):
    if arm in HANDOFF_FROM:
        return FirstUsableHandoff(problem, seed, arm, spec, output)
    if arm == FULL_NATIVE:
        return FullBudgetLSBO(problem, BASE, seed, spec, output)
    if arm == "original_lsbo":
        return NativeLSBO(problem, BASE, seed, spec, output)
    if arm in ("logcei", "scbo_adapted"):
        return ExternalOptimizer(problem, seed, arm, spec)
    if arm == "ra_explicit":
        return StableOnehotDiscoveryBO(problem, BASE, seed, ra_settings(spec, latent=False),
                                       cold_global=True, protect=True, feedback="interval", coverage_scope="cold")
    engine = {"ra_joint_kernel": JointKernelRA, "ra_stop_status": StopStatusRA}.get(arm, StableCorpusDiscoveryBO)
    settings = ra_settings(spec)
    if "boundary_ablation" in spec:
        ablation = spec["boundary_ablation"]
        fraction = ablation["feasibility_fraction"] if arm == "ra_feasibility" else ablation["information_fraction"]
        settings = replace(settings, boundary_fraction=fraction)
    return engine(problem, BASE, seed, settings, output, allocation="onehot_risk")


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
            roster[role] = candidate_pool(problem, seed+offset, spec[role], seen)[0] if spec[role] else []
            seen.update(map(digest, roster[role]))
        save(unit/"roster.json", roster)
        initial = [ledger.evaluate(problem, row, f"{seed}/initial/{i}", "initial")
                   for i, row in enumerate(roster["initial"])]
        for arm in spec["arms"]:
            if "handoff" in spec:
                torch.manual_seed(seed)
            folder = unit/arm
            folder.mkdir()
            optimizer = engine_for(problem, seed, arm, spec, folder)
            history, remaining = list(initial), list(roster["pool"])
            started = time.monotonic()
            for i in range(spec["horizon"]):
                backend = active_backend(arm, history)
                candidate, info = optimizer.propose(history, remaining, i)
                if backend not in NATIVE_ARMS:
                    remaining.remove(candidate)
                if digest(candidate) in {row.observation.design_id for row in history}:
                    raise ValueError("Method proposed a repeated design")
                append(folder/"proposals.jsonl", {
                    "candidate": candidate, "prefix": digest([row.to_dict() for row in history]),
                    "mode": info.get("mode", info.get("phase", info.get("original_proposal", {}).get("source"))),
                    "selected_kernel": info.get("shared_kernel_index"),
                    "coverage_reason": info.get("coverage_reason"),
                    **({"backend": backend} if "handoff" in spec else {}),
                    **({"boundary_slot": info.get("boundary_slot"), "model_mode": info.get("model_mode")}
                       if "boundary_ablation" in spec else {}),
                    **({"model_history_digest": info.get("history_digest")}
                       if "stop_evidence_ablation" in spec else {}),
                })
                result = ledger.evaluate(problem, candidate, f"{seed}/{arm}/{i}", "online")
                optimizer.accept(result)
                history.append(result)
                if getattr(optimizer, "stop_reason", None):
                    break
            save(folder/"run.json", {"online_calls": len(history)-len(initial),
                                    "stop_reason": getattr(optimizer, "stop_reason", None),
                                    "elapsed_s": time.monotonic()-started,
                                    **native_receipt(optimizer)})
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


def comparisons(trajectories, spec, *, reference_method=None, baselines=None):
    method = reference_method or spec.get("comparison_reference", "ra_separate")
    reference = {r["seed"]: r for r in trajectories if r["method"] == method}
    output = []
    for baseline in baselines if baselines is not None else (arm for arm in spec["arms"] if arm != method):
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
            if "handoff" in spec:
                result["reference"] = method
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
    training_steps, boundary_steps, stop_evidence_steps = [], [], []
    ablation = spec.get("boundary_ablation")
    stop_evidence = "stop_evidence_ablation" in spec

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
        paths = {}
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
            holdout_start = heldout[0][0] if heldout else len(calls)+1
            handoff_at = None
            cold_steps = Counter()
            for i, p in enumerate(proposals):
                candidate = p["candidate"]
                key = digest(candidate)
                if key in seen or p["prefix"] != digest([row.to_dict() for row in history]):
                    raise ValueError("Proposal uses a different observed prefix")
                backend = active_backend(arm, history)
                if p.get("backend", arm) != backend:
                    raise ValueError("Proposal uses the wrong engine for its observed prefix")
                if backend not in NATIVE_ARMS:
                    available.remove(key)
                    if ablation:
                        cold_steps[boundary_proposal_kind(arm, i, p, ablation)] += 1
                    elif stop_evidence:
                        cold_steps[stop_evidence_proposal_kind(backend, p, history)] += 1
                    elif "primary" in spec:
                        cold_steps[training_proposal_kind(backend, p)] += 1
                elif arm in HANDOFF_FROM and handoff_at is None:
                    handoff_at = i
                seen.add(key)
                index, row = use(f"{seed}/{arm}/{i}", "online", candidate)
                if not previous < index < holdout_start:
                    raise ValueError("Online evaluation violates time or holdout order")
                previous = index
                history.append(row)
            if arm in NATIVE_ARMS or handoff_at is not None:
                native = receipt["native_settings"]
                if (native["initial"] != spec["initial"]+(handoff_at or 0)
                        or native["budget"] != spec["initial"]+spec["horizon"]
                        or native["gp_steps"] != spec["native_gp_steps"] or native["max_consecutive_solver_failures"] != 5):
                    raise ValueError("Original LSBO settings differ from the native frozen configuration")
                if stop is not None and (len(proposals) < 5 or any(row.observation.objective.kind == "exact"
                                                                  for row in history[-5:])):
                    raise ValueError("Native stop is not supported by the final five missing objectives")
            if arm == spec["calibration_path"]:
                calibration.extend(score_calibration(spec, seed, history, heldout, unit))
            online = history[len(anchors):]
            paths[arm] = online
            if "primary" in spec:
                steps = stop_evidence_steps if stop_evidence else boundary_steps if ablation else training_steps
                steps.append({"seed": seed, "method": arm, "cold_steps": dict(cold_steps)})
            trajectories.append({"seed": seed, "method": arm, "stop_reason": stop, "elapsed_s": receipt["elapsed_s"],
                                 **({"handoff_after_online": handoff_at} if arm in HANDOFF_FROM else {}),
                                 **trajectory_metrics(anchors, online, spec["horizon"], bool(stop))})
        if "handoff" in spec:
            check_handoff_paths(anchors, paths)
        if ablation:
            check_boundary_prefix(paths, ablation["information_fraction"])
        if stop_evidence and paths[STOP_EVIDENCE_ARMS[0]][0] != paths[STOP_EVIDENCE_ARMS[1]][0]:
            raise ValueError("Stop-evidence ablation changed the common initial coverage decision")
    if used != set(calls):
        raise ValueError("Unexpected charged calls in ledger")
    paired = comparisons(trajectories, spec)
    primary = spec.get("primary")
    return {"status": "VERIFIED_RA_LSBO_COMPARISON", "source_sha": sha, "roles": dict(roles),
            "total_calls": len(calls), "max_calls": sum(spec["budget"].values()), "rcaide_mda_calls": 0,
            "trajectories": trajectories, "comparisons": paired,
            "calibration": calibration, "calibration_comparison": compare_calibration(calibration),
            **({"native_stop_comparison": comparisons(
                trajectories, spec, reference_method=FULL_NATIVE, baselines=("original_lsbo",))}
               if FULL_NATIVE in spec["arms"] else {}),
            **({"primary": {**primary, **next(row[primary["metric"]] for row in paired
                                             if row["comparator"] == primary["comparator"]
                                             and row["cohort"] == primary["cohort"])},
                **({"boundary_steps": boundary_steps} if ablation else
                   {"stop_evidence_steps": stop_evidence_steps} if stop_evidence else
                   {"training_steps": training_steps})}
               if primary else {})}


def stop_evidence_proposal_kind(backend, proposal, history):
    """Verify successful GP fits used the declared view of the raw ledger prefix."""
    skipped = proposal["mode"] in {"scheduled_coverage", "model_failure_coverage"}
    receipt = proposal["model_history_digest"]
    if skipped:
        if receipt is not None:
            raise ValueError("Coverage fallback cannot report a successful GP training view")
        return proposal["mode"]
    masked = mask_stopped_constraints(history, BenchmarkAircraft.analytical_constraints)
    view = masked if backend == "ra_stop_status" else history
    if receipt != digest([row.to_dict() for row in view]):
        raise ValueError("GP training view disagrees with the frozen stop-evidence ablation")
    if masked == history:
        return "fit_without_stop_evidence"
    return "fit_with_masked_stop_evidence" if backend == "ra_stop_status" else "fit_with_stop_evidence"


def is_boundary_slot(iteration, fraction):
    return math.floor((iteration+1)*fraction) > math.floor(iteration*fraction)


def boundary_proposal_kind(arm, iteration, proposal, ablation):
    """Audit the score applied at each cold decision and count realized replacements."""
    mode = proposal["mode"]
    if iteration % ablation["coverage_period"] == 0:
        if mode != "scheduled_coverage":
            raise ValueError("Boundary ablation changed the planned coverage schedule")
        return mode
    if mode == "model_failure_coverage":
        return mode
    nominal_slot = is_boundary_slot(iteration, ablation["information_fraction"])
    fraction = (ablation["information_fraction"] if arm == BOUNDARY_ARMS[0]
                else ablation["feasibility_fraction"])
    boundary = is_boundary_slot(iteration, fraction)
    expected = BOUNDARY_INFORMATION if boundary else "feasible_discovery"
    chosen = proposal["model_mode"] if mode == "degenerate_gate_coverage" else mode
    if proposal["boundary_slot"] is not boundary or chosen != expected:
        raise ValueError("Cold proposal disagrees with the frozen information ablation")
    if mode == "feasible_discovery" and nominal_slot:
        return "information_replaced_by_feasibility"
    return mode


def check_boundary_prefix(paths, fraction):
    """Both arms must agree before an information slot can change their paths."""
    reference, comparator = (paths[arm] for arm in BOUNDARY_ARMS)
    first_slot = next((i for i in range(len(reference)) if is_boundary_slot(i, fraction)), len(reference))
    settled_before_slot = any(row.usable_feasible for row in reference[:first_slot])
    prefix = len(reference) if settled_before_slot else first_slot
    if reference[:prefix] != comparator[:prefix]:
        raise ValueError("Boundary ablation changed decisions before its first intervention")


def training_proposal_kind(backend, proposal):
    """Distinguish scored decisions from coverage and the registered failure fallback."""
    kernel = proposal["selected_kernel"]
    mode = proposal["mode"]
    skipped = mode in {"scheduled_coverage", "model_failure_coverage"}
    if backend == "ra_joint_kernel" and not skipped:
        if type(kernel) is not int or not 0 <= kernel < len(C_GRID)*len(D_GRID):
            raise ValueError("Shared training decision is missing a valid kernel selection")
    elif kernel is not None:
        raise ValueError("Unexpected shared kernel selection")
    return mode if skipped else "model"


def check_handoff_paths(initial, paths):
    """Audit the matched trajectories up to the fixed policy intervention."""
    if "original_lsbo" in paths:
        native = paths["original_lsbo"]
        if native != paths[FULL_NATIVE][:len(native)]:
            raise ValueError("Lifting the native stop cap changed a pre-stop decision")
    handoffs = [arm for arm in HANDOFF_FROM if arm in paths]
    warm = any(row.usable_feasible for row in initial)
    for arm in handoffs:
        if warm:
            expected = paths[FULL_NATIVE] if FULL_NATIVE in paths else paths[handoffs[0]]
            observed = paths[arm]
        else:
            cold = HANDOFF_FROM[arm]
            if cold not in paths:
                continue
            original = paths[cold]
            first = next((i for i, row in enumerate(original, 1) if row.usable_feasible), len(original))
            expected = original[:first]
            observed = paths[arm][:first]
        if observed != expected:
            raise ValueError("Paired handoff changed its declared cold or initial-warm trajectory")


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
