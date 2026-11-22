"""Paired X-57 cold-start runs with a single paid shared initialization."""

from collections import Counter
from dataclasses import asdict
import importlib
import importlib.metadata
import importlib.util
from itertools import combinations
from pathlib import Path
import sys
import time

from .budget import digest
from .risk_core.observations import finite
from .x57_cold_risk import propose
from .x57_cold_sampling import RosterShortfall, active_names, freeze_roster
from .x57_cold_worker import dump, sha256

ARMS = ("lsbo_full_budget", "structured_sign", "structured_interval")
INITIAL = 32
MAX_ONLINE = 288
CHECKPOINTS = (32, 48, 64, 128, 256, 320)
NATIVE_STOP = "Repeated missing/failed objective returns; stopped before wasting remaining budget"
DEPENDENCIES = ("numpy", "scipy", "torch", "botorch", "gpytorch", "RCAIDE_LEADS")


def identity(problem):
    root = problem.base / "lsbo"
    rcaide = importlib.util.find_spec("RCAIDE")
    return {
        "contract_id": problem.contract_id, "base_project": str(problem.base),
        "config_sha256": sha256(problem.config_path), "config": problem.problem.config,
        "python_executable": sys.executable, "python_version": sys.version,
        "dependencies": {name: importlib.metadata.version(name) for name in DEPENDENCIES},
        "rcaide_module": {"path": rcaide.origin, "sha256": sha256(Path(rcaide.origin))},
        "lsbo_sha256": {f.name: sha256(f) for f in sorted(root.glob("*.py"))},
        "evaluator_sha256": problem.source_hashes,
        "required_constraints": problem.required_constraints,
        "analytical_constraints": problem.analytical_constraints,
        "variables": [asdict(v) for v in problem.space.variables],
    }


class ObservationAdapter:
    """Feed recorded observations through unmodified native update logic."""

    def __init__(self, physical, evaluation_class, *, native_feedback=False):
        self.physical = physical
        self.evaluation_class = evaluation_class
        self.native_feedback = native_feedback
        self.pending = None

    def __getattr__(self, name):
        return getattr(self.physical, name)

    def evaluate(self, candidate):
        row = self.pending
        if row is None or self.space.key(candidate) != row["candidate_id"]:
            raise ValueError("No journal result for this native update")
        self.pending = None
        feedback, raw = row["feedback"], row["worker_result"]
        if self.native_feedback and (feedback["completion"] is True
                                     or feedback["outcome"] == "numerical_nonconvergence"):
            return self.evaluation_class(**raw)
        metrics = {"worker_dir": row["worker_dir"]}
        # Retain the native diagnostic only for a converged original-model result.
        source = raw.get("metrics")
        if (feedback["completion"] is True and isinstance(source, dict)
                and source.get("solver_discharge_margin_source") == "converged_original_discharge_model"
                and finite(source.get("solver_discharge_margin_W"))):
            metrics["solver_discharge_margin_W"] = source["solver_discharge_margin_W"]
        return self.evaluation_class(feedback["objective_Wh"], feedback["completion"] is True,
                                     dict(feedback["constraints"]), raw.get("status", feedback["outcome"]),
                                     metrics)


def update_native(native, candidate, row, phase, proposal):
    native.problem.pending = row
    try:
        native.evaluate(candidate, phase, proposal)
    except RuntimeError as error:
        if str(error) != NATIVE_STOP or native.problem.pending is not None:
            raise
        return True
    finally:
        native.problem.pending = None
    return False


def endpoint(rows, target_Wh, checkpoints=CHECKPOINTS):
    usable = [(row["evaluation"], row["feedback"]["objective_Wh"])
              for row in rows if row["feedback"]["usable"]]
    hit = next((index for index, energy in usable if energy <= target_Wh), None)
    count = len(rows)
    return {
        "first_usable": usable[0][0] if usable else None, "first_target": hit,
        "target_censor_at": count if hit is None else None,
        "best_usable_Wh": min((energy for _, energy in usable), default=None),
        "evaluations": count,
        "online_evaluations": max(0, count - INITIAL),
        "worker_charges": sum(row["charged"] for row in rows if row["arm"] != "shared_initial"),
        "worker_seconds": sum(row["elapsed_seconds"] for row in rows),
        "failures": dict(Counter(row["feedback"]["outcome"] for row in rows if not row["feedback"]["usable"])),
        "checkpoints": {str(node): {
            "online": max(0, node - INITIAL),
            "reached": count >= node,
            "attained": None if count < node else hit is not None and hit <= node,
            "best_usable_Wh": (min((energy for index, energy in usable if index <= node), default=None)
                               if count >= node else None),
        } for node in checkpoints},
    }


def grouped_summary(seed_results):
    groups = ("all", "0", "1-2", "3+")
    result = {}
    for group in groups:
        rows = [seed for seed in seed_results if group == "all" or seed["initial_group"] == group]
        result[group] = {"seeds": len(rows), "arms": {}}
        for arm in ARMS:
            tracks = [seed["arms"][arm] for seed in rows if arm in seed["arms"]]
            result[group]["arms"][arm] = {}
            for node in CHECKPOINTS:
                observations = [track["checkpoints"][str(node)] for track in tracks]
                reached = sum(row["reached"] for row in observations)
                attained = sum(row["attained"] is True for row in observations)
                result[group]["arms"][arm][str(node)] = {
                    "attained": attained, "total_seeds": len(rows), "observed_at_node": reached,
                    "attainment_rate_observed": attained / reached if reached else None,
                    "attainment_rate_min_all": attained / len(rows) if rows else None,
                    "attainment_rate_max_all": (attained + len(rows) - reached) / len(rows) if rows else None,
                }
        result[group]["paired_best_energy_Wh"] = {}
        for reference, arm in combinations(ARMS, 2):
            comparison = {}
            for node in CHECKPOINTS:
                values = []
                for seed in rows:
                    if reference not in seed["arms"] or arm not in seed["arms"]:
                        continue
                    left, right = [seed["arms"][name]["checkpoints"][str(node)]["best_usable_Wh"]
                                   for name in (reference, arm)]
                    values.append({"seed": seed["seed"], "delta_Wh": (
                        right - left if left is not None and right is not None else None)})
                comparison[str(node)] = values
            result[group]["paired_best_energy_Wh"][f"{arm}_minus_{reference}"] = comparison
    return result


def run_campaign(problem, dispatcher, output, seeds, target_Wh, online, *, fast_mock=False):
    if (not finite(target_Wh) or target_Wh <= 0 or type(online) is not int or not 0 <= online <= MAX_ONLINE
            or not 1 <= len(seeds) <= 6 or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("Require a positive frozen target, up to six distinct seeds and 0-288 online calls")
    if fast_mock and dispatcher.backend != "mock":
        raise ValueError("Reduced optimizer settings are only for mock validation")
    native_module = importlib.import_module("lsbo.optimizer")
    knobs = dict(pretrain_epochs=3, update_epochs=2, gp_steps=3, branches=3,
                 points_per_branch=3, restarts=1, raw_samples=8, local_branches=1,
                 proposal_pool_size=24, exploration_probability=0.) if fast_mock else {}
    risk_pool_size = 32 if fast_mock else 512
    started = time.perf_counter()
    rosters = {}
    for seed in seeds:
        try:
            rosters[seed] = freeze_roster(problem.problem, seed)
        except RosterShortfall as error:
            dump(output / "sampling_shortfall.json", {"seed": seed, "layouts": error.audit})
            raise
    records = {}
    for seed, (roster, audit) in rosters.items():
        points = [{"candidate_id": problem.space.key(p), "candidate": p.to_dict(),
                   "active_variables": active_names(problem.space, p)} for p in roster]
        records[str(seed)] = {"sha256": digest(points), "points": points, "sampling": audit}
    dump(output / "plan.json", {
        "backend": dispatcher.backend, "identity": identity(problem), "arm_names": ARMS,
        "initial": INITIAL, "online": online, "official_max_calls": len(seeds) * (INITIAL + len(ARMS) * online),
        "target_Wh": target_Wh, "checkpoints": CHECKPOINTS, "rosters": records,
        "risk_pool_size": risk_pool_size,
        "physical_worker_sha256": dispatcher.expected_worker,
        "settings": {seed: asdict(native_module.Settings(seed=seed, initial=INITIAL,
                                                         budget=INITIAL + online, **knobs)) for seed in seeds},
        "native_stop_policy": "log and continue on native consecutive missing-objective stop",
        "infrastructure_policy": "stop current arm, or all arms of the seed during shared initialization",
    })
    seed_results = []
    for seed, (roster, _) in rosters.items():
        charge_before = dispatcher.count
        seed_started = time.perf_counter()
        initial = []
        initial_failed = False
        for index, candidate in enumerate(roster, 1):
            row = dispatcher.evaluate(candidate, seed=seed, arm="shared_initial", evaluation=index,
                                      online=0, phase="initial",
                                      proposal={"source": "frozen_layout_lhs", "roster_sha256": records[str(seed)]["sha256"]})
            initial.append(row)
            if row["feedback"]["outcome"] == "infrastructure_error":
                initial_failed = True
                break
        initial_usable = sum(row["feedback"]["usable"] for row in initial)
        group = ("incomplete" if initial_failed else "0" if initial_usable == 0
                 else "1-2" if initial_usable < 3 else "3+")
        seed_report = {"seed": seed, "initial_usable": initial_usable, "initial_group": group,
                       "initial_call_ids": [row["call_id"] for row in initial],
                       "initial_worker_charges": sum(row["charged"] for row in initial), "arms": {}}
        if not initial_failed:
            for arm in ARMS:
                arm_started = time.perf_counter()
                settings = native_module.Settings(seed=seed, initial=INITIAL, budget=INITIAL + online, **knobs)
                adapter = ObservationAdapter(problem.problem, native_module.Evaluation,
                                             native_feedback=arm == "lsbo_full_budget")
                native = native_module.Optimizer(adapter, settings, output / f"seed-{seed}" / arm)
                history, status = [], "complete"

                def observe(candidate, row, phase, proposal):
                    history.append(row)
                    if update_native(native, candidate, row, phase, proposal) and (
                            native.solver_failures == settings.max_consecutive_solver_failures):
                        dispatcher.event("native_missing_objective_stop", call_id=row["call_id"],
                                         seed=seed, arm=arm, evaluation=row["evaluation"],
                                         consecutive=native.solver_failures, continued=True)

                for candidate, row in zip(roster, initial):
                    observe(candidate, row, "initial", row["proposal"])
                for index in range(1, online + 1):
                    begin = time.perf_counter()
                    rejected_before = Counter(native.precheck_rejections)
                    try:
                        candidate, phase, proposal = propose(native, history, arm,
                                                              native_module.mixed_candidate_pool, risk_pool_size)
                    except Exception as error:
                        status = "proposal_error"
                        dispatcher.event("proposal_error", seed=seed, arm=arm, online=index,
                                         error_type=type(error).__name__, message=str(error), charged=False)
                        break
                    rejections = dict(native.precheck_rejections - rejected_before)
                    if rejections:
                        dispatcher.event("proposal_precheck_rejections", seed=seed, arm=arm,
                                         online=index, reasons=rejections, charged=False)
                    proposal["proposal_seconds"] = time.perf_counter() - begin
                    row = dispatcher.evaluate(candidate, seed=seed, arm=arm, evaluation=INITIAL + index,
                                              online=index, phase=phase, proposal=proposal)
                    observe(candidate, row, phase, proposal)
                    if row["feedback"]["outcome"] == "infrastructure_error":
                        status = "infrastructure_stop"
                        break
                arm_report = {**endpoint(history, target_Wh), "status": status,
                              "wall_seconds_excluding_shared_init": time.perf_counter() - arm_started,
                              "call_ids": [row["call_id"] for row in history]}
                native.save_summary(status)
                dump(native.output / "paired_metrics.json", arm_report)
                seed_report["arms"][arm] = arm_report
        seed_report["worker_charges"] = dispatcher.count - charge_before
        seed_report["wall_seconds"] = time.perf_counter() - seed_started
        seed_results.append(seed_report)
    report = {"backend": dispatcher.backend, "worker_charges": dispatcher.count,
              "rcaide_calls": dispatcher.count if dispatcher.backend == "rcaide" else 0,
              "worker_seconds": dispatcher.elapsed, "wall_seconds": time.perf_counter() - started,
              "seeds": seed_results, "attainment": grouped_summary(seed_results)}
    dump(output / "report.json", report)
    return report
