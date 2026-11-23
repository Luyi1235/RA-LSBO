"""Bridge the X-57 ledger to screened ArchSBO and the repository's LogCEI."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import random
from threading import RLock

import numpy as np

from .experiments.comparison.methods import ExternalOptimizer
from .risk_core.observations import OutputObservation, RunObservation
from .x57_cold_sampling import violations
from .x57_energy_stop import STATUS as ENGINEERING_STOP, verified_stop

ARMS = ("archsbo_screened", "logcei_screened_online")
TOTAL_BUDGET = 320
RANDOM_LOCK = RLock()


@dataclass(frozen=True)
class Record:
    candidate: dict
    observation: RunObservation
    usable_feasible: bool


def record_from_row(problem, row):
    feedback = row["feedback"]
    candidate = row["candidate"]
    point = problem.candidate_class(**candidate)
    if (problem.canonical(candidate) != candidate or problem.space.key(point) != row["candidate_id"]
            or problem.precheck(candidate) != row["precheck"]
            or set(feedback["constraints"]) != set(problem.required_constraints)):
        raise ValueError("Feedback violates the common candidate or constraint contract")
    identity = f"{row.get('source_id', row.get('journal', row.get('worker_dir', 'ledger')))}:{row['call_id']}"
    outcome = feedback["outcome"]
    status = {True: "completed", False: "hidden_failure", None: "unknown"}[feedback["completion"]]
    if outcome == "infrastructure_error":
        status = outcome
    objective = (OutputObservation("exact", identity, value=feedback["objective_Wh"])
                 if feedback["objective_kind"] == "exact" else
                 OutputObservation("missing", identity, reason=outcome))
    constraints = []
    for name in problem.required_constraints:
        kind = feedback["constraint_kinds"][name]
        if kind == "exact":
            evidence = OutputObservation("exact", identity, value=feedback["constraints"][name])
        elif kind == "interval" and name == "usable_energy" and outcome == ENGINEERING_STOP and (
                name in feedback.get("known_constraint_violations", []) and verified_stop(row["worker_result"])):
            evidence = OutputObservation("interval", identity, lower=0., lower_inclusive=False)
        elif kind == "missing":
            evidence = OutputObservation("missing", identity, reason=outcome)
        else:
            raise ValueError(f"Unsupported {name} evidence: {kind}")
        constraints.append((name, evidence))
    observation = RunObservation(row["candidate_id"], problem.contract_id, identity, status, objective,
                                 tuple(constraints), failure_cause=outcome if status == "hidden_failure" else None)
    usable = (status == "completed" and objective.kind == "exact"
              and all(item.constraint_satisfaction() is True for _, item in constraints))
    if (usable != feedback["usable"] or (objective.kind == "exact") != (status == "completed")):
        raise ValueError("Feedback changed completion, objective or usability through projection")
    return Record(candidate, observation, usable)


class X57Screen:
    def __init__(self, problem, arch_problem):
        self.problem, self.arch_problem = problem, arch_problem
        self.constraint_names = problem.analytical_constraints
        self.counts = Counter()

    def assess(self, x, context):
        margins = [self.problem.problem.precheck(self.problem.candidate_class(**self.arch_problem.decode(vector)))
                   for vector in x]
        if any(set(m) != set(self.constraint_names) for m in margins):
            raise ValueError("Mismatch between analytical screen and native ArchSBO constraints")
        values = np.asarray([[m[name] for name in self.constraint_names] for m in margins],
                            dtype=float).reshape(len(margins), len(self.constraint_names))
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite analytical constraint during ArchSBO screening")
        self.counts[context + "_checked"] += len(values)
        self.counts[context + "_rejected"] += int((values > 0).any(axis=1).sum())
        return values


class LogCEIProblem:
    allow_unknown_constraint_evidence = True

    def __init__(self, problem):
        self.problem = problem

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def precheck(self, candidate):
        return self.problem.problem.precheck(self.problem.candidate_class(**candidate))


class X57External:
    """Sequential ask/tell with explicit worker ownership outside the optimizer."""

    def __init__(self, problem, arm, seed, initial, total_budget=TOTAL_BUDGET, pool_size=512):
        if arm not in ARMS or len(initial) not in (8, 32) or not len(initial) < total_budget <= TOTAL_BUDGET:
            raise ValueError("Invalid external trajectory settings")
        self.problem, self.arm, self.seed = problem, arm, seed
        self.total_budget, self.initial_size, self.pool_size = total_budget, len(initial), pool_size
        self.initial, self.online = list(initial), []
        self.records = [record_from_row(problem, row) for row in initial]
        self.seen = {record.observation.design_id for record in self.records}
        self.pending = None
        if (len(self.seen) != len(initial)
                or any(r.observation.status == "infrastructure_error" for r in self.records)
                or any((r["seed"], r["evaluation"], r["online"], r["charged"]) != (seed, i, 0, True)
                       for i, r in enumerate(initial, 1))):
            raise ValueError("A shared prefix must contain unique settled observations")
        if arm == "logcei_screened_online":
            self.native = ExternalOptimizer(LogCEIProblem(problem), seed, "logcei",
                                            {"horizon": total_budget, "pool": pool_size})
        else:
            self._init_archsbo()

    @contextmanager
    def _arch_rng(self):
        with RANDOM_LOCK:
            outer_numpy, outer_python = np.random.get_state(), random.getstate()
            np.random.set_state(self.numpy_state)
            random.setstate(self.python_state)
            try:
                yield
            finally:
                self.numpy_state, self.python_state = np.random.get_state(), random.getstate()
                np.random.set_state(outer_numpy)
                random.setstate(outer_python)

    def _init_archsbo(self):
        from run_aircraft_archsbo_probe import SealedAircraft, warm_start
        from run_aircraft_archsbo_screened import screened_algorithm

        class X57Architecture(SealedAircraft):
            def outputs(self, record):
                obs = record.observation
                f = [obs.objective.value if obs.objective.kind == "exact" else np.nan]
                # Native validity handling drops any row with a missing output from
                # the numerical GP. The unit sign code is never a physical target.
                g = [item.value if item.kind == "exact" else
                     (1. if item.constraint_satisfaction() is False else np.nan)
                     for item in (obs.constraint(name) for name in self.constraint_names)]
                return f, g

        learned = [name for name in self.problem.required_constraints
                   if name not in self.problem.analytical_constraints]
        self.arch = X57Architecture(self.problem, constraints=learned)
        self.screen = X57Screen(self.problem, self.arch)
        self.numpy_state = np.random.RandomState(self.seed).get_state()
        self.python_state = random.Random(self.seed).getstate()
        factory = partial(screened_algorithm, screen=self.screen)
        with self._arch_rng():
            self.native = warm_start(self.arch, self.records, self.seed, factory=factory,
                                     followups=self.total_budget - self.initial_size)

    def _logcei_pool(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, len(self.online)]))
        pool, keys = [], set(self.seen)
        rejected = Counter()
        for _ in range(30 * self.pool_size):
            point = self.problem.space.sample(rng)
            key = self.problem.space.key(point)
            if key in keys:
                continue
            candidate = point.to_dict()
            reasons = violations(self.problem.problem.precheck(point))
            if reasons:
                rejected.update(reasons)
                continue
            pool.append(candidate)
            keys.add(key)
            if len(pool) == self.pool_size:
                break
        if len(pool) != self.pool_size:
            raise RuntimeError("LogCEI screened pool supply exhausted")
        return pool, dict(rejected)

    def ask(self):
        if self.pending is not None or len(self.records) >= self.total_budget:
            raise RuntimeError("Settle the current proposal before requesting one within budget")
        self.problem.verify_frozen()
        if self.arm == "logcei_screened_online":
            pool, rejected = self._logcei_pool()
            candidate, info = self.native.propose(self.records, pool, len(self.online))
            info["precheck_rejected_by_constraint"] = rejected
            info["candidate_count"] = len(pool)
            native = None
        else:
            before = self.screen.counts.copy()
            with self._arch_rng():
                native = self.native.ask()
            if native is None or len(native) != 1:
                raise RuntimeError("ArchSBO must produce exactly one proposal")
            x, active = self.arch.correct_x(native.get("X"))
            native.set("X", x, "is_active", active)
            candidate = self.arch.decode(x[0])
            if active[0].tolist() != [self.problem.space.active(v, candidate["discrete"])
                                     for v in self.problem.space.variables]:
                raise ValueError("Native ArchSBO activity mask differs from the physical design")
            info = {"native_surrogate_trained": bool(self.native.infill_obj.was_trained),
                    "screen_counts": dict(self.screen.counts - before)}
        key = self.problem.space.key(self.problem.candidate_class(**candidate))
        if (self.problem.canonical(candidate) != candidate or key in self.seen
                or violations(self.problem.precheck(candidate))):
            raise RuntimeError("External optimizer proposed a duplicate or unscreened design")
        self.pending = (key, candidate, native)
        return candidate, info

    def tell(self, row):
        if self.pending is None or self.pending[:2] != (row["candidate_id"], row["candidate"]):
            raise ValueError("Feedback does not match the pending proposal")
        if (row["seed"], row["arm"], row["evaluation"], row["online"]) != (
                self.seed, self.arm, len(self.records) + 1, len(self.online) + 1):
            raise ValueError("Feedback is out of order or belongs to another trajectory")
        record = record_from_row(self.problem, row)
        if row.get("charged") is not True or record.observation.status == "infrastructure_error":
            raise ValueError("Checkpoint the worker journal and suspend on infrastructure errors")
        if self.arm == "logcei_screened_online":
            self.native.accept(record)
        else:
            pending = self.pending[2]
            f, g = self.arch.outputs(record)
            pending.set("F", np.asarray([f]), "G", np.asarray([g]), "H", np.empty((1, 0)))
            pending[0].evaluated.update(("F", "G", "H"))
            with self._arch_rng():
                self.native.tell(infills=pending)
            self.native.evaluator.n_eval += 1
            if self.native.evaluator.n_eval != len(self.records) + 1:
                raise RuntimeError("ArchSBO evaluation count differs from its settled trajectory")
        self.records.append(record)
        self.online.append(row)
        self.seen.add(row["candidate_id"])
        self.pending = None
        return record

    def checkpoint(self):
        if self.pending is not None:
            raise RuntimeError("Only settled boundaries can be checkpointed")
        return dict(arm=self.arm, seed=self.seed, total_budget=self.total_budget, pool_size=self.pool_size,
                    contract_id=self.problem.contract_id, initial=list(self.initial), online=list(self.online))

    @classmethod
    def resume(cls, problem, checkpoint):
        if checkpoint["contract_id"] != problem.contract_id:
            raise ValueError("Resume requires the same physical contract")
        adapter = cls(problem, checkpoint["arm"], checkpoint["seed"], checkpoint["initial"],
                      checkpoint["total_budget"], checkpoint["pool_size"])
        for row in checkpoint["online"]:
            adapter.ask()
            adapter.tell(row)
        return adapter
