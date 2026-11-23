"""Bridge the X-57 ledger to native ArchSBO and the repository's LogCEI."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from inspect import signature
import random
from threading import RLock

import numpy as np

from .experiments.comparison.methods import ExternalOptimizer
from .risk_core.observations import OutputObservation, RunObservation
from .x57_cold_sampling import violations
from .x57_energy_stop import STATUS as ENGINEERING_STOP, verified_stop

ARMS = ("archsbo_native", "logcei_screened_online")
TOTAL_BUDGET = 320
RANDOM_LOCK = RLock()


def native_kpls_minimum(n_var):
    from sb_arch_opt.algo.arch_sbo.api import get_arch_sbo_gp

    components = signature(get_arch_sbo_gp).parameters["kpls_n_dim"].default
    return components + 1 if components is not None and n_var > components else None


def native_kpls_shortfall(error, minimum, complete):
    return (minimum is not None and 1 < complete < minimum
            and f"at least {minimum} points (currently {complete})" in str(error))


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
        from sb_arch_opt.algo.arch_sbo.api import get_arch_sbo_gp

        class X57Architecture(SealedAircraft):
            def outputs(self, record):
                obs = record.observation
                f = [obs.objective.value if obs.objective.kind == "exact" else np.nan]
                g = [item.value if item.kind == "exact" else np.nan
                     for item in (obs.constraint(name) for name in self.constraint_names)]
                return f, g

        self.arch = X57Architecture(self.problem)
        self.numpy_state = np.random.RandomState(self.seed).get_state()
        self.python_state = random.Random(self.seed).getstate()
        with self._arch_rng():
            self.native = warm_start(self.arch, self.records, self.seed,
                                     factory=get_arch_sbo_gp, factory_options={},
                                     termination=("n_eval", self.total_budget))

    def _arch_tell(self, population, f, g, *, charged):
        population.set("F", np.asarray([f]), "G", np.asarray([g]), "H", np.empty((1, 0)))
        population[0].evaluated.update(("F", "G", "H"))
        if charged:
            self.native.evaluator.n_eval += 1
        with self._arch_rng():
            self.native.tell(infills=population)

    def native_complete_rows(self):
        outputs = (self.arch.outputs(record) for record in self.records)
        return sum(1 for f, g in outputs if np.isfinite(f).all() and np.isfinite(g).all())

    def _ask_archsbo(self):
        rejections = []
        # The evaluation boundary may refuse a physical launch. Feed the
        # observed margins back through standard tell and request a new point.
        for _ in range(100):
            with self._arch_rng():
                native = self.native.ask()
            if native is None or len(native) != 1:
                raise RuntimeError(f"ArchSBO did not produce one proposal; precheck rejections: {rejections}")
            x, active = self.arch.correct_x(native.get("X"))
            native.set("X", x, "is_active", active)
            candidate = self.arch.decode(x[0])
            point = self.problem.candidate_class(**candidate)
            key = self.problem.space.key(point)
            if active[0].tolist() != [self.problem.space.active(v, candidate["discrete"])
                                     for v in self.problem.space.variables]:
                raise ValueError("Native ArchSBO activity mask differs from the physical design")
            if self.problem.canonical(candidate) != candidate or key in self.seen:
                raise RuntimeError("Native ArchSBO proposed a duplicate or noncanonical design")
            margins = self.problem.problem.precheck(point)
            if not violations(margins):
                return candidate, native, {
                    "native_surrogate_trained": bool(self.native.infill_obj.was_trained),
                    "precheck_rejections": rejections,
                }
            self._arch_tell(native, [np.nan], [margins.get(name, np.nan) for name in self.arch.constraint_names],
                            charged=False)
            self.seen.add(key)
            rejections.append({"candidate_id": key, "precheck": margins, "charged": False})
        raise RuntimeError(f"Native ArchSBO exhausted 100 dispatch attempts; precheck rejections: {rejections}")

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
            candidate, native, info = self._ask_archsbo()
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
            f, g = self.arch.outputs(record)
            self._arch_tell(self.pending[2], f, g, charged=True)
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
