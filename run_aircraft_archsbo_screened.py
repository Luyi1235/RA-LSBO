"""Qualify the native ArchSBO search with the common exact aircraft constraints."""

import argparse
from collections import Counter
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sb_arch_opt.algo.arch_sbo.algo import SBOInfill, SurrogateInfillOptimizationProblem
from sb_arch_opt.algo.arch_sbo.hc_strategy import get_hc_strategy
from sb_arch_opt.algo.arch_sbo.infill import HCInfill, get_default_infill
from sb_arch_opt.algo.arch_sbo.models import ModelFactory
from sb_arch_opt.sampling import HierarchicalSampling

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.runner import Ledger, save
from run_aircraft_archsbo_probe import SealedAircraft, warm_start
from run_aircraft_coverage_factorial import SIZES
from run_aircraft_external_baseline_pilot import plan
from run_aircraft_risk_cold_start import discovery_outcome, read, source_commit
from run_aircraft_timing_interval_value import evidence_hashes, read_ledger, structural_margins
from run_aircraft_timing_usable_value import UsableAircraft


PILOT_SOURCE = "f49503200b8f34f71f2c643ba0a7a181176345ba"
STATIC = ("span", "packing", "stall", "root_bending")
FALLBACK_BATCHES = 4
FALLBACK_BATCH_SIZE = 100
ARCHSBO_VERSION = "1.7.1"


class AircraftScreen:
    def __init__(self, aircraft, problem):
        self.aircraft = aircraft
        self.problem = problem
        self.counts = Counter()

    def margins(self, candidate):
        margins = self.aircraft.precheck(candidate)
        return {**margins, **structural_margins(self.aircraft, candidate)}

    def assess(self, x, context):
        margins = np.asarray([[m[name] for name in STATIC]
                              for m in (self.margins(self.problem.decode(row)) for row in x)])
        if margins.shape != (len(x), len(STATIC)) or not np.isfinite(margins).all():
            raise ValueError("Missing or invalid exact aircraft margins")
        self.counts[context+"_checked"] += len(x)
        self.counts[context+"_rejected"] += int(np.any(margins > 0., axis=1).sum())
        return margins


class ScreenedHCInfill(HCInfill):
    def __init__(self, infill, hc_strategy, screen):
        self.screen = screen
        super().__init__(infill, hc_strategy)

    def get_n_infill_constraints(self):
        return super().get_n_infill_constraints()+len(STATIC)

    def _evaluate(self, x, is_active):
        f, learned = super()._evaluate(x, is_active)
        known = self.screen.assess(x, "surrogate")
        return f, known if learned is None else np.column_stack((learned, known))

    def select_infill(self, population, infill_problem, n_infill):
        if not len(population):
            return population
        population = population[np.all(self.screen.assess(population.get("X"), "final") <= 0., axis=1)]
        if not len(population):
            return population
        selected = super().select_infill(population, infill_problem, n_infill)
        if len(selected) and np.any(self.screen.assess(selected.get("X"), "refined") > 0.):
            return self.select_infill_solutions(population, infill_problem, n_infill)
        return selected


class ScreenedSBOInfill(SBOInfill):
    def __init__(self, model, infill, *, screen, **kwargs):
        super().__init__(model, infill, **kwargs)
        self.screen = screen

    def _get_infill_problem(self, infill=None, force_new_points=None):
        if force_new_points is None:
            force_new_points = self.force_new_points
        wrapped = ScreenedHCInfill(self.infill if infill is None else infill, self.hc_strategy, self.screen)
        x_exist = self.total_pop.get("X") if force_new_points else None
        return SurrogateInfillOptimizationProblem(wrapped, self.problem, x_exist=x_exist), wrapped

    def _get_random_infill_points(self, n_infill):
        if n_infill != 1:
            raise ValueError("The sealed aircraft qualification uses one proposal per step")
        existing = self.total_pop.get("X")
        normalization = ModelFactory.get_continuous_normalization(self.problem)
        for _ in range(FALLBACK_BATCHES):
            sampled = HierarchicalSampling().do(self.problem, FALLBACK_BATCH_SIZE)
            if not len(sampled):
                continue
            mask = np.all(self.screen.assess(sampled.get("X"), "fallback") <= 0., axis=1)
            eligible = self.eliminate_duplicates.do(sampled[mask], self.total_pop)
            if len(eligible):
                distances = cdist(normalization.forward(eligible.get("X")),
                                  normalization.forward(existing), metric="cityblock")
                index = int(np.argmax(distances.min(axis=1)))
                self.screen.counts["fallback_selected"] += 1
                return eligible[[index]]
        raise RuntimeError("Native fallback found no new design passing the shared exact screen")


def screened_algorithm(problem, init_size, init_sampling, kpls_n_dim, *, screen):
    model, normalization = ModelFactory(problem).get_md_kriging_model(kpls_n_comp=kpls_n_dim)
    infill, batch = get_default_infill(problem)
    return ScreenedSBOInfill(model, infill, screen=screen, normalization=normalization,
                            hc_strategy=get_hc_strategy(kpls_n_dim=kpls_n_dim)).algorithm(
                                infill_size=batch, init_size=init_size, init_sampling=init_sampling)


def run_prefix(aircraft, records, seed, *, ledger=None):
    remaining = tuple(name for name in aircraft.required_constraints if name not in STATIC)
    problem = SealedAircraft(aircraft, constraints=remaining)
    screen = AircraftScreen(aircraft, problem)
    followups = 1 if ledger is None else ledger.maximum
    algorithm = warm_start(problem, records, seed, factory=partial(screened_algorithm, screen=screen),
                           followups=followups)
    seen = {digest(record.candidate) for record in records}
    steps = []
    for step in range(1, followups+1):
        previous = screen.counts.copy()
        proposal = algorithm.ask()
        if proposal is None or len(proposal) != 1:
            raise RuntimeError("Expected one native ArchSBO proposal")
        x, active = problem.correct_x(proposal.get("X"))
        candidate = problem.decode(x[0])
        key = digest(candidate)
        margins = screen.margins(candidate)
        if key in seen or not all(np.isfinite(v) and v <= 0. for v in margins.values()):
            raise RuntimeError("ArchSBO proposed a duplicate or failed the common exact screen")
        entry = {"step": step, "candidate": candidate, "candidate_id": key, "margins": margins,
                 "screen_counts": dict(screen.counts-previous),
                 "native_surrogate_trained": bool(algorithm.infill_obj.was_trained)}
        if ledger is not None:
            record = ledger.evaluate(aircraft, candidate, f"followup/{step}", "online")
            f, g = problem.outputs(record)
            proposal.set("X", x, "F", np.asarray([f]), "G", np.asarray([g]), "H", np.empty((1, 0)),
                         "is_active", active)
            proposal[0].evaluated.update(("F", "G", "H"))
            algorithm.evaluator.n_eval += 1
            algorithm.tell(infills=proposal)
            if algorithm.evaluator.n_eval != len(records)+ledger.count:
                raise RuntimeError("Mission evaluation count differs from the native ledger")
            entry.update(outcome=discovery_outcome(record.to_dict()), usable=record.usable_feasible,
                         paid_calls=ledger.count)
        seen.add(key)
        steps.append(entry)
    return {"seed": seed, "initial": len(records),
            "initial_exact_rows": int(sum(np.isfinite(problem.outputs(r)[0][0]) for r in records)),
            "steps": steps}


def qualification(folder, sha, output, *, online):
    if version("sb-arch-opt") != ARCHSBO_VERSION:
        raise ValueError(f"This adapter qualifies SBArchOpt {ARCHSBO_VERSION}")
    header = read(folder/"plan.json")
    spec = plan(header["spec"]["name"])
    aircraft = UsableAircraft(spec)
    if (header != {"source_sha": PILOT_SOURCE, "spec": spec, "contract": aircraft.contract_id}
            or read(folder/"seal.json") != evidence_hashes(folder)):
        raise ValueError("Expected the sealed PR147 pilot produced with the fixed pilot source")
    calls = read_ledger(folder, aircraft.contract_id)
    if online:
        output.mkdir(parents=True)
    ledger = Ledger(output, 2) if online else None
    rows = []
    for seed in spec["seeds"][:1] if online else spec["seeds"]:
        records = [calls[f"{seed}/initial/{i}"][1] for i in range(1, max(SIZES)+1)]
        for size in SIZES[:1] if online else SIZES:
            rows.append(run_prefix(aircraft, records[:size], seed, ledger=ledger))
            print(spec["name"], seed, size, flush=True)
    result = {"status": "VERIFIED_ARCHSBO_SCREEN_INTEGRATION", "profile": spec["name"],
              "source_sha": sha, "pilot_source_sha": PILOT_SOURCE, "pilot_seal": read(folder/"seal.json"),
              "task": "online-smoke" if online else "probe",
              "versions": {name: version(name) for name in ("sb-arch-opt", "pymoo", "smt", "numpy", "scipy",
                                                           "numba", "scikit-learn")},
              "new_aircraft_calls": ledger.count if online else 0, "rcaide_mda_calls": 0, "rows": rows}
    if online:
        settled = read_ledger(output, aircraft.contract_id)
        steps = rows[0]["steps"]
        if len(settled) != len(steps) or any(
                (number, record.observation.design_id, record.usable_feasible, role)
                != (step["paid_calls"], step["candidate_id"], step["usable"], "online")
                for (number, record, role), step in zip(settled.values(), steps)):
            raise RuntimeError("Sealed aircraft calls differ from screened paid proposals")
        save(output/"analysis.json", result)
        save(output/"seal.json", evidence_hashes(output))
    else:
        save(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("probe", "online-smoke"))
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder, output = args.input.resolve(), args.output.resolve()
    if output.exists() or any(p in (output, *output.parents) for p in (Path(__file__).resolve().parent, folder)):
        parser.error("Use a new output outside the code worktree and sealed input")
    result = qualification(folder, source_commit(args.expected_sha, clean=True), output,
                           online=args.task == "online-smoke")
    print(result["status"], result["task"], result["profile"], len(result["rows"]), result["new_aircraft_calls"])


if __name__ == "__main__":
    main()
