"""Ask official ArchSBO for a proposal from sealed aircraft feedback, without evaluation."""

import argparse
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pymoo.core.population import Population
from pymoo.core.variable import Choice, Integer, Real
from sb_arch_opt.algo.arch_sbo.api import get_arch_sbo_gp
from sb_arch_opt.problem import ArchOptProblemBase

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.runner import save
from run_aircraft_coverage_factorial import SIZES
from run_aircraft_external_baseline_pilot import plan
from run_aircraft_risk_cold_start import read, source_commit
from run_aircraft_timing_interval_value import evidence_hashes, read_ledger, structural_margins
from run_aircraft_timing_usable_value import UsableAircraft


class SealedAircraft(ArchOptProblemBase):
    def __init__(self, aircraft, *, constraints=None):
        self.space = aircraft.space
        self.candidate_class = aircraft.candidate_class
        self.discrete = aircraft.space.discrete
        self.continuous = aircraft.space.continuous
        self.constraint_names = aircraft.required_constraints if constraints is None else tuple(constraints)
        variables = [
            Choice(options=list(range(len(v.values)))) if v.kind == "nominal"
            else Integer(bounds=(0, len(v.values)-1))
            for v in self.discrete
        ] + [Real(bounds=v.bounds) for v in self.continuous]
        super().__init__(variables, n_obj=1, n_ieq_constr=len(self.constraint_names))

    def encode(self, candidate):
        return [v.values.index(candidate["discrete"][v.name]) for v in self.discrete] + [
            candidate["continuous"][v.name] for v in self.continuous]

    def decode(self, vector):
        split = len(self.discrete)
        discrete = {v.name: v.values[int(round(x))] for v, x in zip(self.discrete, vector[:split])}
        continuous = {v.name: float(x) for v, x in zip(self.continuous, vector[split:])}
        return self.space.canonical(self.candidate_class(continuous, discrete)).to_dict()

    def _is_conditionally_active(self):
        return [bool(v.active_when) for v in self.discrete+self.continuous]

    def _correct_x(self, x, is_active):
        for vector, mask in zip(x, is_active):
            discrete = self.decode(vector)["discrete"]
            mask[:] = [self.space.active(v, discrete) for v in self.discrete+self.continuous]

    def _arch_evaluate(self, *_args, **_kwargs):
        raise RuntimeError("This probe can only use sealed aircraft observations")

    def outputs(self, record):
        obs = record.observation
        constraints = dict(obs.constraints)
        if (obs.status != "completed" or obs.objective.kind != "exact"
                or any(name not in constraints or constraints[name].kind != "exact"
                       for name in self.constraint_names)):
            return [np.nan], [np.nan]*len(self.constraint_names)
        return [obs.objective.value], [constraints[name].value for name in self.constraint_names]


def warm_start(problem, records, seed, *, factory=get_arch_sbo_gp, followups=1,
               factory_options=None, termination=None):
    candidates = [record.candidate for record in records]
    x, active = problem.correct_x(np.asarray([problem.encode(candidate) for candidate in candidates]))
    if any(problem.decode(vector) != candidate for vector, candidate in zip(x, candidates)):
        raise ValueError("Architecture correction changed a sealed physical candidate")
    f, g = map(np.asarray, zip(*(problem.outputs(record) for record in records)))
    population = Population.new(X=x, F=f, G=g, H=np.empty((len(x), 0)), is_active=active)
    for individual in population:
        individual.evaluated.update(("F", "G", "H"))
    if factory_options is None:
        # Reproduce the reduced-aircraft probe settings.
        factory_options = {"kpls_n_dim": None}
    algorithm = factory(problem, init_size=len(x), init_sampling=population, **factory_options)
    algorithm.setup(problem, seed=seed, termination=termination or ("n_gen", followups+1))
    initial = algorithm.ask()
    if len(initial) != len(x) or not np.array_equal(initial.get("X"), x):
        raise ValueError("ArchSBO warm start differs from the shared prefix")
    algorithm.tell(infills=initial)
    algorithm.evaluator.n_eval = len(records)
    return algorithm


def ask(aircraft, records, seed):
    problem = SealedAircraft(aircraft)
    candidates = [record.candidate for record in records]
    algorithm = warm_start(problem, records, seed)
    proposal = algorithm.ask()
    if proposal is None or len(proposal) != 1:
        raise ValueError("ArchSBO did not return one native proposal")
    candidate = problem.decode(proposal.get("X")[0])
    precheck = aircraft.precheck(candidate)
    structural = structural_margins(aircraft, candidate)
    return {
        "observed_statuses": dict(Counter(r.observation.status for r in records)),
        "exact_output_rows": int(np.isfinite(f).sum()),
        "native_surrogate_trained": bool(algorithm.infill_obj.was_trained),
        "candidate": candidate,
        "candidate_id": digest(candidate),
        "is_unseen": candidate not in candidates,
        "passes_screen": all(np.isfinite(v) and v <= 0. for v in (*precheck.values(), *structural.values())),
        "geometry_margins": precheck,
        "structural_margins": structural,
    }


def probe(folder, sha):
    header = read(folder/"plan.json")
    spec = plan(header["spec"]["name"])
    aircraft = UsableAircraft(spec)
    if (header != {"source_sha": sha, "spec": spec, "contract": aircraft.contract_id}
            or read(folder/"seal.json") != evidence_hashes(folder)):
        raise ValueError("Input is not a sealed pilot from the same code SHA")
    ledger = read_ledger(folder, aircraft.contract_id)
    rows = []
    for seed in spec["seeds"]:
        records = [ledger[f"{seed}/initial/{i}"][1] for i in range(1, max(SIZES)+1)]
        for size in SIZES:
            row = ask(aircraft, records[:size], seed)
            rows.append({"seed": seed, "initial": size, **row})
    return {"status": "COMPLETED_NATIVE_ARCHSBO_PROBE", "profile": spec["name"], "source_sha": sha,
            "versions": {name: version(name) for name in ("sb-arch-opt", "pymoo", "smt", "numpy", "scipy",
                                                           "numba", "scikit-learn")},
            "new_aircraft_calls": 0, "rcaide_mda_calls": 0, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output, folder = args.output.resolve(), args.input.resolve()
    if any(path in (output, *output.parents) for path in (Path(__file__).resolve().parent, folder)):
        parser.error("Write outside the code worktree and the sealed input")
    report = probe(folder, source_commit(args.expected_sha, clean=True))
    save(output, report)
    print(report["status"], report["profile"], len(report["rows"]))


if __name__ == "__main__":
    main()
