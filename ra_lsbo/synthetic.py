"""Deterministic hierarchical test problem with serial failures and censored outputs."""

from .budget import digest
from .features import load_base
from .legacy import normalize_evaluation


VARIABLES = [
    {"name": "layout", "kind": "nominal", "default": "single", "values": ["single", "distributed"]},
    {"name": "count", "kind": "ordinal", "default": 2, "values": [2, 4],
     "active_when": {"layout": ["distributed"]}},
    {"name": "x", "kind": "continuous", "default": .7, "bounds": [0., 1.]},
    {"name": "y", "kind": "continuous", "default": .65, "bounds": [0., 1.]},
    {"name": "spacing", "kind": "continuous", "default": .2, "bounds": [.05, .4],
     "active_when": {"layout": ["distributed"]}},
]


class SyntheticProblem:
    required_constraints = ("packing", "envelope", "load")
    analytical_constraints = ("packing",)

    def __init__(self, base_project, scenario="mixed"):
        if scenario not in ("mixed", "sparse", "no_initial_feasible"):
            raise ValueError("Unknown synthetic scenario")
        modules, _ = load_base(base_project)
        self.space = modules["space"].DesignSpace.from_dict({"variables": VARIABLES})
        self.candidate_class = modules["space"].Candidate
        self.scenario = scenario
        self.contract_id = digest({"schema": VARIABLES, "scenario": scenario,
                                   "equations_version": "hierarchical_serial_v1"})

    def canonical(self, raw):
        # This boundary does not repair invalid active continuous coordinates.
        candidate = self.space.canonical(self.candidate_class(**raw))
        for variable in self.space.continuous:
            if self.space.active(variable, candidate.discrete):
                if not variable.bounds[0] <= raw["continuous"][variable.name] <= variable.bounds[1]:
                    raise ValueError("Active candidate out of bounds")
        return candidate.to_dict()

    def precheck(self, raw):
        raw = self.canonical(raw)
        c, d = raw["continuous"], raw["discrete"]
        return {"packing": c["spacing"]*(d["count"]-1)-.9 if d["layout"] == "distributed" else -1.}

    def evaluate(self, raw):
        raw = self.canonical(raw)
        c, d = raw["continuous"], raw["discrete"]
        checks = self.precheck(raw)
        failure = None
        if checks["packing"] > 0:
            status = "analytical_constraint_rejected"
        elif c["x"] < .10:
            failure, status = "geometry", "geometry_generation_failure"
        elif d["layout"] == "distributed" and c["y"] < .12:
            failure, status = "network", "network_configuration_failure"
        elif c["x"]+c["y"] < .60:
            failure, status = "mission", "mission_solver_nonconvergence"
        else:
            status = "ok"
        objective = None
        if status == "ok":
            radius = .085 if self.scenario == "sparse" else .28
            checks.update(envelope=((c["x"]-.7)/radius)**2 + ((c["y"]-.65)/.31)**2-1,
                          load=c["x"]+c["y"]-1.48)
            objective = (100 + 40*(c["x"]-.8)**2 + 25*(c["y"]-.7)**2
                         + (2*(c["spacing"]-.18)**2 + .1*(d["count"]-2)
                            if d["layout"] == "distributed" else 1.))
            if any(value > 0 for value in checks.values()):
                status = "constraint_violation"
        return normalize_evaluation({
            "objective": objective, "solver_success": objective is not None,
            "constraints": checks, "status": status,
            "metrics": {"failure_stage": failure, "mda_evaluated": status != "analytical_constraint_rejected"},
        }, self.required_constraints, self.analytical_constraints).to_dict()
