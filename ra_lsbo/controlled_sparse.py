"""A separate sparse controlled process with an interior mission-failure hole."""

import hashlib
from pathlib import Path

from .budget import digest
from .legacy import normalize_evaluation
from .mechanism_system import FiveModeProblem
from .records import HIDDEN_MODES


class SparseInteriorFiveModeProblem(FiveModeProblem):
    """Keep the five stages; move the mission risk into the useful search region."""

    def __init__(self, base_project):
        super().__init__(base_project)
        self.contract_id = digest({
            "parent_contract": self.contract_id,
            "controlled_case": "sparse_interior_mission_20261021_v1",
            "envelope_radii": [.12, .15], "hole_center": [.78, .69], "hole_radius": .055,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def evaluate(self, raw):
        value = self.canonical(raw)
        c, d = value["continuous"], value["discrete"]
        constraints = self.precheck(value)
        if constraints["packing"] > 0:
            return normalize_evaluation({
                "status": "analytical_constraint_rejected", "constraints": constraints,
                "solver_success": False, "objective": None}, self.required_constraints,
                self.analytical_constraints).to_dict()
        inside_hole = (c["x"]-.78)**2+(c["y"]-.69)**2 < .055**2
        predicates = (
            c["x"] < .08,
            d["layout"] == "distributed" and c["y"] < .12,
            c["x"] > .88 and d["layout"] == "distributed",
            c["x"]+c["y"] < .62 or inside_hole,
            c["y"] > .86)
        failed = next((i for i, flag in enumerate(predicates) if flag), None)
        objective = None
        if failed is None:
            constraints.update(
                envelope=((c["x"]-.69)/.12)**2+((c["y"]-.63)/.15)**2-1.,
                load=c["x"]+c["y"]-1.48)
            objective = (100+40*(c["x"]-.81)**2+25*(c["y"]-.72)**2
                         +(.1*(d["count"]-2)+2*(c["spacing"]-.18)**2
                           if d["layout"] == "distributed" else 1.))
        status = (HIDDEN_MODES[failed] if failed is not None else
                  "constraint_violation" if any(v > 0 for v in constraints.values()) else "ok")
        risk = normalize_evaluation({
            "status": status, "constraints": constraints, "objective": objective,
            "solver_success": objective is not None, "metrics": {"mda_evaluated": True}},
            self.required_constraints, self.analytical_constraints).to_dict()
        if failed is not None:
            for name in HIDDEN_MODES[:failed]:
                risk["hidden_labels"][name] = 0
        risk["evidence"]["controlled_stage_trace"] = [
            {"stage": name, "executed": failed is None or i <= failed,
             "failed": i == failed if failed is not None else False}
            for i, name in enumerate(HIDDEN_MODES)]
        risk["evidence"]["controlled_case"] = "sparse_interior_mission_20261021_v1"
        return risk


def enable_sparse_case():
    from . import system_experiment
    system_experiment.FiveModeProblem = SparseInteriorFiveModeProblem
