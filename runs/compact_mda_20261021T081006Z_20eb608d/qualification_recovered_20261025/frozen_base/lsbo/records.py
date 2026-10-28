from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

from .space import Candidate


@dataclass
class Evaluation:
    objective: float | None
    solver_success: bool
    constraints: dict = field(default_factory=dict)
    status: str = "ok"
    metrics: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def has_objective(self):
        return self.solver_success and self.objective is not None and math.isfinite(self.objective)

    def feasible(self, required_constraints):
        return self.has_objective() and all(
            name in self.constraints and self.constraints[name] is not None
            and math.isfinite(self.constraints[name]) and self.constraints[name] <= 0
            for name in required_constraints)

    def to_dict(self):
        def clean(value):
            if isinstance(value, dict):
                return {str(k): clean(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [clean(v) for v in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        return clean(asdict(self))


@dataclass
class Observation:
    index: int
    candidate: Candidate
    result: Evaluation
    phase: str
    proposal: dict = field(default_factory=dict)

    def to_dict(self, required_constraints):
        return {"evaluation": self.index, "candidate": self.candidate.to_dict(),
                "result": self.result.to_dict(), "feasible": self.result.feasible(required_constraints),
                "phase": self.phase, "proposal": self.proposal}
