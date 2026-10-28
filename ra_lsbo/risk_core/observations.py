"""Keep numerical evidence, inequality evidence, and missingness distinct."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional, Tuple


def finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def named(value):
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class OutputObservation:
    kind: str
    evidence_id: str
    value: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    reason: Optional[str] = None

    def __post_init__(self):
        if self.kind not in {"exact", "interval", "missing"} or not named(self.evidence_id):
            raise ValueError("An observation needs a supported kind and evidence source")
        if type(self.lower_inclusive) is not bool or type(self.upper_inclusive) is not bool:
            raise ValueError("Endpoint inclusion flags must be boolean")
        if self.kind == "exact":
            if not finite(self.value) or self.lower is not None or self.upper is not None:
                raise ValueError("Exact evidence needs one finite value and no interval")
            if self.reason is not None:
                raise ValueError("Exact evidence cannot have a missingness reason")
        elif self.kind == "interval":
            if self.value is not None or self.reason is not None:
                raise ValueError("Inequality evidence is not a numeric or missing target")
            if self.lower is None and self.upper is None:
                raise ValueError("An interval must contain actual bound information")
            if any(v is not None and not finite(v) for v in (self.lower, self.upper)):
                raise ValueError("Use null for an unbounded endpoint")
            if (self.lower is not None and self.upper is not None
                    and self.lower >= self.upper):
                raise ValueError("Interval bounds must be strictly ordered")
        else:
            if any(v is not None for v in (self.value, self.lower, self.upper)):
                raise ValueError("A missing output cannot carry a training target")
            if not named(self.reason):
                raise ValueError("Missing observations need an explicit reason")

    def constraint_satisfaction(self):
        """Return evidence about g <= 0; an overlapping interval stays unknown."""
        if self.kind == "exact":
            return self.value <= 0
        if self.kind == "interval":
            if self.upper is not None and self.upper <= 0:
                return True
            if (self.lower is not None and
                    (self.lower > 0 or (self.lower == 0 and not self.lower_inclusive))):
                return False
        return None


@dataclass(frozen=True)
class RunObservation:
    design_id: str
    contract_id: str
    evidence_id: str
    status: str
    objective: OutputObservation
    constraints: Tuple[Tuple[str, OutputObservation], ...]
    passed_stages: Tuple[str, ...] = ()
    failed_stage: Optional[str] = None
    failure_cause: Optional[str] = None

    def __post_init__(self):
        if not all(named(v) for v in (self.design_id, self.contract_id, self.evidence_id)):
            raise ValueError("Design, physical contract and evidence identities are required")
        if self.status not in {"completed", "hidden_failure", "infrastructure_error", "unknown"}:
            raise ValueError("Unsupported run status")
        if not isinstance(self.objective, OutputObservation):
            raise ValueError("Objective evidence must be explicit")
        if not isinstance(self.constraints, tuple):
            raise ValueError("Use an immutable named output tuple")
        names = []
        for item in self.constraints:
            if (not isinstance(item, tuple) or len(item) != 2 or not named(item[0])
                    or not isinstance(item[1], OutputObservation)):
                raise ValueError("Each constraint needs a unique name and observation")
            names.append(item[0])
        if len(set(names)) != len(names):
            raise ValueError("Duplicate constraint names")
        if (not isinstance(self.passed_stages, tuple)
                or not all(named(s) for s in self.passed_stages)
                or len(set(self.passed_stages)) != len(self.passed_stages)):
            raise ValueError("Passed stages must be explicit, unique stage names")
        if self.failed_stage is not None and not named(self.failed_stage):
            raise ValueError("Invalid failure stage")
        if self.failed_stage in self.passed_stages:
            raise ValueError("A stage cannot both pass and fail in one declared trace")
        if self.status == "completed" and (
                self.failed_stage is not None or self.failure_cause is not None):
            raise ValueError("A completed run cannot report an unresolved terminal failure")
        if self.status == "hidden_failure" and not named(self.failure_cause):
            raise ValueError("Known hidden failure needs a supported cause")
        if self.failure_cause is not None and not named(self.failure_cause):
            raise ValueError("Invalid failure cause")

    def completion_label(self):
        if self.status == "completed":
            return 1
        if self.status == "hidden_failure":
            return 0
        return None

    def constraint(self, name):
        for key, value in self.constraints:
            if key == name:
                return value
        raise KeyError(name)


def validated_runs(runs):
    runs = tuple(runs)
    if not all(isinstance(run, RunObservation) for run in runs):
        raise ValueError("Expected explicit run observations")
    if len({run.contract_id for run in runs}) > 1:
        raise ValueError("One model training batch cannot mix physical contracts")
    return runs


def output_rows(runs, name, role="constraint"):
    """Export only supported output evidence; never invent a failed-run margin."""
    if role not in {"objective", "constraint"}:
        raise ValueError("Unsupported output role")
    result = []
    for run in validated_runs(runs):
        observation = run.objective if role == "objective" else run.constraint(name)
        # Infrastructure failures need separate source validation before their
        # outputs may enter a physical response model.
        if (run.status in {"infrastructure_error", "unknown"}
                or observation.kind == "missing"):
            continue
        result.append({
            "design_id": run.design_id, "contract_id": run.contract_id,
            "run_evidence_id": run.evidence_id, "output_evidence_id": observation.evidence_id,
            "kind": observation.kind, "value": observation.value,
            "lower": observation.lower, "upper": observation.upper,
            "lower_inclusive": observation.lower_inclusive,
            "upper_inclusive": observation.upper_inclusive,
        })
    return result


def completion_rows(runs):
    result = []
    for run in validated_runs(runs):
        label = run.completion_label()
        if label is not None:
            result.append({
                "design_id": run.design_id, "contract_id": run.contract_id,
                "evidence_id": run.evidence_id, "label": label,
                "passed_stages": run.passed_stages, "failed_stage": run.failed_stage,
                "failure_cause": run.failure_cause,
            })
    return result
