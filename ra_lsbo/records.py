"""Observed failure events and missingness, not predicted risk probabilities."""

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Optional


SCHEMA_VERSION = "ra_evaluation_v1"
HIDDEN_MODES = (
    "geometry_generation_failure",
    "network_configuration_failure",
    "motor_design_failure",
    "mission_solver_nonconvergence",
    "energy_audit_failure",
)


def finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass
class RiskEvaluation:
    schema_version: str
    raw_status: str
    state: str
    objective: Optional[float]
    objective_observed: bool
    constraints: Dict[str, Optional[float]]
    constraint_observed: Dict[str, bool]
    analytical_constraints: List[str]
    hidden_labels: Dict[str, Optional[int]]
    hidden_success: Optional[int]
    failure_modes: List[str]
    feasible_label: Optional[bool]
    mda_evaluated: Optional[bool]
    budget_charged: bool
    elapsed_seconds: Optional[float]
    evidence: dict
    issues: List[str]

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported evaluation schema")
        if self.state not in {"feasible", "infeasible", "incomplete", "hidden_failure",
                              "infrastructure_error", "precheck_rejected", "unknown"}:
            raise ValueError("Unsupported terminal state")
        for flag in (self.objective_observed, self.budget_charged):
            if type(flag) is not bool:
                raise ValueError("Observation and budget flags must be boolean")
        for flag in (self.mda_evaluated, self.feasible_label):
            if flag is not None and type(flag) is not bool:
                raise ValueError("Optional boolean flags must be boolean or null")
        if set(self.constraints) != set(self.constraint_observed):
            raise ValueError("Constraint values and observation masks must agree")
        if not set(self.analytical_constraints).issubset(self.constraints):
            raise ValueError("Analytical constraints must be required constraints")
        if ((self.objective_observed and not finite_number(self.objective))
                or (not self.objective_observed and self.objective is not None)):
            raise ValueError("Observed objective must be finite, otherwise null")
        for name, observed in self.constraint_observed.items():
            value = self.constraints[name]
            if (type(observed) is not bool or (observed and not finite_number(value))
                    or (not observed and value is not None)):
                raise ValueError("Observed constraint must be finite, otherwise null")
        if set(self.hidden_labels) != set(HIDDEN_MODES):
            raise ValueError("Missing or unknown hidden failure head")
        labels = list(self.hidden_labels.values()) + [self.hidden_success]
        if any(v is not None and (type(v) is not int or v not in (0, 1)) for v in labels):
            raise ValueError("Labels must be 0, 1 or null")
        if self.objective_observed != (self.hidden_success == 1):
            raise ValueError("Objective validity and terminal success must agree")
        if self.hidden_success == 1 and any(v != 0 for v in self.hidden_labels.values()):
            raise ValueError("Successful terminal evaluation cannot have hidden failures")
        if self.hidden_success == 0 and 1 not in self.hidden_labels.values():
            raise ValueError("Known hidden failure needs an observed failure mode")
        expected_label = {
            "feasible": True, "infeasible": False, "incomplete": None,
            "hidden_failure": False, "infrastructure_error": None,
            "precheck_rejected": False, "unknown": None,
        }[self.state]
        if self.feasible_label is not expected_label:
            raise ValueError("Terminal state and feasibility label disagree")
        completed = self.state in ("feasible", "infeasible", "incomplete")
        if completed != self.objective_observed:
            raise ValueError("Terminal state and objective validity disagree")
        if (self.state == "hidden_failure") != (self.hidden_success == 0):
            raise ValueError("Hidden failure state and success label disagree")
        if self.state == "precheck_rejected" and (self.mda_evaluated is not False or self.budget_charged):
            raise ValueError("Precheck rejection cannot consume an MDA evaluation")
        violated = any(v is not None and v > 0 for v in self.constraints.values())
        if self.state == "infeasible" and not violated:
            raise ValueError("Completed infeasible state needs an observed violation")
        if self.state == "incomplete" and (violated or all(self.constraint_observed.values())):
            raise ValueError("Incomplete state needs missing constraints and no observed violation")
        if self.feasible_label is True and (
            not self.objective_observed or self.hidden_success != 1
            or not all(self.constraint_observed.values())
            or any(v > 0 for v in self.constraints.values())
        ):
            raise ValueError("Feasible evaluation lacks complete valid observations")

    def to_dict(self):
        return asdict(self)
