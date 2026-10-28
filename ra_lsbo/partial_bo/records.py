"""Keep model termination, output validity, and physical feasibility separate."""

from dataclasses import asdict, dataclass

from ..budget import digest
from ..risk_core.observations import OutputObservation, RunObservation


@dataclass(frozen=True)
class DesignRecord:
    candidate: dict
    observation: RunObservation
    diagnostics: dict

    def __post_init__(self):
        if digest(self.candidate) != self.observation.design_id:
            raise ValueError("Observation belongs to another canonical design")

    @property
    def physical_feasible(self):
        if self.diagnostics.get("certified_physical_infeasible") is True:
            return False
        labels = [value.constraint_satisfaction() for _, value in self.observation.constraints]
        if False in labels:
            return False
        return True if labels and all(value is True for value in labels) else None

    @property
    def usable_feasible(self):
        return (self.observation.status == "completed" and self.physical_feasible is True
                and self.observation.objective.kind == "exact")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        packet = dict(value["observation"])
        packet["objective"] = OutputObservation(**packet["objective"])
        packet["constraints"] = tuple((name, OutputObservation(**observation))
                                      for name, observation in packet["constraints"])
        packet["passed_stages"] = tuple(packet["passed_stages"])
        return cls(value["candidate"], RunObservation(**packet), value["diagnostics"])

    def legacy_row(self, evaluation):
        observation = self.observation
        exact = observation.objective.kind == "exact"
        return {"evaluation": evaluation, "candidate": self.candidate,
                "evaluator_contract_sha256": observation.contract_id,
                "risk": {"objective": observation.objective.value if exact else None,
                         "objective_observed": exact,
                         "constraints": {name: value.value if value.kind == "exact" else None
                                         for name, value in observation.constraints},
                         "raw_status": self.diagnostics.get("termination_reason", observation.status),
                         "evidence": {"partial_observation_adapter": True}}}


def without_intervals(record):
    from dataclasses import replace
    def numerical_only(value):
        return (OutputObservation("missing", value.evidence_id, reason="interval_ablation")
                if value.kind == "interval" else value)
    observation = replace(
        record.observation, objective=numerical_only(record.observation.objective),
        constraints=tuple((name, numerical_only(value))
                          for name, value in record.observation.constraints))
    return DesignRecord(record.candidate, observation, record.diagnostics)
