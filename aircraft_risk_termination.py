"""Controlled feedback coarsening with no change to the reduced-aircraft truth."""
from dataclasses import replace
import hashlib
from pathlib import Path

from ra_lsbo.budget import digest
from ra_lsbo.risk_core.observations import OutputObservation

STRUCTURAL_GATES = ("stall", "root_bending")
POLICIES = ("native", "structural-first")


def coarsen(record, analytical, contract):
    """Emulate an early structural stop from a trusted, already computed margin."""
    packet = record.observation
    if packet.status in {"unknown", "infrastructure_error"} or record.diagnostics.get("numerical_failure"):
        return replace(record, observation=replace(packet, contract_id=contract))
    for name in STRUCTURAL_GATES:
        margin = packet.constraint(name)
        if margin.kind == "exact" and margin.constraint_satisfaction() is False:
            evidence = digest([packet.evidence_id, contract, name])
            outputs = tuple((key, value if key in analytical else
                             OutputObservation("interval", evidence+":"+key, lower=0., lower_inclusive=False)
                             if key == name else
                             OutputObservation("missing", evidence+":"+key, reason="structural_stop"))
                            for key, value in packet.constraints)
            observation = replace(packet, contract_id=contract, evidence_id=evidence, status="completed",
                                  objective=OutputObservation("missing", evidence+":energy", reason="structural_stop"),
                                  constraints=outputs, failed_stage=None, passed_stages=(), failure_cause=None)
            return replace(record, observation=observation, diagnostics={
                "mission_completed": False, "termination_reason": "controlled_structural_stop",
                "controlled_trigger": name, "model_family": "reduced_aircraft", "new_physical_mda": False})
    return replace(record, observation=replace(packet, contract_id=contract))


def with_termination_policy(problem, policy):
    if policy == "native":
        return problem
    if policy != "structural-first":
        raise ValueError("Unknown controlled termination policy")
    return StructuralFirstFeedback(problem)


class StructuralFirstFeedback:
    """Keep canonicalization and the exact precheck of the underlying evaluator."""

    def __init__(self, problem):
        self._problem = problem
        self.settings = {**problem.settings, "termination_feedback": "structural-first",
                         "structural_gate_order": list(STRUCTURAL_GATES),
                         "feedback_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self.contract_id = digest([problem.contract_id, self.settings])

    def __getattr__(self, name):
        return getattr(self._problem, name)

    def evaluate(self, candidate):
        return coarsen(self._problem.evaluate(candidate), self.analytical_constraints, self.contract_id)
