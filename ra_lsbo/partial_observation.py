"""Narrow, source-bound recovery of an audited mission with missing lift output."""

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path

from .legacy import normalize_evaluation as legacy_normalize
from .records import finite_number


RULE_ID = "audited_mission_partial_lift_20261021_v1"
LIFT_VERSION = "native_delta_lift_concept_screen_20261004_v1"
LIFT_SOURCE_SHA256 = "7f9cccaa216bdb40d974f252d97c1c4a8f509ce4a1e2cb38ba278e58122d8b5b"
LIFT_SOURCE = Path(__file__).resolve().parents[2]/"LSBO"/"X57-LSBO-refactored"/"problems/x57/simplified_powered_lift.py"
MESSAGE = "ValueError: Invalid equivalent low-speed lift limit"


def partial_lift_proof(raw):
    metrics = raw.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    origin, lift = metrics.get("failure_origin", {}), metrics.get("simplified_powered_lift", {})
    if not isinstance(origin, dict) or not isinstance(lift, dict):
        return None
    if (raw.get("status") != "evaluation_error:ValueError"
            or metrics.get("failure_stage") != "procedure.post_process"
            or metrics.get("failure_message") != MESSAGE
            or origin.get("function") != "corrected_limit"
            or Path(origin.get("file", "")).resolve() != LIFT_SOURCE.resolve()
            or metrics.get("postprocess_version") != "nasa_charge_soc_energy_audit_fail_closed_v1"
            or metrics.get("energy_audit_status") != "passed"
            or lift.get("version") != LIFT_VERSION or lift.get("status") != "failed"
            or lift.get("failure") != MESSAGE):
        return None
    segments = metrics.get("mission_solver_segments", [])
    if (not isinstance(segments, list) or not segments
            or any(not isinstance(s, dict) or s.get("converged") is not True for s in segments)):
        return None
    energy = metrics.get("mission_depletion_Wh")
    dry, stall, delta = (lift.get(name) for name in ("dry_clmax", "dry_stall_speed_mps", "delta_cl"))
    if (not all(finite_number(v) for v in (energy, dry, stall, delta))
            or energy <= 0 or dry <= 0 or stall <= 0 or dry+delta > 0):
        return None
    if hashlib.sha256(LIFT_SOURCE.read_bytes()).hexdigest() != LIFT_SOURCE_SHA256:
        return None
    return {
        "rule": RULE_ID, "physical_helper_sha256": LIFT_SOURCE_SHA256,
        "validated_mission_segments": len(segments), "audited_mission_depletion_Wh": energy,
        "invalid_equivalent_lift_limit": dry+delta,
        "interpretation": "Observed mission energy; final engineering feasibility remains unobserved"}


def normalize_with_partial_lift(raw, required_constraints, analytical_constraints=()):
    proof = partial_lift_proof(raw)
    if proof is None:
        return legacy_normalize(raw, required_constraints, analytical_constraints)
    carrier = deepcopy(raw)
    carrier.update(status="ok", solver_success=True, objective=proof["audited_mission_depletion_Wh"],
                   constraints={name: None for name in required_constraints},
                   constraint_observed={name: False for name in required_constraints})
    value = legacy_normalize(carrier, required_constraints, analytical_constraints)
    evidence = dict(value.evidence, partial_observation_proof=proof)
    return replace(
        value, raw_status=raw["status"], evidence=evidence,
        failure_modes=["unavailable_output:stall_speed"],
        issues=value.issues+["audited_objective_retained_invalid_lift_output_censored"])


def enable_partial_lift_normalization():
    """Install the declared observation rule in a fresh evaluation worker only."""
    from . import x57_live
    x57_live.normalize_evaluation = normalize_with_partial_lift
