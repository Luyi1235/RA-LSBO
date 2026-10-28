"""Conservative translation of an LSBO Evaluation dict without running MDA."""

from .records import HIDDEN_MODES, SCHEMA_VERSION, RiskEvaluation, finite_number


PRECHECK = {"geometry_rejected", "analytical_constraint_rejected"}
INFRASTRUCTURE = {"timeout", "worker_no_result", "worker_nonzero_exit",
                  "invalid_worker_response"}
MODE_BY_STATUS = {mode: mode for mode in HIDDEN_MODES}
MODE_BY_STATUS.update({
    "mission_not_converged": "mission_solver_nonconvergence",
    "evaluation_error:DomainSolveError": "mission_solver_nonconvergence",
})


def normalize_evaluation(raw, required_constraints, analytical_constraints=()):
    required = list(required_constraints)
    analytical = list(analytical_constraints)
    if (not required or any(not isinstance(name, str) or not name for name in required)
            or len(set(required)) != len(required)):
        raise ValueError("Required constraints must be nonempty and unique")
    if not set(analytical).issubset(required):
        raise ValueError("Analytical constraints must be a subset")
    status = raw.get("status", "unknown")
    metrics = raw.get("metrics", {})
    margins = raw.get("constraints", {})
    if not isinstance(status, str) or not isinstance(metrics, dict) or not isinstance(margins, dict):
        raise ValueError("Invalid legacy evaluation fields")
    issues, modes = [], []
    hidden = dict.fromkeys(HIDDEN_MODES)
    hidden_success = None
    evidence = {key: metrics[key] for key in (
        "failure_stage", "failure_message", "failure_origin", "solver_error_file",
        "energy_audit_status", "mission_solver_version", "worker_dir",
        "solver_discharge_margin_W", "solver_discharge_margin_source",
    ) if key in metrics}
    mode = MODE_BY_STATUS.get(status)
    failed_segments = [
        s.get("segment", "unknown") for s in metrics.get("mission_solver_segments", [])
        if s.get("converged") is False
    ]
    if failed_segments:
        evidence["failed_segments"] = failed_segments
    if status in PRECHECK:
        state = "precheck_rejected"
    elif status in INFRASTRUCTURE:
        state = "infrastructure_error"
    elif status.startswith(("worker_error:", "exception:")):
        # An exception alone does not establish an aircraft-dependent failure.
        state = "unknown"
        issues.append("unclassified_worker_or_interface_error")
    else:
        if metrics.get("energy_audit_status") == "failed":
            mode = "energy_audit_failure"
        elif failed_segments:
            mode = "mission_solver_nonconvergence"
        if mode:
            state = "hidden_failure"
            modes.append(mode)
            hidden[mode] = 1
            hidden_success = 0
        elif (status in ("ok", "constraint_violation")
              and raw.get("solver_success") is True and finite_number(raw.get("objective"))):
            state = "completed"
            hidden_success = 1
            hidden = dict.fromkeys(HIDDEN_MODES, 0)
        else:
            state = "unknown"
            issues.append("insufficient_evidence_for_terminal_outcome")

    objective_observed = state == "completed"
    if not objective_observed and raw.get("objective") is not None:
        issues.append("discarded_untrusted_objective")
    objective = float(raw["objective"]) if objective_observed else None
    supplied_mask = raw.get("constraint_observed", {})
    if not isinstance(supplied_mask, dict):
        raise ValueError("constraint_observed must be a mapping")
    if any(type(value) is not bool for value in supplied_mask.values()):
        raise ValueError("Constraint observation flags must be boolean")
    constraints, observed = {}, {}
    for name in required:
        value = margins.get(name)
        # Legacy final outputs are valid on completion. On failure, numeric
        # leftovers are not trusted without an explicit per-output validity flag.
        default = objective_observed or (state == "precheck_rejected" and name in analytical)
        trusted = supplied_mask.get(name, default)
        if state in ("infrastructure_error", "unknown"):
            trusted = False
        valid = trusted and finite_number(value)
        observed[name] = valid
        constraints[name] = float(value) if valid else None
        if value is not None and not valid:
            issues.append("untrusted_or_nonfinite_constraint:" + name)
        if valid and value > 0:
            modes.append("constraint:" + name)

    violations = any(observed[k] and constraints[k] > 0 for k in required)
    if state in ("precheck_rejected", "hidden_failure") or violations:
        feasible = False
    elif objective_observed and all(observed.values()):
        feasible = True
    else:
        feasible = None
    if state == "completed":
        state = "feasible" if feasible is True else "infeasible" if feasible is False else "incomplete"
        if feasible is None:
            issues.append("completed_but_required_constraints_missing")
        if (status == "ok" and violations) or (status == "constraint_violation" and feasible is True):
            issues.append("status_constraint_mismatch")
    mda = metrics.get("mda_evaluated")
    if mda is not None and type(mda) is not bool:
        raise ValueError("mda_evaluated must be boolean or null")
    if status in PRECHECK:
        if mda is True:
            raise ValueError("Precheck rejection cannot claim an MDA evaluation")
        mda = False
    elapsed = raw.get("elapsed_seconds")
    elapsed = float(elapsed) if finite_number(elapsed) and elapsed >= 0 else None
    return RiskEvaluation(
        SCHEMA_VERSION, status, state, objective, objective_observed,
        constraints, observed, analytical, hidden, hidden_success, modes,
        feasible, mda, status not in PRECHECK, elapsed, evidence, issues,
    )
