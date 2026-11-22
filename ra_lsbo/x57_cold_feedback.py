"""Map X-57 worker outcomes to usable observations and completion evidence."""

from .risk_core.observations import OutputObservation, finite

RISK_METRIC = "negative_minimum_admissible_terminal_power_W"
RISK_SCALE_W = 1000.
CERTIFICATE_SOURCE = "mock_converged_physical_limit_v1"
NUMERICAL = {"mission_not_converged", "evaluation_error:DomainSolveError"}


def certificate(raw, backend):
    """The real evaluator has no qualified certificate producer in v1."""
    metrics = raw.get("metrics")
    evidence = metrics.get("completion_certificate") if isinstance(metrics, dict) else None
    if backend != "mock" or not isinstance(evidence, dict):
        return None
    if (evidence.get("metric") != RISK_METRIC or evidence.get("source") != CERTIFICATE_SOURCE
            or evidence.get("validated_original_model") is not True):
        return None
    lower, upper = evidence.get("lower_W"), evidence.get("upper_W")
    if (not finite(lower) or lower < 0
            or (upper is not None and (not finite(upper) or upper <= lower))):
        return None
    return {"lower_W": float(lower), "upper_W": None if upper is None else float(upper)}


def normalize(raw, required, checks, backend):
    status = raw.get("status")
    metrics = raw.get("metrics")
    received = raw.get("constraints")
    constraints = dict(checks)
    objective = None
    completed = None
    bounds = None
    if (status in ("ok", "constraint_violation") and raw.get("solver_success") is True
            and finite(raw.get("objective")) and isinstance(metrics, dict)
            and isinstance(received, dict) and all(finite(received.get(name)) for name in required)
            and all(abs(received[k] - v) <= 1e-9 * max(1., abs(v)) for k, v in checks.items())):
        constraints.update((name, float(received[name])) for name in required)
        completed, objective = True, float(raw["objective"])
        outcome = ("completed_usable" if all(constraints[name] <= 0 for name in required)
                   else "completed_constraint_violation")
    elif status == "physical_early_stop" and raw.get("solver_success") is False:
        bounds = certificate(raw, backend)
        if bounds is not None:
            outcome, completed = "physical_early_stop", False
        elif backend == "mock" and isinstance(metrics, dict) and metrics.get("physical_stop_verified") is True:
            # The physical classification has independent evidence in the mock protocol.
            outcome, completed = "physical_early_stop", False
        else:
            outcome = "unverified_termination"
    elif status in NUMERICAL and raw.get("solver_success") is False:
        outcome = "numerical_nonconvergence"
    else:
        outcome = "infrastructure_error"
    return {
        "outcome": outcome, "completion": completed, "usable": outcome == "completed_usable",
        "objective_Wh": objective, "objective_kind": "exact" if completed else "missing",
        "constraints": {name: constraints.get(name) for name in required},
        "constraint_kinds": {name: "exact" if name in constraints else "missing" for name in required},
        "risk_metric": RISK_METRIC, "risk_interval_W": bounds,
    }


def risk_observation(row, use_interval):
    feedback = row["feedback"]
    completed = feedback["completion"]
    if completed is None:
        return None
    if completed:
        return OutputObservation("interval", row["call_id"], upper=0., upper_inclusive=False)
    bounds = feedback["risk_interval_W"] if use_interval else None
    lower = bounds["lower_W"] / RISK_SCALE_W if bounds else 0.
    upper = bounds["upper_W"] / RISK_SCALE_W if bounds and bounds["upper_W"] is not None else None
    return OutputObservation("interval", row["call_id"], lower=lower, upper=upper)
