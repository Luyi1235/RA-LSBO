"""PR86-C objective observation model, separate from the existing two-risk-head GP.

No privileged objective labels. The input API accepts only observable rows.
"""
import math
import time
from unittest.mock import patch

from risk_learning.core import digest, features, known_label


def exact_feasible_objectives(rows):
    """Feasible exact objective values, using *observable* evidence only."""
    values = []
    for r in rows:
        if not all(known_label(r, h) for h in ("region", "resource")):
            continue
        o = r["objective"]
        if o["kind"] == "exact" and math.isfinite(float(o["value"])):
            values.append(float(o["value"]))
    return values


def outcome_state(rows):
    values = exact_feasible_objectives(rows)
    return dict(
        n=len(rows), feasible_found=bool(values),
        best_exact_feasible=min(values) if values else None,
        exact_feasible_count=len(values),
        exact_objective_count=sum(r["objective"]["kind"] == "exact" for r in rows),
        censored_region_count=sum(r["region"]["kind"] == "interval" for r in rows),
        missing_objective_count=sum(r["objective"]["kind"] == "missing" for r in rows),
    )


def _bill_key(bill):
    for k in ("objective_gp_head_started", "objective_gp_head_completed",
              "objective_gp_head_errors", "objective_gp_kernel_started",
              "objective_gp_kernel_completed", "objective_gp_kernel_errors",
              "objective_gp_kernel_candidates_rejected"):
        bill.setdefault(k, 0)


def predict_objective(observed_rows, queries, seed, bill):
    """Fit only exactly observed objective data; return mean/variance or None."""
    _bill_key(bill)
    exact = [r for r in observed_rows if r["objective"]["kind"] == "exact"]
    receipt = dict(
        training_rows=len(observed_rows), exact_training_rows=len(exact),
        training_input_sha256=digest([(r["candidate_id"], r["objective"]) for r in exact]),
        query_ids=[digest(c) for c in queries], privileged_training=False,
        missing_labels_imputed=False,
    )
    if len(exact) < 2:
        receipt["status"] = "INSUFFICIENT_EXACT_OBJECTIVES"
        return None, receipt

    import numpy as np
    from ra_lsbo.experiments.simplified_aircraft import dualview_optimizer as native
    from ra_lsbo.risk_core.observations import OutputObservation

    X = np.asarray([features(r["candidate"]) for r in exact], dtype=float)
    Q = np.asarray([features(c) for c in queries], dtype=float)
    O = [OutputObservation(kind="exact", value=float(r["objective"]["value"]),
                           evidence_id=digest(["objective",r["candidate_id"]])) for r in exact]
    original = native.EvidenceGP

    def counted(*args, **kwargs):
        bill["objective_gp_kernel_started"] += 1
        try:
            fitted = original(*args, **kwargs)
        except Exception:
            bill["objective_gp_kernel_errors"] += 1
            raise
        bill["objective_gp_kernel_completed"] += 1
        return fitted

    bill["objective_gp_head_started"] += 1
    start = time.perf_counter()
    try:
        with patch.object(native, "EvidenceGP", counted):
            fitted = native.StableGroupedHead(
                X, O, continuous_count=3, seed=seed, name="objective",
                objective=True, reference_scale=1.0)
        predicted = fitted.predict(Q)
    except native.ModelNumericalFailure as error:
        bill["objective_gp_head_errors"] += 1
        receipt.update(status="MODEL_NUMERICAL_FAILURE", message=str(error),
                       kernel_failures=error.failures)
        raise
    bill["objective_gp_head_completed"] += 1
    diag = fitted.diagnostics()
    bill["objective_gp_kernel_candidates_rejected"] += sum(
        entry["status"] != "ok" for entry in diag["candidates"])
    means = [float(v) for v in predicted["mean"]]
    variances = [float(v) for v in predicted["variance"]]
    if not all(math.isfinite(v) for v in means+variances) or any(v < 0 for v in variances):
        raise FloatingPointError("Invalid objective GP prediction")
    receipt.update(status="OK", elapsed_seconds=time.perf_counter()-start,
                   head=diag, objective_fit_only_exact=True)
    return dict(mean=means, variance=variances), receipt
