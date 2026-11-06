"""Pure, same-design opportunity diagnostics; no policy or evaluator access."""
import hashlib
import json
import math


def fingerprint(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _unique(rows):
    ids = [fingerprint(x) for x in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidates in a declared pool")
    return ids


def snapshot(case, seed, iteration, history, original_pool, extended_pool,
             candidate, meta, incumbent):
    original_ids = _unique(original_pool)
    all_ids = _unique(extended_pool)
    chosen = fingerprint(candidate)
    if (all_ids[:len(original_ids)] != original_ids or chosen not in original_ids or
            set(all_ids) & {fingerprint(r.candidate) for r in history}):
        raise ValueError("Audit pool must extend the original pool without history repeats")
    model = []
    probabilities = meta.get("candidate_gp_original_risk")
    if probabilities is not None:
        pool = meta["pool"]
        ids = _unique(pool)
        eligible = meta["eligible"]
        utilities = meta["candidate_net_utility"]
        if (len(ids) != len(probabilities) or len(ids) != len(eligible) or
                len(ids) != len(utilities) or not set(ids) <= set(original_ids)):
            raise ValueError("Actual GP predictions are not bound to the original pool")
        for token, risk, allowed, utility in zip(ids, probabilities, eligible, utilities):
            if (type(allowed) is not bool or not isinstance(risk, (int, float)) or
                    not math.isfinite(risk) or not 0 <= risk <= 1 or
                    not isinstance(utility, (int, float)) or not math.isfinite(utility)):
                raise ValueError("Invalid genuine GP receipt")
            model.append({"id": token, "p_fail": float(risk), "eligible": allowed,
                          "utility": float(utility)})
    model_selected = meta.get("credibility_decision") is not None
    if model_selected:
        ids = [m["id"] for m in model]
        if chosen not in ids or not model[ids.index(chosen)]["eligible"]:
            raise ValueError("Actual selected GP candidate lacks aligned forecast")
        r = meta["credibility_decision"]
        if abs(r["selected_predicted_failure"] - model[ids.index(chosen)]["p_fail"]) > 1e-10:
            raise ValueError("Selected receipt differs from original GP-only risk")
    return {"case": case, "seed": seed, "iteration": iteration,
            "pre_outcome_history_hash": fingerprint([r.to_dict() for r in history]),
            "history_size": len(history), "incumbent": incumbent,
            "original_pool": original_pool, "audit_pool": extended_pool,
            "selected_id": chosen, "selection_mode": meta.get("mode"),
            "selected_has_model_receipt": model_selected, "model_predictions": model,
            "audit_truth_available_to_policy": False}


def summarize(snap, truth):
    """Oracle is a within-frozen-pool diagnostic, NOT a deployable competitor."""
    ids = _unique(snap["audit_pool"])
    by_id = {}
    for row in truth:
        if row["id"] in by_id or type(row.get("feasible")) is not bool:
            raise ValueError("Duplicate or unknown audit truth")
        objective = row.get("feasible_objective")
        if row["feasible"]:
            if type(objective) not in (int, float) or not math.isfinite(objective):
                raise ValueError("Feasible truth requires exact objective")
        elif objective is not None:
            raise ValueError("Never impute an objective to an infeasible design")
        by_id[row["id"]] = row
    if set(by_id) != set(ids):
        raise ValueError("Audit must cover exactly the sealed candidate pool")
    original = _unique(snap["original_pool"])
    models = snap["model_predictions"]
    if len({m["id"] for m in models}) != len(models) or not {m["id"] for m in models} <= set(original):
        raise ValueError("Malformed model prediction roster")
    eligible = [m["id"] for m in models if m["eligible"]]
    incumbent = snap["incumbent"]

    def opportunity(subset):
        values = [by_id[i]["feasible_objective"] for i in subset if by_id[i]["feasible"]]
        best = min(values) if values else None
        return {"candidates": len(subset), "feasible": len(values), "best": best,
                "improves_known_incumbent": None if incumbent is None else
                    sum(v < incumbent for v in values),
                "best_improvement": None if incumbent is None or best is None else
                    max(0., incumbent - best)}

    def risk_score(items):
        if not items:
            return {"n": 0, "brier": None, "mean_predicted_failure": None,
                    "actual_failure_rate": None, "low_risk_count": 0,
                    "low_risk_failures": 0}
        outcomes = [int(not by_id[m["id"]]["feasible"]) for m in items]
        low = [(m, y) for m, y in zip(items, outcomes) if m["p_fail"] <= .2]
        return {"n": len(items),
                "brier": sum((m["p_fail"] - y)**2 for m, y in zip(items, outcomes)) / len(items),
                "mean_predicted_failure": sum(m["p_fail"] for m in items) / len(items),
                "actual_failure_rate": sum(outcomes) / len(items),
                "low_risk_count": len(low), "low_risk_failures": sum(y for _, y in low)}

    chosen = by_id[snap["selected_id"]]
    original_summary = opportunity(original)
    ranked = sorted((m for m in models if m["eligible"]),
                    key=lambda m: -m["utility"])  # stable original-order ties
    k = min(8, len(ranked))
    selected_prediction = ([m for m in models if m["id"] == snap["selected_id"]]
                           if snap["selected_has_model_receipt"] else [])
    return {"case": snap["case"], "seed": snap["seed"], "iteration": snap["iteration"],
            "snapshot_sha256": fingerprint(snap), "selection_mode": snap["selection_mode"],
            "original": original_summary, "expanded": opportunity(ids),
            "model_pool": opportunity([m["id"] for m in models]),
            "eligible": opportunity(eligible), "top8_utility": opportunity([m["id"] for m in ranked[:k]]),
            "selected": chosen,
            "selected_feasible_gap_to_original_best": (
                chosen["feasible_objective"] - original_summary["best"] if chosen["feasible"] else None),
            "risk_all_model_candidates": risk_score(models),
            "risk_actual_selected_with_receipt": risk_score(selected_prediction),
            "oracle_is_within_pool_only": True,
            "expanded_pool_has_no_gp_scores": True,
            "same_state_not_alternative_on_policy_performance": True}
