"""Matched energy acquisition with explicit risk and a shared cold fallback."""

import math


def energy_scores(rule, improvement, probability, coverage):
    if improvement is None:
        return coverage
    if len(improvement) != len(probability):
        raise ValueError("Energy and feasibility forecasts must align")
    if rule == "ei":
        return improvement
    if rule == "cei":
        return [ei*p for ei, p in zip(improvement, probability)]
    raise ValueError("Unknown energy acquisition rule")


def energy_receipt(rule, receipt, scores, prefix, model_due, checked_label):
    """Reproduce the saved rule and validate its eligibility from real feedback."""
    if set(receipt) != {"best", "exact_count", "objective_model", "representation",
                        "representation_updated", "improvement", "probability", "coverage"}:
        raise ValueError("Missing energy acquisition receipt")
    trusted = [r for r in prefix
               if r["observation"]["status"] == "completed"
               and not r["diagnostics"].get("numerical_failure")
               and r["observation"]["objective"]["kind"] == "exact"]
    best = min((r["observation"]["objective"]["value"] for r in trusted
                if checked_label(r) is True), default=None)
    if best != receipt["best"] or len(trusted) != receipt["exact_count"]:
        raise ValueError("Energy prefix evidence does not support the incumbent or objective fit")
    if receipt["representation_updated"] is not model_due or not receipt["representation"]:
        raise ValueError("Objective representation schedule mismatch")
    ready = best is not None and len(trusted) >= 2
    improvement, probability, coverage = (receipt[key] for key in ("improvement", "probability", "coverage"))
    if ready:
        if (not receipt["objective_model"] or coverage is not None
                or not isinstance(improvement, list) or not isinstance(probability, list)
                or len(improvement) != len(scores) or len(probability) != len(scores)
                or not all(math.isfinite(x) and x >= 0 for x in improvement)
                or not all(0 <= x <= 1 for x in probability)):
            raise ValueError("Invalid EI or feasibility forecast")
    elif (receipt["objective_model"] is not None or improvement is not None or probability is not None
          or not isinstance(coverage, list) or len(coverage) != len(scores) or not all(math.isfinite(x) and x >= 0 for x in coverage)):
        raise ValueError("Invalid shared coverage fallback")
    if any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
           for a, b in zip(scores, energy_scores(rule, improvement, probability, coverage))):
        raise ValueError("Energy acquisition scores do not replay")


def energy_pairs(trajectories):
    comparisons = (
        ("cei_latent", "ei_latent"),
        ("cei_explicit", "ei_explicit"),
        ("cei_latent", "cei_explicit"),
        ("ei_latent", "ei_explicit"),
    )
    lookup = {(row["seed"], row["arm"]): row for row in trajectories}
    local_trial = any(row["arm"] == "cei_latent_local" for row in trajectories)
    if local_trial:
        comparisons = (("cei_latent_local", "cei_latent"),)
    rows = []
    for seed in sorted({seed for seed, _ in lookup}):
        for arm, baseline in comparisons:
            left, right = lookup[seed, arm], lookup[seed, baseline]
            a, b = left["best_feasible_Wh"], right["best_feasible_Wh"]
            outcome = ("tie" if a == b else "win" if b is None or (a is not None and a < b) else "loss")
            rows.append({"seed": seed, "arm": arm, "baseline": baseline, "outcome": outcome,
                         "arm_best_Wh": a, "baseline_best_Wh": b,
                         "delta_Wh": a-b if a is not None and b is not None else None,
                         "arm_first_usable_call": left["first_usable_feasible_call"],
                         "baseline_first_usable_call": right["first_usable_feasible_call"],
                         "online_feasible_delta": left["online_feasible"]-right["online_feasible"],
                         "online_infeasible_delta": left["online_infeasible"]-right["online_infeasible"],
                         "online_unknown_delta": left["online_unknown"]-right["online_unknown"]})
            if local_trial:
                if left["switch_best_feasible_Wh"] != right["switch_best_feasible_Wh"]:
                    raise ValueError("The local-search arms did not share the switch incumbent")
                rows[-1].update(switch_best_feasible_Wh=left["switch_best_feasible_Wh"],
                                arm_augmentation_mode=left["augmentation_mode"],
                                arm_local_evaluations=left["local_augmentation_selected"],
                                baseline_global_addition_evaluations=right["augmentation_selected"])
    return rows
