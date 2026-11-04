"""PR54 pure candidate policy: semantic-only vs budget-aware dual-view BO.

Consumes only posterior forecasts available BEFORE the next evaluation.
Does not fit GP/VAE, evaluate a physical design, or read future outcomes.
The *observed* cap is enforced separately by the sequential trial driver.
Predicted risk charges are fallible and offer NO probabilistic safety claim.
"""
import math

SCHEMA = "ra_lsbo_pr54_observed_cap_and_proxy_budget_v1"
POLICIES = ("semantic_only", "budgeted")
EPS = 1e-12


def observed_failures(records, initial, iteration):
    """Count observed unusable online outputs, not initial failures or predictions."""
    if (type(initial) is not int or initial < 1 or
            type(iteration) is not int or iteration < 0 or
            len(records) != initial + iteration):
        raise ValueError("Observed failure ledger does not match settled online prefix")
    failures = 0
    for row in records[initial:]:
        packet = row.observation
        if (packet.status != "completed" or
                row.physical_feasible is None):
            raise ValueError("Unknown/untrusted physical outcome cannot enter budget ledger")
        failures += not row.usable_feasible
    return int(failures)


def _finite_probability(x):
    if not isinstance(x, (int,float)) or not math.isfinite(float(x)) or not 0.<=x<=1.:
        raise ValueError("Nonfinite or invalid GP risk-proxy probability")
    return float(x)


def _score(mode, p, ei, uncertainty):
    """Use constraint probability ONLY, not the one-class record-completion GP."""
    if mode == "budgeted_boundary_information":
        q = min(1.-EPS,max(EPS,p))
        return -q*math.log(q)-(1.-q)*math.log(1.-q) + (
            .01*uncertainty/(1.+uncertainty))
    score = math.log(max(p,1e-300))
    if mode in ("probability_weighted_feasible_improvement","balanced_feasible_improvement"):
        if ei is None:
            raise ValueError("Feasible EI mode without an objective GP")
        score += math.log(max(ei,1e-300))
    return score


def choose_semantic_or_budget(meta, ei, uncertainty, *, policy,
                              iteration, horizon, failure_cap, failures_seen):
    """Return a new selected index plus an audited, non-guaranteed risk decision.

    The ORIGINAL acquisition's support restoration and boundary phase are
    retained as auditable scheduling decisions. Constraint-only probability
    replaces the record-status completion factor in *ranking*. Eligibility is
    recomputed from constraints' optimistic violation, not completion GP.
    If the predicted dynamic allowance is infeasible, select minimum predicted
    risk among supported candidates (never pretend to have passed the gate).
    """
    if policy not in POLICIES:
        raise ValueError("Unknown opt-in candidate policy")
    if (type(iteration) is not int or type(horizon) is not int or
            type(failure_cap) is not int or type(failures_seen) is not int or
            not 0 <= iteration < horizon or not 1 <= failure_cap <= horizon or
            not 0 <= failures_seen < failure_cap):
        raise ValueError("No online slots, or observed failure cap exhausted")
    heads = meta.get("per_constraint_probability")
    violation = meta.get("optimistic_violation")
    legacy_scores = meta.get("scores")
    if not isinstance(heads,dict) or not heads or not isinstance(violation,list) or not violation:
        raise ValueError("Missing original constrained posterior selection arrays")
    n = len(violation)
    if (not isinstance(legacy_scores,list) or len(legacy_scores)!=n or
            not all(isinstance(v,list) and len(v)==n for v in heads.values())):
        raise ValueError("Misaligned GP risk posterior arrays")
    if ei is not None and len(ei)!=n:
        raise ValueError("Misaligned expected improvement array")
    if len(uncertainty)!=n:
        raise ValueError("Misaligned uncertainty array")
    probabilities=[]
    for i in range(n):
        probabilities.append(math.prod(_finite_probability(values[i])
                          for values in heads.values()))
    margins=[]
    for v in violation:
        if not isinstance(v,(int,float)) or not math.isfinite(float(v)) or v<0:
            raise ValueError("Expected finite nonnegative optimistic violation")
        margins.append(float(v))
    uncertain=[]
    for v in uncertainty:
        if not isinstance(v,(int,float)) or not math.isfinite(float(v)) or v<0:
            raise ValueError("Invalid nonnegative uncertainty bonus")
        uncertain.append(float(v))
    improvements = None
    if ei is not None:
        improvements=[]
        for v in ei:
            if not isinstance(v,(int,float)) or not math.isfinite(float(v)) or v<0:
                raise ValueError("Negative or nonfinite feasible EI")
            improvements.append(float(v))
    close = [i for i,v in enumerate(margins) if v <= 1e-10]
    fallback = None
    if not close:
        minimum = min(margins)
        close = [i for i,v in enumerate(margins)
                 if math.isclose(v,minimum,rel_tol=1e-10,abs_tol=1e-12)]
        fallback = "no_constraint_optimistic_support"
    mode = meta.get("mode")
    if not isinstance(mode,str):
        raise ValueError("Missing registered original phase")
    scores = [_score(mode, p, improvements[i] if improvements is not None else None, uncertain[i])
              for i,p in enumerate(probabilities)]
    boundary = mode == "budgeted_boundary_information"
    remaining = horizon-iteration
    allowance = (failure_cap-failures_seen)/remaining
    risk = [1.0-p for p in probabilities]
    supported = close
    if policy == "budgeted":
        acceptable = [i for i in supported if risk[i] <= allowance+1e-12]
        if acceptable:
            selected = max(acceptable,key=lambda i:(scores[i],-i))
            response = "predicted_allowance_filter"
        else:
            selected = min(supported,key=lambda i:(risk[i],-scores[i],i))
            response = "no_candidate_below_allowance_min_predicted_failure"
            fallback = response
    else:
        selected = max(supported,key=lambda i:(scores[i],-i))
        response = "constraint_only_ranking"
    if not math.isfinite(scores[selected]):
        raise ValueError("Nonfinite selected acquisition score")
    updated = {
        **meta,
        "mode":mode,
        "scores":[round(s,15) if i in supported else None for i,s in enumerate(scores)],
        "legacy_scores_before_semantic_fix":legacy_scores,
        "original_selector_index_before_semantic_fix": next(
            (i for i,v in enumerate(legacy_scores) if v is not None and
             v==max(w for w in legacy_scores if w is not None)),None),
        "eligible":[i in supported for i in range(n)],
        "fallback":fallback if fallback else meta.get("fallback"),
        "feasibility_proxy_no_status":probabilities,
        "risk_semantics_rule":"constraint_only_not_record_completion",
        "risk_policy":policy,
        "risk_budget": {
            "schema":SCHEMA,
            "observed_online_failures_before_proposal":failures_seen,
            "online_failure_cap":failure_cap,
            "remaining_online_calls":remaining,
            "remaining_failure_allowance":failure_cap-failures_seen,
            "predicted_per_call_failure_allowance":allowance,
            "selected_predicted_failure":risk[selected],
            "selected_predicted_feasible_proxy":probabilities[selected],
            "candidate_risk_within_allowance":risk[selected] <= allowance+1e-12,
            "candidate_filter_applied":policy=="budgeted",
            "selection_rule":response,
            "initial_calls_not_charged_to_online_cap":True,
            "hard_observed_stop_separate":True,
            "probability_guarantee":False,
        },
        "probability_guarantee":False,
    }
    return selected,updated
