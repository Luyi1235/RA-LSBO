"""PR60 exploratory branch-evidence risk dispersion acquisition (PURE Python).

The PR54 latent-objective GP / onehot-risk GP, phase schedule, and observed
six-failure stop remain byte-identical. The new engine only changes how each
genuine pre-outcome GP candidate is ranked.

A Beta(1,1) Bernoulli posterior describes the observed failure frequency of
each DISCRETE branch from all ALREADY-SETTLED shared initial/online records.
This is a model-misspecification *heuristic*, NOT a credible safety bound:
adaptive samples are not IID Bernoulli within a branch; individual continuous
designs also differ. Never infer safety or calibration from this posterior.

Variants:
  branch_mean:      max(original GP risk, branch Beta posterior mean), same
                    per-call predicted allowance and min-risk fallback.
  branch_dispersion:max(GP risk, Beta mean+ONE posterior std), same gate/fallback.
  branch_explore:   same risk as branch_dispersion; when NO candidate meets
                    its heuristic allowance, select largest branch uncertainty
                    instead of the original min-risk fallback.

None receives cross-seed empirical data or the PR59 holdout outcomes.
"""
from collections import defaultdict
import math

from ..budget import digest
from .risk_budget54_policy import _score, _finite_probability, observed_failures

VARIANTS = ("branch_mean", "branch_dispersion", "branch_explore")
SCHEMA = "ra_lsbo_pr60_branch_posterior_uncertainty_acquisition_v1"
BETA_PRIOR_ALPHA = 1.
BETA_PRIOR_BETA = 1.
DISPERSION_MULTIPLIER = 1.  # Registered before PR60 on-policy outcomes.
PROB_TOL = 1.e-12


def _positive_count(record):
    if record.observation.status != "completed" or record.physical_feasible is None:
        raise ValueError("Risk-evidence branch labels must be settled and physically known")
    return int(not record.usable_feasible)


def branch_posterior(records, candidates):
    """Compute branch risk information from observations strictly PRE proposal.

    No labels or scores of unobserved candidates are read. Branch IDs use the
    original hierarchical discrete choices, not latent coordinates.
    """
    if not candidates:
        raise ValueError("Need nonempty registered candidate pool")
    counts = defaultdict(lambda: [0,0])
    for r in records:
        branch = digest(r.candidate["discrete"])
        failed = _positive_count(r)
        counts[branch][0] += failed
        counts[branch][1] += (1-failed)
    output = []
    for c in candidates:
        branch = digest(c["discrete"])
        failures, successes = counts[branch]
        a, b = BETA_PRIOR_ALPHA + failures, BETA_PRIOR_BETA + successes
        total = a + b
        mean = a / total
        sd = math.sqrt(a*b/(total**2*(total+1)))
        output.append({
            "branch": branch, "observed_failures":failures,
            "observed_successes":successes,
            "n_settled_in_branch":failures+successes,
            "posterior_failure_mean":mean, "posterior_failure_sd":sd,
            "posterior_failure_mean_plus_sd":min(1.,mean+DISPERSION_MULTIPLIER*sd),
        })
    return output


def choose_branch_evidence(meta, candidates, records, expected_improvement,
                           predictive_uncertainty, *, variant,
                           iteration, horizon, failure_cap):
    """Select an ACTUAL candidate index from real pre-outcome posterior arrays.

    Reuses original PR54 support and period scheduling; only the risk-value
    and optionally its no-acceptable-candidate fallback are changed.
    """
    if variant not in VARIANTS:
        raise ValueError("Unregistered PR60 risk-UQ ablation")
    if (type(iteration) is not int or type(horizon) is not int or
            type(failure_cap) is not int or not 0<=iteration<horizon or
            not 1<=failure_cap<=horizon):
        raise ValueError("Invalid immutable PR54 online horizon/failure rule")
    n = len(candidates)
    if n < 1:
        raise ValueError("Empty original GP-selected candidate pool")
    original_risk = meta.get("feasibility_proxy_no_status")
    eligible = meta.get("eligible")
    mode = meta.get("mode")
    if (not isinstance(original_risk,list) or len(original_risk)!=n or
            not isinstance(eligible,list) or len(eligible)!=n or
            any(type(x) is not bool for x in eligible) or not any(eligible) or
            type(mode) is not str):
        raise ValueError("Misaligned original GP risk/optimistic-support arrays")
    if (not isinstance(predictive_uncertainty,list) or len(predictive_uncertainty)!=n or
            (expected_improvement is not None and (
                not isinstance(expected_improvement,list) or
                len(expected_improvement)!=n))):
        raise ValueError("PR60 posterior objective/uncertainty candidate arrays differ")
    support=[i for i,x in enumerate(eligible) if x]
    base_fail=[1. - _finite_probability(x) for x in original_risk]
    post=branch_posterior(records,candidates)
    corrected=[]
    scores=[]
    for i,evidence in enumerate(post):
        empirical = (evidence["posterior_failure_mean"]
                     if variant=="branch_mean"
                     else evidence["posterior_failure_mean_plus_sd"])
        proposed=max(base_fail[i],empirical)
        corrected.append(min(1.,max(0.,proposed)))
        ei=None if expected_improvement is None else float(expected_improvement[i])
        unc=float(predictive_uncertainty[i])
        if not math.isfinite(unc) or unc<0 or (ei is not None and (
                not math.isfinite(ei) or ei<0)):
            raise ValueError("Nonfinite original EI or GP uncertainty")
        scores.append(_score(mode,1.-corrected[i],ei,unc))
    n_initial = len(records)-iteration
    failures_seen=observed_failures(records,n_initial,iteration)
    allowance=(failure_cap-failures_seen)/(horizon-iteration)
    accepted=[i for i in support if corrected[i] <= allowance+PROB_TOL]
    if accepted:
        index=max(accepted,key=lambda i:(scores[i],-i))
        rule="empirical_branch_risk_under_predicted_allowance"
    elif variant=="branch_explore":
        # The original pure minimum-risk fallback in PR58 failed 42/47 times.
        # This explicit experimental arm spends its unavoidable analytical
        # evaluation on the most uncertain observed discrete branch, not on
        # a falsely safety-labelled design. Uncertainty is ONLY branch-level.
        index=max(support,key=lambda i:(
            post[i]["posterior_failure_sd"],
            float(predictive_uncertainty[i])/(1.+float(predictive_uncertainty[i])),
            scores[i],-i))
        rule="no_candidate_below_allowance_max_branch_information"
    else:
        index=min(support,key=lambda i:(corrected[i],-scores[i],i))
        rule="no_candidate_below_allowance_min_corrected_proxy"
    if not math.isfinite(scores[index]):
        raise ValueError("Nonfinite selected new acquisition")
    receipt={
        **meta.get("risk_budget", {}),
        "schema":SCHEMA,
        "risk_policy":variant,
        "selected_predicted_failure":corrected[index],
        "selected_predicted_feasible_proxy":1.-corrected[index],
        "selected_original_GP_failure_proxy":base_fail[index],
        "selected_branch_posterior":post[index],
        "candidate_risk_within_allowance":corrected[index]<=allowance+PROB_TOL,
        "candidate_filter_applied":True,
        "selection_rule":rule,
        "remaining_online_calls":horizon-iteration,
        "observed_online_failures_before_proposal":failures_seen,
        "remaining_failure_allowance":failure_cap-failures_seen,
        "predicted_per_call_failure_allowance":allowance,
        "fallback_not_a_safety_guarantee":not bool(accepted),
        "branch_posterior_is_not_safe_or_calibrated":True,
        "new_aircraft_calls":0,
        "probability_guarantee":False,
    }
    prior_scores=meta.get("scores")
    if not isinstance(prior_scores,list) or len(prior_scores)!=n:
        raise ValueError("Unregistered original risk acquisition trace")
    return index,{
        **meta,
        "risk_policy":variant,
        "scores":[scores[i] if i in support else None for i in range(n)],
        "previous_PR54_risk_scores":prior_scores,
        "eligible":eligible,
        "fallback":(rule if not accepted else meta.get("fallback")),
        "risk_budget":receipt,
        "feasibility_proxy_no_status":[1.-v for v in corrected],
        "empirical_branch_risk":post,
        "conservative_risk_proxy_for_candidate":[float(v) for v in corrected],
        "previous_PR54_min_risk_selection_index":meta.get("previous_PR54_min_risk_selection_index"),
        "probability_guarantee":False,
    }
