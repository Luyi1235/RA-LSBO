"""PR63 candidate-level credibility × optimization-value acquisition (pure Python).

Only the new opt-in PR63 research subclass uses this. The exact PR54 GP/VAE,
kernel evidence, objective EI, risk marginal product and original phase
schedule remain unchanged. There is NO hard risk-probability gate and therefore
no "none can pass -> high-failure fallback" mechanism.

Local credibility is a spatial, same-discrete-branch evidence HEURISTIC:
past settled observations are weighted by continuous active-coordinate
distance, NOT pooled as IID branch Bernoulli trials. It is neither a calibrated
risk probability nor a statistical confidence/safety bound.

The acquisition is an explicit soft benefit-cost decision, not "make the risk
upper bound larger":
  utility(x) = normalized_gain(x) * (1 - risk_proxy(x))
             - failure_price(t) * risk_proxy(x)
             + 0.08 * epistemic_ambiguity(x)
                    * normalized_gain(x) * (1-risk_proxy(x))**2   [value arm only]

"would_defer" records when the best net value <=0 relative to a zero-cost
outside action. Under the predeclared equal-SIX-evaluation trial it does NOT
skip an evaluation; actual abstention belongs to a separate cost-based study.

No unseen candidate truth, test-seed labels, or PR59/60/61 output is ingested.
"""
import math
import statistics

from .risk_budget54_policy import _finite_probability, observed_failures

SCHEMA="ra_lsbo_pr63_candidate_local_credibility_soft_value_v1"
VARIANTS=("soft_gp","local_blend","local_value")
LOCAL_LENGTH_SCALE=.25
LOCAL_PRIOR_MASS=1.
LOCAL_RISK_PRIOR=.5
TRUST_PSEUDO_MASS=2.
FAILURE_PRICE_BASE=.15
FAILURE_PRICE_BUDGET_SLOPE=.25
INFORMATION_VALUE_COEFFICIENT=.08
NO_PROBABILITY_GUARANTEE=True


def _coordinates(problem, candidate):
    """Continuous metric on the *active* subspace of this exact branch."""
    active=[]
    discrete=candidate["discrete"]
    values=candidate["continuous"]
    for variable in problem.space.continuous:
        if problem.space.active(variable,discrete):
            name=variable.name
            low,high=variable.bounds
            if high<=low or not math.isfinite(float(values[name])):
                raise ValueError("Invalid active coordinate or physical bound")
            z=(float(values[name])-float(low))/(float(high)-float(low))
            if not 0 <= z <= 1:
                raise ValueError("Active normalized coordinate outside declared support")
            active.append((name,z))
    return active


def local_credibility(problem, records, candidate, gp_risk):
    """Use ONLY settled records, and do not homogenize continuous designs.

    The support mass is sum(exp(-0.5*d(x,x_i)^2/.25^2)) for same discrete
    branch, with distance on active, normalized continuous coordinates.
    Failures from distant points carry little weight.
    """
    gp_risk=_finite_probability(gp_risk)
    if not records:
        raise ValueError("PR63 requires an observed historical prefix")
    present=_coordinates(problem,candidate)
    branch=candidate["discrete"]
    mass=weighted_fail=mass_sq=0.
    same_branch=0
    nearest=None
    for row in records:
        if row.observation.status!="completed" or row.physical_feasible is None:
            raise ValueError("No untrusted/unknown physical outcome in local risk support")
        if row.candidate["discrete"]!=branch:
            continue
        previous=dict(_coordinates(problem,row.candidate))
        if len(previous)!=len(present) or any(k not in previous for k,_ in present):
            raise ValueError("Historical active mask differs inside same discrete branch")
        distance=math.sqrt(sum((value-previous[k])**2 for k,value in present)/
                           max(1,len(present)))
        if nearest is None or distance<nearest:
            nearest=distance
        weight=math.exp(-.5*(distance/LOCAL_LENGTH_SCALE)**2)
        mass+=weight
        mass_sq+=weight*weight
        weighted_fail+=weight*int(not row.usable_feasible)
        same_branch+=1
    empirical=(LOCAL_RISK_PRIOR*LOCAL_PRIOR_MASS+weighted_fail)/(
        LOCAL_PRIOR_MASS+mass)
    trust=mass/(mass+TRUST_PSEUDO_MASS) if mass else 0.
    # A convex blend is NOT a Bayesian posterior; it is a frozen, testable
    # candidate-specific regularizer whose reliability is still unverified.
    adjusted=(1.-trust)*gp_risk+trust*empirical
    disagreement=abs(gp_risk-empirical)
    ambiguity=min(1.,math.sqrt(empirical*(1.-empirical)/(LOCAL_PRIOR_MASS+mass))+
                  (1.-trust)*.25+trust*disagreement)
    return {
        "same_discrete_branch_observations":same_branch,
        "weighted_local_support_mass":mass,
        "local_effective_sample_size":mass*mass/mass_sq if mass_sq>0 else 0.,
        "nearest_same_branch_distance":nearest,
        "local_empirical_failure_heuristic":empirical,
        "local_trust_weight":trust,
        "gp_local_failure_disagreement":disagreement,
        "candidate_local_ambiguity":ambiguity,
        "gp_original_failure_proxy":gp_risk,
        "candidate_blended_failure_proxy":max(0.,min(1.,adjusted)),
        "statistically_calibrated":False,
        "independent_bernoulli_assumption_made":False,
    }


def _normalized_gains(mode, prob_fail, ei, support):
    n=len(prob_fail)
    if mode=="budgeted_boundary_information":
        return [4.*p*(1.-p) for p in prob_fail]
    if mode in ("probability_weighted_feasible_improvement",
                "balanced_feasible_improvement"):
        if ei is None or len(ei)!=n:
            raise ValueError("PR63 EI exploitation without aligned objective EI forecasts")
        values=[]
        for x in ei:
            if type(x) not in (int,float) or not math.isfinite(x) or x<0:
                raise ValueError("PR63 invalid original EI")
            values.append(float(x))
        positive=sorted(v for i,v in enumerate(values) if support[i] and v>0)
        if not positive:
            return [1.]*n  # Original may re-enter feasible discovery.
        scale=statistics.median(positive)
        return [v/(v+scale) if v>0 else 0. for v in values]
    if mode=="feasible_discovery":
        return [1.]*n
    raise ValueError("Unsupported unregistered historical acquisition phase")


def choose_local_value(problem,meta,candidates,records,expected_improvement,
                       predictive_uncertainty,*,variant,iteration,
                       initial_count=12,horizon=12,failure_cap=6):
    """Choose a real on-policy candidate with SOFT cost of observed failures.

    Must be called with the exact candidates passed to that GP predict().
    All permitted support restoration and coverage are handled by the
    original engine; unsupported candidates are NOT evaluated here.
    """
    if variant not in VARIANTS:
        raise ValueError("Unregistered PR63 candidate credibility ablation")
    if (type(initial_count) is not int or type(horizon) is not int or
            type(failure_cap) is not int or type(iteration) is not int or
            initial_count!=12 or horizon!=12 or failure_cap!=6 or
            not 0<=iteration<6 or len(records)!=initial_count+iteration):
        raise ValueError("PR63 frozen original 12/6/12 online prefix contract changed")
    n=len(candidates)
    if n<1:
        raise ValueError("Empty actual GP prediction candidate set")
    raw_prob=meta.get("feasibility_proxy_no_status")
    supported=meta.get("eligible")
    mode=meta.get("mode")
    if (not isinstance(raw_prob,list) or len(raw_prob)!=n or
            not isinstance(supported,list) or len(supported)!=n or
            any(type(v) is not bool for v in supported) or
            not any(supported) or
            type(mode) is not str or
            not isinstance(predictive_uncertainty,list) or
            len(predictive_uncertainty)!=n):
        raise ValueError("Misaligned original GP feasibility/optimistic support")
    original=[1.-_finite_probability(p) for p in raw_prob]
    for u in predictive_uncertainty:
        if type(u) not in (int,float) or not math.isfinite(u) or u<0:
            raise ValueError("Nonfinite original GP uncertainty field")
    evidence=[local_credibility(problem,records,c,g)
              for c,g in zip(candidates,original)]
    gains=_normalized_gains(mode,original,expected_improvement,supported)
    spent=observed_failures(records,initial_count,iteration)
    # Unlike PR54/60: never exclude all candidates on a predicted-risk floor.
    price=FAILURE_PRICE_BASE+FAILURE_PRICE_BUDGET_SLOPE*(spent/failure_cap)
    utils=[]
    extras=[]
    adjusted=[]
    for e,g in zip(evidence,gains):
        local_r=e["candidate_blended_failure_proxy"]
        risk=e["gp_original_failure_proxy"] if variant=="soft_gp" else local_r
        info=(INFORMATION_VALUE_COEFFICIENT*e["candidate_local_ambiguity"]*
              g*(1.-risk)**2 if variant=="local_value" else 0.)
        u=g*(1.-risk)-price*risk+info
        if not math.isfinite(u):
            raise ValueError("Nonfinite candidate expected net-value proxy")
        utils.append(float(u))
        extras.append(float(info))
        adjusted.append(float(risk))
    support_indices=[i for i,yes in enumerate(supported) if yes]
    index=max(support_indices,key=lambda i:(utils[i],-i))
    outside=utils[index]<=0.
    base=meta.get("risk_budget",{})
    receipt={
        **base,
        "schema":SCHEMA,"risk_policy":variant,
        "selection_rule":"no_hard_gate_max_expected_net_value",
        "candidate_filter_applied":False,
        "no_hard_risk_gate":True,
        "selected_predicted_failure":adjusted[index],
        "selected_predicted_feasible_proxy":1.-adjusted[index],
        "selected_original_GP_failure_proxy":original[index],
        "selected_local_evidence":evidence[index],
        "selected_objective_gain_normalized":gains[index],
        "selected_information_bonus":extras[index],
        "selected_net_utility":utils[index],
        "failure_cost_weight":price,
        "would_defer_in_unconstrained_utility_model":outside,
        "actual_action":"evaluate_under_frozen_six_call_protocol",
        "outside_option_utility":0.,
        "abstention_not_performed_in_equal_budget_study":True,
        "candidate_risk_within_allowance":None,
        "predicted_per_call_failure_allowance":None,
        "probability_guarantee":False,
        "risk_proxy_is_not_calibrated":True,
    }
    return index,{
        **meta,
        "risk_policy":variant,
        "scores":[u if supported[i] else None for i,u in enumerate(utils)],
        "previous_PR54_scores_before_soft_value":meta.get("scores"),
        "eligible":supported,
        "fallback":meta.get("fallback"),
        "feasibility_proxy_no_status":[1.-v for v in adjusted],
        "candidate_net_utility":utils,
        "candidate_local_support_mass":[e["weighted_local_support_mass"] for e in evidence],
        "candidate_local_ambiguity":[e["candidate_local_ambiguity"] for e in evidence],
        "candidate_gp_original_risk":original,
        "credibility_decision":receipt,
        "risk_budget":receipt,
        "probability_guarantee":False,
    }
