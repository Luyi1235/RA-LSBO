"""PR86-C frozen risk-vs-objective acquisition rules (pure, no truth access)."""
import math
import random

from risk_learning.core import digest
from pr86c_protocol import METHODS

W_OBJECTIVE = 0.65
W_INFO = 0.35


def entropy(p):
    p = float(p)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Probability must be finite within [0,1]")
    if p in (0.0, 1.0):
        return 0.0
    return -p*math.log(p)-(1.0-p)*math.log1p(-p)


def percentile_ranks(values):
    """Average rank for ties; identical scores never get arbitrary rank bonuses."""
    n = len(values)
    if n == 0 or any(not math.isfinite(float(v)) for v in values):
        raise ValueError("Invalid candidate scores")
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: float(values[i]))
    out = [0.0]*n
    pos = 0
    while pos < n:
        end = pos+1
        while end < n and float(values[order[end]]) == float(values[order[pos]]):
            end += 1
        rank = (pos+end-1)/(2.0*(n-1))
        for k in range(pos, end):
            out[order[k]] = rank
        pos = end
    return out


def expected_improvement(mean, variance, best):
    """Minimization EI under Gaussian objective prediction."""
    m, v, b = float(mean), float(variance), float(best)
    if not all(math.isfinite(a) for a in (m, v, b)) or v < 0:
        raise ValueError("Invalid EI inputs")
    gap = b-m
    if v == 0:
        return max(gap, 0.0)
    std = math.sqrt(v)
    z = gap/std
    cdf = 0.5*math.erfc(-z/math.sqrt(2.0))
    pdf = math.exp(-0.5*z*z)/math.sqrt(2.0*math.pi)
    return max(0.0, gap*cdf+std*pdf)


def score_candidates(method, forecast, objective, incumbent):
    if method not in METHODS or method == "iid_random":
        raise ValueError("Score rule requires nonrandom registered method")
    ps = forecast["probabilities"]
    regions, resources, joints = (ps[k] for k in ("region", "resource", "joint"))
    n = len(joints)
    if n == 0 or any(len(a) != n for a in (regions, resources)):
        raise ValueError("Forecast must align with nonempty candidate pool")
    for a, b, joint in zip(regions, resources, joints):
        if not all(math.isfinite(float(p)) and 0 <= float(p) <= 1 for p in (a,b,joint)):
            raise ValueError("Nonfinite risk probability")
        if abs(float(joint)-float(a)*float(b)) > 1e-10:
            raise ValueError("Joint must be registered independent-head product")

    if method == "boundary":
        return [float(p)*(1.0-float(p)) for p in joints], "joint_boundary"
    if method == "marginal_entropy":
        return [entropy(a)+entropy(b) for a,b in zip(regions,resources)], "two_marginal_entropy"

    if incumbent is None or objective is None:
        # Cold-start/no trained objective: no synthetic objective or privileged baseline.
        eis = [1.0]*n
        cold = True
    else:
        if set(objective) != {"mean", "variance"} or any(len(objective[k]) != n for k in objective):
            raise ValueError("Objective posterior length mismatch")
        eis = [expected_improvement(m,v,incumbent)
               for m,v in zip(objective["mean"],objective["variance"])]
        cold = False
    cei = [float(p)*ei for p,ei in zip(joints,eis)]
    if method == "constrained_ei":
        return cei, "cold_feasibility" if cold else "constrained_expected_improvement"
    info = [ei*entropy(p) for ei,p in zip(eis,joints)]
    cei_rank, info_rank = percentile_ranks(cei), percentile_ranks(info)
    return [W_OBJECTIVE*a+W_INFO*b for a,b in zip(cei_rank,info_rank)], (
        "cold_feasibility_and_joint_entropy_rank" if cold else
        "objective_relevance_weighted_joint_uncertainty_rank")


def choose(method, candidates, forecast, objective, incumbent, seed, step):
    """Select one unevaluated candidate using only pre-outcome surrogate quantities."""
    if method not in METHODS or not candidates:
        raise ValueError("Unknown method or exhausted candidate pool")
    if method == "iid_random":
        rng = random.Random(seed+104729*step)
        scores = [rng.random() for _ in candidates]
        rule = "uniform_without_replacement"
    else:
        scores, rule = score_candidates(method, forecast, objective, incumbent)
    if len(scores) != len(candidates) or any(not math.isfinite(float(v)) for v in scores):
        raise ValueError("Invalid score vector")
    index = max(range(len(candidates)), key=lambda i:(float(scores[i]),digest(candidates[i])))
    receipt = dict(method=method, rule=rule, step=step, selected_index=index,
                   selected_candidate_sha256=digest(candidates[index]),
                   score_digest=digest(scores), scores=scores,
                   cold_fallback=(incumbent is None or objective is None)
                   if method in ("constrained_ei","objective_weighted_boundary") else None,
                   incumbent=incumbent)
    return index, receipt
