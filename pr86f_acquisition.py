"""Three frozen selectors; the first two delegate to the unmodified reference."""
import math
from pr86f_protocol import METHODS


def scores(method, forecast, objective, incumbent):
    from pr86c_acquisition import score_candidates, expected_improvement, entropy
    if method not in METHODS:
        raise ValueError("Unregistered method")
    # Calling the frozen baseline also validates aligned probabilities/product.
    if method != "direct_weighted_boundary":
        return score_candidates(method, forecast, objective, incumbent)
    cei, baseline_rule = score_candidates("constrained_ei", forecast, objective, incumbent)
    p = forecast["probabilities"]["joint"]
    cold = incumbent is None or objective is None
    ei = [1.0]*len(p) if cold else [expected_improvement(m,v,incumbent)
         for m,v in zip(objective["mean"],objective["variance"])]
    values = [0.65*c + 0.35*(e*entropy(q)) for c,e,q in zip(cei,ei,p)]
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Invalid direct acquisition vector")
    return values, "cold_direct_feasibility_entropy" if cold else "direct_ei_joint_entropy"


def choose(method, candidates, forecast, objective, incumbent):
    from risk_learning.core import digest
    values, rule = scores(method, forecast, objective, incumbent)
    if len(values) != len(candidates) or not candidates:
        raise ValueError("Invalid candidate score alignment")
    index = max(range(len(values)), key=lambda i:(values[i],digest(candidates[i])))
    return index, dict(method=method, rule=rule, scores=values,
        score_digest=digest(values), selected_index=index,
        selected_id=digest(candidates[index]), incumbent=incumbent,
        cold_fallback=objective is None or incumbent is None,
        all_scores_zero=all(v == 0 for v in values))
