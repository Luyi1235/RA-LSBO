"""H2 prequential diagnostics with proper Gaussian MIXTURE scoring.

Unlike the H1 single-head evaluator, a two-component SOC mixture must
NOT be scored by inserting its moment-matched mean/variance into a
single-Gaussian NLL. Exact NLL is a log mixture of densities; interval
NLL is a log mixture of interval probability masses. Exact central
coverage uses the exact mixture CDF at the observed target value.
"""
from collections import Counter
import math
from statistics import median

import numpy as np
from scipy.special import logsumexp,ndtr

from ...partial_bo.records import DesignRecord
from .h2_risk_controller import proper_nll


def score_mixture(components,index,observation):
    weights=np.asarray(components["weights"],float)
    means=np.asarray(components["mean"],float)
    vars_=np.asarray(components["variance"],float)
    if (weights.shape!=(2,) or means.ndim!=2 or vars_.shape!=means.shape
        or means.shape[0]!=2 or not 0<=index<means.shape[1] or
        not np.isfinite(weights).all() or not np.isclose(weights.sum(),1.)
        or (weights<0).any() or not np.isfinite(means).all()
        or not np.isfinite(vars_).all() or (vars_<=0).any()):
        raise ValueError("Original two-view Gaussian mixture information differs")
    predictions=[
        {"mean":float(means[i,index]),"variance":float(vars_[i,index])}
        for i in range(2)]
    scores=[proper_nll(p,observation) for p in predictions]
    if observation.kind=="missing":
        return {"kind":"missing","nll":None,"brier":None,"covered95":None}
    if any(s is None for s in scores) or scores[0]["type"]!=scores[1]["type"]:
        raise ValueError("Mismatched component SOC observation likelihoods")
    nll=-float(logsumexp([math.log(w)-s["nll"] for w,s in zip(weights,scores) if w>0]))
    if not math.isfinite(nll):
        raise FloatingPointError("Mixture SOC predictive score nonfinite")
    p=float(weights@ndtr(-means[:,index]/np.sqrt(vars_[:,index])))
    label=observation.constraint_satisfaction()
    result={"kind":observation.kind,
            "score_kind":scores[0]["type"],"nll":nll,
            "brier":None if label is None else (p-float(label))**2,
            "p_feasible":p,"covered95":None}
    if observation.kind=="exact":
        y=observation.value
        cdf=float(weights@ndtr((y-means[:,index])/np.sqrt(vars_[:,index])))
        result["covered95"]=bool(.025<=cdf<=.975)
        result["standardized_component_error"]=[
            (y-pred["mean"])/math.sqrt(pred["variance"]) for pred in predictions]
    return result


def h2_prequential_soc(proposals,online):
    if len(proposals)!=len(online):
        raise ValueError("H2 selected forecast and actual evaluation lengths differ")
    counts=Counter()
    exact=[]
    interval=[]
    briers=[]
    covered=0
    for proposal,record in zip(proposals,online):
        if proposal["candidate"]!=record.candidate:
            raise ValueError("SOC forecast not paired to actual candidate")
        m=proposal["metadata"]
        if m.get("posterior_fit_skipped") or m.get("coverage_reason")=="model_failure":
            counts["unscored_coverage_or_model_failure"]+=1
            continue
        forecast=m["forecasts"]["constraints"]["soc"]
        if "components" not in forecast:
            raise ValueError("H2 fitted SOC forecast omitted component distribution")
        score=score_mixture(forecast["components"],m["selected_index"],
                            record.observation.constraint("soc"))
        if score["kind"]=="exact":
            exact.append(score["nll"])
            covered+=int(score["covered95"])
            counts["exact"]+=1
        elif score["kind"]=="interval":
            interval.append(score["nll"])
            counts["interval"]+=1
        else:
            counts["missing"]+=1
        if score["brier"] is not None:briers.append(score["brier"])
    return {
        "schema":"h2_exact_two_gaussian_SOC_prequential_v1",
        "exact":len(exact),"interval":len(interval),
        "skipped_no_forecast":counts["unscored_coverage_or_model_failure"],
        "missing":counts["missing"],
        "exact_density_nll_mean":sum(exact)/len(exact) if exact else None,
        "exact_density_nll_median":median(exact) if exact else None,
        "exact_95pct_coverage_numerator":covered,
        "interval_mass_nll_mean":sum(interval)/len(interval) if interval else None,
        "interval_mass_nll_median":median(interval) if interval else None,
        "resolved_feasibility_brier_mean":sum(briers)/len(briers) if briers else None,
        "resolved_feasibility_brier_n":len(briers),
        "uncertainty_semantics":"exact two-component Gaussian mixture, not moment matched",
        "inference_scope":"on-policy adaptive selected points, not independently calibrated",
    }
