"""Read-only, per-trial prequential SOC diagnostics from saved BO forecasts.

The posterior forecast at iteration t was sealed BEFORE the true evaluation
of that design. This function never fits GP/VAE or calls the evaluator.
Exact density NLL and interval mass NLL must never be pooled together.
"""
from math import erf, log, pi, sqrt
from statistics import median
import math

from scipy.special import log_ndtr
from ...partial_bo.models import log_interval_probability


def soc_prequential(proposals, online):
    if len(proposals)!=len(online):
        raise ValueError("Saved forecast and observation lengths disagree")
    exact_nll=[]
    interval_nll=[]
    absolute_z=[]
    coverage_90=0
    coverage_95=0
    counts={"forecasted":0,"skipped_no_forecast":0,
            "exact":0,"interval":0,"missing":0,
            "zero_or_invalid_variance":0,"model_failure_fallback":0}
    for prop,record in zip(proposals,online):
        if prop["candidate"] != record.candidate:
            raise ValueError("Forecast candidate is not the settled design")
        meta=prop["metadata"]
        counts["model_failure_fallback"]+=int(meta.get("coverage_reason")=="model_failure")
        pred=meta.get("forecasts")
        if pred is None:
            counts["skipped_no_forecast"]+=1
            continue
        constraints=pred.get("constraints",{})
        if "soc" not in constraints:
            raise ValueError("SOC absent from a registered posterior prediction")
        index=meta["selected_index"]
        p=constraints["soc"]
        try:
            mean=float(p["mean"][index])
            variance=float(p["variance"][index])
        except (IndexError,KeyError,TypeError) as error:
            raise ValueError("Saved selected-index SOC forecast invalid") from error
        if not math.isfinite(mean) or not math.isfinite(variance) or variance<=0:
            counts["zero_or_invalid_variance"]+=1
            continue
        counts["forecasted"]+=1
        sd=sqrt(variance)
        observation=record.observation.constraint("soc")
        if observation.kind=="exact":
            z=(observation.value-mean)/sd
            nll=.5*log(2*pi*variance)+.5*z*z
            if not math.isfinite(nll):
                raise ValueError("Nonfinite SOC exact predictive density")
            exact_nll.append(nll)
            absolute_z.append(abs(z))
            coverage_90+=int(abs(z)<=1.6448536269514722)
            coverage_95+=int(abs(z)<=1.959963984540054)
            counts["exact"]+=1
        elif observation.kind=="interval":
            lower=-math.inf if observation.lower is None else (observation.lower-mean)/sd
            upper=math.inf if observation.upper is None else (observation.upper-mean)/sd
            if lower >= upper:raise ValueError("Invalid standardized SOC observation interval")
            logp= (float(log_ndtr(-lower)) if upper==math.inf else
                   float(log_ndtr(upper)) if lower==-math.inf else
                   log_interval_probability(lower,upper))
            if not math.isfinite(logp) or logp>1e-10:
                raise ValueError("Nonfinite or impossible interval predictive mass")
            interval_nll.append(-logp)
            counts["interval"]+=1
        else:
            counts["missing"]+=1
    return {
        **counts,
        "exact_density_nll_mean":sum(exact_nll)/len(exact_nll) if exact_nll else None,
        "exact_density_nll_median":median(exact_nll) if exact_nll else None,
        "exact_abs_standardized_error_median":median(absolute_z) if absolute_z else None,
        "exact_abs_standardized_error_max":max(absolute_z) if absolute_z else None,
        "exact_90pct_coverage_numerator":coverage_90,
        "exact_95pct_coverage_numerator":coverage_95,
        "interval_mass_nll_mean":sum(interval_nll)/len(interval_nll) if interval_nll else None,
        "interval_mass_nll_median":median(interval_nll) if interval_nll else None,
        "epistemic_limit":"Adaptive on-policy selected points; not iid, not physical safety calibration",
    }
