"""PR86-B acquisition rules and anchor-only calibration; dependency-free."""
import math
import random
from risk_learning.core import features, digest

METHODS = ("iid_random", "maximin", "latent_uncertainty", "boundary", "calibration_aware_voi")
# Five-point Gauss-Hermite rule transformed to standard-normal expectation.
_NODES = (-2.8569700138728056, -1.355626179974266, 0.0, 1.355626179974266, 2.8569700138728056)
_WEIGHTS = (0.0112574113277207, 0.2220759220056126, 0.5333333333333333,
            0.2220759220056126, 0.0112574113277207)

def _cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _entropy(p):
    p = min(max(float(p), 1e-15), 1.0-1e-15)
    return -p*math.log(p) - (1.0-p)*math.log(1.0-p)

def bald_binary_information(mean, variance, observation_noise):
    """I(latent margin; observed sign) under a probit observation approximation."""
    m, v, noise = float(mean), float(variance), float(observation_noise)
    if not all(math.isfinite(x) for x in (m, v, noise)) or v < 0 or noise < 0:
        raise ValueError("Invalid Gaussian information inputs")
    total = v + noise
    if total == 0:
        return 0.0
    p = _cdf(-m / math.sqrt(total))
    predictive_entropy = _entropy(p)
    if v == 0:
        return 0.0
    if noise == 0:
        return predictive_entropy
    conditional_entropy = 0.0
    root = math.sqrt(v)
    for z, w in zip(_NODES, _WEIGHTS):
        conditional_entropy += w * _entropy(_cdf(-(m + root*z)/math.sqrt(noise)))
    return max(0.0, predictive_entropy - conditional_entropy)

def _distance(a, b):
    x, y = features(a), features(b)
    return math.sqrt(sum((u-v)**2 for u, v in zip(x, y)))

def _rank_percentiles(values):
    order = sorted(range(len(values)), key=lambda i: (float(values[i]), i))
    out = [0.0] * len(values)
    denom = max(1, len(values)-1)
    for rank, i in enumerate(order):
        out[i] = rank / denom
    return out

def anchor_prevalence(anchor_observations):
    if not anchor_observations:
        raise ValueError("At least one random anchor is required")
    n = len(anchor_observations)
    heads = {}
    for head in ("region", "resource"):
        k = sum(1 for row in anchor_observations
                if row[head]["kind"] == "exact" and row[head]["value"] <= 0)
        heads[head] = (k+1.0)/(n+2.0)
    joint = sum(1 for row in anchor_observations
                if all(row[h]["kind"] == "exact" and row[h]["value"] <= 0
                       for h in ("region", "resource")))
    return {"region": heads["region"], "resource": heads["resource"],
            "joint": (joint+1.0)/(n+2.0), "n": n}

def anchor_intercept_calibrate(probabilities, target_prevalence):
    """Logit intercept shift; only anchor prevalence and unlabeled query x are used."""
    if not probabilities or not 0 < target_prevalence < 1:
        raise ValueError("Nonempty predictions and interior anchor prevalence required")
    ps = [min(max(float(p), 1e-12), 1.0-1e-12) for p in probabilities]
    if any(not math.isfinite(p) for p in ps):
        raise ValueError("Nonfinite probability")
    logits = [math.log(p/(1.0-p)) for p in ps]
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo+hi)/2.0
        avg = sum(1.0/(1.0+math.exp(-max(-60.0,min(60.0,z+mid)))) for z in logits)/len(logits)
        if avg < target_prevalence:
            lo = mid
        else:
            hi = mid
    shift = (lo+hi)/2.0
    return [1.0/(1.0+math.exp(-max(-60.0,min(60.0,z+shift)))) for z in logits], shift

def calibrate_prediction(prediction, anchor_rates):
    raw = prediction["probabilities"]
    calibrated = {}
    shifts = {}
    for head in ("region", "resource"):
        calibrated[head], shifts[head] = anchor_intercept_calibrate(raw[head], anchor_rates[head])
    calibrated["joint"] = [a*b for a,b in zip(calibrated["region"], calibrated["resource"])]
    return calibrated, shifts

def choose(method, candidates, evaluated, forecast, seed, step, anchor_fraction=0.25, calibration_candidates=None):
    """Return (index, receipt); forecast arrays align with candidates."""
    if method not in METHODS or not candidates:
        raise ValueError("Unknown method or empty candidate pool")
    n = len(candidates)
    if method == "iid_random":
        rng = random.Random(seed + 104729*step)
        scores = [rng.random() for _ in candidates]
        rule = "uniform_without_replacement"
    elif method == "maximin":
        if not evaluated:
            raise ValueError("Maximin requires evaluated anchors")
        scores = [min(_distance(c, old) for old in evaluated) for c in candidates]
        rule = "maximum_minimum_hierarchical_feature_distance"
    else:
        gaussian = forecast.get("gaussian", {})
        probs = forecast.get("probabilities", {})
        if set(gaussian) != {"region", "resource"}:
            raise ValueError("Native Gaussian head forecasts required")
        if method == "latent_uncertainty":
            ranks = [_rank_percentiles(gaussian[h]["variance"]) for h in ("region", "resource")]
            scores = [(ranks[0][i]+ranks[1][i])/2.0 for i in range(n)]
            rule = "mean_percentile_rank_of_latent_variance"
        elif method == "boundary":
            joint = probs["joint"]
            scores = [float(p)*(1.0-float(p)) for p in joint]
            rule = "joint_predictive_boundary_uncertainty_p_times_one_minus_p"
        else:
            period = round(1.0/anchor_fraction)
            if (step+1) % period == 0:
                anchor_pool = calibration_candidates if calibration_candidates is not None else candidates
                if not anchor_pool:
                    raise ValueError("Independent random calibration-anchor pool exhausted")
                rng = random.Random(seed + 104729*step)
                scores = [rng.random() for _ in anchor_pool]
                index = max(range(len(anchor_pool)), key=lambda i: (float(scores[i]), digest(anchor_pool[i])))
                return index, {"method": method, "rule": "reserved_random_calibration_anchor",
                    "step": step, "selected_index": index,
                    "selected_candidate_sha256": digest(anchor_pool[index]),
                    "scores": [float(x) for x in scores], "selected_from": "calibration_anchor_pool",
                    "anchor_fraction": anchor_fraction}
            else:
                scores = []
                for i in range(n):
                    value = 0.0
                    for head in ("region", "resource"):
                        row = gaussian[head]
                        value += bald_binary_information(row["mean"][i], row["variance"][i],
                                                         row["observation_noise_variance"])
                    scores.append(value)
                rule = "sum_of_marginal_bald_mutual_information"
    if len(scores) != n or any(not math.isfinite(float(x)) for x in scores):
        raise ValueError("Invalid acquisition score vector")
    index = max(range(n), key=lambda i: (float(scores[i]), digest(candidates[i])))
    return index, {"method": method, "rule": rule, "step": step, "selected_index": index,
                   "selected_candidate_sha256": digest(candidates[index]),
                   "scores": [float(x) for x in scores], "selected_from": "acquisition_pool",
                   "anchor_fraction": anchor_fraction if method == "calibration_aware_voi" else None}

def metrics(labels, probabilities):
    if len(labels) != len(probabilities):
        raise ValueError("Label/probability length mismatch")
    if not labels:
        return {"n": 0, "positives": 0, "brier": None, "log_loss": None,
                "auroc": None, "auprc": None, "ece": None, "false_feasible_rate": None, "bins": []}
    if any(type(y) is not bool for y in labels):
        raise ValueError("Boolean labels required")
    if any(not math.isfinite(float(p)) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid probabilities")
    n, pos = len(labels), sum(labels)
    eps = 1e-15
    brier = sum((float(p)-float(y))**2 for y,p in zip(labels,probabilities))/n
    loss = -sum(float(y)*math.log(max(float(p),eps)) + (1-float(y))*math.log(max(1-float(p),eps))
                for y,p in zip(labels,probabilities))/n
    positives = [p for y,p in zip(labels,probabilities) if y]
    negatives = [p for y,p in zip(labels,probabilities) if not y]
    auc = (sum((a>b)+0.5*(a==b) for a in positives for b in negatives)/(len(positives)*len(negatives))
           if positives and negatives else None)
    prc = None
    if pos:
        order = sorted(range(n), key=lambda i: (-probabilities[i], i))
        tp, area = 0, 0.0
        for rank, i in enumerate(order, 1):
            if labels[i]:
                tp += 1
                area += (tp/rank)/pos
        prc = area
    bins, ece = [], 0.0
    for j in range(10):
        ids = [i for i,p in enumerate(probabilities) if min(int(float(p)*10),9)==j]
        mp = sum(float(probabilities[i]) for i in ids)/len(ids) if ids else None
        rate = sum(labels[i] for i in ids)/len(ids) if ids else None
        if ids:
            ece += len(ids)/n*abs(mp-rate)
        bins.append({"bin": j, "n": len(ids), "mean_probability": mp, "observed_rate": rate})
    false_feasible = (sum(1 for y,p in zip(labels,probabilities) if p >= .5 and not y)/(n-pos)
                      if n > pos else None)
    return {"n": n, "positives": pos, "brier": brier, "log_loss": loss, "auroc": auc,
            "auprc": prc, "ece": ece, "false_feasible_rate": false_feasible, "bins": bins}

def brier_skill_score(model_brier, baseline_brier):
    if model_brier is None or baseline_brier is None or baseline_brier <= 0:
        return None
    return 1.0 - model_brier/baseline_brier
