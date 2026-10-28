"""Diagnostics for strictly out-of-prefix binary predictions."""

import math


def binary_metrics(probabilities, labels, bins=5):
    if len(probabilities) != len(labels) or type(bins) is not int or bins < 1:
        raise ValueError("Invalid metric lengths or bin count")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Probabilities must be finite and in [0,1]")
    if any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError("Labels must be binary integers")
    if not labels:
        return {"samples": 0, "brier": None, "log_loss": None, "ece": None, "reliability": []}
    n = len(labels)
    groups = [[] for _ in range(bins)]
    for p, y in zip(probabilities, labels):
        groups[min(int(p*bins), bins-1)].append((p, y))
    reliability, ece = [], 0.
    for index, group in enumerate(groups):
        probability = sum(p for p, y in group)/len(group) if group else None
        observed = sum(y for p, y in group)/len(group) if group else None
        if group:
            ece += len(group)/n * abs(probability-observed)
        reliability.append({"lower": index/bins, "upper": (index+1)/bins, "samples": len(group),
                            "mean_probability": probability, "observed_fraction": observed})
    clipped = [min(max(p, 1e-12), 1-1e-12) for p in probabilities]
    return {"samples": n, "positives": sum(labels), "negatives": n-sum(labels),
            "brier": sum((p-y)**2 for p, y in zip(probabilities, labels))/n,
            "log_loss": -sum(y*math.log(p)+(1-y)*math.log1p(-p) for p, y in zip(clipped, labels))/n,
            "ece": ece, "bins": bins, "reliability": reliability}
