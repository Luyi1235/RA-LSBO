"""Constraint supervision ablation with a common Gaussian-process likelihood."""
from collections import Counter, defaultdict
from dataclasses import asdict
from statistics import mean, median

from ra_lsbo.budget import digest
from ra_lsbo.risk_core.observations import OutputObservation

MODES = ("bounds", "sign", "exact")


def constraint_feedback(value, mode, *, trusted=True):
    if mode not in MODES:
        raise ValueError(f"Unknown constraint feedback mode {mode}")
    if not trusted:
        return OutputObservation("missing", value.evidence_id, reason="untrusted_evaluation")
    if value.kind != "interval" or mode == "bounds":
        return value
    if mode == "sign":
        sign = value.constraint_satisfaction()
        if sign is True:
            return OutputObservation("interval", value.evidence_id, upper=0.)
        if sign is False:
            return OutputObservation("interval", value.evidence_id, lower=0., lower_inclusive=False)
    return OutputObservation("missing", value.evidence_id, reason="interval_withheld")


def feedback_columns(records, names, mode):
    columns = {name: [] for name in names}
    for record in records:
        packet = record["observation"]
        trusted = (packet["status"] not in {"unknown", "infrastructure_error"}
                   and not record["diagnostics"].get("numerical_failure"))
        constraints = dict(packet["constraints"])
        for name, column in columns.items():
            column.append(constraint_feedback(OutputObservation(**constraints[name]), mode, trusted=trusted))
    return columns


def feedback_receipt(columns, mode):
    return {"mode": mode, "identity": digest({name: [asdict(row) for row in column]
                                               for name, column in columns.items()}),
            "kinds": {name: dict(Counter(row.kind for row in column)) for name, column in columns.items()}}


def summarize_discovery(pairs):
    groups = defaultdict(list)
    for row in pairs:
        if not row["initial_usable"]:
            groups[row["arm"], row["baseline"]].append(row)
    summaries = []
    comparisons = dict.fromkeys((row["arm"], row["baseline"]) for row in pairs)
    for arm, baseline in comparisons:
        cohort = groups[arm, baseline]
        deltas = [row["capped_calls_delta"] for row in cohort]
        wins = Counter(row["outcome"] for row in cohort)
        leave_one_out = [(sum(deltas)-value)/(len(deltas)-1) for value in deltas] if len(deltas) > 1 else []
        summaries.append({"arm": arm, "baseline": baseline, "initial_unsolved": len(cohort),
                          "wins": wins["win"], "losses": wins["loss"], "ties": wins["tie"],
                          "arm_discovered": sum(row["arm_discovered"] for row in cohort),
                          "baseline_discovered": sum(row["baseline_discovered"] for row in cohort),
                          "mean_capped_calls_delta": mean(deltas) if deltas else None,
                          "median_capped_calls_delta": median(deltas) if deltas else None,
                          "leave_one_out_range": [min(leave_one_out), max(leave_one_out)] if leave_one_out else None})
    return summaries
