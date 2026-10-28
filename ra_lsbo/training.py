"""Per-head training selections. No imputation, predictions or model fitting."""

from .records import HIDDEN_MODES, RiskEvaluation


def training_rows(observations):
    result = {
        "objective": [],
        "constraints": {},
        "hidden_success": [],
        "hidden_modes": {name: [] for name in HIDDEN_MODES},
        "final_feasibility": [],
    }
    for row in observations:
        value = row["risk"]
        RiskEvaluation(**value)
        base = {"evaluation": row["evaluation"], "candidate": row["candidate"]}
        for key in ("source_journal_sha256", "evaluator_contract_sha256"):
            if key in row:
                base[key] = row[key]
        if value["objective_observed"]:
            result["objective"].append(dict(base, target=value["objective"]))
        for name, observed in value["constraint_observed"].items():
            if name in value["analytical_constraints"]:
                continue
            table = result["constraints"].setdefault(name, [])
            if observed:
                table.append(dict(base, target=value["constraints"][name]))
        if value["hidden_success"] is not None:
            result["hidden_success"].append(dict(base, target=value["hidden_success"]))
        for name, label in value["hidden_labels"].items():
            if label is not None:
                result["hidden_modes"][name].append(dict(base, target=label))
        # Calibrate the expensive evaluator only conditional on cheap screening.
        if value["state"] != "precheck_rejected" and value["feasible_label"] is not None:
            result["final_feasibility"].append(dict(base, target=int(value["feasible_label"])))
    return result
