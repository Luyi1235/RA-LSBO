"""Paired end-to-end endpoints on a matched aircraft evaluation budget."""
from collections import Counter
from statistics import mean, median

from aircraft_risk_protocol import SIGN_LSBO_PAIRS


def budget_trajectory(records, outcomes, checkpoints):
    rows, best = [], None
    for n, (record, outcome) in enumerate(zip(records, outcomes, strict=True), 1):
        if outcome == "usable":
            energy = record["observation"]["objective"]["value"]
            best = energy if best is None else min(best, energy)
        if n in checkpoints:
            rows.append({"n": n, "best_feasible_Wh": best})
    return rows


def paired_endpoints(trajectories):
    by_seed = {}
    for row in trajectories:
        by_seed.setdefault(row["seed"], {})[row["arm"]] = row
    pairs = []
    for seed, arms in sorted(by_seed.items()):
        for arm, baseline in SIGN_LSBO_PAIRS:
            left, right = arms[arm], arms[baseline]
            if left["initial_usable"] != right["initial_usable"]:
                raise ValueError("Paired search arms must share the initial usable count")
            for a, b in zip(left["budget_trajectory"], right["budget_trajectory"], strict=True):
                if a["n"] != b["n"]:
                    raise ValueError("Paired search budgets differ")
                x, y = a["best_feasible_Wh"], b["best_feasible_Wh"]
                # Any observed usable design outranks nondiscovery.
                outcome = "tie" if x == y else "win" if y is None or (x is not None and x < y) else "loss"
                pairs.append({"seed": seed, "arm": arm, "baseline": baseline,
                              "initial_usable": left["initial_usable"], "n": a["n"],
                              "arm_best_Wh": x, "baseline_best_Wh": y, "outcome": outcome,
                              "delta_Wh": x-y if x is not None and y is not None else None})
    return pairs


def terminal_summary(pairs, horizon):
    summaries = []
    for arm, baseline in SIGN_LSBO_PAIRS:
        rows = [r for r in pairs if r["arm"] == arm and r["baseline"] == baseline and r["n"] == horizon]
        for cohort, group in (("all", rows), ("initial_unsolved", [r for r in rows if not r["initial_usable"]])):
            counts = Counter(r["outcome"] for r in group)
            deltas = [r["delta_Wh"] for r in group if r["delta_Wh"] is not None]
            net = counts["win"]-counts["loss"]
            leave_one_out = [(net-{"win": 1, "loss": -1, "tie": 0}[r["outcome"]])/(len(group)-1)
                             for r in group] if len(group) > 1 else []
            summaries.append({"arm": arm, "baseline": baseline, "cohort": cohort,
                              "seeds": len(group), "wins": counts["win"], "losses": counts["loss"],
                              "ties": counts["tie"], "net_win_fraction": net/len(group) if group else None,
                              "leave_one_out_net_win": [min(leave_one_out), max(leave_one_out)] if leave_one_out else None,
                              "arm_discovered": sum(r["arm_best_Wh"] is not None for r in group),
                              "baseline_discovered": sum(r["baseline_best_Wh"] is not None for r in group),
                              "both_discovered": len(deltas), "mean_delta_Wh": mean(deltas) if deltas else None,
                              "median_delta_Wh": median(deltas) if deltas else None})
    return summaries
