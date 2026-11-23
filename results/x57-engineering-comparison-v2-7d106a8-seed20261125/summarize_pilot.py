"""Verify the X-57 pilot ledger and summarize best energy and fallback events."""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def event_key(row):
    return row["journal"], row["call_id"]


def summarize(root):
    plan_path = root / "pilot_plan.json"
    plan = load(plan_path)
    report = load(root / "pilot_report.json")
    shared = load(root / "shared_initial.json")
    events = [json.loads(line) for line in (root / "calls.jsonl").read_bytes().splitlines()]
    starts = [r for r in events if r["event"] == "worker_started"]
    ends = [r for r in events if r["event"] in ("settled", "interrupted")]
    end_by_key = {event_key(r): r for r in ends}
    charges = Counter(r["arm"] for r in starts)
    intervals = sorted((datetime.fromisoformat(timestamp), change)
                       for r in starts for timestamp, change in (
                           (r["started_at_utc"], 1), (end_by_key[event_key(r)]["finished_at_utc"], -1)))
    active = peak = 0
    for _, change in intervals:
        active += change
        peak = max(peak, active)
    require(hashlib.sha256(plan_path.read_bytes()).hexdigest() == report["plan_sha256"],
            "Plan differs from the report")
    require(Counter(map(event_key, starts)) == Counter(map(event_key, ends))
            and len(ends) == len(end_by_key) == len(starts)
            and all(r["charged"] and end_by_key[event_key(r)]["charged"]
                    and end_by_key[event_key(r)]["candidate_id"] == r["candidate_id"] for r in starts),
            "Worker starts and chargeable settlements do not align")
    require(report["worker_starts"] == len(starts) <= plan["max_worker_calls"]
            and report["peak_workers"] == peak <= plan["worker_concurrency_limit"],
            "Report has a different total charge or worker concurrency")
    require(report["rcaide_calls"] == (len(starts) if plan["backend"] == "rcaide" else 0),
            "RCAIDE charge total disagrees with the backend")
    require(charges["shared_initial"] == len(shared) == report["shared_initial_starts"]
            and [r["candidate_id"] for r in shared] == [r["candidate_id"] for r in plan["tasks"]],
            "Shared results differ from the frozen roster")
    require(sum(r["feedback"]["usable"] for r in shared) == report["shared_initial_usable"],
            "Shared usable count differs from the report")

    arms = {}
    for name in plan["arms"]:
        arm = report["arms"][name]
        online = [r for r in ends if r["arm"] == name]
        history = arm["history"]
        require(len(online) == charges[name] == report["online_starts"][name]
                and len(history) == len(shared) + len(online),
                f"Online budget or history differs for {name}")
        for i, (row, saved) in enumerate(zip(shared + online, history), 1):
            require(row["candidate_id"] == saved["candidate_id"] and row["feedback"] == saved["feedback"]
                    and row["evaluation"] == saved["evaluation"] == i
                    and saved["charged"] == (i > len(shared)),
                    f"Replay alignment failed for {name} evaluation {i}")
        for previous, following in zip(online, online[1:]):
            following_start = next(r for r in starts if event_key(r) == event_key(following))
            require(previous["finished_at_utc"] <= following_start["started_at_utc"],
                    f"Online worker calls overlap within {name}")

        best = first_usable = first_online = None
        trajectory = []
        for row in shared + online:
            feedback = row["feedback"]
            if feedback["usable"]:
                energy = feedback["objective_Wh"]
                best = energy if best is None else min(best, energy)
                if first_usable is None:
                    first_usable = row["evaluation"]
                if first_online is None and row["online"]:
                    first_online = row["online"]
            trajectory.append({"evaluation": row["evaluation"], "online": row["online"],
                               "candidate_id": row["candidate_id"], "outcome": feedback["outcome"],
                               "objective_Wh": feedback["objective_Wh"], "best_usable_Wh": best})

        fallback_events, native_paths = [], []
        fits, fitted_partial = 0, set()
        for row in online:
            proposal = row["proposal"]
            fit = proposal.get("engineering_energy_fit")
            if fit is not None:
                fits += 1
                fitted_partial.update(obs["evidence_id"] for obs in fit["partial_observations"])
            reasons = {k: v for k, v in proposal.items() if k.endswith("_fallback")}
            if reasons:
                fallback_events.append({"evaluation": row["evaluation"], "online": row["online"], **reasons})
            if proposal.get("risk_mode") == "native_recovery_or_exploration":
                native_paths.append({"evaluation": row["evaluation"], "online": row["online"],
                                     "source": proposal.get("source")})
        stop_events = [{k: r[k] for k in ("event", "call_id", "source_id") if k in r} for r in events
                       if r["journal"] == name and r["event"].startswith("native_")]
        require(best == arm["best_usable_Wh"] and first_usable == arm["first_usable_call"]
                and first_online == arm["first_usable_online"] and fits == arm["energy_acquisition_fits"]
                and sorted(fitted_partial) == arm["fitted_partial_ids"],
                f"Published endpoints or fit counts differ for {name}")
        arms[name] = {"first_usable_call": first_usable, "first_usable_online": first_online,
                      "best_usable_Wh": best, "worker_starts_online": charges[name],
                      "usable_online": sum(r["feedback"]["usable"] for r in online),
                      "available_partial_feedback": arm["available_partial_feedback"],
                      "partial_feedback_fitted": len(fitted_partial), "fitted_partial_ids": sorted(fitted_partial),
                      "engineering_energy_fits": fits, "fallbacks": fallback_events,
                      "native_recovery_or_exploration": native_paths, "native_stop_events": stop_events,
                      "trajectory": trajectory}
    sign, interval = (arms[f"ra_lsbo_energy_{mode}"] for mode in ("sign", "interval"))
    return {"status": report["status"], "code_sha": report["code_sha"], "seed": plan["seed"],
            "distance_nmi": plan["distance_nmi"], "rcaide_calls": report["rcaide_calls"],
            "worker_starts": len(starts), "peak_workers": peak, "wall_seconds": report["wall_seconds"],
            "shared_initial_usable": report["shared_initial_usable"], "shared_initial_starts": len(shared),
            "shared_initial_usable_positions": [r["evaluation"] for r in shared if r["feedback"]["usable"]],
            "worker_outcomes": dict(Counter(r["feedback"]["outcome"] for r in ends)),
            "matched_ra_candidate_steps": sum(a["candidate_id"] == b["candidate_id"]
                                              for a, b in zip(sign["trajectory"][len(shared):],
                                                              interval["trajectory"][len(shared):])),
            "arms": arms}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.archive), indent=2, allow_nan=False))
