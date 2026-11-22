"""Prepare and evaluate the fixed 20-call qualification roster."""

from collections import Counter
import json
import math
import statistics
import time

import numpy as np

from .budget import digest
from .x57_cold_campaign import identity
from .x57_cold_sampling import LAYOUT, LAYOUTS, active_names, violations
from .x57_cold_worker import dump

N_QUALIFICATION = 20


def qualification_plan(problem, output, worker_identity):
    space = problem.space
    rng = np.random.default_rng(20261122)
    points = []
    for layout in LAYOUTS:
        default = space.default()
        reference_discrete = {**default.discrete, LAYOUT: layout}
        nominal = space.from_unit(space.unit(default), reference_discrete)
        pool = {}
        for index in range(2048):
            point = (space.sample(rng, reference_discrete) if index % 2 == 0
                     else space.sample(rng))
            if point.discrete[LAYOUT] != layout:
                continue
            checks = problem.problem.precheck(point)
            if not violations(checks):
                pool.setdefault(space.key(point), (point, checks))

        def take(key, tag, reverse=False):
            available = [(identifier, p, checks) for identifier, (p, checks) in pool.items()]
            identifier, point, checks = (max if reverse else min)(available, key=lambda item: key(*item[1:]))
            del pool[identifier]
            return record(point, checks, tag)

        def record(point, checks, tag):
            return {"role": tag, "layout": layout, "candidate": point.to_dict(),
                    "candidate_id": space.key(point), "precheck": checks,
                    "active_variables": active_names(space, point)}

        checks = problem.problem.precheck(nominal)
        if violations(checks):
            ref = take(lambda p, _: (space.branch_key(p.discrete) != space.branch_key(nominal.discrete),
                                     np.linalg.norm((space.unit(p) - space.unit(nominal)) * space.mask(p.discrete))),
                       "nearest_screened_reference")
        else:
            ref = record(nominal, checks, "configured_reference")
            pool.pop(ref["candidate_id"], None)
        design = [ref]
        boundaries = ("flutter", "span_min", "span_max" if layout == LAYOUTS[0] else "prop_disk_clearance")
        for field in boundaries:
            design.append(take(lambda _, checks: abs(checks[field]), "boundary:" + field))
        design.append(take(lambda p, _: min(abs(p.continuous["wing_area"] - v)
                                          for v in next(var for var in space.continuous
                                                        if var.name == "wing_area").bounds),
                           "boundary:wing_area"))
        for _ in range(3):
            anchors = [space.unit(problem.candidate_class(**row["candidate"])) for row in design]
            design.append(take(lambda p, _: min(np.linalg.norm((space.unit(p) - a) * space.mask(p.discrete))
                                               for a in anchors),
                               "interior_diversity", reverse=True))
        design.extend([{**design[0], "role": "repeat_reference", "repeat_of": design[0]["candidate_id"]},
                       {**design[1], "role": "repeat_boundary", "repeat_of": design[1]["candidate_id"]}])
        points.append(design)
    roster = [row for pair in zip(*points) for row in pair]
    plan = {"schema": "x57_cold_qualification_v1", "worker": worker_identity,
            "identity": identity(problem), "points": roster, "roster_sha256": digest(roster),
            "max_rcaide_calls": N_QUALIFICATION, "operator_approved": False}
    dump(output / "qualification_plan.json", plan)
    return plan


def run_qualification(problem, dispatcher, output, plan):
    points = plan["points"]
    if (plan["schema"] != "x57_cold_qualification_v1" or dispatcher.backend != "rcaide"
            or digest(plan["identity"]) != digest(identity(problem))
            or plan["roster_sha256"] != digest(points)
            or len(points) != N_QUALIFICATION or plan["max_rcaide_calls"] != N_QUALIFICATION):
        raise ValueError("Qualification plan identity or budget changed")
    candidates = [problem.candidate_class(**problem.canonical(row["candidate"])) for row in points]
    for point, row in zip(candidates, points):
        if (problem.space.key(point) != row["candidate_id"] or point.discrete[LAYOUT] != row["layout"]
                or problem.problem.precheck(point) != row["precheck"] or violations(row["precheck"])):
            raise ValueError("Qualification candidate changed since preparation")
    rows = []
    stop = None
    started = time.perf_counter()
    for i, (point, row) in enumerate(zip(candidates, points), 1):
        result = dispatcher.evaluate(point, seed=None, arm="qualification", evaluation=i, online=0,
                                     phase="qualification", proposal={"source": row["role"]})
        rows.append(result)
        if result["feedback"]["outcome"] == "infrastructure_error":
            stop = {"call_id": result["call_id"], "evaluation": i, "charged": result["charged"],
                    "worker_status": result["worker_result"].get("status")}
            break
    layouts = {}
    for layout in LAYOUTS:
        selected = [r for r in rows if r["candidate"]["discrete"][LAYOUT] == layout]
        repeats = []
        for row in selected:
            if not row["proposal"]["source"].startswith("repeat"):
                continue
            original = next(prior for prior in selected if prior["candidate_id"] == row["candidate_id"])
            before, after = original["feedback"], row["feedback"]
            repeats.append({"first_call": original["call_id"], "repeat_call": row["call_id"],
                            "both_usable": before["usable"] and after["usable"],
                            "same_outcome": before["outcome"] == after["outcome"],
                            "energy_delta_Wh": (after["objective_Wh"] - before["objective_Wh"]
                                                if before["completion"] and after["completion"] else None)})
        seconds = sorted(r["elapsed_seconds"] for r in selected if r["charged"])
        layouts[layout] = {
            "outcomes": dict(Counter(r["feedback"]["outcome"] for r in selected)),
            "usable": sum(r["feedback"]["usable"] for r in selected),
            "verified_intervals": sum(r["feedback"]["risk_interval_W"] is not None for r in selected),
            "elapsed_seconds": seconds, "median_seconds": statistics.median(seconds) if seconds else None,
            "p90_seconds": seconds[math.ceil(.9 * len(seconds)) - 1] if seconds else None,
            "repeat_comparisons": repeats,
        }
    report = {"status": "infrastructure_stop" if stop else "complete", "stop": stop,
              "planned_calls": N_QUALIFICATION, "rcaide_calls": dispatcher.count, "settled_results": len(rows),
              "wall_seconds": time.perf_counter() - started, "layouts": layouts}
    dump(output / "qualification_report.json", report)
    return report
