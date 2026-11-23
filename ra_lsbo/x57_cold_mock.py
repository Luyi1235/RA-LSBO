"""Deterministic subprocess mock of the X-57 worker response protocol."""

import json
from pathlib import Path
import sys
import time

from .x57_cold_feedback import CERTIFICATE_SOURCE, RISK_METRIC
from .x57_cold_worker import dump
from .x57_energy_stop import STATUS, stopped_metrics


def result(payload):
    step = payload["mock"]
    constraints = {k: -.1 for k in step["required"]}
    constraints.update(step["precheck"])
    scenario = step["profile"]
    evaluation, online = step["evaluation"], step["online"]
    if scenario == "cold-start-debug":
        distance = payload["config"]["mission"]["cruise_distance_nmi"]
        if not online and evaluation % 4 == 1:
            segment = {"segment": "DER", "converged": True, "electrical_summary": [
                {"bus": "bus", "terminal_energy_Wh": 48000., "duration_s": 3600.,
                 "minimum_terminal_power_W": 48000., "maximum_terminal_power_W": 48000.,
                 "initial_charge_soc": 1., "final_charge_soc": .05}]}
            return dict(objective=None, status=STATUS, solver_success=False, constraints={},
                        metrics=stopped_metrics([segment], payload["config"]["mission"]))
        if not online and evaluation % 4 == 2 and distance == 47.5:
            return dict(objective=None, status="mission_not_converged", solver_success=False,
                        constraints={}, metrics={"mission_profile": payload["config"]["mission"]})
        if not online and evaluation % 4 == 3:
            constraints["usable_energy"] = .005
        else:
            constraints["usable_energy"] = -.05
        return dict(objective=46350. - 70. * online, solver_success=True, constraints=constraints,
                    status="constraint_violation" if constraints["usable_energy"] > 0 else "ok", metrics={})
    if scenario == "distance-diagnostic":
        distance = payload["config"]["mission"]["cruise_distance_nmi"]
        if distance == 50:
            constraints["soc"] = .01
        return dict(objective=30000. + 150. * distance, status="ok" if distance < 50 else "constraint_violation",
                    solver_success=True, constraints=constraints, metrics={})
    initial_case = (evaluation + (evaluation - 1) // 4) % 4
    if scenario == "infrastructure" and online == 1:
        return dict(objective=None, status="worker_error:MockCrash", solver_success=False,
                    constraints={}, metrics={"message": "mock infrastructure failure"})
    if scenario == "engineering-pilot" and not online and evaluation == 4:
        segment = {"segment": "DER", "converged": True, "electrical_summary": [
            {"bus": "bus", "terminal_energy_Wh": 48000., "duration_s": 3600.,
             "minimum_terminal_power_W": 48000., "maximum_terminal_power_W": 48000.,
             "initial_charge_soc": 1., "final_charge_soc": .05}]}
        return dict(objective=None, status="mission_not_converged", solver_success=False, constraints={},
                    metrics={"mda_evaluated": True, "usable_terminal_energy_Wh": 47000.,
                             "mission_profile": payload["config"]["mission"], "mission_solver_segments": [segment]})
    if scenario == "no-usable" or (not online and initial_case == 0):
        return dict(objective=None, status="mission_not_converged", solver_success=False,
                    constraints={"solver": 1.}, metrics={"minimum_trial_terminal_power_W": -500})
    if not online and initial_case in (1, 2):
        valid = initial_case == 1
        return dict(objective=None, status="physical_early_stop", solver_success=False, constraints={},
                    metrics={"physical_stop_verified": True, "partial_energy_Wh": 17000.,
                             "completion_certificate": {
                                 "metric": RISK_METRIC, "source": CERTIFICATE_SOURCE,
                                 "validated_original_model": True,
                                 "lower_W": 150. if valid else -150., "upper_W": 900.,
                             }})
    energy = 40500. - 190. * online + 65. * payload["candidate"]["continuous"]["wing_area"]
    if not online:
        constraints["soc"] = .01
    return dict(objective=energy, status="ok" if online else "constraint_violation",
                solver_success=True, constraints=constraints, metrics={})


if __name__ == "__main__":
    request, response = map(Path, sys.argv[1:3])
    payload = json.loads(request.read_text(encoding="utf-8"))
    if payload["mock"]["profile"] == "distance-diagnostic":
        (response.parent / "solver_work").mkdir()
        time.sleep(2)
    dump(response, result(payload))
