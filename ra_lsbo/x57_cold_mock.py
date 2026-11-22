"""Deterministic subprocess mock of the X-57 worker response protocol."""

import json
from pathlib import Path
import sys

from .x57_cold_feedback import CERTIFICATE_SOURCE, RISK_METRIC
from .x57_cold_worker import dump


def result(payload):
    step = payload["mock"]
    constraints = {k: -.1 for k in step["required"]}
    constraints.update(step["precheck"])
    scenario = step["profile"]
    evaluation, online = step["evaluation"], step["online"]
    initial_case = (evaluation + (evaluation - 1) // 4) % 4
    if scenario == "infrastructure" and online == 1:
        return dict(objective=None, status="worker_error:MockCrash", solver_success=False,
                    constraints={}, metrics={"message": "mock infrastructure failure"})
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
    dump(response, result(json.loads(request.read_text(encoding="utf-8"))))
