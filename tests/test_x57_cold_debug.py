from types import SimpleNamespace

import numpy as np
import pytest

from ra_lsbo.x57_cold_feedback import normalize, risk_observation
from ra_lsbo.x57_energy_stop import MISSION_SEGMENTS, STATUS, first_crossing, prefix_energy, stopped_metrics
from ra_lsbo.x57_energy_surrogate import energy_training_row
from ra_lsbo.x57_native_slot import ordered_branches
from run_x57_energy_stop_worker import EnergyStop, install


def segment(tag, initial, final, energy):
    return {"segment": tag, "converged": True, "electrical_summary": [{
        "bus": "bus", "duration_s": 3600., "minimum_terminal_power_W": energy,
        "maximum_terminal_power_W": energy, "terminal_energy_Wh": energy,
        "initial_charge_soc": initial, "final_charge_soc": final,
    }]}


def test_engineering_stop_unwinds_the_frozen_boundary_and_keeps_natural_completion_unknown():
    diagnostics = {"mission_solver_segments": []}
    started = []
    postprocessed = []

    jobs = {tag: SimpleNamespace(tag=tag, process=SimpleNamespace(post_process={
        "native_postprocess": lambda job: postprocessed.append(job.tag)}))
            for tag in MISSION_SEGMENTS}
    nexus = SimpleNamespace(missions=SimpleNamespace(base_mission=SimpleNamespace(segments=jobs)))
    module = SimpleNamespace(install_mission_solver=lambda _nexus, _diagnostics: None)
    install(module)
    module.install_mission_solver(nexus, diagnostics)

    def checked(report):
        started.append(report["segment"])
        diagnostics["mission_solver_segments"].append(report)
        job = jobs[report["segment"]]
        for step in job.process.post_process.values():
            step(job)

    checked(segment("DER", 1., .8, 17000.))
    checked(segment("ICA", .8, .6, 30000.))
    with pytest.raises(EnergyStop):
        checked(segment("climb_1", .6, .4, 1000.))
        checked(segment("climb_2", .4, .3, 500.))
    assert started == ["DER", "ICA", "climb_1"]
    assert postprocessed == started
    metrics = stopped_metrics(diagnostics["mission_solver_segments"], {"cruise_distance_nmi": 47.5})
    raw = {"status": STATUS, "solver_success": False, "objective": None, "constraints": {}, "metrics": metrics}
    feedback = normalize(raw, ["usable_energy"], {}, "rcaide")
    row = {"feedback": feedback, "worker_result": raw, "call_id": "test"}
    assert feedback["known_constraint_violations"] == ["usable_energy"]
    assert feedback["constraints"]["usable_energy"] is None
    assert feedback["completion"] is None and feedback["objective_kind"] == "missing"
    assert risk_observation(row, True) is None
    sign, bound = [energy_training_row(row, "test", 47.5, prefix_energy, interval)[0]
                   for interval in (False, True)]
    assert sign.lower == 0. and sign.lower_inclusive is False
    assert bound.lower == pytest.approx(48000. / 47000. - 1.)
    metrics["engineering_energy_stop"]["terminal_energy_lower_Wh"] = 49000.
    assert normalize(raw, ["usable_energy"], {}, "rcaide")["outcome"] == "infrastructure_error"


def test_final_segment_crossing_preserves_exact_mission_evaluation():
    jobs = [segment(tag, 1. - i / 20., 1. - (i + 1) / 20., 4000. if i < 10 else 10000.)
            for i, tag in enumerate(MISSION_SEGMENTS)]
    assert first_crossing(jobs) is None
    jobs[5]["converged"] = False
    assert prefix_energy(jobs)[0] == 20000.


def test_reserved_slot_replaces_only_the_last_elite_when_needed():
    points = [SimpleNamespace(discrete=label) for label in ("a", "a", "b", "c", "d")]
    space = SimpleNamespace(branch_key=lambda key: key)
    scores = np.array([10., 9.5, 9., 8., 7.])
    assert ordered_branches(points, scores, space, {"a", "b"}, 2, False)[:2] == [0, 2]
    assert ordered_branches(points, scores, space, {"a", "b"}, 2, True)[:2] == [0, 3]
    assert ordered_branches(points, scores, space, {"a"}, 2, True)[:2] == [0, 2]
