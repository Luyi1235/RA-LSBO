import pytest

from run_x57_distance_diagnostic import elapsed_concurrency, observation, prefix_energy, run_wave


def segment(name, initial, final, energy=100.):
    return {"segment": name, "converged": True, "electrical_summary": [{
        "bus": "bus", "duration_s": 360., "minimum_terminal_power_W": 800.,
        "maximum_terminal_power_W": 1200., "terminal_energy_Wh": energy,
        "initial_charge_soc": initial, "final_charge_soc": final,
    }]}


def test_unconverged_trial_cannot_enter_terminal_energy_bound():
    failed = segment("climb_1", .8, .7)
    failed["converged"] = False
    segments = [segment("DER", 1., .9), segment("ICA", .9, .8), failed, segment("climb_2", .7, .6)]
    feedback = {"outcome": "numerical_nonconvergence", "completion": None, "usable": False,
                "constraints": {"flutter": -.1, "soc": None}, "objective_Wh": None,
                "objective_kind": "missing", "risk_interval_W": None}
    result = observation({"feedback": feedback, "elapsed_seconds": 12., "worker_result": {
        "status": "evaluation_error:DomainSolveError", "metrics": {
            "mission_solver_segments": segments, "final_soc": .2, "terminal_energy_Wh": 1000.,
        },
    }})
    assert result["failed_segment"] == "climb_1"
    assert result["terminal_energy_lower_Wh"] == 200.
    assert result["prefix_end_soc"] == .8
    assert result["completion"] is None and result["objective_kind"] == "missing"
    assert result["objective_Wh"] is result["terminal_energy_Wh"] is result["final_soc"] is None


@pytest.mark.parametrize("invalid", [segment("ICA", .91, .8), segment("ICA", .9, .8, 150.),
                                     segment("climb_1", .9, .8)])
def test_prefix_stops_on_discontinuous_or_invalid_source(invalid):
    lower, accepted = prefix_energy([segment("DER", 1., .9), invalid])
    assert lower == 100. and [p["segment"] for p in accepted] == ["DER"]


def test_infrastructure_fault_stops_queued_work():
    tasks = [{"task_id": str(i)} for i in range(12)]
    started = []

    def launch(task):
        started.append(task["task_id"])
        return {"status": "infrastructure_error"}

    assert run_wave(tasks, 1, launch) == {"0": {"status": "infrastructure_error"}}
    assert started == ["0"]


def test_concurrency_counts_overlap_not_total_launches():
    rows = [
        {"started_at_utc": "a", "finished_at_utc": "c"},
        {"started_at_utc": "b", "finished_at_utc": "d"},
        {"started_at_utc": "c", "finished_at_utc": "e"},
        {"started_at_utc": None, "finished_at_utc": None},
    ]
    assert elapsed_concurrency(rows) == 2
