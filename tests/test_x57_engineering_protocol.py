import json
from types import SimpleNamespace

import torch

from ra_lsbo.x57_energy_surrogate import engineering_log_probability
from run_x57_cold_start import BASE, CONFIG
import run_x57_engineering_energy_pilot as experiment


def test_cold_score_includes_soc_even_when_energy_favors_the_other_point():
    features = torch.tensor([[0.], [1.]], dtype=torch.double)
    energy = SimpleNamespace(log_probability=lambda x: torch.tensor([-.01, -.5]))
    soc = SimpleNamespace(posterior=lambda x: SimpleNamespace(mean=.1 - .2 * x,
                                                              variance=torch.full_like(x, .01)))
    score = engineering_log_probability(features, energy, ("soc",), {"soc": (soc, 0.)}, (0., 1.))
    assert score.argmax().item() == 1


def test_x57_matched_three_arm_mock(tmp_path, monkeypatch):
    config = json.loads(CONFIG.read_bytes())
    config["mission"]["cruise_distance_nmi"] = 45
    path = tmp_path / "mission.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "run"
    output.mkdir()
    args = SimpleNamespace(output=output, execute_rcaide=False, seed=20261125, online=4,
                           workers=3, offline_report_sha256=None)

    native_update = experiment.update_native

    def simulate_native_stop(native, point, row, phase, metadata):
        stop = native_update(native, point, row, phase, metadata)
        return stop or row["online"] == 1

    monkeypatch.setattr(experiment, "update_native", simulate_native_stop)
    report = experiment.pilot(args, BASE, {}, {}, path, {}, set())
    assert report["status"] == "complete"
    assert report["rcaide_calls"] == 0 and report["worker_starts"] == 20
    assert report["shared_initial_starts"] == 8
    assert report["peak_workers"] <= 3
    assert report["online_starts"] == dict.fromkeys(experiment.ARMS, 4)
    sign = report["arms"]["ra_lsbo_energy_sign"]
    interval = report["arms"]["ra_lsbo_energy_interval"]
    assert sign["fitted_partial_ids"] == interval["fitted_partial_ids"]
    assert len(sign["fitted_partial_ids"]) == 1
    for a, b in zip(sign["history"][:8], interval["history"][:8]):
        av, bv = a["energy_evidence"], b["energy_evidence"]
        assert a["candidate_id"] == b["candidate_id"] and a["feedback"] == b["feedback"]
        if av["kind"] == "interval":
            assert {**av, "lower": bv["lower"], "lower_inclusive": True} == bv
            assert av["lower"] == 0 < bv["lower"]
        else:
            assert av == bv
    events = [json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()]
    assert sum(e["event"] == "native_missing_objective_stop" for e in events) == 3
    assert any(e.get("proposal", {}).get("risk_mode") == "risk_weighted_native_search" for e in events)
