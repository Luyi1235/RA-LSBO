import pytest

from ra_lsbo.experiments.mathematical_benchmarks import run_all


def plan():
    return {"schema": "ra_lsbo_external_comparison_v2", "suite": "mathematical",
            "arms": run_all.METHODS,
            "profile": {"name": "pilot", "initial_policies": ("random",),
                        "cases": ("branin_sparse", "rosenbrock_sparse"),
                        "seeds": tuple(range(9701, 9721)), "initial": 12, "horizon": 30},
            "budget": {"maximum_model_calls": 7716, "maximum_rcaide_mda_calls": 0}}


def test_one_click_runs_all_phases_in_order(monkeypatch, tmp_path):
    calls = []
    output = tmp_path / "new_campaign"
    monkeypatch.setattr(run_all, "prepare", lambda path, profile, suite:
                        calls.append(("prepare", path, profile, suite)) or plan())
    monkeypatch.setattr(run_all, "run", lambda path, cap:
                        calls.append(("run", path, cap)) or {"charged_model_calls": 7716})
    monkeypatch.setattr(run_all, "audit", lambda path:
                        calls.append(("audit", path)) or {"passed": True, "checked_proposals": 7200})
    monkeypatch.setattr(run_all, "report", lambda path:
                        calls.append(("report", path)) or path / "report.md")
    assert run_all.execute(output=output) == output
    assert calls == [("prepare", output, "pilot", "mathematical"), ("run", output, 7716),
                     ("audit", output), ("report", output)]


def test_one_click_stops_on_unexpected_budget(monkeypatch, tmp_path):
    invalid = plan()
    invalid["budget"]["maximum_model_calls"] = 7717
    monkeypatch.setattr(run_all, "prepare", lambda *args: invalid)
    monkeypatch.setattr(run_all, "run", lambda *args: pytest.fail("unexpected run"))
    with pytest.raises(ValueError, match="protocol and budget"):
        run_all.execute(output=tmp_path / "campaign")


def test_one_click_stops_after_failed_run_or_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(run_all, "prepare", lambda *args: plan())
    monkeypatch.setattr(run_all, "run", lambda *args: (_ for _ in ()).throw(RuntimeError("failed run")))
    monkeypatch.setattr(run_all, "audit", lambda *args: pytest.fail("unexpected audit"))
    monkeypatch.setattr(run_all, "report", lambda *args: pytest.fail("unexpected report"))
    with pytest.raises(RuntimeError, match="failed run"):
        run_all.execute(output=tmp_path / "campaign")
    monkeypatch.setattr(run_all, "run", lambda *args: {"charged_model_calls": 10})
    monkeypatch.setattr(run_all, "audit", lambda *args: {"passed": False})
    with pytest.raises(RuntimeError, match="Audit did not pass"):
        run_all.execute(output=tmp_path / "campaign2")
