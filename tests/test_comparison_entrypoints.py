from pathlib import Path
import importlib
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("suite", ("mathematical_benchmarks", "simplified_aircraft"))
@pytest.mark.parametrize("entrypoint", ("compare", "study"))
def test_entrypoint_runs_as_script_and_module(suite, entrypoint, tmp_path):
    script = ROOT / "ra_lsbo" / "experiments" / suite / (entrypoint + ".py")
    module = f"ra_lsbo.experiments.{suite}.{entrypoint}"
    direct = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            capture_output=True, text=True)
    modular = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT,
                             capture_output=True, text=True)
    for result in (direct, modular):
        assert result.returncode == 0, result.stderr
        assert "prepare" in result.stdout
        assert "--output" in result.stdout


@pytest.mark.parametrize("suite,plan_suite", (("mathematical_benchmarks", "mathematical"),
                                             ("simplified_aircraft", "simplified_aircraft")))
def test_default_entrypoint_runs_full_pilot(monkeypatch, suite, plan_suite):
    module = importlib.import_module(f"ra_lsbo.experiments.{suite}.compare")
    calls = []

    def prepare(folder, profile, selected_suite):
        calls.append(("prepare", folder, profile, selected_suite))
        return {"arms": ("ra_lsbo", "random"), "budget": {"maximum_model_calls": 7716}}

    def run(folder):
        calls.append(("run", folder))
        return {"charged_model_calls": 200}

    def audit(folder):
        calls.append(("audit", folder))

    def report(folder):
        calls.append(("report", folder))
        return folder / "report.md"

    for name, function in (("prepare", prepare), ("run", run), ("audit", audit), ("report", report)):
        monkeypatch.setattr(module, name, function)
    module.main([])
    assert [call[0] for call in calls] == ["prepare", "run", "audit", "report"]
    folder = calls[0][1]
    assert calls[0][2:] == ("pilot", plan_suite)
    assert calls[1] == ("run", folder)
    assert all(call[1] == folder for call in calls)


@pytest.mark.parametrize("suite,maximum", (("mathematical", 7716),
                                           ("simplified_aircraft", 4820)))
def test_pilot_uses_twenty_seeds_and_declares_complete_budget(suite, maximum):
    from ra_lsbo.experiments.comparison.protocol import build_plan

    plan = build_plan("pilot", suite)
    assert plan["profile"]["seeds"] == tuple(range(9701, 9721))
    assert plan["profile"]["initial"] == 12
    assert plan["profile"]["horizon"] == 30
    assert plan["budget"]["maximum_model_calls"] == maximum


def test_runner_accepts_registered_budget_or_enforces_explicit_limit(monkeypatch, tmp_path):
    from ra_lsbo.experiments.comparison import runner

    monkeypatch.setattr(runner, "load_plan", lambda folder: {
        "budget": {"maximum_model_calls": 7716}})

    def reached_execution(*args):
        raise RuntimeError("execution reached")

    monkeypatch.setattr(runner.os, "open", reached_execution)
    with pytest.raises(RuntimeError, match="execution reached"):
        runner.run(tmp_path)
    with pytest.raises(ValueError, match="exceeds approved limit"):
        runner.run(tmp_path, 500)
