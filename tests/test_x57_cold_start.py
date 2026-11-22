from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ra_lsbo.x57_cold_campaign import ARMS, INITIAL, MAX_ONLINE, endpoint
from ra_lsbo.x57_cold_feedback import (
    CERTIFICATE_SOURCE, RISK_METRIC, normalize, risk_observation,
)
from ra_lsbo.x57_cold_qualification import N_QUALIFICATION, qualification_plan
from ra_lsbo.x57_cold_sampling import LAYOUTS, RosterShortfall, active_names, freeze_roster
from ra_lsbo.x57_cold_worker import Dispatcher
from ra_lsbo.x57_live import X57LiveProblem, evaluator_files
from run_x57_cold_start import BASE, CONFIG, ROOT, main


@pytest.fixture
def live(tmp_path):
    return X57LiveProblem(BASE, CONFIG, tmp_path / "workers")


def test_freeze_layout_activation_and_screening(live):
    space = live.space
    roster, audit = freeze_roster(live.problem, 0)
    assert [p.discrete["prop_layout"] for p in roster] == list(LAYOUTS) * 16
    assert len({space.key(p) for p in roster}) == 32
    assert all(all(v <= 0 for v in live.problem.precheck(p).values()) for p in roster)
    assert all("propeller_count" not in active_names(space, p) for p in roster[::2])
    assert all("propeller_count" in active_names(space, p) for p in roster[1::2])
    assert {p.discrete["propeller_count"] for p in roster[1::2]} == {2, 4, 6, 8, 10, 12}
    for point in roster:
        active = active_names(space, point)
        wingtip = point.discrete["prop_layout"] == LAYOUTS[0]
        assert ("tip_prop_rpm" in active) == wingtip
        assert ("prop_span_spacing" in active) != wingtip
        assert ("winglet_fraction" in active) == (point.discrete["has_winglet"] == "yes")
    for layout in LAYOUTS:
        receipt = audit[layout]
        assert receipt["after"]["count"] == 16
        assert receipt["screened_supply"] >= 16
        assert receipt["before"]["continuous"]["wing_area"]["bins"] == list(range(16))
        winglet = receipt["before"]["continuous"]["winglet_fraction"]
        assert winglet["bins"] == list(range(winglet["active_count"]))
        assert set(receipt["after"]["discrete"]["has_winglet"]) == {"no", "yes"}
    repeated, _ = freeze_roster(live.problem, 0)
    assert [space.key(p) for p in repeated] == [space.key(p) for p in roster]


def test_shortfall_reports_original_branch(live, monkeypatch):
    precheck = live.problem.precheck
    monkeypatch.setattr(live.problem, "precheck", lambda p: (
        {"blocked": 1.} if p.discrete["prop_layout"] == LAYOUTS[1] else precheck(p)))
    with pytest.raises(RosterShortfall) as raised:
        freeze_roster(live.problem, 0, max_designs=8)
    blocked = raised.value.audit[LAYOUTS[1]]
    assert blocked["screened_supply"] == 0
    assert blocked["rejections_by_reason"] == {"blocked": 128}


def test_worker_feedback_preserves_information_boundaries():
    required = ("geometry", "soc")
    checks = {"geometry": -.5}
    completed = normalize({"status": "constraint_violation", "solver_success": True,
                           "objective": 91., "constraints": {"geometry": -.5, "soc": .2}, "metrics": {}},
                          required, checks, "mock")
    assert completed["completion"] is True and not completed["usable"]
    assert completed["objective_kind"] == "exact"
    assert risk_observation({"call_id": "one", "feedback": completed}, True).upper == 0.

    cert = {"metric": RISK_METRIC, "source": CERTIFICATE_SOURCE, "validated_original_model": True,
            "lower_W": 200., "upper_W": 500.}
    raw = {"status": "physical_early_stop", "solver_success": False,
           "objective": None, "constraints": {"soc": 900.},
           "metrics": {"physical_stop_verified": True, "completion_certificate": cert}}
    known = normalize(raw, required, checks, "mock")
    row = {"call_id": "two", "feedback": known}
    assert (risk_observation(row, True).lower, risk_observation(row, True).upper) == (.2, .5)
    assert (risk_observation(row, False).lower, risk_observation(row, False).upper) == (0., None)
    assert known["constraints"] == {"geometry": -.5, "soc": None}
    for invalid in ({**cert, "metric": "unrelated"}, {**cert, "lower_W": -1.},
                    {**cert, "upper_W": 100.}, {**cert, "validated_original_model": False}):
        raw["metrics"]["completion_certificate"] = invalid
        rejected = normalize(raw, required, checks, "mock")
        assert rejected["completion"] is False and rejected["risk_interval_W"] is None
        rejected_row = {"call_id": "invalid", "feedback": rejected}
        assert risk_observation(rejected_row, False) == risk_observation(rejected_row, True)
    for status, expected in (("mission_not_converged", "numerical_nonconvergence"),
                             ("worker_error:ImportError", "infrastructure_error"),
                             ("timeout", "infrastructure_error")):
        raw.update(status=status)
        parsed = normalize(raw, required, checks, "mock")
        assert parsed["outcome"] == expected and parsed["objective_Wh"] is None
        assert risk_observation({"call_id": "unknown", "feedback": parsed}, True) is None
    raw["status"] = "physical_early_stop"
    assert normalize(raw, required, checks, "rcaide")["outcome"] == "unverified_termination"


@pytest.mark.parametrize("scenario,online,status,charges", [
    ("mixed", 4, "complete", 44),
    ("no-usable", 1, "complete", 35),
    ("infrastructure", 2, "infrastructure_stop", 35),
])
def test_mock_worker_closed_loop(tmp_path, scenario, online, status, charges):
    output = tmp_path / scenario
    result = subprocess.run([sys.executable, "-B", "run_x57_cold_start.py", "run",
                             "--output", str(output), "--target-wh", "41000",
                             "--online", str(online), "--fast-mock", "--mock-profile", scenario],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == (0 if status == "complete" else 2), result.stdout[-5000:] + result.stderr
    ledger = [json.loads(line) for line in (output / "calls.jsonl").read_bytes().splitlines()]
    settled = [r for r in ledger if r["event"] == "settled"]
    starts = {r["call_id"] for r in ledger if r["event"] == "worker_started"}
    assert len(starts) == charges == sum(r["charged"] for r in settled)
    assert starts == {r["call_id"] for r in settled}
    initial = [r["call_id"] for r in settled if r["arm"] == "shared_initial"]
    assert len(initial) == 32
    report = json.loads((output / "report.json").read_bytes())
    assert report["worker_charges"] == charges and report["rcaide_calls"] == 0
    assert report["seeds"][0]["initial_group"] == "0"
    for arm, track in report["seeds"][0]["arms"].items():
        assert track["status"] == status and track["call_ids"][:32] == initial
        paid = [r for r in settled if r["arm"] == arm]
        assert all(r["online"] + INITIAL == r["evaluation"] for r in paid)
        assert track["call_ids"][32:] == [r["call_id"] for r in paid]
        if scenario == "mixed":
            hit = next(r["evaluation"] for r in paid if r["feedback"]["objective_Wh"] <= 41000.)
            assert track["first_usable"] == 33 and track["first_target"] == hit
            if arm != ARMS[0]:
                assert [r["proposal"]["risk_stage"] for r in paid] == ["completion"] * 3 + ["performance"]
                assert paid[-1]["proposal"]["risk_mode"] == "risk_weighted_native_search"
                assert all(r["proposal"]["n_finite_information_rows"] ==
                           (8 if arm == "structured_interval" else 0) for r in paid)
        else:
            assert track["first_usable"] is None and track["best_usable_Wh"] is None
            if scenario == "no-usable" and arm != ARMS[0]:
                assert paid[0]["proposal"]["risk_fallback"] == "no_supported_completion_feedback"
    if scenario == "no-usable":
        assert Counter(r["arm"] for r in ledger if r["event"] == "native_missing_objective_stop") == dict.fromkeys(ARMS, 1)
    if scenario == "mixed":
        first = [r["proposal"] for r in settled if r["evaluation"] == 33 and r["arm"] in ARMS[1:]]
        assert first[0]["candidate_list_sha256"] == first[1]["candidate_list_sha256"]
        assert first[0]["p_complete"] != first[1]["p_complete"]


def test_dispatch_launch_failure_and_timeout(live, tmp_path, monkeypatch):
    candidate = freeze_roster(live.problem, 0)[0][0]
    fields = dict(seed=0, arm="shared_initial", evaluation=1, online=0, phase="initial", proposal={})
    timeout_dir = tmp_path / "timeout"
    timeout_dir.mkdir()
    timeout = Dispatcher(live, timeout_dir, backend="mock", worker_root=ROOT, timeout=0.)
    result = timeout.evaluate(candidate, **fields)
    assert result["charged"] and result["worker_result"]["status"] == "timeout"
    assert '"event": "interrupted"' in timeout.journal.read_text(encoding="utf-8")

    def denied(*args, **kwargs):
        raise OSError("synthetic launch failure")

    monkeypatch.setattr(subprocess, "Popen", denied)
    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    failed = Dispatcher(live, failed_dir, backend="mock", worker_root=ROOT)
    result = failed.evaluate(candidate, **fields)
    assert failed.count == 0 and not result["charged"]
    assert result["feedback"]["outcome"] == "infrastructure_error"


def test_endpoints_use_cumulative_nodes_and_retain_unobserved():
    assert 6 * (INITIAL + 3 * MAX_ONLINE) == 5376
    rows = [{"evaluation": i, "feedback": {"usable": i >= 40, "objective_Wh": 40000 - i,
                "outcome": "completed_usable" if i >= 40 else "numerical_nonconvergence"},
             "arm": "shared_initial" if i <= 32 else ARMS[0], "charged": True, "elapsed_seconds": 1.}
            for i in range(1, 49)]
    result = endpoint(rows, 39955)
    assert result["first_usable"] == 40 and result["first_target"] == 45
    assert result["checkpoints"]["32"]["attained"] is False
    assert result["checkpoints"]["48"] == {"online": 16, "reached": True, "attained": True,
                                           "best_usable_Wh": 39952}
    assert result["checkpoints"]["64"]["attained"] is None


@pytest.mark.parametrize("failure,expected_calls,expected_worker_status", [
    (None, 20, None),
    (1, 0, "worker_launch_error"),
    (20, 20, "worker_nonzero_exit"),
])
def test_qualification_reports_and_exits_on_infrastructure_stop(
        live, tmp_path, monkeypatch, failure, expected_calls, expected_worker_status):
    frozen_worker = tmp_path / "physical-base"
    shutil.copytree(BASE, frozen_worker, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    qualification_plan(live, tmp_path, {"worker_sha256": evaluator_files(frozen_worker)})
    output = tmp_path / "qualification-run"
    attempts = []

    def fake_worker(command, *, cwd, **kwargs):
        assert cwd == frozen_worker
        assert command[-3] == "problems.x57.worker"
        request, response = map(Path, command[-2:])
        payload = json.loads(request.read_bytes())
        candidate = live.candidate_class(**payload["candidate"])
        attempts.append(candidate)
        if failure == len(attempts):
            if failure == 1:
                raise OSError("synthetic launch failure")
            return SimpleNamespace(pid=101, wait=lambda **_: 1, returncode=1)
        checks = live.problem.precheck(candidate)
        margins = {**dict.fromkeys(live.required_constraints, -0.1), **checks}
        response.write_text(json.dumps({"objective": 40000., "solver_success": True,
                                        "status": "ok", "constraints": margins, "metrics": {}}), encoding="utf-8")
        return SimpleNamespace(pid=100, wait=lambda **_: 0, returncode=0)

    monkeypatch.setattr(subprocess, "Popen", fake_worker)
    monkeypatch.setattr(sys, "argv", ["run_x57_cold_start.py", "qualification-run", "--execute-rcaide",
                                     "--plan", str(tmp_path / "qualification_plan.json"),
                                     "--output", str(output)])
    if failure is None:
        main()
    else:
        with pytest.raises(SystemExit) as stopped:
            main()
        assert stopped.value.code == 2
    report = json.loads((output / "qualification_report.json").read_bytes())
    assert report["planned_calls"] == N_QUALIFICATION == 20
    assert report["rcaide_calls"] == expected_calls
    ledger = [json.loads(line) for line in (output / "calls.jsonl").read_bytes().splitlines()]
    assert sum(row["event"] == "worker_started" for row in ledger) == expected_calls
    if failure is None:
        assert report["status"] == "complete" and report["stop"] is None
        assert all(v["usable"] == 10 and len(v["repeat_comparisons"]) == 2
                   for v in report["layouts"].values())
    else:
        assert report["status"] == "infrastructure_stop"
        assert report["stop"]["worker_status"] == expected_worker_status
        assert report["stop"]["evaluation"] == len(attempts) == failure
