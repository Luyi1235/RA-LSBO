"""Pure evidence semantics plus in-memory/temporary SQLite integration; no aircraft imports."""
import csv
import json
from pathlib import Path
import sqlite3

import pytest

from confirm12.risk_event_semantics import (
    REQUIRED, constraint_sign, digest, label_record, run,
)


def observation(kind="exact", value=-0.1, lower=None, upper=None, **kwargs):
    return dict(kind=kind, value=value if kind == "exact" else None,
                lower=lower, upper=upper, **kwargs)


def make_record(*, status="completed", reason="complete", objective="exact", mission=True,
                soc=None, failure_cause=None):
    outputs = [(name, soc if name == "soc" and soc is not None else observation())
               for name in sorted(REQUIRED)]
    return dict(candidate=dict(continuous=dict(x=1.), discrete=dict(config="test")),
                observation=dict(status=status, objective=observation(objective, -1., lower=12. if objective == "interval" else None),
                                 constraints=outputs, failure_cause=failure_cause,
                                 design_id="", contract_id="contract"),
                diagnostics=dict(termination_reason=reason, mission_completed=mission))


def test_three_distinct_events_not_silently_equated():
    success = label_record(make_record())
    assert success["status_completion"] == 1
    assert success["mission_completed"] is True
    assert success["usable_feasible"] is True

    truncated = label_record(make_record(reason="physical_soc_reserve_abort", mission=False,
                                         objective="interval", soc=observation("interval", lower=0.1)))
    assert truncated["status_completion"] == 1
    assert truncated["mission_completed"] is False
    assert truncated["physical_feasibility"] is False
    assert truncated["usable_feasible"] is False
    assert truncated["status_completed_mission_incomplete"] is True
    assert not truncated["semantic_issues"]

    incomplete = label_record(make_record(reason="physical_battery_power_domain", mission=False,
                                          objective="interval", soc=observation("missing")))
    assert incomplete["physical_feasibility"] is None
    assert incomplete["mission_and_usable_feasible"] is False


def test_unknowns_and_interval_boundaries_remain_unknown():
    assert constraint_sign(observation("interval", lower=-1., upper=1.)) is None
    assert constraint_sign(observation("interval", lower=0., lower_inclusive=False)) is False
    assert constraint_sign(observation("interval", upper=0.)) is True
    assert constraint_sign(observation("missing")) is None
    assert constraint_sign(observation("exact", 0.)) is True
    assert constraint_sign(observation("exact", 0.01)) is False
    hidden = label_record(make_record(status="hidden_failure", reason="physical_discharge_domain",
                                      objective="interval", mission=None, soc=observation("missing"),
                                      failure_cause="battery_discharge_domain"))
    assert hidden["status_completion"] == 0
    assert hidden["mission_completed"] is False
    assert hidden["physical_feasibility"] is None
    assert hidden["usable_feasible"] is False
    infra = label_record(make_record(status="infrastructure_error", reason=None,
                                     objective="missing", mission=None))
    assert infra["status_completion"] is None
    assert infra["mission_completed"] is None


def test_contradictory_mission_evidence_is_reported_not_relabelled():
    row = label_record(make_record(reason="physical_soc_reserve_abort", mission=True,
                                   objective="interval"))
    assert "mission_true_without_completed_exact_target" in row["semantic_issues"]
    assert "mission_diagnostic_termination_disagreement" in row["semantic_issues"]


def test_sqlite_read_only_fixture_and_csv(tmp_path):
    root = tmp_path / "original"
    root.mkdir()
    plan = {"source_commit": "synthetic_test_commit"}
    (root / "plan.json").write_text(json.dumps(plan), encoding="utf8")
    dbpath = root / "real_calls.sqlite"
    good = make_record()
    bad = make_record(reason="physical_soc_reserve_abort", mission=False,
                      objective="interval", soc=observation("interval", lower=0.1))
    rows = [good, bad]
    with sqlite3.connect(str(dbpath)) as db:
        db.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE calls (
          id TEXT,case_name TEXT,seed INTEGER,role TEXT,arm TEXT,idx INTEGER,
          candidate_hash TEXT,contract TEXT,state TEXT,record TEXT,record_hash TEXT);
        """)
        db.executemany("INSERT INTO meta VALUES (?,?)", [
            ("scope", "REAL_AIRCRAFT_USER_INITIATED"),
            ("plan_sha", digest(plan)), ("state", "RUNNING_REAL")])
        for idx, value in enumerate(rows):
            cid = f"cid{idx}"
            value["observation"]["design_id"] = digest(value["candidate"])
            value["diagnostics"]["confirm12_real_call"] = {"call_id": cid}
            db.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                cid, "aircraft_ordinary", 1, "initial", "", idx,
                digest(value["candidate"]), "contract", "settled", json.dumps(value), digest(value)))
    before = dbpath.stat()
    out = tmp_path / "result"
    report = run(root, out, strict_identity=False)
    assert report["calls_scanned"] == 2
    assert report["original_ledger_state_unchanged"] == "RUNNING_REAL"
    assert report["by_case_role_arm"]["aircraft_ordinary/initial/shared_initial"]["status_completed_mission_incomplete"] == 1
    with (out / "risk_event_records.csv").open(encoding="utf8", newline="") as f:
        saved = list(csv.DictReader(f))
    assert len(saved) == 2
    assert saved[1]["status_completion"] == "1"
    assert saved[1]["mission_completed"] == "0"
    assert saved[1]["objective_exact"] == "0"
    assert dbpath.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError, match="fresh output path"):
        run(root, out, strict_identity=False)
