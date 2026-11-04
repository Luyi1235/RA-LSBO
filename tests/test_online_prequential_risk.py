"""No-physical GP regression of prequential risk record parsing and seed isolation."""
import json
import math
import sqlite3
from pathlib import Path

import pytest

from confirm12.risk_event_semantics import digest
from confirm12.online_prequential_risk import (
    fixed_splits, selected_probabilities, read_arm, calibrate, summarize,
    _cluster_delta, _rate_threshold, HORIZON,
)


def roster():
    arms = ["ra_lsbo", "full_latent"]
    return {"max_calls": 151040, "workers": 12,
            "cases": {case: {
                "seeds": list(range(n, n + 1024)),
                "arms_by_seed": {str(s): arms for s in range(n, n+1024)}
            } for case, n in (("aircraft_ordinary", 500001),
                              ("aircraft_sparse", 600001))}}


def test_splits_outcome_blind_and_disjoint_from_pr51():
    entries = fixed_splits(roster(), 64)
    assert len(entries) == 256
    for case in ("aircraft_ordinary", "aircraft_sparse"):
        these = [x for x in entries if x[0] == case]
        cal = [s for _, split, s in these if split == "calibration"]
        test = [s for _, split, s in these if split == "test"]
        assert len(cal) == len(test) == 64
        assert len(set(cal + test)) == 128
        root = min(roster()["cases"][case]["seeds"])
        assert all(x >= root + 640 for x in cal + test)
    with pytest.raises(ValueError):
        fixed_splits(roster(), 65)


def meta_for(candidate, *, skipped=False):
    heads = {"soc": [.7], "stall": [.8], "root_bending": [.9],
             "current": [.98], "voltage": [.99], "power": [.95]}
    result = {"pool": [candidate], "selected_index": 0, "probability_guarantee": False}
    if skipped:
        result["posterior_fit_skipped"] = True
    else:
        result.update(completion_probability=[.96],
                      per_constraint_probability=heads,
                      joint_probability_independence_proxy=[
                          .96 * math.prod(v[0] for v in heads.values())],
                      posterior_fit_skipped=False)
    return result


def test_selected_probability_joint_identity_and_zero_imputation():
    candidate = {"continuous": {"x": .3}, "discrete": {"k": "v"}}
    chosen, reason = selected_probabilities(meta_for(candidate), digest(candidate))
    assert reason is None and chosen is not None
    assert chosen["soc_mission"] == .7
    assert chosen["constraint_product"] == pytest.approx(
        math.prod([.7, .8, .9, .98, .99, .95]))
    assert chosen["legacy_completion_product"] == pytest.approx(
        chosen["constraint_product"] * .96)
    empty, reason = selected_probabilities(
        meta_for(candidate, skipped=True), digest(candidate))
    assert empty is None and reason == "posterior_fit_skipped"
    with pytest.raises(ValueError, match="selected-index"):
        selected_probabilities(meta_for(candidate), digest({"wrong": 3}))
    broken = meta_for(candidate)
    broken["joint_probability_independence_proxy"] = [.9]
    with pytest.raises(ValueError, match="joint"):
        selected_probabilities(broken, digest(candidate))


def fake_record(idx, cid, contract, *, mission=True):
    candidate = {"continuous": {"x": float(idx)}, "discrete": {"k": "v"}}
    constraints = []
    for name in ("span", "packing", "stall", "root_bending",
                 "soc", "current", "voltage", "power"):
        observed = {"kind": "exact", "value": -.1}
        if not mission and name == "soc":
            observed = {"kind": "interval", "lower": .1, "upper": None}
        constraints.append((name, observed))
    return {
        "candidate": candidate,
        "observation": {
            "design_id": digest(candidate), "contract_id": contract,
            "status": "completed", "failure_cause": None,
            "objective": {"kind": "exact", "value": 1000.} if mission
                         else {"kind": "interval", "lower": 500., "upper": None},
            "constraints": constraints,
        },
        "diagnostics": {
            "mission_completed": mission,
            "termination_reason": "complete" if mission else "physical_soc_reserve_abort",
            "confirm12_real_call": {"call_id": cid},
        },
    }


def test_real_sqlite_receipts_and_proposal_prefix_are_bound(tmp_path):
    case, seed, arm, contract = "aircraft_ordinary", 500001, "ra_lsbo", "frozen_contract"
    folder = tmp_path / "trials" / case / str(seed) / arm
    folder.mkdir(parents=True)
    originals = []
    path = tmp_path / "real_calls.sqlite"
    with sqlite3.connect(str(path)) as conn:
        conn.execute("CREATE TABLE calls (id TEXT, idx INT, candidate_hash TEXT, "
                     "contract TEXT, state TEXT, record TEXT, record_hash TEXT, "
                     "case_name TEXT, seed INT, role TEXT, arm TEXT)")
        for i in range(12 + HORIZON):
            cid = "call_" + str(i)
            record = fake_record(i, cid, contract, mission=(i % 3 != 1))
            role = "initial" if i < 12 else "online"
            index = i if i < 12 else i - 12
            conn.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         (cid, index, digest(record["candidate"]), contract,
                          "settled", json.dumps(record), digest(record),
                          case, seed, role, "" if role == "initial" else arm))
            originals.append(record)
    proposals = []
    history = originals[:12]
    for i in range(HORIZON):
        actual = originals[12+i]
        meta = meta_for(actual["candidate"], skipped=(i == 0))
        meta.update(raw_history_digest=digest(history),
                    mode="scheduled_coverage" if i == 0 else "feasible_discovery",
                    cold_start=False)
        proposals.append({
            "iteration": i, "candidate": actual["candidate"],
            "candidate_sha256": digest(actual["candidate"]),
            "prefix_digest": digest(history), "metadata": meta,
        })
        history = history + [actual]
    (folder / "proposals.jsonl").write_text(
        "\n".join(json.dumps(x) for x in proposals) + "\n", encoding="utf8")
    with sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True) as db:
        rows = read_arm(db, tmp_path, case, "test", seed, arm, contract)
    assert len(rows) == HORIZON
    assert rows[0]["prediction_available"] is False
    assert rows[1]["prediction_available"] is True
    assert rows[1]["iteration"] == 1
    assert rows[0]["historical_feasible_prevalence"] > 0
    assert rows[-1]["feasible_true"] in (0, 1)
    proposals[2]["prefix_digest"] = "tampered"
    (folder / "proposals.jsonl").write_text(
        "\n".join(json.dumps(x) for x in proposals) + "\n", encoding="utf8")
    with sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True) as db:
        with pytest.raises(ValueError, match="training prefix"):
            read_arm(db, tmp_path, case, "test", seed, arm, contract)


def scored(case, seed, split, feasible, raw):
    return {
        "case": case, "arm": "ra_lsbo", "split": split, "seed": seed, "iteration": 1,
        "mode": "feasible_discovery", "prediction_available": True,
        "feasible_true": feasible, "mission_true": feasible,
        "predicted_constraint_product": raw,
        "predicted_legacy_completion_product": raw * .96,
        "predicted_soc": raw, "predicted_record_completion": .96,
        "historical_feasible_prevalence": .5,
        "historical_mission_prevalence": .5,
        "online_calibrated_constraint_product": None,
        "online_calibrated_legacy_completion_product": None,
        "online_calibrated_soc_mission": None,
        "online_calibrated_record_completion_as_mission": None,
    }


def test_disjoint_seed_calibrator_and_paired_metrics():
    cal = [scored("aircraft_sparse", 50+i, "calibration",
                  int(i % 4 != 0), .7) for i in range(20)]
    hold = [scored("aircraft_sparse", 101+i, "test",
                   int(i % 3 != 0), .8) for i in range(20)]
    shifts, coverage = calibrate(cal, hold)
    assert len(shifts) == 4
    assert coverage["aircraft_sparse/ra_lsbo/constraint_product"]["calibration_predictions"] == 20
    assert all(r["online_calibrated_constraint_product"] is not None for r in hold)
    s = summarize(hold)
    model = s["aircraft_sparse/ra_lsbo/all"]["models"]["constraint_product"]
    assert model["n"] == 20
    assert model["paired_calibrated_minus_raw_brier"]["clusters"] == 20
    assert model["raw"]["known_negatives"] == 7
    assert model["threshold_0_8_raw"]["accepted_selected_calls"] == 20
    assert model["risk_charge"]["no_failure_budget_guarantee"] is True
    bad = [dict(r) for r in hold]
    bad[0]["seed"] = cal[0]["seed"]
    with pytest.raises(ValueError, match="disjointness"):
        calibrate(cal, bad)
