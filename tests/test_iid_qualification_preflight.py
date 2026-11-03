import json
import math
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import beta

from ra_lsbo.experiments import iid_qualification as q


class SameRng:
    def integers(self, n):
        return 0

    def random(self, n):
        return np.full(n, .5)


def test_configuration_grammar_and_canonicalization():
    assert len(q.CONFIGURATIONS) == len(set(q.CONFIGURATIONS)) == 12
    for seed in range(20):
        for item in q.sample(seed, 30)["records"]:
            c, d = item["candidate"]["continuous"], item["candidate"]["discrete"]
            assert (d["prop_layout"], d["propeller_count"], d["parallel_strings"]) in q.CONFIGURATIONS
            assert all(lo <= c[name] <= hi for name, (lo, hi) in q.BOUNDS.items())
            if d["prop_layout"] == "wingtip":
                assert d["propeller_count"] == 2
                assert c["dep_diameter"] == .8 and c["prop_spacing"] == .7
            else:
                assert c["tip_diameter"] == 1.72
            assert all(v <= 0 for v in item["screen"].values())


def test_repeat_designs_preserved_and_each_charged():
    rows = q.sample(123, 3, rng=SameRng())["records"]
    assert len({x["candidate_sha256"] for x in rows}) == 1
    assert [x["index"] for x in rows] == [0, 1, 2]
    assert all(x["charge_role"] == "one_call_per_index_even_if_duplicate" for x in rows)


def test_separate_streams_determinism_and_prefix_stability():
    first = q.sample(21001, 20)
    assert first == q.sample(21001, 20)
    assert first["records"] == q.sample(21001, 30)["records"][:20]
    assert first["records"] != q.sample(21002, 20)["records"]
    assert q.sample(21001, 20) == first  # second stream cannot change first


def test_rejections_preserve_all_attempt_counts_and_exhaustion():
    calls = []
    def reject_once(candidate):
        calls.append(candidate)
        return {"span": 1. if len(calls) == 1 else -1., "packing": -1.}
    result = q.sample(1, 2, precheck=reject_once)
    assert result["attempts"] == 3 and result["rejections"] == 1
    assert [r["attempts"] for r in result["records"]] == [2, 1]
    with pytest.raises(RuntimeError, match="exhausted"):
        q.sample(1, 1, max_attempts_per_sample=3,
                 precheck=lambda _: {"span": 1., "packing": -1.})


@pytest.mark.parametrize("screen", [
    {"span": float("nan"), "packing": -1.}, {"span": -1.},
    {"span": -1., "packing": float("inf")},
])
def test_invalid_screen_is_an_error(screen):
    with pytest.raises(ValueError):
        q.sample(1, 1, precheck=lambda _: screen)


def test_screen_matches_historical_analytic_geometry_without_truth(monkeypatch):
    from ra_lsbo.partial_bo.problems import SimplifiedElectricAircraft
    monkeypatch.setattr(SimplifiedElectricAircraft, "evaluate", lambda *a: pytest.fail("truth called"))
    for seed in range(4):
        for row in q.sample(seed, 50)["records"]:
            candidate = row["candidate"]
            expected = SimplifiedElectricAircraft.geometry(None, candidate)[3]
            assert q.screen(candidate) == pytest.approx(expected, abs=1e-14)


def test_unknown_and_partial_observation_contract():
    good = dict.fromkeys(q.REQUIRED, True)
    assert q.outcome_contract("completed", True, "exact", good) == "known_feasible"
    for status in ("unknown", "infrastructure_error", "numerical_failure", "hidden_failure"):
        assert q.outcome_contract(status, False, "missing", {"soc": False}) == "abort_unknown"
    assert q.outcome_contract("physical_abort", False, "interval", {"soc": False}) == "known_infeasible"
    assert q.outcome_contract("physical_abort", False, "interval", {"soc": None}) == "abort_unknown"
    assert q.outcome_contract("completed", True, "exact", {"soc": True}) == "abort_unknown"
    with pytest.raises(ValueError):
        q.outcome_contract("completed", True, "exact", {"soc": 1})


def test_ci_threshold_is_precommitted():
    assert beta.ppf(.975, 20, 281) < .1
    assert beta.ppf(.975, 21, 280) >= .1
    assert q.build_plan("a" * 40)["thresholds"]["sparse_minimum_feasible"] == 1


def test_import_and_plan_do_not_import_evaluator():
    script = ("import sys; from ra_lsbo.experiments import iid_qualification as q; "
              "q.sample(21001, 5); assert 'ra_lsbo.partial_bo.problems' not in sys.modules; "
              "assert not any(x == 'RCAIDE' or x.startswith('RCAIDE.') for x in sys.modules)")
    subprocess.run([sys.executable, "-c", script], check=True)


def test_plan_contains_no_labels_or_executable_evaluator():
    plan = q.build_plan("a" * 40)
    assert plan["execution_authorized"] is False
    assert plan["budget"]["actual_simplified_aircraft_calls"] == 0
    assert all(len(t["records"]) == 300 for t in plan["tasks"].values())
    assert set(plan["tasks"]) == set(q.SEEDS)
    text = json.dumps(plan["tasks"])
    assert '"objective"' not in text and '"observation"' not in text
    assert not hasattr(q, "run") and not hasattr(q, "evaluate")


def test_plan_cli_write_once_and_no_evaluation(tmp_path):
    path = tmp_path / "plan.json"
    command = [sys.executable, "-m", "ra_lsbo.experiments.iid_qualification", "--output", str(path)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    before = path.read_bytes()
    plan = json.loads(before)
    assert plan["budget"]["actual_simplified_aircraft_calls"] == 0
    assert plan["execution_authorized"] is False
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0
    assert path.read_bytes() == before
