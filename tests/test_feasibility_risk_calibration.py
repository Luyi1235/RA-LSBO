"""Pure-Python PR51 tests: no actual aircraft, GP or VAE models are invoked."""
import math

import pytest

from confirm12.feasibility_risk_calibration import (
    PROXIES, SIZES, _calibrate, _event, _logit, _score, _validate_disjoint,
    fit_intercept, fixed_splits, laplace_prevalence, probabilities, summarize,
)


def fake_plan():
    return {"cases": {
        "aircraft_ordinary": {"seeds": list(range(500001, 501025))},
        "aircraft_sparse": {"seeds": list(range(600001, 601025))},
    }}


def test_fixed_seed_roster_is_disjoint_and_excludes_pr48_and_pr49_targets():
    result = fixed_splits(fake_plan(), folds=8)
    assert len(result) == 16
    for case in ("aircraft_ordinary", "aircraft_sparse"):
        matches = [r for r in result if r["case"] == case]
        assert len(matches) == 8
        all_used = [seed for row in matches
                    for key in ("train", "calibration", "test") for seed in row[key]]
        assert len(all_used) == len(set(all_used)) == 8 * 32
        base = min(fake_plan()["cases"][case]["seeds"])
        assert all(base + 128 + i not in all_used for i in range(8))
        assert all(base + 256 + i not in all_used for i in range(8))
    with pytest.raises(ValueError):
        fixed_splits(fake_plan(), folds=9)


def test_three_risk_proxies_are_not_independent_events():
    heads = dict(soc=.2, stall=.8, root_bending=.7, current=.9,
                 power=.95, voltage=.98)
    p = probabilities(.97, heads)
    assert set(p) == set(PROXIES)
    assert p["mission_status_proxy"] == .97
    assert p["mission_soc_proxy"] == .2
    assert p["feasible_legacy_completion_product"] < p["feasible_constraint_product"]
    assert p["feasible_constraint_product"] < p["feasible_soc_stall_bending"]
    assert abs(p["feasible_legacy_completion_product"] -
               .97 * math.prod(heads.values())) < 1e-12


def test_monotone_one_parameter_calibration_is_fitted_to_calibration_data_only():
    probs = [.5] * 80
    truth = [1] * 64 + [0] * 16
    b = fit_intercept(probs, truth)
    assert b > 0
    assert _calibrate(.3, b) < _calibrate(.8, b)
    assert _score([_calibrate(x, b) for x in probs], truth)["log_loss"] < \
           _score(probs, truth)["log_loss"]
    assert laplace_prevalence([1, 1, 0, 1]) == 4 / 6
    assert _calibrate(.3, 0.) == pytest.approx(.3)
    with pytest.raises(ValueError):
        _logit(float("nan"))
    with pytest.raises(ValueError):
        fit_intercept([.4], [None])


def test_binary_metric_brier_ece_and_false_positives():
    result = _score([.8, .1, .55], [1, 0, 0])
    assert result["brier"] == pytest.approx((.2**2 + .1**2 + .55**2) / 3)
    assert result["false_positive_at_0_5"] == 1
    assert result["known_negatives"] == 2
    assert sum(bin["count"] for bin in result["bins"]) == 3


def test_scoring_keeps_same_test_rows_paired():
    rows = []
    for view in ("onehot", "latent"):
        for i, (y, p) in enumerate(((1, .8), (0, .2))):
            rows.append({
                "case": "aircraft_sparse", "fold": 0, "training_size": 12,
                "view": view, "proxy": "mission_soc_proxy",
                "target_event": "mission", "target_seed": 101,
                "true_event": y, "raw_probability": p,
                "calibrated_probability": p,
                "train_prevalence_baseline": .5,
            })
    result = summarize(rows)
    assert len(result) == 2
    key = "aircraft_sparse/12/mission/mission_soc_proxy/onehot"
    assert result[key]["raw"]["brier"] == pytest.approx(.04)
    assert result[key]["training_prevalence_baseline"]["brier"] == pytest.approx(.25)
    assert result[key]["test_seeds"] == 1


class _Record:
    def __init__(self, candidate, mission, soc_valid):
        self.candidate = candidate
        self._mission = mission
        self._soc_valid = soc_valid

    def to_dict(self):
        controls = ("span", "packing", "stall", "root_bending",
                    "soc", "current", "voltage", "power")
        constraints = []
        for head in controls:
            if head == "soc" and not self._soc_valid:
                value = {"kind": "interval", "lower": .01, "upper": None,
                         "lower_inclusive": True}
            else:
                value = {"kind": "exact", "value": -.1}
            constraints.append((head, value))
        objective = ({"kind": "exact", "value": 1000.} if self._mission
                     else {"kind": "interval", "lower": 500., "upper": None})
        return {"candidate": self.candidate,
                "observation": {
                    "status": "completed", "objective": objective,
                    "constraints": constraints, "failure_cause": None},
                "diagnostics": {
                    "termination_reason": "complete" if self._mission
                    else "physical_soc_reserve_abort",
                    "mission_completed": self._mission}}


def test_semantic_event_is_mission_not_record_status_and_feasibility():
    r = _Record({"x": 2}, False, False)
    assert _event(r) == {"mission": 0, "feasible": 0}
    t = _Record({"x": 3}, True, True)
    assert _event(t) == {"mission": 1, "feasible": 1}
    with pytest.raises(ValueError, match="SOC/mission"):
        _event(_Record({"x": 4}, False, True))


def test_cross_partition_duplicate_rejected_not_intra_partition():
    a = _Record({"x": 1}, True, True)
    b = _Record({"x": 2}, False, False)
    c = _Record({"x": 3}, True, True)
    assert len(_validate_disjoint([a, a], [b], [c])) == 3
    with pytest.raises(ValueError, match="Overlap"):
        _validate_disjoint([a], [b, a], [c])
