import pytest

from ra_lsbo.x57_cold_feedback import usable_energy_feedback


def test_same_violation_evidence_with_different_lower_endpoints():
    sign = usable_energy_feedback("prefix", 49350., 47000., use_interval=False)
    interval = usable_energy_feedback("prefix", 49350., 47000., use_interval=True)
    assert sign.evidence_id == interval.evidence_id
    assert sign.constraint_satisfaction() is interval.constraint_satisfaction() is False
    assert sign.lower == 0 and not sign.lower_inclusive
    assert interval.lower == pytest.approx(.05) and interval.lower_inclusive
    assert sign.upper is interval.upper is sign.value is interval.value is None


@pytest.mark.parametrize("lower", [None, 46999., 47000., float("nan")])
def test_prefix_at_or_below_cap_does_not_prove_either_sign(lower):
    sign = usable_energy_feedback("prefix", lower, 47000., use_interval=False)
    interval = usable_energy_feedback("prefix", lower, 47000., use_interval=True)
    assert sign == interval
    assert sign.kind == "missing" and sign.constraint_satisfaction() is None
