"""Zero-truth tests of actual archived H1 optimizer instantiation.

These tests MUST NOT grant production approval or call an aircraft evaluator.
"""
import pytest

from confirm12.real_legacy_adapters import (
    LEGACY_METHODS, build_historical_engine, confirmation_readiness,
)


@pytest.mark.parametrize("legacy_method", sorted(LEGACY_METHODS))
def test_historical_engine_is_a_real_legacy_optimizer_without_truth(
    monkeypatch, tmp_path, legacy_method,
):
    from ra_lsbo.experiments.core.problems import AircraftCase
    from ra_lsbo.structure_corpus import CorpusRepresentation

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected physical evaluation or VAE fit")

    monkeypatch.setattr(AircraftCase, "evaluate", forbidden)
    monkeypatch.setattr(CorpusRepresentation, "fit", forbidden)

    problem, engine, info = build_historical_engine(
        "aircraft_ordinary", 390001, legacy_method,
        tmp_path / ("new_" + legacy_method),
    )
    assert problem is not None
    assert callable(engine.propose) and callable(engine.accept)
    assert info.legacy_method == legacy_method
    assert (info.initial, info.horizon, info.pool) == (12, 18, 128)
    assert info.record_kind == "offline_interface_probe_no_evaluation"
    assert info.experiment_qualified is False


@pytest.mark.parametrize(
    "bad", ("ra_lsbo", "full_latent", "full_onehot",
            "reverse_representation", "interval_as_missing", "cbob_adapted"),
)
def test_no_silent_aliasing_of_unqualified_confirmation_methods(tmp_path, bad):
    with pytest.raises(ValueError, match="no automatic confirmatory remapping"):
        build_historical_engine("aircraft_ordinary", 390001, bad, tmp_path / "never_used")


def test_bad_case_or_seed_cannot_start_any_method(tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        build_historical_engine("aircraft_fake", 1, "random", tmp_path / "never_used")
    with pytest.raises(ValueError, match="Seed"):
        build_historical_engine("aircraft_sparse", -1, "random", tmp_path / "never_used")


def test_write_once_output_path(tmp_path):
    with pytest.raises(FileExistsError, match="new historical"):
        build_historical_engine("aircraft_ordinary", 3, "random", tmp_path)


def test_all_proposed_confirmation_arms_remain_unqualified():
    states = confirmation_readiness()
    assert len(states) == 7
    assert all(item.startswith("BLOCKED:") for item in states.values())
