"""P0 H2-M regression gates: stable Gaussian moments and causal optimizer replay.

All candidate outcomes are synthetic RunObservation fixtures. No VAE/GP fitting,
simplified-aircraft evaluator, RCAIDE, or physical budget is invoked here.
"""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.experiments.simplified_aircraft import h2_pilot
from ra_lsbo.experiments.simplified_aircraft.h2_optimizer import CausalH2CorpusBO
from ra_lsbo.experiments.simplified_aircraft.h2_risk_controller import component_mixture
from ra_lsbo.experiments.simplified_aircraft.dualview_protocol import model_settings, profile_for
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
from ra_lsbo.experiments.simplified_aircraft.corpus_optimizer import CorpusDiscoveryBO
from ra_lsbo.partial_bo.problems import candidate_pool
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.risk_core.observations import OutputObservation, RunObservation


def test_h2_mixture_variance_stays_positive_at_large_shared_offset():
    one = {"mean": np.array([1e9, 1e9 + 1e-3]),
           "variance": np.array([1e-6, 1e-8])}
    latent = {"mean": np.array([1e9, 1e9 - 1e-3]),
              "variance": np.array([1e-6, 1e-8])}
    mixed = component_mixture(one, latent, .5, .2)
    center = mixed["mean"]
    expected = (.5 * (one["variance"] + (one["mean"] - center) ** 2)
                + .5 * (latent["variance"] + (latent["mean"] - center) ** 2))
    np.testing.assert_allclose(mixed["variance"], expected, rtol=1e-12, atol=0.)
    assert mixed["variance"][0] == pytest.approx(1e-6, rel=1e-12)
    assert np.all(mixed["variance"] > 0)
    assert np.isfinite(mixed["variance"]).all()


def test_h2_mixture_moments_preserve_ordinary_scale_semantics():
    one = {"mean": np.array([-.3, .4]), "variance": np.array([.02, .05])}
    latent = {"mean": np.array([.8, -.2]), "variance": np.array([.04, .07])}
    w = .37
    result = component_mixture(one, latent, w, .15)
    expected_mean = w * one["mean"] + (1-w) * latent["mean"]
    expected_var = (w * (one["variance"] + (one["mean"] - expected_mean) ** 2)
                    + (1-w) * (latent["variance"] + (latent["mean"] - expected_mean) ** 2))
    np.testing.assert_allclose(result["mean"], expected_mean, atol=1e-15)
    np.testing.assert_allclose(result["variance"], expected_var, rtol=1e-13)
    assert result["scale"] == .15
    assert result["components"]["weights"].tolist() == pytest.approx([w, 1-w])


def test_h2_actual_optimizer_adapter_fifth_proposal_uses_four_settled_scores():
    """Fake ONLY the upstream GP/VAE proposer; run real H2 propose/accept/replay."""
    problem = problem_for("aircraft_ordinary")
    candidates, _ = candidate_pool(problem, 812403, 6)
    assert len(candidates) >= 5
    engine = CausalH2CorpusBO(
        problem, BASE, 17, model_settings("hybrid_guard", profile_for("pilot")),
        "/tmp/h2-p0-synthetic-only-never-written")
    history, proposals, online, weights_at_proposal = [], [], [], []

    def fake_parent_propose(records, _pool, iteration):
        candidate = candidates[iteration]
        prefix = digest([r.to_dict() for r in records])
        engine.pending = {
            "design": digest(candidate), "contract": problem.contract_id,
            "records": list(records), "prefix_digest": prefix,
            "branch": engine._branch(candidate), "prior_branch_best": None}
        weight = engine.h2.weight
        mix = component_mixture(
            {"mean": np.array([-.3]), "variance": np.array([.02])},
            {"mean": np.array([.4]), "variance": np.array([.02])},
            weight, .1)
        return candidate, {
            "raw_history_digest": prefix, "selected_index": 0,
            "posterior_fit_skipped": False, "coverage_reason": None,
            "forecasts": {"constraints": {"soc": {
                "components": {
                    "weights": mix["components"]["weights"].tolist(),
                    "mean": mix["components"]["mean"].tolist(),
                    "variance": mix["components"]["variance"].tolist()}}}}}

    with patch.object(CorpusDiscoveryBO, "propose", side_effect=fake_parent_propose):
        for iteration in range(5):
            candidate, metadata = engine.propose(history, candidates, iteration)
            weight = metadata["H2_controller_before"]["onehot_weight"]
            weights_at_proposal.append(weight)
            proposal = {
                "candidate": candidate, "prefix_digest": digest([r.to_dict() for r in history]),
                "metadata": metadata}
            proposals.append(proposal)
            # The selected forecast is committed BEFORE the synthetic observation exists.
            assert engine.h2.pending is not None
            assert engine.h2.pending["weight_at_proposal"] == weight
            observation = RunObservation(
                design_id=digest(candidate), contract_id=problem.contract_id,
                evidence_id=f"p0-synthetic-{iteration}", status="completed",
                objective=OutputObservation("exact", f"objective-{iteration}", value=10000.),
                constraints=tuple((name, OutputObservation(
                    "exact", f"{name}-{iteration}", value=-.05))
                    for name in problem.required_constraints))
            record = DesignRecord(candidate, observation, {"synthetic_only": True})
            engine.accept(record)
            online.append(record)
            history.append(record)

    assert weights_at_proposal[:4] == [0.5] * 4
    assert weights_at_proposal[4] > .5
    assert engine.h2.n_scored == 5
    audit = h2_pilot.replay_h2_controller(proposals, online)
    assert audit["settled_events"] == audit["scored_events"] == 5
    assert audit["final_state"] == engine.h2.snapshot()
    assert audit["new_GP_fits"] == audit["new_aircraft_calls"] == 0

    # A recorded fifth-step decision cannot be silently rewritten as a 0.5 prior.
    tampered = deepcopy(proposals)
    tampered[4]["metadata"]["H2_controller_before"]["onehot_weight"] = .5
    with pytest.raises(ValueError, match="controller"):
        h2_pilot.replay_h2_controller(tampered, online)
