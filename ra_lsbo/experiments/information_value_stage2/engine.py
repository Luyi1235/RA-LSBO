"""Opt-in all-support myopic soft-GP control for PR67.

The only changed decision element is the optimistic-violation eligible mask.
The original GP/VAE, candidate predictions, EI normalization reference set,
soft-GP risk/utility formula, phase schedule, and post-proposal coverage remain.
"""
import math

from ...budget import digest
from ..soc_revision.acquisition import quantities
from ..risk_budget63_local_value_engine import LocalCredibilityValueBO
from ..risk_budget63_local_value_policy import (
    _normalized_gains, local_credibility, FAILURE_PRICE_BASE,
    FAILURE_PRICE_BUDGET_SLOPE,
)
from ..risk_budget54_policy import observed_failures
from .protocol import all_candidate_support


class AllSupportSoftGPBO(LocalCredibilityValueBO):
    """PR63 soft-GP selector scored over every legal GP-predicted candidate."""

    def __init__(self, problem, base, seed, output, *, initial_count=12, failure_cap=6):
        super().__init__(problem, base, seed, output, variant="soft_gp",
                         initial_count=initial_count, failure_cap=failure_cap)

    def _choose_acquisition(self, predictions, best, iteration, acquisition, records):
        # Call the exact frozen PR54 parent selector only to obtain its aligned
        # forecasts, phase and original eligible set. No historical labels added.
        original_index, original = super(LocalCredibilityValueBO, self)._choose_acquisition(
            predictions, best, iteration, acquisition, records)
        candidates = self._pr63_last_predicted_candidates
        if not isinstance(candidates, list) or len(candidates) != len(original.get("eligible", [])):
            raise ValueError("PR67 forecasts are not bound to the exact posterior candidate list")
        n = len(candidates)
        original_mask = list(original["eligible"])
        all_mask = all_candidate_support(original_mask, n)
        raw_feasible = original.get("feasibility_proxy_no_status")
        if not isinstance(raw_feasible, list) or len(raw_feasible) != n:
            raise ValueError("PR67 original feasibility forecasts are misaligned")
        risk = [1.0 - float(p) for p in raw_feasible]
        values = quantities(predictions, best, original["kappa"], acquisition.completion_threshold)
        ei = None if values["ei"] is None else values["ei"].tolist()

        # Crucial isolation: normalize EI against the ORIGINAL eligible set,
        # exactly as PR63 does. Only the final candidate eligibility filter is
        # removed; newly admitted candidates do not rescale existing scores.
        gains = _normalized_gains(original["mode"], risk, ei, original_mask)
        spent = observed_failures(records, 12, iteration)
        failure_price = FAILURE_PRICE_BASE + FAILURE_PRICE_BUDGET_SLOPE * (spent / 6.0)
        evidence = [local_credibility(self.problem, records, candidate, r)
                    for candidate, r in zip(candidates, risk)]
        utilities = [g * (1.0 - r) - failure_price * r for g, r in zip(gains, risk)]
        if any(not math.isfinite(u) for u in utilities):
            raise ValueError("PR67 generated nonfinite soft utility")
        selected = max(range(n), key=lambda i: (utilities[i], -i))
        receipt = {
            "schema": "ra_lsbo_pr63_candidate_local_credibility_soft_value_v1",
            "risk_policy": "soft_gp_all_support",
            "selection_rule": "no_hard_gate_max_expected_net_value_all_legal_candidates",
            "candidate_filter_applied": False,
            "no_hard_risk_gate": True,
            "selected_predicted_failure": risk[selected],
            "selected_predicted_feasible_proxy": 1.0 - risk[selected],
            "selected_original_GP_failure_proxy": risk[selected],
            "selected_local_evidence": evidence[selected],
            "selected_objective_gain_normalized": gains[selected],
            "selected_information_bonus": 0.0,
            "selected_net_utility": utilities[selected],
            "failure_cost_weight": failure_price,
            "would_defer_in_unconstrained_utility_model": utilities[selected] <= 0.0,
            "actual_action": "evaluate_under_frozen_six_call_protocol",
            "outside_option_utility": 0.0,
            "abstention_not_performed_in_equal_budget_study": True,
            "candidate_risk_within_allowance": None,
            "predicted_per_call_failure_allowance": None,
            "probability_guarantee": False,
            "risk_proxy_is_not_calibrated": True,
        }
        receipt["selected_candidate_sha256"] = digest(candidates[selected])
        meta = {
            **original,
            "risk_policy": "soft_gp_all_support",
            "scores": [float(u) for u in utilities],
            "eligible": all_mask,
            "feasibility_proxy_no_status": [1.0 - r for r in risk],
            "candidate_net_utility": utilities,
            "candidate_local_support_mass": [e["weighted_local_support_mass"] for e in evidence],
            "candidate_local_ambiguity": [e["candidate_local_ambiguity"] for e in evidence],
            "candidate_gp_original_risk": risk,
            "credibility_decision": receipt,
            "risk_budget": receipt,
            "fallback": None,
            "support_ablation": {
                "rule": "all_legal_gp_candidates",
                "original_support_fallback": original.get("fallback"),
                "original_eligible_count": sum(original_mask),
                "all_candidate_count": n,
                "original_eligible_mask": original_mask,
                "optimistic_violation_support_filter_removed": True,
                "EI_normalization_reference_unchanged": True,
                "hard_probability_risk_gate_added": False,
            },
            "pre_ablation_candidate_index": original_index,
            "probability_guarantee": False,
        }
        return selected, meta
