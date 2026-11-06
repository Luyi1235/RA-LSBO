"""Opt-in all-support myopic soft-GP control for PR67.

The only changed decision element is the optimistic-violation eligible mask.
The original GP/VAE, candidate predictions, EI, soft-GP risk/utility formula,
phase schedule, and post-proposal coverage handling remain inherited.
"""
from ..soc_revision.acquisition import quantities
from ..risk_budget63_local_value_engine import LocalCredibilityValueBO
from ..risk_budget63_local_value_policy import choose_local_value
from .protocol import all_candidate_support


class AllSupportSoftGPBO(LocalCredibilityValueBO):
    """PR63 soft-GP selector scored over every legal GP-predicted candidate."""

    def __init__(self, problem, base, seed, output, *, initial_count=12, failure_cap=6):
        super().__init__(problem, base, seed, output, variant="soft_gp",
                         initial_count=initial_count, failure_cap=failure_cap)

    def _choose_acquisition(self, predictions, best, iteration, acquisition, records):
        # Bypass only LocalCredibilityValueBO's eligible-only wrapper and call
        # the exact frozen PR54 parent selector to obtain aligned GP forecasts.
        original_index, original = super(LocalCredibilityValueBO, self)._choose_acquisition(
            predictions, best, iteration, acquisition, records)
        candidates = self._pr63_last_predicted_candidates
        if not isinstance(candidates, list) or len(candidates) != len(original.get("eligible", [])):
            raise ValueError("PR67 forecasts are not bound to the exact posterior candidate list")
        values = quantities(predictions, best, original["kappa"], acquisition.completion_threshold)
        ei = None if values["ei"] is None else values["ei"].tolist()
        original_mask = list(original["eligible"])
        expanded = dict(original)
        expanded["eligible"] = all_candidate_support(original_mask, len(candidates))
        index, meta = choose_local_value(
            self.problem, expanded, candidates, records, ei,
            values["uncertainty"].tolist(), variant="soft_gp", iteration=iteration,
            initial_count=12, horizon=12, failure_cap=6)
        meta["support_ablation"] = {
            "rule": "all_legal_gp_candidates",
            "original_eligible_count": sum(original_mask),
            "all_candidate_count": len(candidates),
            "original_eligible_mask": original_mask,
            "optimistic_violation_support_filter_removed": True,
            "hard_probability_risk_gate_added": False,
        }
        meta["pre_ablation_budgeted_candidate_index"] = original_index
        return index, meta
