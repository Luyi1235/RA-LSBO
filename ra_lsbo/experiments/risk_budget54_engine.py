"""PR54 opt-in risk decision integrated into the EXISTING real VAE/GP dual view.

The original RA-LSBO source and PR45 physical evidence remain untouched;
the only shared change is an opt-in selector hook whose default delegates to
the identical frozen choose_acquisition function.
"""
from dataclasses import replace

from .risk_budget54_policy import (
    POLICIES, choose_semantic_or_budget, observed_failures,
)
from .simplified_aircraft.dualview_optimizer import StableCorpusDiscoveryBO
from .simplified_aircraft.dualview_pilot import model_settings, profile_for
from ..soc_revision.acquisition import quantities


class ObservedFailureCapReached(RuntimeError):
    """No new evaluation may be dispatched after this observed online cap."""


def settings_for_math54():
    # Identical PR45 kernel 4x2 and VAE 2000 structures/30-20 epochs; only
    # horizon and local candidate augmentation are opt-in experiment choices.
    return replace(model_settings("hybrid_onehot",profile_for("pilot")),
                   horizon=12,local=False)


class BudgetedDualViewMathBO(StableCorpusDiscoveryBO):
    """Latent objective + One-hot risk, corrected selection, optional budget.

    Importantly, "semantic_only" and "budgeted" share the SAME surrogate,
    objectives, training and phase scheduling. The latter differs only in
    candidate selection under remaining observed failure allowance.
    """
    def __init__(self, problem, base, seed, output, *, policy,
                 initial_count=12, failure_cap=6):
        if policy not in POLICIES or initial_count != 12 or failure_cap != 6:
            raise ValueError("Unregistered PR54 policy or observation/stop budget")
        self.pr54_policy = policy
        self.pr54_initial = initial_count
        self.pr54_cap = failure_cap
        super().__init__(problem,base,seed,settings_for_math54(),output,
                         allocation="onehot_risk")

    def propose(self, records,global_pool,iteration):
        failed = observed_failures(records,self.pr54_initial,iteration)
        if failed >= self.pr54_cap:
            raise ObservedFailureCapReached("hard_observed_online_failure_cap")
        candidate, meta = super().propose(records,global_pool,iteration)
        # The inherited DiscoveryBO can replace a model selection with a
        # deterministic cold-start/degenerate-gate coverage decision. The
        # originally model-selected candidate's risk receipt must NOT be
        # mislabeled as a forecast of the final coverage design.
        bypass = meta.get("coverage_reason") is not None or (
            "coverage" in str(meta.get("mode", "")))
        if bypass and "risk_budget" in meta:
            original = meta.pop("risk_budget")
            meta["risk_selection_bypassed"] = {
                "reason":meta.get("coverage_reason",meta.get("mode")),
                "former_model_selection_rule":original["selection_rule"],
                "recorded_probability_for_actual_coverage_candidate":False,
            }
        elif "risk_budget" in meta:
            index = meta.get("selected_index")
            proxy = meta["feasibility_proxy_no_status"]
            if (type(index) is not int or not 0<=index<len(proxy) or
                    abs(proxy[index]-meta["risk_budget"]["selected_predicted_feasible_proxy"])>1e-12):
                raise ValueError("Budget prediction receipt does not bind to selected candidate")
        return candidate, meta

    def _choose_acquisition(self,predictions,best,iteration,acquisition,records):
        _, frozen = super()._choose_acquisition(
            predictions,best,iteration,acquisition,records)
        values = quantities(predictions,best,frozen["kappa"],
                            acquisition.completion_threshold)
        failed = observed_failures(records,self.pr54_initial,iteration)
        expected = None if values["ei"] is None else values["ei"].tolist()
        return choose_semantic_or_budget(
            frozen,expected,values["uncertainty"].tolist(),
            policy=self.pr54_policy,iteration=iteration,horizon=12,
            failure_cap=self.pr54_cap,failures_seen=failed)
