"""PR63 opt-in true GP/VAE candidate-local credible-value acquisition integration.

The original PR54 dual-view surrogate and frozen acquisition files stay
byte-identical. Like PR60, this subclass intercepts the exact candidate
list immediately after an ORIGINAL posterior.predict() call and overrides
only the model-based acquisition decision. Any subsequent deterministic
coverage/fallback does NOT retain a fake selected-GP-risk receipt.
"""
from ..budget import digest
from ..soc_revision.acquisition import quantities
from .risk_budget54_engine import BudgetedDualViewMathBO
from .risk_budget63_local_value_policy import VARIANTS, choose_local_value


class _BoundedPoolPosterior:
    def __init__(self,owner,raw):
        self.owner=owner
        self.raw=raw

    def predict(self,candidates):
        forecast=self.raw.predict(candidates)
        self.owner._pr63_last_predicted_candidates=list(candidates)
        return forecast

    @property
    def best(self):
        return self.raw.best

    @best.setter
    def best(self,v):
        self.raw.best=v

    def __getattr__(self,name):
        return getattr(self.raw,name)


class LocalCredibilityValueBO(BudgetedDualViewMathBO):
    """Three new arms differ ONLY in risk blending and small value-of-info."""

    def __init__(self,problem,base,seed,output,*,variant,
                 initial_count=12,failure_cap=6):
        if variant not in VARIANTS or initial_count!=12 or failure_cap!=6:
            raise ValueError("Unregistered PR63 candidate credibility policy")
        self.pr63_variant=variant
        self._pr63_last_predicted_candidates=None
        super().__init__(problem,base,seed,output,policy="budgeted",
                         initial_count=initial_count,failure_cap=failure_cap)

    def _build_posterior(self,records):
        return _BoundedPoolPosterior(self,super()._build_posterior(records))

    def _choose_acquisition(self,predictions,best,iteration,acquisition,records):
        # Only obtain ORIGINAL pre-outcome GP arrays, phase and support.
        # Its transient hard-gated index is never dispatched/evaluated here.
        original_index,original=super()._choose_acquisition(
            predictions,best,iteration,acquisition,records)
        candidates=self._pr63_last_predicted_candidates
        if (not isinstance(candidates,list) or
                len(candidates)!=len(original["eligible"])):
            raise ValueError("PR63 GP risk forecast is not bound to its candidate pool")
        values=quantities(predictions,best,original["kappa"],
                          acquisition.completion_threshold)
        ei=None if values["ei"] is None else values["ei"].tolist()
        index,meta=choose_local_value(
            self.problem,original,candidates,records,ei,
            values["uncertainty"].tolist(),variant=self.pr63_variant,
            iteration=iteration,initial_count=12,horizon=12,failure_cap=6)
        meta["pre_soft_PR54_budgeted_candidate_index"]=original_index
        return index,meta

    def propose(self,records,global_pool,iteration):
        candidate,meta=super().propose(records,global_pool,iteration)
        # The inherited DiscoveryBO may OVERWRITE model selection with an
        # entirely different scheduled/degenerate/model-failure coverage
        # design. Discard all candidate-level trust receipts in that case.
        bypass=meta.get("coverage_reason") is not None or (
            "coverage" in str(meta.get("mode","")))
        if bypass:
            if "credibility_decision" in meta:
                old=meta.pop("credibility_decision")
                meta["pr63_coverage_bypass"]={
                    "reason":meta.get("coverage_reason",meta.get("mode")),
                    "model_selected_utility_not_attributed_to_coverage":old["selected_net_utility"],
                    "selected_coverage_risk_forecast_available":False,
                }
            if "risk_budget" in meta:
                raise ValueError("PR63 coverage retained a model risk selection receipt")
        else:
            r=meta.get("credibility_decision")
            if not isinstance(r,dict) or not meta.get("risk_budget"):
                raise ValueError("PR63 actual GP-selected point lacks soft utility evidence")
            idx=meta.get("selected_index")
            pool=meta.get("pool")
            if (type(idx) is not int or not isinstance(pool,list) or
                    not 0<=idx<len(pool) or pool[idx]!=candidate or
                    abs(r["selected_predicted_failure"]-
                        (1.-meta["feasibility_proxy_no_status"][idx]))>1e-10 or
                    r["candidate_filter_applied"] is not False):
                raise ValueError("PR63 candidate utility is not associated with actual selected design")
            r["selected_candidate_sha256"]=digest(candidate)
            r["model_selected_coverage_bypassed"]=False
        return candidate,meta
