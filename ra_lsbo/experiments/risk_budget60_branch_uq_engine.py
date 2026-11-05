"""PR60 opt-in branch evidence selector using the ORIGINAL PR54 GP/VAE engine.

No change to PR54 acquisition source SHA or group GP source. Only the new
PR60 research class uses this wrapper. The candidate pool is intercepted
at the exact posterior predict(...) call, avoiding guessed selection indices.
"""
from .risk_budget54_engine import BudgetedDualViewMathBO
from .risk_budget60_branch_uq_policy import VARIANTS, choose_branch_evidence
from ..soc_revision.acquisition import quantities


class _TrackedPosterior:
    def __init__(self, owner, underlying):
        self.owner=owner
        self.underlying=underlying

    def predict(self,candidates):
        result=self.underlying.predict(candidates)
        self.owner._pr60_last_predicted_pool=list(candidates)
        return result

    @property
    def best(self):
        return self.underlying.best

    @best.setter
    def best(self,value):
        self.underlying.best=value

    def __getattr__(self,name):
        return getattr(self.underlying,name)


class BranchUQDualViewMathBO(BudgetedDualViewMathBO):
    def __init__(self, problem, base, seed, output, *, variant,
                 initial_count=12, failure_cap=6):
        if variant not in VARIANTS or initial_count != 12 or failure_cap != 6:
            raise ValueError("PR60 registered branch uncertainty variant only")
        self.pr60_variant=variant
        self._pr60_last_predicted_pool=None
        super().__init__(problem,base,seed,output,policy="budgeted",
                         initial_count=initial_count,failure_cap=failure_cap)

    def _build_posterior(self,records):
        return _TrackedPosterior(self,super()._build_posterior(records))

    def _choose_acquisition(self,predictions,best,iteration,acquisition,records):
        original_index,original = super()._choose_acquisition(
            predictions,best,iteration,acquisition,records)
        pool=self._pr60_last_predicted_pool
        if not isinstance(pool,list) or len(pool)!=len(original["eligible"]):
            raise ValueError("PR60 GP forecast did not bind to its current candidate list")
        values=quantities(predictions,best,original["kappa"],
                          acquisition.completion_threshold)
        expected=None if values["ei"] is None else values["ei"].tolist()
        idx,adjusted=choose_branch_evidence(
            original,pool,records,expected,values["uncertainty"].tolist(),
            variant=self.pr60_variant,iteration=iteration,
            horizon=12,failure_cap=6)
        adjusted["previous_PR54_min_risk_selection_index"]=original_index
        return idx,adjusted
