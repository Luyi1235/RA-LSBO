"""PR64 real soft-utility outside-option action: abstain before evaluator call."""
from .risk_budget63_local_value_engine import LocalCredibilityValueBO


class SoftGPAbstainBO(LocalCredibilityValueBO):
    """Identical PR63 soft_gp proposal, except U<=0 performs a real stop."""

    def __init__(self,problem,base,seed,output,*,initial_count=12,failure_cap=6):
        super().__init__(problem,base,seed,output,variant="soft_gp",
                         initial_count=initial_count,failure_cap=failure_cap)

    def propose(self,records,global_pool,iteration):
        candidate,meta=super().propose(records,global_pool,iteration)
        receipt=meta.get("credibility_decision")
        if receipt is None:
            # Coverage/no-model: never fabricate a probability or infer U<=0.
            meta["pr64_action"]="evaluate_no_model_abstention_unavailable"
            return candidate,meta
        if receipt.get("would_defer_in_unconstrained_utility_model") is True:
            receipt["actual_action"]="abstain_stop_before_evaluator"
            receipt["abstention_performed"]=True
            meta["pr64_action"]="abstain_stop_before_evaluator"
            meta["abstained_candidate"]=candidate
            meta["abstained_candidate_sha256"]=receipt["selected_candidate_sha256"]
            # Returning None is the runner's explicit no-evaluation contract.
            return None,meta
        receipt["actual_action"]="evaluate"
        receipt["abstention_performed"]=False
        meta["pr64_action"]="evaluate"
        return candidate,meta
