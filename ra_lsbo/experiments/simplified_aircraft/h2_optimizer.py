"""Opt-in H2 integration with the existing risk-decoupled aircraft BO.

This source defines a candidate algorithm only; importing it never performs
aircraft evaluations or retrains any GP. New prospective campaigns must
register this class and its separate comparator/seed/budget protocol.
"""
from copy import deepcopy

import numpy as np

from ...budget import digest
from ...partial_bo.models import PrefixFeatures
from ...structure_corpus import feature_snapshot
from .dualview_optimizer import HeterogeneousRiskPosterior,StableCorpusDiscoveryBO
from .h2_risk_controller import (
    CausalSOCController,H2Settings,component_mixture)


class H2RiskPosterior(HeterogeneousRiskPosterior):
    """Latent objective, onehot constraints and past-score-mixed SOC CDF."""

    def __init__(self,problem,records,seed,settings,latent,risk,weight,snapshot):
        self.h2_weight=float(weight)
        self.h2_snapshot=deepcopy(snapshot)
        super().__init__(
            problem,records,seed,settings,latent,risk,
            dual_soc=True,view_policy="h2_causal_soc_score_mixture_v1")

    def predict(self,candidates):
        x_lat=self.objective_features.transform(candidates)
        x_one=self.risk_features.transform(candidates)
        constraints={}
        for name,head in self.constraints.items():
            one=head.predict(x_one)
            if name=="soc":
                if self.dual_soc_head is None:
                    raise RuntimeError("H2 requires a second latent SOC head")
                lat=self.dual_soc_head.predict(x_lat)
                if not np.isclose(head.scale,self.dual_soc_head.scale):
                    raise ValueError("SOC two-view output scales differ")
                value=component_mixture(one,lat,self.h2_weight,head.scale)
            else:
                value={**one,"scale":head.scale}
            constraints[name]=value
        return {
            "completion":self.completion.predict(x_one),
            "constraints":constraints,
            "objective":None if self.objective is None else self.objective.predict(x_lat),
        }

    def diagnostics(self):
        result=super().diagnostics()
        result["model"]="ra_lsbo_h2_causal_risk_v1"
        result["controller_snapshot_before_prediction"]=deepcopy(self.h2_snapshot)
        result["SOC_view_weight_onehot"]=self.h2_weight
        result["SOC_probability_rule"]="componentwise_gaussian_mixture_CDF"
        result["SOC_margin_rule"]="moment_matched_Gaussian_optimistic_geometry"
        return result


class CausalH2CorpusBO(StableCorpusDiscoveryBO):
    """Prefix-causal SOC weight update after each actual settled online call."""

    def __init__(self,problem,base,seed,settings,output,controller_settings=None):
        self.h2=CausalSOCController(controller_settings or H2Settings())
        self.last_h2_settlement=None
        super().__init__(problem,base,seed,settings,output,allocation="soc_dual_guard")

    def _build_posterior(self,records):
        latent=feature_snapshot(self.problem,self.base,records,self.session)
        onehot=PrefixFeatures(self.problem,self.base,records,self.seed,False,30)
        return H2RiskPosterior(self.problem,records,self.seed,self.settings,
                               latent,onehot,self.h2.weight,self.h2.snapshot())

    def propose(self,records,global_pool,iteration):
        if self.h2.pending is not None or len(self.h2.events)!=iteration:
            raise ValueError("H2 controller must have settled all earlier proposals")
        prefix=digest([r.to_dict() for r in records])
        snapshot=self.h2.snapshot()
        candidate,meta=super().propose(records,global_pool,iteration)
        if meta.get("coverage_reason")=="model_failure":
            committed=self.h2.register(digest(candidate),prefix,skip="model_failure")
        elif meta.get("posterior_fit_skipped"):
            committed=self.h2.register(digest(candidate),prefix,skip="coverage")
        else:
            forecast=meta["forecasts"]["constraints"]["soc"]
            comp=forecast["components"]
            selected=meta["selected_index"]
            if (len(comp["weights"])!=2 or
                not np.isclose(comp["weights"][0],snapshot["onehot_weight"])
                or not np.isclose(comp["weights"][1],1-snapshot["onehot_weight"])):
                raise ValueError("H2 selected SOC components do not match past weight")
            committed=self.h2.register(
                digest(candidate),prefix,
                onehot={"mean":comp["mean"][0][selected],
                        "variance":comp["variance"][0][selected]},
                latent={"mean":comp["mean"][1][selected],
                        "variance":comp["variance"][1][selected]})
        meta["H2_controller_before"]=snapshot
        meta["H2_prior_forecast_commitment"]=committed
        meta["H2_controller_source"]="past_settled_SOC_only"
        return candidate,meta

    def accept(self,record):
        if self.h2.pending is None:
            raise ValueError("No H2 prior forecast for settled optimization")
        p=deepcopy(self.h2.pending)
        # The original RA-LSBO validator must accept this exact record first.
        # Nonmatching physical feedback never advances H2 view weights.
        super().accept(record)
        self.last_h2_settlement=self.h2.settle(
            record.observation.design_id,
            p["prefix"],record.observation.constraint("soc"))

    def controller_events(self):
        return deepcopy(self.h2.events)
