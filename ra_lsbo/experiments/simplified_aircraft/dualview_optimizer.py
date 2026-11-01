"""RA-LSBO heterogeneous risk-head candidate: latent energy, explicit-structure risk.

Research-only opt-in algorithm. It keeps the frozen 2000-structure persistent
latent objective representation and the original acquisition/phase rules.
Completion and constraints use hierarchical one-hot; an optional SOC guard
uses both encodings and the less-optimistic Gaussian feasibility probability.

All mixed-likelihood GP candidates use the explicitly enabled stable-tail
moments; a totally failed response head causes a *recorded* deterministic
coverage decision, not a fabricated target, optimistic prior, or pilot abort.
Unknown/infrastructure/numerical physical results still stop after settlement.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import math
import numpy as np
from scipy.special import ndtr

from ...budget import digest
from ...partial_bo.models import PrefixFeatures, EvidenceGP
from ...risk_core.mixed_gp import MixedGPSettings
from ...risk_core.observations import OutputObservation
from ...structure_corpus import feature_snapshot
from ...partial_refinement.posterior import RefinementPosterior, group_divisors
from ..discovery_stages.optimizer import DiscoveryBO
from ..discovery_stages.policies import coverage_pick, filter_near, coverage_enabled
from .corpus_optimizer import CorpusDiscoveryBO, SCHEMA as CORPUS_TRAINING_SCHEMA
from .corpus_optimizer import COUNT as STRUCTURE_COUNT

C_GRID=(.05,.15,.4,1.)
D_GRID=(.5,1.5)
FEATURE_VIEW_POLICY = "explicit risk/latent energy"
FALLBACK_TAG="all_kernel_candidates_failed_or_prediction_numeric_failure"


class ModelNumericalFailure(RuntimeError):
    """No acceptable response-head posterior; eligible for registered coverage.

    Contains a bounded, serializable candidate-level failure record. Physical
    evaluator failures and unexpected software exceptions are NOT this class.
    """
    def __init__(self, head, failures):
        self.head=str(head)
        self.failures=deepcopy(failures)
        super().__init__("No valid GP for response head: "+self.head)


def _normalize(observations, objective=False, completion=False, reference_scale=1.):
    values=np.asarray([o.value for o in observations if o.kind=="exact"],float)
    center=float(values.mean()) if objective and len(values) else 0.
    if objective and len(values)>1:
        scale=max(float(values.std()),reference_scale*.01)
    elif len(values):
        scale=max(float(np.median(np.abs(values))),.05)
    else:
        scale=float(reference_scale)
    if completion:
        center,scale=0.,1.
    if not np.isfinite(center) or not np.isfinite(scale) or scale<=0:
        raise ValueError("Invalid physical output normalization")
    result=[]
    for o in observations:
        if o.kind=="exact":
            result.append(replace(o,value=(o.value-center)/scale))
        elif o.kind=="interval":
            result.append(replace(o,
                lower=None if o.lower is None else (o.lower-center)/scale,
                upper=None if o.upper is None else (o.upper-center)/scale))
        else:
            result.append(o)
    return center,scale,result


class StableGroupedHead:
    """Exact existing grouped Matérn scale grid, with stable interval moments.

    No GP hyperparameter grid extension or last-minute fit retry. The log
    evidence winner is selected from successfully converged candidates.
    """
    def __init__(self, inputs, observations, *, continuous_count, seed, name,
                 objective=False, completion=False, reference_scale=1.,
                 continuous_grid=C_GRID, discrete_grid=D_GRID):
        inputs=np.asarray(inputs,float)
        if inputs.ndim!=2 or len(inputs)!=len(observations) or not np.isfinite(inputs).all():
            raise ValueError("Finite aligned GP feature rows required")
        if tuple(continuous_grid)!=C_GRID or tuple(discrete_grid)!=D_GRID:
            raise ValueError("The preregistered 4x2 kernel grid must not change")
        self.center,self.scale,normalized=_normalize(
            observations,objective,completion,reference_scale)
        self.name=name
        self.candidates=[]
        fitted=[]
        for c in C_GRID:
            for d in D_GRID:
                divisor=group_divisors(inputs.shape[1],continuous_count,c,d)
                row={"continuous_length":c,"discrete_length":d}
                try:
                    model=EvidenceGP(inputs/divisor,normalized,seed=seed,
                        settings=MixedGPSettings(
                            length_scale=1.,
                            exact_noise=1e-3,
                            interval_noise=1. if completion else 0.,
                            jitter=1e-8,
                            stable_tail_moments=True,
                            record_ep_trace=False))
                    evidence=float(model.log_evidence())
                    if not np.isfinite(evidence):
                        raise FloatingPointError("Nonfinite EP marginal evidence")
                    row.update(status="ok",log_evidence=evidence)
                    fitted.append((model,divisor))
                except (np.linalg.LinAlgError,FloatingPointError,RuntimeError) as error:
                    row.update(status=type(error).__name__,error_message=str(error),
                               log_evidence=None)
                    fitted.append(None)
                self.candidates.append(row)
        successful=[(i,item) for i,item in enumerate(fitted) if item is not None]
        if not successful:
            raise ModelNumericalFailure(self.name,self.candidates)
        self.selected_index=max(successful,key=lambda t:self.candidates[t[0]]["log_evidence"])[0]
        self.model,self.divisor=fitted[self.selected_index]
        self.identity=digest({"name":name,"center":self.center,"scale":self.scale,
                              "features_dim":inputs.shape[1],"candidate_grid":self.candidates,
                              "selected_index":self.selected_index,
                              "model":self.model.to_dict(),
                              "stable_tail":True})

    def predict(self, raw):
        try:
            result=self.model.predict(np.asarray(raw,float)/self.divisor)
            mean=self.center+self.scale*np.asarray(result["mean"])
            var=self.scale*self.scale*np.asarray(result["variance"])
            if not np.isfinite(mean).all() or not np.isfinite(var).all() or np.any(var<0):
                raise FloatingPointError("Nonfinite or negative mixed GP predictive variance")
            return {"mean":mean,"variance":var}
        except (np.linalg.LinAlgError,FloatingPointError,RuntimeError) as error:
            raise ModelNumericalFailure(self.name,[
                {"phase":"prediction","status":type(error).__name__,
                 "error_message":str(error)}]) from error

    def diagnostics(self):
        return {"identity":self.identity,"name":self.name,
            "center":self.center,"scale":self.scale,
            "selected":deepcopy(self.candidates[self.selected_index]),
            "candidates":deepcopy(self.candidates),
            "counts":deepcopy(self.model.diagnostics),
            "likelihood":"unmodified_exact_plus_interval_EP",
            "tail_moments":"explicit_stable_extreme",
            "evidence_interpretation":"approximate EP evidence, not calibration"}


def _probability_feasible(pred):
    mean=np.asarray(pred["mean"],float)
    variance=np.asarray(pred["variance"],float)
    if np.any(variance<0) or not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise ModelNumericalFailure("soc_dualview_fusion",[
            {"phase":"feasibility","error_message":"Nonfinite Gaussian moments"}])
    z=np.divide(-mean,np.sqrt(variance),out=np.zeros_like(mean),where=variance>0)
    return np.where(variance>0,ndtr(z),(mean<=0).astype(float))


def conservative_soc_moments(onehot,latent):
    """Pick the lower P(g<=0) view for each candidate (zero observation leakage).

    The original acquisition accepts Gaussian mean/variance. Selecting a
    view-specific pair retains a proper Gaussian component at each query;
    it does not claim the piecewise policy is a coherent joint GP or calibrated.
    """
    p_one=_probability_feasible(onehot)
    p_lat=_probability_feasible(latent)
    if p_one.shape!=p_lat.shape:
        raise ValueError("Dual risk predictions have incompatible shapes")
    use_lat=p_lat<p_one
    return {
        "mean":np.where(use_lat,latent["mean"],onehot["mean"]),
        "variance":np.where(use_lat,latent["variance"],onehot["variance"]),
        "selected_latent":use_lat.astype(int),
        "view_disagreement_feasibility":np.abs(p_lat-p_one),
    }


class HeterogeneousRiskPosterior(RefinementPosterior):
    """Prefix-only heterogeneous view allocation, no target leakage."""
    def __init__(self, problem, records, seed, settings, objective_features,
                 risk_features, *, dual_soc=False, view_policy=None):
        if settings.average_kernel:
            raise ValueError("This registered experiment uses kernel evidence selection")
        if any(r.observation.contract_id!=problem.contract_id for r in records):
            raise ValueError("Heterogeneous posterior training contract changed")
        self.problem=problem
        self.features=objective_features
        self.objective_features=objective_features
        self.risk_features=risk_features
        self.view_policy=view_policy or FEATURE_VIEW_POLICY
        self.dual_soc=bool(dual_soc)
        self.history_digest=digest([r.to_dict() for r in records])
        x_obj=objective_features.transform([r.candidate for r in records])
        x_risk=risk_features.transform([r.candidate for r in records])
        packets=[r.observation for r in records]
        nc=len(problem.space.continuous)

        def fit(x,obs,name,offset,**kwargs):
            return StableGroupedHead(x,obs,continuous_count=nc,
                seed=seed+offset,name=name,
                continuous_grid=settings.continuous_grid,
                discrete_grid=settings.discrete_grid,**kwargs)

        def supported(packet,value,name):
            if packet.status in {"unknown","infrastructure_error"}:
                return OutputObservation("missing",packet.evidence_id+":"+name,
                                         reason="untrusted_terminal_status")
            return value
        labels=[]
        for packet in packets:
            completion=packet.completion_label()
            labels.append(OutputObservation("missing",packet.evidence_id+":completion",
                    reason="untrusted_terminal_status") if completion is None else
                OutputObservation("interval",packet.evidence_id+":completion",upper=0.)
                if completion else
                OutputObservation("interval",packet.evidence_id+":completion",
                                  lower=0.,lower_inclusive=False))
        self.completion=fit(x_risk,labels,"completion",0,completion=True)
        self.constraints={}
        self.dual_soc_head=None
        for number,name in enumerate(problem.required_constraints):
            if name in problem.analytical_constraints:
                continue
            data=[supported(p,p.constraint(name),name) for p in packets]
            self.constraints[name]=fit(x_risk,data,"constraint:"+name,number+1)
            if self.dual_soc and name=="soc":
                self.dual_soc_head=fit(x_obj,data,"constraint:soc:latent",number+1)
        outputs=[supported(p,p.objective,"objective") for p in packets]
        self.objective=fit(x_obj,outputs,"objective",100,objective=True,
            reference_scale=getattr(problem,"objective_reference_scale",1.)) if any(
                v.kind!="missing" for v in outputs) else None
        self.best=min((r.observation.objective.value for r in records
            if r.usable_feasible and r.observation.objective.kind=="exact"),default=None)

    def predict(self,candidates):
        xo=self.objective_features.transform(candidates)
        xr=self.risk_features.transform(candidates)
        forecasts={}
        for name,head in self.constraints.items():
            pred=head.predict(xr)
            if name=="soc" and self.dual_soc_head is not None:
                guarded=conservative_soc_moments(pred,self.dual_soc_head.predict(xo))
                pred=guarded
            forecasts[name]={**pred,"scale":head.scale}
        return {"completion":self.completion.predict(xr),
                "constraints":forecasts,
                "objective":None if self.objective is None else self.objective.predict(xo)}

    def diagnostics(self):
        heads={"completion":self.completion.diagnostics()}
        heads.update({name:head.diagnostics() for name,head in self.constraints.items()})
        if self.dual_soc_head is not None:
            heads["soc:latent_guard"]=self.dual_soc_head.diagnostics()
        if self.objective is not None:
            heads["objective"]=self.objective.diagnostics()
        return {"model":"risk_decoupled_dualview_v1","history_digest":self.history_digest,
                # Preserve the frozen corpus-checkpoint audit contract: the
                # objective VAE/feature snapshot is the persistent one.
                "feature_identity":self.objective_features.identity,
                "objective_feature_identity":self.objective_features.identity,
                "risk_feature_identity":self.risk_features.identity,
                "view_policy":self.view_policy,"soc_dual_guard":self.dual_soc,
                "soc_dual_rule":"min_per_candidate_P_g_le_0" if self.dual_soc else None,
                "heads":heads,"best_observed":self.best}


def _registered_failure_proposal(engine, records, global_pool, iteration, error):
    """Deterministic fallback for model failure ONLY, not physical evaluation.

    No surrogate prior or invented constraints are substituted. A new
    candidate is still physically evaluated and all fallback occurrences
    become an explicit outcome of the prospective method.
    """
    if engine.pending is not None or iteration!=engine.next_iteration:
        raise ValueError("Fallback needs a clean, matching online iteration")
    pool,_,rejected=filter_near(engine.problem,global_pool,records)
    if not pool:
        raise RuntimeError("No legal novel candidates for registered failure coverage")
    index,rule=coverage_pick(engine.problem,pool,records,engine.seed,iteration)
    candidate=pool[index]
    cold=not any(r.usable_feasible for r in records)
    prefix=digest([r.to_dict() for r in records])
    metadata={
        "mode":"model_failure_coverage","coverage_reason":"model_failure",
        "coverage_rule":rule,
        "model_failure":{"head":error.head,"candidate_failures":deepcopy(error.failures),
                         "policy":FALLBACK_TAG},
        "global_pool_size":len(global_pool),
        "novel_global_pool_size":len(pool),
        "augmented_pool_size":len(pool),
        "global_near_rejections":rejected,
        "local_reports":[],"local_enabled":False,
        "coverage_scope":"cold","coverage_active":coverage_enabled("cold",cold),
        "coverage_scheduled":False,
        "posterior_fit_skipped":False,
        "posterior_fit_failed":True,
        "selected_index":index,"pool":pool,
        "selected_from_local_augmentation":False,
        "probability_guarantee":False,
    }
    return engine._finish(candidate,records,metadata,prefix)


class StableOnehotDiscoveryBO(DiscoveryBO):
    """Frozen acquisition with stable-tail onehot heads and registered fallback."""
    def _build_posterior(self,records):
        one=PrefixFeatures(self.problem,self.base,records,self.seed,False,30)
        return HeterogeneousRiskPosterior(self.problem,records,self.seed,
            self.settings,one,one,view_policy="onehot_objective_and_risk")

    def propose(self,records,global_pool,iteration):
        try:
            return super().propose(records,global_pool,iteration)
        except ModelNumericalFailure as error:
            return _registered_failure_proposal(self,records,global_pool,iteration,error)


class StableCorpusDiscoveryBO(CorpusDiscoveryBO):
    """Corpus objective and risk features, optionally risk-decoupled/guarded."""

    def __init__(self, problem, base, seed, settings, output, *, allocation):
        if allocation not in {"latent_all","onehot_risk","soc_dual_guard"}:
            raise ValueError("Unregistered risk-view allocation")
        self.allocation=allocation
        super().__init__(problem,base,seed,settings,output)

    def _build_posterior(self, records):
        latent=feature_snapshot(self.problem,self.base,records,self.session)
        risk=(latent if self.allocation=="latent_all" else
              PrefixFeatures(self.problem,self.base,records,self.seed,False,30))
        return HeterogeneousRiskPosterior(self.problem,records,self.seed,
            self.settings,latent,risk,dual_soc=self.allocation=="soc_dual_guard",
            view_policy=self.allocation)

    def propose(self,records,global_pool,iteration):
        try:
            return super().propose(records,global_pool,iteration)
        except ModelNumericalFailure as error:
            # CorpusDiscoveryBO catches and marks all exceptions fatal.
            # ONLY this preregistered head-numerical failure is recoverable.
            self.failed=False
            candidate,meta=_registered_failure_proposal(
                self,records,global_pool,iteration,error)
            self.last_requested_iteration=iteration
            prefix=digest([r.to_dict() for r in records])
            # Even when no GP survives, bind the representation snapshot
            # that was actually used for that failed proposal. The immutable
            # checkpoint auditor must not mistake the fallback for a fit skip.
            meta["feature_identity"]=feature_snapshot(
                self.problem,self.base,records,self.session).identity
            meta["corpus_training"]={
                "schema":CORPUS_TRAINING_SCHEMA,
                "corpus_identity":self.session.corpus["identity"],
                "draws":STRUCTURE_COUNT,
                "latest":deepcopy(self.latest_training),
                "fit_count":len(self.session.events),
                "cumulative_gradient_updates":self.session.total_updates,
                "cumulative_fit_seconds":float(sum(self.session.fit_seconds)),
                "current_gp_prefix":prefix,
                "representation_fit_on_this_proposal":iteration%8==0,
                "gp_fit_skipped":False,
                "representation_fit_included_in_proposal_time":True,
                "reconstruction_controls_selection":False}
            return candidate,meta
