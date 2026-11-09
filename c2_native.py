"""Independent C2 posterior-head composer; never edits the frozen C1 fitter."""
from copy import deepcopy
import math
from unittest.mock import patch
from c1_native import Posterior
from c1_protocol import digest
from risk_learning.core import features
from risk_learning.native import training_payload

def _head(rows,name,seed,bill):
 from ra_lsbo.experiments.simplified_aircraft import dualview_optimizer as native
 from ra_lsbo.risk_core.observations import OutputObservation
 public=deepcopy(rows)
 if name in ("region","resource"):
  subset=training_payload(public,"native_censored")
 else:
  subset=[dict(candidate=r["candidate"],candidate_id=r["candidate_id"],objective=r["objective"])
          for r in public if r["objective"]["kind"]=="exact"]
  if not subset: return None,dict(status="NO_EXACT_OBJECTIVE")
 x=__import__("numpy").asarray([features(r["candidate"]) for r in subset])
 observations=[OutputObservation(evidence_id=digest(["c1",name,r["candidate_id"]]),**r[name]) for r in subset]
 exact=[r[name]["value"] for r in subset if r[name]["kind"]=="exact"]
 center=math.fsum(exact)/len(exact) if name=="objective" and exact else 0.
 if name=="objective" and len(exact)>1:
  scale=max(math.sqrt(math.fsum((v-center)**2 for v in exact)/len(exact)),.01)
 elif exact:
  v=sorted(abs(v) for v in exact);n=len(v);scale=max(v[n//2] if n%2 else .5*(v[n//2-1]+v[n//2]),.05)
 else: scale=1.
 original=native.EvidenceGP
 def counted(*a,**kw):
  bill["gp_kernel_started"]+=1
  try: result=original(*a,**kw)
  except Exception: bill["gp_kernel_errors"]+=1;raise
  bill["gp_kernel_completed"]+=1;return result
 bill["gp_head_started"]+=1
 try:
  with patch.object(native,"EvidenceGP",counted):
   head=native.StableGroupedHead(x,observations,continuous_count=3,seed=seed,
      name=name,objective=(name=="objective"),reference_scale=1.)
 except native.ModelNumericalFailure:
  bill["gp_head_errors"]+=1;raise
 bill["gp_head_completed"]+=1
 diag=head.diagnostics()
 bill["kernel_candidates_rejected"]+=sum(c["status"]!="ok" for c in diag["candidates"])
 if digest(rows)!=digest(public): raise ValueError("Head fitter mutated training rows")
 return head,dict(status="OK",training_sha256=digest(public),head=diag)

class CompositePosterior(Posterior):
 """Prediction composes two risk heads and an optional objective head."""
 def __init__(self,heads,identity,provenance):
  if not {"region","resource"}.issubset(heads) or set(heads)-{"region","resource","objective"}:
   raise ValueError("Composite requires both risk heads and at most one objective head")
  super().__init__(heads,identity);self.provenance=provenance

def fit_composite(rows,seed,bill,*,update_risk,update_objective,frozen=None):
 if type(update_risk)is not bool or type(update_objective)is not bool: raise ValueError("Boolean update factors required")
 if frozen is None and not (update_risk and update_objective):
  raise ValueError("Initial composite must fit both factors")
 heads={};provenance={}
 for index,name in enumerate(("region","resource")):
  if update_risk:
   h,r=_head(rows,name,seed+index,bill)
   heads[name]=h;provenance[name]=dict(mode="updated",training_sha256=r["training_sha256"],identity=r["head"]["identity"])
  else:
   if frozen is None: raise ValueError("Missing frozen risk head")
   heads[name]=frozen.heads[name];provenance[name]=dict(mode="frozen",identity=heads[name].identity)
 if update_objective:
  h,r=_head(rows,"objective",seed+2,bill)
  if h is not None: heads["objective"]=h
  provenance["objective"]=(dict(mode="updated",training_sha256=r["training_sha256"],identity=r["head"]["identity"])
                           if h is not None else dict(mode="unavailable_no_exact_labels"))
 else:
  if frozen is None: raise ValueError("Missing frozen objective state")
  h=frozen.heads.get("objective")
  if h is not None: heads["objective"]=h
  provenance["objective"]=(dict(mode="frozen",identity=h.identity) if h is not None
                           else dict(mode="frozen_unavailable"))
 if not heads["region"] or not heads["resource"]: raise ValueError("Composite missing risk head")
 if frozen is not None:
  for name,update in (("region",update_risk),("resource",update_risk),("objective",update_objective)):
   if not update:
    old=frozen.heads.get(name);new=heads.get(name)
    if new is not old: raise ValueError("Frozen head object was not reused: "+name)
 identity=digest(dict(training_sha256=digest(rows),seed=seed,update_risk=update_risk,
  update_objective=update_objective,heads={k:v.identity for k,v in heads.items()}))
 return CompositePosterior(heads,identity,provenance),dict(status="OK",model_id=identity,
  update_risk=update_risk,update_objective=update_objective,heads=provenance,
  training_sha256=digest(rows))
