"""C2 factorial protocol: independent risk-head and objective-head updates."""
import hashlib, json
TASKS=tuple("tradeoff_s%d"%i for i in range(4))
ARMS=("risk_frozen_objective_frozen","risk_updated_objective_frozen",
      "risk_frozen_objective_updated","risk_updated_objective_updated",
      "risk_only_updated","uniform_hash")
PROFILES={
 "ci":dict(tasks=(TASKS[0],TASKS[3]),repeats=1,steps=2,block=940000),
 "smoke":dict(tasks=TASKS,repeats=1,steps=3,block=950000),
 "full":dict(tasks=TASKS,repeats=16,steps=24,block=960000)}
INITIAL=48
POOL=256
CELLS=("00","10","01","11")

def digest(x):
 return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def plan(profile):
 if profile not in PROFILES: raise ValueError("Unknown profile")
 p=PROFILES[profile]
 units=[dict(task=t,repeat=r,seed=p["block"]+1000*TASKS.index(t)+32*r)
        for t in p["tasks"] for r in range(p["repeats"])]
 n=len(units);h=p["steps"]
 return dict(schema="c2_head_factorial_v1",profile=profile,units=units,tasks=list(p["tasks"]),
  arms=list(ARMS),initial=INITIAL,pool=POOL,steps=h,checkpoints=[k for k in (6,12,24) if k<=h] or [h],
  initial_calls=n*INITIAL,online_calls=n*len(ARMS)*h,total_math_calls=n*(INITIAL+len(ARMS)*h),
  max_gp_heads=n*(3+8*(h-1)),max_gp_kernels=8*n*(3+8*(h-1)),
  trials=n*len(ARMS),training_shared_once=True,same_design_across_arms_charged_per_arm=True,
  no_label_resampling=True,no_unselected_evaluation=True,new_vae_fits=0,new_fantasy_fits=0,
  aircraft_calls=0,rcaide_calls=0,seed_block=p["block"],
  primary=dict(cells=list(CELLS),endpoint="final best feasible objective",stratum="initial_warm",
   effects=["risk_main","objective_main","risk_objective_interaction"],
   sign_tests_per_task=3,holm_family=12,alpha=.05),
  limits=["Known-task mechanism attribution, not held-out-task generalization.",
   "Factorial marginal effects are intention-to-treat contrasts over adaptive trajectories.",
   "Missing any of four feasible outcomes makes that unit's objective factorial effects missing; never impute zero.",
   "Failure counts and exact objective labels are separate cost endpoints, not an unregistered safety penalty."])

def factorial(y00,y10,y01,y11):
 vals=(y00,y10,y01,y11)
 if any(v is None for v in vals): return dict(complete=False,risk=None,objective=None,interaction=None)
 if any(type(v) not in (int,float) for v in vals): raise ValueError("Finite objective outcomes required")
 if any(v!=v or abs(v)==float("inf") for v in vals): raise ValueError("Nonfinite objective")
 return dict(complete=True,
  risk=.5*((y10-y00)+(y11-y01)),
  objective=.5*((y01-y00)+(y11-y10)),
  interaction=(y11-y10)-(y01-y00))

def ei(mean,variance,incumbent):
 import math
 if any(type(x) not in (int,float) or not math.isfinite(x) for x in (mean,variance,incumbent)) or variance<0:
  raise ValueError("Invalid EI moments")
 d=incumbent-mean
 if variance==0:return max(0.,d)
 sd=math.sqrt(variance);z=d/sd
 return max(0.,d*.5*math.erfc(-z/math.sqrt(2.))+sd*math.exp(-.5*z*z)/math.sqrt(2.*math.pi))

def select_c2(arm,ids,available,prediction,incumbent,salt):
 if arm not in ARMS or not ids or len(ids)!=len(set(ids)) or not available:
  raise ValueError("Invalid C2 selection request")
 if any(type(i)is not int or not 0<=i<len(ids) for i in available) or len(set(available))!=len(available):
  raise ValueError("Invalid available indices")
 scores=[0.]*len(ids);mode="uniform_hash"
 if arm!="uniform_hash":
  if prediction is None:mode="model_failure_hash"
  elif arm=="risk_only_updated":
   scores=list(prediction["probabilities"]["joint"]);mode="risk_only"
  elif incumbent is None or prediction.get("objective") is None:
   mode="cold_shared_hash" if incumbent is None else "no_objective_model_hash"
  else:
   o=prediction["objective"]
   scores=[ei(m,v,incumbent)*q for m,v,q in zip(o["mean"],o["variance"],prediction["probabilities"]["joint"])]
   mode="risk_ei"
 tie=[digest(["c2_outcome_blind_tie_v1",salt,i]) for i in ids]
 j=min(available,key=lambda i:(-scores[i],tie[i],ids[i]))
 return dict(mode=mode,scores=scores,selected_index=j,selected_candidate_id=ids[j],incumbent=incumbent)

def bill():
 return {k:0 for k in ("math_started","math_settled","math_errors","gp_head_started",
 "gp_head_completed","gp_head_errors","gp_kernel_started","gp_kernel_completed",
 "gp_kernel_errors","kernel_candidates_rejected","prediction_started",
 "prediction_completed","prediction_errors","vae_fits","fantasy_fits","aircraft_calls","rcaide_calls")}
