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
  primary=dict(cells=CELLS,endpoint="final best feasible objective",stratum="initial_warm",
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
