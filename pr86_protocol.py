"""PR86-A frozen cold-start learnability protocol."""
import hashlib,json,math,random
from risk_learning.core import TASKS,digest,candidates,features,known_label
POLICIES=("iid_uniform","class_balanced","branch_stratified","coverage_maximin")
METHODS=("native_censored","privileged_exact")
SIZES=(12,24,48,96,192)
PROFILES={
 "ci":dict(tasks=(TASKS[0],TASKS[2]),repeats=1,sizes=(12,24),test=64,block=1100000),
 "smoke":dict(tasks=TASKS,repeats=1,sizes=(12,24,48),test=128,block=1110000),
 "full":dict(tasks=TASKS,repeats=3,sizes=SIZES,test=512,block=1120000)}
POOL=192

def plan(profile):
 if profile not in PROFILES:raise ValueError("Unknown PR86 profile")
 p=PROFILES[profile]
 units=[dict(task=t,repeat=r,train_seed=p["block"]+1000*i+32*r,
             test_seed=p["block"]+1000*i+32*r+1)
        for i,t in enumerate(TASKS) if t in p["tasks"] for r in range(p["repeats"])]
 n=len(units);sizes=list(p["sizes"]);models=n*len(POLICIES)*len(sizes)*len(METHODS)
 calls=n*(POOL+p["test"])
 return dict(schema="pr86_cold_start_learnability_v1",profile=profile,units=units,
  tasks=list(p["tasks"]),policies=list(POLICIES),methods=list(METHODS),
  sizes=sizes,train_pool=POOL,test_count=p["test"],
  train_pool_calls=n*POOL,test_calls=n*p["test"],total_math_calls=calls,
  model_rows=models,max_gp_heads=models*2,max_gp_kernels=models*16,
  seed_block=p["block"],nested_prefixes=True,shared_fixed_test_set=True,
  test_labels_fit_isolation=True,online_optimization=False,
  aircraft_calls=0,rcaide_calls=0,vae_fits=0,fantasy_fits=0,
  primary_metrics=["joint_brier","joint_auroc"],
  primary_method="native_censored",
  privileged_exact_role="diagnostic_information_upper_bound_only",
  limits=["Offline supervised learnability diagnostic; not an optimizer.",
   "All 192 training-pool labels and all held-out test labels are paid for.",
   "Sampling policies select from the same labelled 192-row training pool.",
   "Test labels never enter sampling, fitting, normalization, kernel selection or stopping.",
   "Policy contrasts are descriptive because class-balanced sampling uses observed training labels.",
   "No pooled cross-task raw margin RMSE."])

def _dist(a,b):
 x=features(a);y=features(b)
 return sum((u-v)**2 for u,v in zip(x,y))**.5

def sample_orders(pool,observed_rows,seed):
 if len(pool)!=POOL or len(observed_rows)!=POOL:raise ValueError("Expected fixed 192-row pool")
 ids=list(range(POOL));rng=random.Random(seed)
 iid=ids[:];rng.shuffle(iid)
 labels=[all(known_label(r,h) for h in ("region","resource")) for r in observed_rows]
 groups={k:[i for i,y in enumerate(labels) if y==k] for k in (False,True)}
 for g in groups.values():rng.shuffle(g)
 balanced=[]
 while groups[False] or groups[True]:
  for k in (True,False):
   if groups[k]:balanced.append(groups[k].pop())
 branches={b:[i for i,c in enumerate(pool) if c["branch"]==b] for b in range(4)}
 for g in branches.values():rng.shuffle(g)
 strat=[]
 while any(branches.values()):
  for b in range(4):
   if branches[b]:strat.append(branches[b].pop())
 # Deterministic maximin: outcome-blind geometry, fixed hashed first point.
 start=min(ids,key=lambda i:digest(["pr86-maximin-start",seed,i]))
 remaining=set(ids);remaining.remove(start);maximin=[start]
 min_d={i:_dist(pool[i],pool[start]) for i in remaining}
 while remaining:
  nxt=min(remaining,key=lambda i:(-min_d[i],digest(["pr86-maximin-tie",seed,i])))
  remaining.remove(nxt);maximin.append(nxt)
  for i in remaining:min_d[i]=min(min_d[i],_dist(pool[i],pool[nxt]))
 return dict(iid_uniform=iid,class_balanced=balanced,
             branch_stratified=strat,coverage_maximin=maximin),dict(
              observed_joint_feasible=sum(labels),observed_joint_infeasible=POOL-sum(labels),
              branch_counts={str(b):sum(c["branch"]==b for c in pool) for b in range(4)})
def coverage(rows):
 xs=[features(r["candidate"] if "candidate" in r else r) for r in rows]
 branches={str(b):sum((r["candidate"] if "candidate" in r else r)["branch"]==b for r in rows) for b in range(4)}
 if len(xs)<2:nearest=None
 else:
  ds=[]
  for i,x in enumerate(xs):
   ds.append(min(sum((a-b)**2 for a,b in zip(x,y))**.5 for j,y in enumerate(xs) if i!=j))
  nearest=dict(mean=sum(ds)/len(ds),median=sorted(ds)[len(ds)//2],maximum=max(ds))
 return dict(branch_counts=branches,branches_covered=sum(v>0 for v in branches.values()),
  nearest_neighbor_distance=nearest)
