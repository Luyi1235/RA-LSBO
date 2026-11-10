"""Dependency-free held-out risk metrics for PR86."""
import math
def metrics(labels,probs,bins=10):
 if len(labels)!=len(probs):raise ValueError("Label/probability alignment")
 if any(type(y)is not bool for y in labels):raise ValueError("Boolean labels required")
 if any(not math.isfinite(p) or not 0<=p<=1 for p in probs):raise ValueError("Invalid probability")
 n=len(labels);pos=sum(labels);neg=n-pos
 if not n:return dict(n=0,positive=0,negative=0,brier=None,log_loss=None,auroc=None,auprc=None,ece=None,bins=[])
 eps=1e-15
 brier=sum((p-float(y))**2 for y,p in zip(labels,probs))/n
 ll=-sum(float(y)*math.log(max(p,eps))+(1-float(y))*math.log(max(1-p,eps)) for y,p in zip(labels,probs))/n
 auc=None
 if pos and neg:
  pp=[p for y,p in zip(labels,probs) if y];nn=[p for y,p in zip(labels,probs) if not y]
  auc=sum((a>b)+.5*(a==b) for a in pp for b in nn)/(pos*neg)
 prc=None
 if pos:
  order=sorted(range(n),key=lambda i:(-probs[i],i));tp=0;last=0.;area=0.
  for rank,i in enumerate(order,1):
   if labels[i]:
    tp+=1;area+=(tp/rank)*(1/pos)
  prc=area
 out=[];ece=0.
 for k in range(bins):
  ix=[i for i,p in enumerate(probs) if min(int(p*bins),bins-1)==k]
  if not ix:out.append(dict(bin=k,n=0,mean_probability=None,observed_rate=None));continue
  mp=sum(probs[i] for i in ix)/len(ix);rate=sum(labels[i] for i in ix)/len(ix)
  ece+=len(ix)/n*abs(mp-rate);out.append(dict(bin=k,n=len(ix),mean_probability=mp,observed_rate=rate))
 return dict(n=n,positive=pos,negative=neg,brier=brier,log_loss=ll,auroc=auc,
  auprc=prc,ece=ece,bins=out)
