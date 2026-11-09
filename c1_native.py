"""Reusable native heads; frozen and updated arms differ only in training prefix."""
import time
from copy import deepcopy
from unittest.mock import patch
from c1_protocol import digest, information


class Posterior:
    def __init__(self, heads, identity):
        self.heads, self.identity = heads, identity

    def predict(self, queries, bill):
        import numpy as np
        from scipy.special import ndtr
        from risk_learning.core import features
        from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import ModelNumericalFailure
        bill['prediction_started'] += 1
        try:
            q = np.asarray([features(c) for c in queries])
            moments = {name: h.predict(q) for name,h in self.heads.items()}
            probabilities, gaussian = {}, {}
            for name in ('region','resource'):
                m,v = moments[name]['mean'],moments[name]['variance']
                p = ndtr(np.divide(-m,np.sqrt(v),out=np.zeros_like(m),where=v>0))
                p = np.where(v>0,p,(m<=0).astype(float))
                probabilities[name] = p.tolist()
                gaussian[name] = dict(mean=m.tolist(),variance=v.tolist())
            probabilities['joint'] = [a*b for a,b in zip(probabilities['region'],probabilities['resource'])]
            objective = None
            if 'objective' in moments:
                objective = {k:moments['objective'][k].tolist() for k in ('mean','variance')}
            result = dict(model_id=self.identity,candidate_ids=[digest(c) for c in queries],
                          probabilities=probabilities,gaussian=gaussian,objective=objective)
        except ModelNumericalFailure as error:
            bill['prediction_errors'] += 1
            return None, dict(kind='ModelNumericalFailure',message=str(error),details=error.failures)
        except Exception:
            bill['prediction_errors'] += 1
            raise
        bill['prediction_completed'] += 1
        return result,None


def fit(rows, seed, bill, *, risk_only=False):
    """Only observable rows; no truth, evaluator, task name or output directory."""
    import numpy as np
    from risk_learning.core import features
    from risk_learning.native import training_payload
    from ra_lsbo.experiments.simplified_aircraft import dualview_optimizer as native
    from ra_lsbo.risk_core.observations import OutputObservation
    start=time.perf_counter()
    public=deepcopy(rows)
    risk_payload=training_payload(public,'native_censored')
    objective_payload=[dict(candidate=r['candidate'],candidate_id=r['candidate_id'],objective=r['objective']) for r in public]
    receipt=dict(schema='c1_native_fit_v1',seed=seed,risk_only=risk_only,
        training_sha256=digest(public),training_information=information(public),
        risk_input_sha256=digest(risk_payload),objective_input_sha256=digest(objective_payload),
        privileged_training=False,heads={},failure=None)
    heads={}; original=native.EvidenceGP
    def counted(*args,**kwargs):
        bill['gp_kernel_started']+=1
        try:
            model=original(*args,**kwargs)
        except Exception:
            bill['gp_kernel_errors']+=1
            raise
        bill['gp_kernel_completed']+=1
        return model
    names=['region','resource']
    exact=[r for r in public if r['objective']['kind']=='exact']
    if exact and not risk_only:
        names.append('objective')
    for index,name in enumerate(names):
        subset=exact if name=='objective' else risk_payload
        x=np.asarray([features(r['candidate']) for r in subset])
        observations=[OutputObservation(evidence_id=digest(['c1',name,r['candidate_id']]),**r[name]) for r in subset]
        bill['gp_head_started']+=1
        try:
            with patch.object(native,'EvidenceGP',counted):
                head=native.StableGroupedHead(x,observations,continuous_count=3,
                    seed=seed+index,name=name,objective=(name=='objective'),reference_scale=1.)
        except native.ModelNumericalFailure as error:
            bill['gp_head_errors']+=1
            receipt.update(status='MODEL_NUMERICAL_FAILURE',failure=dict(head=name,message=str(error),details=error.failures))
            receipt['elapsed_seconds']=time.perf_counter()-start
            receipt['model_id']=None
            return None,receipt
        except Exception:
            bill['gp_head_errors']+=1
            raise
        bill['gp_head_completed']+=1
        diagnostic=head.diagnostics()
        bill['kernel_candidates_rejected']+=sum(c['status']!='ok' for c in diagnostic['candidates'])
        heads[name]=head;receipt['heads'][name]=diagnostic
    identity=digest(dict(training_sha256=receipt['training_sha256'],seed=seed,risk_only=risk_only,
                         heads={k:v['identity'] for k,v in receipt['heads'].items()}))
    receipt.update(status='OK',model_id=identity,elapsed_seconds=time.perf_counter()-start)
    if digest(rows)!=digest(public):
        raise ValueError('Training data mutation')
    return Posterior(heads,identity),receipt
