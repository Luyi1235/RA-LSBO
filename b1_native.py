"""One shared objective GP and one observable censored risk fit per prefix."""
import time
from unittest.mock import patch
from b1_core import digest, objective_payload


def fit_shared(rows, queries, seed, bill):
    from risk_learning.native import predict
    from risk_learning.core import features
    from ra_lsbo.experiments.simplified_aircraft import dualview_optimizer as native
    from ra_lsbo.risk_core.observations import OutputObservation
    import numpy as np

    started=time.perf_counter()
    risk, risk_receipt=predict(rows,queries,'native_censored',seed,bill)
    packet=objective_payload(rows)
    objective_receipt=dict(training_input_sha256=digest(packet),
        training_rows=len(rows), exact_objective_rows=sum(r['objective']['kind']=='exact' for r in rows),
        query_ids=[digest(c) for c in queries], privileged_training=False, head=None)
    receipt=dict(risk=risk_receipt,objective=objective_receipt,
                 same_objective_for_all_policies=True, no_objective_truth_imputation=True)
    if risk is None:
        return None,dict(receipt,status='MODEL_NUMERICAL_FAILURE',elapsed_seconds=time.perf_counter()-started)
    exact=[r for r in packet if r['objective']['kind']=='exact']
    objective=None
    if exact:
        X=np.asarray([features(r['candidate']) for r in exact])
        Q=np.asarray([features(c) for c in queries])
        observations=[OutputObservation('exact',digest(['b1_objective',r['candidate_id']]),
                                        value=r['objective']['value']) for r in exact]
        original=native.EvidenceGP
        def counted(*args,**kwargs):
            bill['gp_kernel_started']+=1
            try:
                fitted=original(*args,**kwargs)
            except Exception:
                bill['gp_kernel_errors']+=1
                raise
            bill['gp_kernel_completed']+=1
            return fitted
        bill['gp_head_started']+=1
        try:
            with patch.object(native,'EvidenceGP',counted):
                head=native.StableGroupedHead(X,observations,continuous_count=3,
                       seed=seed+2,name='b1_objective',objective=True,reference_scale=1.)
            moments=head.predict(Q)
        except native.ModelNumericalFailure as error:
            bill['gp_head_errors']+=1
            objective_receipt.update(error=str(error),failures=error.failures)
            return None,dict(receipt,status='MODEL_NUMERICAL_FAILURE',elapsed_seconds=time.perf_counter()-started)
        bill['gp_head_completed']+=1
        objective_receipt['head']=head.diagnostics()
        bill['kernel_candidates_rejected']+=sum(c['status']!='ok' for c in head.candidates)
        objective=dict(mean=moments['mean'].tolist(),variance=moments['variance'].tolist())
    objective_receipt['status']='OK' if exact else 'NO_OBSERVED_OBJECTIVE_LABELS'
    return dict(risk=risk,objective=objective),dict(receipt,status='OK',elapsed_seconds=time.perf_counter()-started)
