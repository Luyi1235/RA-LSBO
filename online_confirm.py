"""Fresh-seed C1 replication; original optimizer/model/evaluator stay unchanged."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import re
from unittest.mock import patch

PARENT = '62b46f7a30ab5c3475cd59b86c7f3ad51974a1da'
PILOT_RESULT = 'b3ebfec2c7771eef57919a0a96eb7a0223a266ed'
EXPERIMENT = 'c1r_fresh_seed_replication_v1'
PROTECTED = ('ra_lsbo','risk_learning','c1_protocol.py','c1_native.py','c1_study.py',
             'c1_verify.py','run_c1_online.py','check_c1_evidence.py',
             'b1_core.py','b1_native.py','b1_study.py','b1_verify.py')


def plan(profile):
    from c1_protocol import plan as prior_plan, TASKS, ARMS
    if profile not in ('ci','smoke','full'):
        raise ValueError('Unknown replication profile')
    spec=deepcopy(prior_plan(profile))
    block={'ci':910000,'smoke':920000,'full':930000}[profile]
    repeats=16 if profile=='full' else 1
    tasks=list(dict.fromkeys(u['task'] for u in spec['units']))
    units=[dict(task=t,repeat=r,seed=block+1000*TASKS.index(t)+32*r)
           for t in tasks for r in range(repeats)]
    n,h=len(units),spec['steps']
    spec.update(units=units,initial_calls=n*48,online_calls=n*len(ARMS)*h,
        total_math_calls=n*(48+len(ARMS)*h),trials=n*len(ARMS),
        max_gp_heads=n*(3+8*(h-1)),max_gp_kernels=8*n*(3+8*(h-1)),
        replication_id=EXPERIMENT,pilot_code_sha=PARENT,pilot_result_sha=PILOT_RESULT,
        primary_inference=dict(endpoint='k24 initial_warm final best feasible objective',
            contrast=['risk_ei_updated','risk_ei_frozen'],
            method='two-sided exact sign test on non-ties; Holm across four tasks',
            alpha=.05,family=list(TASKS),no_pooling_with_pilot=True,
            inference_only_for_full=True),
        safety_endpoint='paired online infeasible count; report alongside objective, no safety claim',
        task_generalization=False,method_parameters_changed=False)
    return spec


@contextmanager
def plan_scope():
    # Patch ONLY experiment planning in the original runner and verifier.
    # No selector, model, evaluator, history, incumbent or tolerance is changed.
    import c1_study
    import c1_verify
    with patch.object(c1_study,'plan',plan), patch.object(c1_verify,'plan',plan):
        yield


def source(expected):
    from risk_learning.io import git,source_receipt
    root=Path(__file__).resolve().parent
    receipt=source_receipt(root,expected)
    if git(root,'diff','--name-only',PARENT,expected,'--',*PROTECTED):
        raise ValueError('Frozen pilot implementation changed')
    receipt.update(replication_id=EXPERIMENT,pilot_code_sha=PARENT,
                   pilot_result_sha=PILOT_RESULT,original_implementation_unchanged=True)
    return receipt


def run(root,profile,expected):
    from c1_study import run as original_run
    receipt=source(expected)
    with plan_scope():
        return original_run(root,profile,receipt)


def verify(root,expected=None):
    from c1_verify import verify as original_verify
    from risk_learning.io import read
    receipt=read(Path(root)/'source_receipt.json')
    if (receipt.get('replication_id')!=EXPERIMENT or receipt.get('pilot_code_sha')!=PARENT
            or receipt.get('pilot_result_sha')!=PILOT_RESULT
            or receipt.get('original_implementation_unchanged') is not True):
        raise ValueError('Replication source identity differs')
    with plan_scope(), patch('risk_learning.core.truth',side_effect=RuntimeError('No truth in verify')):
        result=original_verify(root,expected)
    return dict(result,replication_id=EXPERIMENT)


def extract(root):
    from risk_learning.io import read
    from c1_protocol import feasible
    root=Path(root);spec=read(root/'plan.json');units=[]
    for u in spec['units']:
        prefix='units/%s/r%d'%(u['task'],u['repeat'])
        with (root/prefix/'records.jsonl').open(encoding='utf-8') as f:
            records={r['ref']:r['value'] for r in (json.loads(line) for line in f)}
        trials={};forecasts={};candidates={}
        for arm in spec['arms']:
            trials[arm]=records[prefix+'/trials/'+arm+'/result.json']
            forecasts[arm]=[];candidates[arm]=[]
            for t in range(spec['steps']):
                p=records[prefix+'/trials/%s/proposal_%03d.json'%(arm,t)]
                r=records[prefix+'/trials/%s/observation_%03d.json'%(arm,t)]
                candidates[arm].append(r['candidate'])
                ref=p['prediction_ref']
                pred=records[ref]['prediction'] if ref is not None else None
                forecasts[arm].append(None if pred is None else dict(
                    p=pred['probabilities']['joint'][p['decision']['selected_index']],
                    feasible=feasible(r['observed'])))
        units.append(dict(task=u['task'],repeat=u['repeat'],trials=trials,
                          selected_forecasts=forecasts,online_candidates=candidates))
    return spec,units


def analysis(root,output,expected,*,pilot=False,evidence_root=None):
    from risk_learning.io import read,write,seal,file_sha,git
    from online_confirm_stats import analyze_units
    if pilot:
        if evidence_root is None or git(evidence_root,'rev-parse','HEAD')!=PILOT_RESULT or git(evidence_root,'status','--porcelain'):
            raise ValueError('Wrong or dirty pinned pilot evidence')
        source(expected)
        root=Path(evidence_root)/'results/phase-c1-online-learning-20261109/full'
        from c1_verify import verify as original_verify
        with patch('risk_learning.core.truth',side_effect=RuntimeError('No truth in pilot review')):
            verification=original_verify(root,PARENT)
    else:
        verification=verify(root,expected)
    output=Path(output)
    if output.exists():
        raise FileExistsError('Analysis output exists; do not overwrite')
    spec,units=extract(root)
    report=analyze_units(units,spec,confirmatory=not pilot and spec['profile']=='full')
    report.update(status='COMPLETE_PILOT_READ_ONLY_REVIEW' if pilot else 'COMPLETE_C1R_ANALYSIS',
        analysis_code_sha=expected,input_result_sha=PILOT_RESULT if pilot else None,
        input_manifest_sha256=file_sha(Path(root)/'manifest.json'),input_plan=spec,
        historical_verification=verification,
        new_bill=dict(math_calls=0,gp_fits=0,vae_fits=0,fantasy_fits=0,aircraft_calls=0,rcaide_calls=0))
    output.mkdir(parents=True)
    write(output/'report.json',report)
    write(output/'trials.json',units)
    seal(output)
    return dict(status=report['status'],primary=report['primary'],
                verification=verification,new_bill=report['new_bill'])


def tamper(root,expected):
    import tempfile
    import shutil
    from risk_learning.io import read,write,seal
    from check_c1_evidence import check
    verify(root,expected)
    with plan_scope():
        result=check(root)
    extra=[]
    for which in ('replication_identity','replication_plan'):
        with tempfile.TemporaryDirectory() as td:
            dst=Path(td)/'copy';shutil.copytree(root,dst)
            p=dst/('source_receipt.json' if which=='replication_identity' else 'plan.json')
            value=read(p);value['replication_id']='tampered';p.unlink();write(p,value)
            (dst/'manifest.json').unlink();seal(dst)
            try:
                verify(dst,expected)
            except ValueError:
                extra.append(which)
            else:
                raise AssertionError('Accepted '+which)
    return dict(status='REJECTED_ALL_REPLICATION_TAMPERING',checks=result['checks']+extra,
                new_math_calls=0,new_gp_fits=0)
