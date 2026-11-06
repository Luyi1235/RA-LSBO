"""Offline quadrature diagnostic on pinned PR70 prefixes, NEVER new BO trials.

Load the original numerical VAE checkpoint, refit only prefix GP heads, require
four-node replay agreement, then change quadrature nodes at the SAME state.
All true evaluators and VAE training methods are blocked during reconstruction.
"""
import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback
from unittest.mock import patch

from quadrature_audit_protocol import BASE, EVIDENCE, RESULT, ARM, RULES, plan, key, require, close, summarize, fingerprint


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, encoding='utf-8').strip()


def locked(root, expected):
    require(isinstance(expected,str) and len(expected)==40 and all(c in '0123456789abcdef' for c in expected), 'Exact SHA required')
    require(git(root,'rev-parse','HEAD') == expected, 'Wrong worktree SHA')
    require(not git(root,'status','--porcelain','--untracked-files=normal'), 'Dirty worktree')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def serial(x):
    if isinstance(x, dict): return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [serial(v) for v in x]
    if hasattr(x,'tolist'): return x.tolist()
    return x


@contextmanager
def model_only_guard(bill, ceiling):
    """Observe real EvidenceGP constructions; prohibit true labels and training."""
    from ra_lsbo.partial_bo.models import EvidenceGP
    from ra_lsbo.partial_bo.problems import HierarchicalBranin, SimplifiedElectricAircraft
    from ra_lsbo.experiments.core.problems import MathematicalCase, AircraftCase
    from ra_lsbo.structure_corpus import CorpusRepresentation
    from ra_lsbo.structured_encoder import StructuredRepresentation
    old_init = EvidenceGP.__init__
    def init(obj,*a,**kw):
        require(bill['gp_started'] < ceiling, 'Registered GP-construction ceiling exceeded')
        bill['gp_started'] += 1
        stage=bill['phase']
        group=bill['by_phase'].setdefault(stage,{'started':0,'completed':0,'errors':0,'seconds':0.})
        group['started'] += 1
        t=time.perf_counter()
        try:
            old_init(obj,*a,**kw)
        except BaseException:
            bill['gp_errors'] += 1; group['errors'] += 1
            raise
        else:
            bill['gp_completed'] += 1; group['completed'] += 1
        finally:
            group['seconds'] += time.perf_counter()-t
    def no_evaluation(*a,**kw):
        bill['blocked_evaluator_attempts'] += 1
        raise RuntimeError('Diagnostic forbids ALL true evaluator calls')
    def no_training(*a,**kw):
        bill['blocked_vae_training_attempts'] += 1
        raise RuntimeError('Diagnostic must load checkpoint, NEVER train VAE')
    with ExitStack() as stack:
        stack.enter_context(patch.object(EvidenceGP,'__init__',init))
        for cls in (HierarchicalBranin,SimplifiedElectricAircraft,MathematicalCase,AircraftCase):
            stack.enter_context(patch.object(cls,'evaluate',no_evaluation))
        for cls in (CorpusRepresentation,StructuredRepresentation):
            stack.enter_context(patch.object(cls,'fit',no_training))
        yield


def restore(problem, base, seed, initial, history, checkpoint, metadata):
    from ra_lsbo.features import load_base
    from ra_lsbo.structured_encoder import hierarchy_space
    from ra_lsbo.structure_corpus import CorpusRepresentation, native_observations, feature_snapshot
    from ra_lsbo.partial_bo.models import PrefixFeatures
    from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import HeterogeneousRiskPosterior
    from ra_lsbo.experiments.risk_budget54_engine import settings_for_math54
    modules,_=load_base(base)
    space=hierarchy_space(modules,[asdict(v) for v in problem.space.variables])
    session=CorpusRepresentation.load_checkpoint(checkpoint,space,native_observations(initial,modules,space))
    require(session.seed==seed and len(session.events)==1, 'Wrong original VAE checkpoint')
    require(session._sealed_state==metadata['corpus_training']['latest']['parameter_state'], 'VAE state differs')
    obj=feature_snapshot(problem,base,history,session)
    risk=PrefixFeatures(problem,base,history,seed,False,30)
    require(obj.identity==metadata['objective_feature_identity'] and risk.identity==metadata['risk_feature_identity'],
            'Feature identity differs; do not silently retrain or relax replay')
    posterior=HeterogeneousRiskPosterior(problem,history,seed,replace(settings_for_math54(),horizon=6,local=False),
                                       obj,risk,view_policy='onehot_risk')
    diag=posterior.diagnostics()
    require(set(diag['heads'])==set(metadata['heads']), 'Head roster differs')
    for name, head in diag['heads'].items():
        for field in ('continuous_length','discrete_length'):
            require(head['selected'][field]==metadata['heads'][name]['selected'][field], 'Selected kernel differs')
        close(head['center'],metadata['heads'][name]['center'],'center:'+name)
        close(head['scale'],metadata['heads'][name]['scale'],'scale:'+name)
    return posterior,diag


def run(output, evidence_root, profile, expected_sha):
    from ra_lsbo.experiments.information_value_stage1.io import write_new, manifest, file_sha
    from pr70_protocol import verify as verify_input
    code=Path(__file__).resolve().parent
    ev=Path(evidence_root).resolve(); out=Path(output).resolve()
    locked(code,expected_sha);locked(ev,EVIDENCE)
    require(not out.exists() and out!=code and code not in out.parents and out!=ev and ev not in out.parents,
            'Output must be NEW and outside both code and evidence trees')
    git(code,'merge-base','--is-ancestor',BASE,expected_sha)
    changes=git(code,'diff','--name-status',BASE,expected_sha).splitlines()
    require(changes and all(x.startswith('A\t') for x in changes), 'All frozen PR70 files must be unchanged')
    source=ev/RESULT
    verified_before=verify_input(source)
    require(read(source/'report.json')['code_sha']==BASE, 'Input is not fixed PR70 code')
    spec=plan(profile)
    out.mkdir(parents=True,exist_ok=False)
    write_new(out/'plan.json',spec)
    write_new(out/'source_receipt.json',{'diagnostic_code_sha':expected_sha,'base_code_sha':BASE,
        'input_result_sha':EVIDENCE,'input_manifest_sha256':file_sha(source/'manifest.json'),
        'verified_input_before':verified_before,'python':platform.python_version(),'platform':platform.platform(),
        'added_source_sha256':{x.split('\t',1)[1]:file_sha(code/x.split('\t',1)[1]) for x in changes}})
    bill={'phase':'imports','gp_started':0,'gp_completed':0,'gp_errors':0,'by_phase':{},
          'blocked_evaluator_attempts':0,'blocked_vae_training_attempts':0,
          'new_true_evaluations':0,'new_vae_training':0,'aircraft_calls':0,'rcaide_calls':0}
    states=[];start=time.perf_counter()
    try:
        from ra_lsbo.partial_bo.records import DesignRecord
        from ra_lsbo.experiments.core.protocol import BASE as model_base
        from ra_lsbo.experiments.risk_budget63_runner import _case
        from pr70_future_pool import NextPoolFactory
        from quadrature_audit_numeric import compute_rule,verify_rule
        with model_only_guard(bill,spec['gp_construction_ceiling']):
            for state in spec['states']:
                case,seed,i=state['case'],state['seed'],state['iteration']
                folder=source/'trials'/case/str(seed)/ARM
                proposals=lines(folder/'proposals.jsonl'); observations=lines(folder/'observations.jsonl')
                initial=[DesignRecord.from_dict(x) for x in read(source/'shared_initial'/case/(str(seed)+'.json'))['records']]
                # Later observations are loaded ONLY by raw integrity checking;
                # model input is explicitly sliced at the original decision.
                history=initial+[DesignRecord.from_dict(x['record']) for x in observations[:i]]
                p=proposals[i];meta=p['metadata'];r=meta['information_value_decision']
                require(p['history_hash']==fingerprint([x.to_dict() for x in history]), 'Wrong prefix')
                require(r['depth']==2 and r['fallback'] is None and r['arm']==ARM,'Not a supported original two-step state')
                require(meta['corpus_training']['latest']['iteration']==0,'Unregistered checkpoint schedule')
                problem=_case(case)
                bill['phase']='prefix_gp_reconstruction'
                posterior,diag=restore(problem,model_base,seed,initial,history,folder/'representation'/'iteration_000',meta)
                candidates=meta['pool']
                forecast=posterior.predict(candidates)
                close(serial(forecast),meta['forecasts'],'original forecasts')
                close(posterior.best,meta['best_observed'],'original incumbent')
                target=out/'states'/('%s_%s_%02d'%(case,seed,i))
                factory=NextPoolFactory(posterior,candidates,problem,case,seed,i,history)
                supports={str(j):factory.support(j) for j in r['first_action_indices']}
                require(supports==r['future_supports'],'Original future supports did not replay')
                write_new(target/'snapshot.json',{'case':case,'seed':seed,'iteration':i,
                    'history_hash':p['history_hash'],'history':[x.to_dict() for x in history],
                    'best':posterior.best,'pool':candidates,'forecasts':serial(forecast),'heads':diag['heads'],
                    'first_indices':r['first_action_indices'],'future_supports':supports,
                    'original_selected_index':r['selected_index'],'original_scores':r['scores'],
                    'original_myopic_scores':r['myopic_scores'],'input_proposal_sha256':file_sha(folder/'proposals.jsonl')})
                row={**state,'best':posterior.best,'rules':{}}
                for n,replicate in RULES:
                    name=key(n,replicate);bill['phase']=name
                    f=NextPoolFactory(posterior,candidates,problem,case,seed,i,history)
                    value=compute_rule(forecast,posterior.best,f,seed=seed,iteration=i,n=n,replicate=replicate,
                                       first_indices=r['first_action_indices'])
                    verify_rule(value,posterior.best,r['first_action_indices'],{k:len(v['model_candidates']) for k,v in supports.items()})
                    write_new(target/(name+'.json'),value)
                    if (n,replicate)==(4,0):
                        require(value['status']=='OK','Original four-node rule failed to replay')
                        close(value['base_scores'],r['myopic_scores'],'original Q1')
                        close(value['scores'],{str(j):r['scores'][j] for j in r['first_action_indices']},'original Q2')
                        require(value['selected_index']==r['selected_index'],'Original four-node action changed')
                    row['rules'][name]={k:value.get(k) for k in ('status','selected_index','scores','first_indices',
                        'winner_gap','normalized_winner_gap','winner_near_tie_at_replay_tolerance',
                        'estimated_advantage_over_myopic','elapsed_seconds','fantasy_fits','numeric_error')}
                states.append(row)
                print('QUADRATURE',case,seed,i,{k:v['selected_index'] for k,v in row['rules'].items()},flush=True)
        require(bill['blocked_evaluator_attempts']==bill['blocked_vae_training_attempts']==0,'Forbidden call attempted')
        require(bill['gp_started']==bill['gp_completed']+bill['gp_errors'],'GP construction bill mismatch')
        locked(code,expected_sha);locked(ev,EVIDENCE)
        verified_after=verify_input(source)
        require(verified_before==verified_after,'Input evidence changed')
        report={'status':'COMPLETE_SAME_STATE_QUADRATURE_DIAGNOSTIC','diagnostic_code_sha':expected_sha,
                'profile':profile,'input_result_sha':EVIDENCE,'original_code_sha':BASE,
                'input_verify_before':verified_before,'input_verify_after':verified_after,
                'state_count':len(states),'states':states,'summary':summarize(states),'bill':bill,
                'elapsed_seconds':time.perf_counter()-start,
                'limits':['Fixed existing-policy prefixes, no new online outcomes.',
                    '64-node rule is not ground truth, convergence proof or a safety bound.',
                    'Two scrambled replicates give sensitivity, NOT a confidence interval.',
                    'Root GP refits occur; VAE checkpoint loading/forward passes are not training.',
                    'Model error, global first-action shortlist error and actual performance are not tested.']}
        write_new(out/'report.json',report);manifest(out)
        return report
    except BaseException as error:
        write_new(out/'failure.json',{'diagnostic_code_sha':expected_sha,'input_result_sha':EVIDENCE,
            'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),
            'completed_states':len(states),'bill':bill})
        manifest(out)
        raise


def verify(output):
    from ra_lsbo.experiments.information_value_stage1.io import verify as verify_manifest
    from quadrature_audit_numeric import verify_rule
    out=Path(output);hashes=verify_manifest(out)
    spec=read(out/'plan.json');report=read(out/'report.json');source=read(out/'source_receipt.json')
    require(spec==plan(spec['profile']),'Unregistered plan')
    require(report['status']=='COMPLETE_SAME_STATE_QUADRATURE_DIAGNOSTIC','Not complete')
    require(report['diagnostic_code_sha']==source['diagnostic_code_sha'],'Code identity mismatch')
    require(report['input_result_sha']==source['input_result_sha']==EVIDENCE,'Input identity mismatch')
    require(len(report['states'])==len(spec['states']),'Missing/extra diagnostic state')
    for row,expected in zip(report['states'],spec['states']):
        require({k:row[k] for k in expected}==expected,'State selection changed')
        folder=out/'states'/('%s_%s_%02d'%(row['case'],row['seed'],row['iteration']))
        snap=read(folder/'snapshot.json')
        require(snap['history_hash']==fingerprint(snap['history']),'Snapshot prefix hash mismatch')
        close(row['best'],snap['best'],'snapshot best')
        for n,replicate in RULES:
            name=key(n,replicate);r=read(folder/(name+'.json'))
            require(r['nodes']==n and r['replicate']==replicate,'Rule identity changed')
            verify_rule(r,snap['best'],snap['first_indices'],{k:len(v['model_candidates']) for k,v in snap['future_supports'].items()})
            require(row['rules'][name]=={k:r.get(k) for k in row['rules'][name]},'Rule summary mismatch')
            if (n,replicate)==(4,0):
                require(r['selected_index']==snap['original_selected_index'],'Baseline action mismatch')
                close(r['scores'],{str(i):snap['original_scores'][i] for i in snap['first_indices']},'baseline Q2')
    require(report['summary']==summarize(report['states']),'Summary mismatch')
    bill=report['bill']
    require(bill['gp_started']==bill['gp_completed']+bill['gp_errors'],'GP totals differ')
    for a,b in (('gp_started','started'),('gp_completed','completed'),('gp_errors','errors')):
        require(bill[a]==sum(x[b] for x in bill['by_phase'].values()),'GP phases do not reconcile')
    require(all(bill[k]==0 for k in ('blocked_evaluator_attempts','blocked_vae_training_attempts',
                  'new_true_evaluations','new_vae_training','aircraft_calls','rcaide_calls')),'Nonzero forbidden work')
    return {**hashes,'status':'VERIFIED_SAME_STATE_QUADRATURE','states':len(report['states']),
            'gp_fits_in_this_verification':0,'new_evaluations_in_this_verification':0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=('run','verify'),default='run')
    p.add_argument('--profile',choices=('smoke','full'),default='smoke')
    p.add_argument('--expected-sha');p.add_argument('--evidence-root',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--input',type=Path)
    a=p.parse_args()
    if a.task=='verify':
        if a.input is None:p.error('--input required')
        result=verify(a.input)
    else:
        if a.output is None or a.evidence_root is None or a.expected_sha is None:
            p.error('--output, --evidence-root and --expected-sha required')
        r=run(a.output,a.evidence_root,a.profile,a.expected_sha)
        result={k:r[k] for k in ('status','diagnostic_code_sha','profile','state_count','bill')}
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
