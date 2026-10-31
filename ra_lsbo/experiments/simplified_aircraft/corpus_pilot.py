"""Frozen five-arm aircraft corpus loop: prepare/run/audit/report only.

The 2000 structural draws are not model evaluations. The held-out 5 s check
cannot update any model or select another design. Old entrypoints stay unchanged.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from ...budget import digest
from ...partial_bo.problems import candidate_pool
from ...partial_bo.records import DesignRecord
from ..comparison.runner import evidence_hashes as json_evidence_hashes
from ..core.methods import settings_for
from ..core.metrics import trajectory_metrics, paired_comparisons
from ..core.protocol import BASE, ROOT, sha
from ..core.runner import Ledger, NATIVE_STOP, append, lines, plain, read, save
from ..discovery_stages.methods import Labelled, optimizer_for, specification
from ..discovery_stages.optimizer import DiscoveryBO
from ..discovery_stages.runner import validate_decision, discovery_summary
from ..discovery_stages.metrics import decision_diagnostics, lexicographic_pairs
from ..discovery_stages.phase_metrics import phase_diagnostics, warm_pairs
from . import reviewed_qualification as q
from .numerical_checks import qualification_report, sensitivity
from .transition import problem_for, runtime, physical_state, convergence

from .corpus_optimizer import LoggedControlBO, CorpusDiscoveryBO, encoder_settings, scheduled_updates
from .corpus_audit import verify_offline_archive, offline_receipt, audit_representation, finite_seconds

SCHEMA = 'ra_lsbo_aircraft_corpus_loop_v1'
DOCUMENT = 'docs/AIRCRAFT_CORPUS_LOOP_PROTOCOL_20261031.md'
CASES = ('aircraft_ordinary', 'aircraft_sparse')
ARMS = ('random', 'ra_old30', 'ra_old480', 'ra_corpus', 'ra_onehot')
REFERENCE = 'ra_corpus'
PROFILES = {
    'smoke': {'name':'smoke','seeds':[9920],'initial':6,'horizon':2,'pool':32,
              'encoder_epochs':30,'native_gp_steps':2},
    'pilot': {'name':'pilot','seeds':list(range(9921,9929)),'initial':12,'horizon':30,'pool':128,
              'encoder_epochs':30,'native_gp_steps':15},
}
# LF-normalized source identity only for cross-platform equation compatibility.
# Every actual experiment additionally seals RAW bytes, runtime and checkout.
MODEL_TEXT = {
    'ra_lsbo/partial_bo/problems.py':'96644d79df3c7a63d59acaaf3f195d60dc41146330216d1546b5381f0988881c',
    'ra_lsbo/experiments/core/problems.py':'2ddf35098d53c75d44eae7f96fff75407c3ab888d491cad432ee8e31c720a33a',
    'runs/compact_mda_20261021T081006Z_20eb608d/qualification_recovered_20261025/frozen_base/problems/x57/nasa_battery.py':
        '4eb98685920879a9fd2930135c0ddd7e1e903ab514f52cbe293cb23f93206a7d',
}


def method_spec(method, profile):
    if method not in ARMS:
        raise ValueError('Unregistered corpus-loop arm')
    if method == 'random':
        return specification('random', profile)
    settings = replace(settings_for('ra_lsbo', profile),
                       latent=method != 'ra_onehot',
                       encoder_epochs=480 if method == 'ra_old480' else 30)
    spec = {**specification('ev_latent',profile), 'settings':asdict(settings),
            'representation':method, 'normalization':'unchanged prefix padding',
            'feature_cost_logged':True}
    if method == 'ra_corpus':
        spec['corpus_training'] = {'count':2000,'with_replacement':True,
            'sampling':'uniform_canonical_with_replacement',
            'initial_and_updates':asdict(encoder_settings()),
            'update_at':scheduled_updates(profile['horizon']),
            'scheduled_even_when_GP_skipped':True,
            'fixed_corpus_reused':True,'new_evaluator_labels':False,
            'optimizer_momenta_retained':False}
    return spec


def engine_for(problem, method, seed, profile, output):
    spec = method_spec(method,profile)
    if method == 'random':
        return optimizer_for(problem,BASE,seed,'random',profile,output)
    settings = replace(settings_for('ra_lsbo',profile),
                       latent=method != 'ra_onehot',
                       encoder_epochs=480 if method == 'ra_old480' else 30)
    if method == 'ra_corpus':
        engine = CorpusDiscoveryBO(problem,BASE,seed,settings,output)
    else:
        engine = LoggedControlBO(problem,BASE,seed,settings,cold_global=True,
                                 protect=True,feedback='interval',coverage_scope='cold')
    return Labelled(engine,method)


def execution_order(plan,case,seed):
    index = CASES.index(case)*len(plan['profile']['seeds'])+plan['profile']['seeds'].index(seed)
    rotation = index % len(ARMS)
    return ARMS[rotation:]+ARMS[:rotation]


def evidence_hashes(folder):
    # Include numeric VAE checkpoints in addition to every JSON/JSONL record.
    folder = Path(folder)
    result = json_evidence_hashes(folder)
    result.update({p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*.npz'))})
    return dict(sorted(result.items()))


def representation_accounting(proposals, method):
    if method == 'random':
        return {'encoder_fit_calls':0,'gradient_updates':0,'encoder_fit_seconds':0.,
                'feature_preparation_seconds':0.,'gp_seconds':0.,'gp_fit_calls':0,
                'cost_interpretation':'component timings, not a controlled hardware benchmark'}
    if method == 'ra_corpus':
        infos=[p['metadata']['persistent_representation'] for p in proposals]
        last=infos[-1]
        return {'encoder_fit_calls':last['encoder_fit_calls'],
                'gradient_updates':last['cumulative_gradient_updates'],
                'encoder_fit_seconds':last['cumulative_fit_seconds'],
                'feature_preparation_seconds':sum(x['snapshot_seconds'] for x in infos),
                'gp_seconds':sum(x['gp_seconds'] or 0. for x in infos),
                'gp_fit_calls':sum(x['gp_seconds'] is not None for x in infos),
                'cost_interpretation':'cumulative training counted once; snapshot time excludes training'}
    infos=[p['metadata'].get('representation_fit') for p in proposals]
    infos=[x for x in infos if x is not None]
    return {'encoder_fit_calls':sum(x['encoder_fit_calls'] for x in infos),
            'gradient_updates':sum(x['gradient_updates'] for x in infos),
            'encoder_fit_seconds':None,
            'feature_preparation_seconds':sum(x['feature_seconds'] for x in infos),
            'gp_seconds':sum(x['gp_seconds'] for x in infos), 'gp_fit_calls':len(infos),
            'cost_interpretation':'feature time INCLUDES reset training; separate fit-only time unavailable'}


def primary_pairs(summaries):
    # Primary comparator per case: 5 s verified incumbent availability first,
    # then refined Wh at that same, nominally selected design. No re-selection.
    lookup={(r['case'],r['seed'],r['method']):r for r in summaries}
    result=[]
    for case in CASES:
        seeds=sorted({r['seed'] for r in summaries if r['case']==case})
        for other in ARMS:
            if other==REFERENCE:continue
            counts=Counter(); differences=[];percentages=[];pairs=[]
            for seed in seeds:
                left,right=lookup[case,seed,REFERENCE],lookup[case,seed,other]
                a,b=left['step_verification'],right['step_verification']
                av,bv=a['verified_usable'],b['verified_usable']
                if av and bv:
                    x,y=a['refined_objective_Wh'],b['refined_objective_Wh']
                    diff=x-y;differences.append(diff)
                    rel=100*diff/y if y!=0 else None
                    if rel is not None:percentages.append(rel)
                    verdict='tie' if np.isclose(x,y,rtol=1e-10,atol=1e-12) else 'win' if x<y else 'loss'
                else:
                    diff=rel=None
                    verdict='win' if av else 'loss' if bv else 'neither_verified'
                counts[verdict]+=1
                pairs.append({'seed':seed,'left_verified':av,'right_verified':bv,
                              'difference_Wh':diff,'relative_difference_pct':rel,'verdict':verdict})
            result.append({'case':case,'left':REFERENCE,'right':other,'n_all':len(seeds),
                'counts':dict(counts),'n_both_verified':len(differences),
                'median_difference_Wh':float(np.median(differences)) if differences else None,
                'median_relative_difference_pct':float(np.median(percentages)) if percentages else None,
                'pairs':pairs,'interpretation':'developmental paired endpoint, no significance claim'})
    return result

def budget_for(profile):
    trials = len(CASES)*len(profile['seeds'])*len(ARMS)
    roles = {'shared_initial':len(CASES)*len(profile['seeds'])*profile['initial'],
             'online':trials*profile['horizon'], 'incumbent_step_verification':trials}
    return {**roles,'maximum_model_calls':sum(roles.values()),'maximum_rcaide_mda_calls':0}


def build_plan(profile_name):
    if profile_name not in PROFILES:
        raise ValueError('Only smoke or exploratory pilot; no confirm, scenarios, resume or MDA')
    profile = deepcopy(PROFILES[profile_name])
    for relative,expected in MODEL_TEXT.items():
        if hashlib.sha256((ROOT/relative).read_bytes().replace(b'\r\n',b'\n')).hexdigest() != expected:
            raise ValueError('Reviewed aircraft equations changed; new qualification required')
    sources = set((ROOT/'ra_lsbo').rglob('*.py')) | set(BASE.rglob('*.py'))
    sources |= {ROOT/DOCUMENT, ROOT/'requirements-models.txt',
                ROOT/'tests/test_aircraft_corpus_loop.py', ROOT/'tests/test_aircraft_nominal_pilot.py',
                ROOT/'tests/test_aircraft_numerical_checks.py',ROOT/'tests/test_structure_corpus.py'}
    tasks = {}
    for case in CASES:
        problem = problem_for(case)
        task = {'settings':deepcopy(problem.settings),
                'contracts':{str(s):problem_for(case,s).contract_id for s in (20.,5.)},'initial':{}}
        for seed in profile['seeds']:
            # Same prospective candidate draws in both missions; no label-based rejection.
            pool,rejected = candidate_pool(problem,44000000+seed,profile['initial'])
            task['initial'][str(seed)] = {'candidates':pool,'exact_screen_rejections':rejected}
        tasks[case] = task
    version = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True)
    return plain({'schema':SCHEMA,'registered_utc':datetime.now(timezone.utc).isoformat(),
        'source_commit':version.stdout.strip() if version.returncode==0 else 'unversioned-checkout',
        'runtime':runtime(),'sources':{str(p):sha(p) for p in sorted(sources)},
        'profile':profile,'arms':ARMS,'reference':REFERENCE,'cases':CASES,'tasks':tasks,
        'methods':{m:method_spec(m,profile) for m in ARMS},'budget':budget_for(profile),
        'reviewed_qualification':q.receipt() if profile_name=='pilot' else None,
        'reviewed_offline_corpus':offline_receipt() if profile_name=='pilot' else None,
        'primary_endpoint':'5s verified availability then same fixed incumbent 5s energy, paired by seed',
        'primary_comparators':['ra_old480','ra_onehot'],
        'method_order':'cyclic rotation by case/seed index; all method random streams separately seeded',
        'structural_diagnostic_does_not_change_budget_or_select_method':True,
        'scope':'exploratory nominal optimization, no external-superiority or safety claims',
        'task_labels':{'aircraft_ordinary':'100 km ordinary','aircraft_sparse':'140 km constrained; historical sparse identifier'},
        'qualification_labels_available_to_optimizer':False,
        'verification':{'integration_step_s':5.,'fed_back_to_optimizer':False,'replacement_selection':False,
                        'relative_energy_screen':.01,'role':'deterministic numerical check, not physical validation'},
        'untrusted_policy':'stop whole campaign after journaling; no silent labels, exclusions or retries',
        'new_rcaide_mda_calls':0,'publication_confirmation':False})


def prepare(folder, profile='smoke', qualification=None, corpus_evidence=None):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError('Use a new write-once result directory')
    diagnostic = None
    if profile == 'pilot':
        if qualification is None or corpus_evidence is None:
            raise ValueError('Complete reviewed qualification AND fixed offline corpus archive required')
        verify_offline_archive(corpus_evidence)
        source, destination = Path(qualification).resolve(), folder.resolve()
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError('New output must be separate from historical qualification')
        diagnostic = qualification_report(qualification)
        if diagnostic['receipt'] != q.receipt():
            raise ValueError('Not the reviewed qualification receipt')
    elif qualification is not None or corpus_evidence is not None:
        raise ValueError('The smoke has no qualification-directory prerequisite')
    plan = build_plan(profile)
    folder.mkdir(parents=True); (folder/'source_snapshot').mkdir()
    for path,identity in plan['sources'].items():
        target=folder/'source_snapshot'/(identity+'.source')
        if not target.exists():target.write_bytes(Path(path).read_bytes())
    if diagnostic is not None:
        save(folder/'qualification_diagnostics.json', diagnostic)
        plan['qualification_diagnostics_sha256'] = sha(folder/'qualification_diagnostics.json')
    else:
        plan['qualification_diagnostics_sha256'] = None
    save(folder/'plan.json',plan)
    save(folder/'registration.json',{'plan_sha256':sha(folder/'plan.json'),
                                   'maximum_model_calls':plan['budget']['maximum_model_calls']})
    return plan


def load_plan(folder):
    folder=Path(folder);plan=read(folder/'plan.json')
    if plan.get('schema') != SCHEMA:
        raise ValueError('Not a new aircraft corpus loop plan')
    if read(folder/'registration.json') != {'plan_sha256':sha(folder/'plan.json'),
                                    'maximum_model_calls':plan['budget']['maximum_model_calls']}:
        raise ValueError('Registration changed')
    expected=build_plan(plan['profile']['name'])
    expected.pop('registered_utc')
    expected['qualification_diagnostics_sha256'] = plan.get('qualification_diagnostics_sha256')
    if plan['profile']['name']=='pilot':
        if sha(folder/'qualification_diagnostics.json') != plan.get('qualification_diagnostics_sha256'):
            raise ValueError('Qualification diagnostics changed')
        if read(folder/'qualification_diagnostics.json')['receipt'] != q.receipt():
            raise ValueError('Qualification receipt changed')
    elif plan.get('qualification_diagnostics_sha256') is not None:
        raise ValueError('Unregistered smoke diagnostics')
    if digest(expected) != digest({k:v for k,v in plan.items() if k!='registered_utc'}):
        raise ValueError('Registered code, environment, commit, configuration or receipt changed')
    for h in plan['sources'].values():
        if sha(folder/'source_snapshot'/(h+'.source'))!=h:raise ValueError('Snapshot changed')
    return plan


def ensure_trusted(row):
    if row.diagnostics.get('numerical_failure') or row.observation.status in {'unknown','infrastructure_error'}:
        raise RuntimeError('Untrusted outcome settled; preserve the campaign and stop without automatic retry')


def pool_for(problem, plan, case, seed, iteration, history):
    return candidate_pool(problem,440000000+seed*10000+CASES.index(case)*100000000+iteration,
                          plan['profile']['pool'],{r.observation.design_id for r in history})


def incumbent(history):
    # Chronological first minimum; validation never changes this choice.
    return min((r for r in history if r.usable_feasible),
               key=lambda r:r.observation.objective.value,default=None)


def verification_result(best, refined):
    if best is None:
        if refined is not None:raise ValueError('Cannot validate an invented incumbent')
        return {'status':'no_nominal_feasible_incumbent','verified_usable':False,'record':None}
    if refined is None or best.candidate!=refined.candidate:
        raise ValueError('Verification must evaluate the same fixed incumbent')
    exact_deltas={n:abs(v.value-refined.observation.constraint(n).value)
        for n,v in best.observation.constraints if v.kind=='exact' and refined.observation.constraint(n).kind=='exact'}
    return plain({'status':'evaluated','verified_usable':physical_state(refined)=='usable_feasible',
        'nominal_design_id':best.observation.design_id,'nominal_objective_Wh':best.observation.objective.value,
        'refined_objective_Wh':refined.observation.objective.value if refined.observation.objective.kind=='exact' else None,
        'screen':convergence([best,refined]),'absolute_constraint_differences':exact_deltas,
        'step_sensitivity':sensitivity([best,refined], (20.,5.)),
        'record':refined.to_dict(),'not_optimizer_feedback':True})


def summary_for(problem,plan,case,seed,method,initial,online,proposals,stop_reason,refined):
    best=incumbent(initial+online)
    return plain({'case':case,'seed':seed,'method':method,'initial_policy':'random',
        **trajectory_metrics(initial,online,plan['profile']['horizon'],stop_reason is not None),
        'stop_reason':stop_reason,'decision_diagnostics':decision_diagnostics(problem,initial,proposals,online),
        'representation_accounting':representation_accounting(proposals,method),
        'phase_diagnostics':phase_diagnostics(initial,proposals,online) if plan['methods'][method]['engine'] in {'discovery_bo_v1','phase_discovery_v1'} else None,
        'output_kinds':dict(Counter(n+':'+v.kind for r in online
            for n,v in [('objective',r.observation.objective),*r.observation.constraints])),
        'step_verification':verification_result(best,refined)})


def validate_stop(method,online,horizon,reason):
    if not 0<len(online)<=horizon:raise ValueError('Invalid horizon')
    stops=[];streak=0
    for i,r in enumerate(online):
        streak=0 if r.observation.objective.kind=='exact' else streak+1
        if streak>=5:stops.append(i)
    if reason is not None:
        if method!='original_lsbo' or reason!=NATIVE_STOP or not stops or stops[0]!=len(online)-1:
            raise ValueError('Unsubstantiated native stop')
    elif len(online)!=horizon or (method=='original_lsbo' and stops):
        raise ValueError('Incomplete trial or ignored native stop')


def manifest(plan,case,seed,method,initial):
    return plain({'case':case,'seed':seed,'method':method,'initial':[r.to_dict() for r in initial],
                  'initial_digest':digest([r.to_dict() for r in initial]),'specification':plan['methods'][method]})


def packet(plan,summaries,count,roles,timings):
    return plain({'schema':SCHEMA,'profile':plan['profile']['name'],'summaries':summaries,
        'discovery_summary':discovery_summary(summaries),'warm_pairs':warm_pairs(summaries),
        'primary_verified_pairs':primary_pairs(summaries),
        'paired_comparisons':paired_comparisons(summaries,reference=REFERENCE),
        'lexicographic_pairs':lexicographic_pairs(summaries,REFERENCE),'timings':timings,
        'charged_model_calls':count,'charged_by_role':dict(roles),
        'maximum_model_calls':plan['budget']['maximum_model_calls'],
        'unused_calls_not_reassigned':plan['budget']['maximum_model_calls']-count,
        'new_rcaide_mda_calls':0,'publication_confirmation':False})


def run(folder,approved_call_limit):
    folder=Path(folder);plan=load_plan(folder);profile=plan['profile']
    maximum=plan['budget']['maximum_model_calls']
    if type(approved_call_limit) is not int or approved_call_limit!=maximum:
        raise ValueError(f'Explicit complete-call cap required: {maximum}')
    fd=os.open(folder/'execution.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    ledger=Ledger(folder,maximum);summaries=[];timings=[]
    for case in CASES:
        problem=problem_for(case)
        for seed in profile['seeds']:
            initial=[]
            for i,c in enumerate(plan['tasks'][case]['initial'][str(seed)]['candidates']):
                row=ledger.evaluate(problem,c,f'{case}/{seed}/initial/{i}','shared_initial')
                initial.append(row);ensure_trusted(row)
            for method in execution_order(plan,case,seed):
                output=folder/'optimization'/case/f'{seed}_{method}';output.mkdir(parents=True)
                save(output/'manifest.json',manifest(plan,case,seed,method,initial))
                start=time.monotonic();engine=engine_for(problem,method,seed,profile,output)
                history=list(initial);online=[];proposals=[];proposal_times=[];accept_times=[]
                for iteration in range(profile['horizon']):
                    pool,rejected=pool_for(problem,plan,case,seed,iteration,history)
                    tick=time.monotonic();candidate,metadata=engine.propose(history,pool,iteration)
                    proposal_times.append(time.monotonic()-tick)
                    prop={'iteration':iteration,'candidate':candidate,'pool':pool,'exact_screen_rejections':rejected,
                          'prefix_digest':digest([r.to_dict() for r in history]),'metadata':metadata}
                    append(output/'proposals.jsonl',prop);proposals.append(prop)
                    row=ledger.evaluate(problem,candidate,f'{case}/{seed}/{method}/online/{iteration}','online')
                    append(output/'observations.jsonl',row.to_dict());online.append(row);ensure_trusted(row)
                    tick=time.monotonic();engine.accept(row);accept_times.append(time.monotonic()-tick)
                    history.append(row)
                    if engine.stop_reason:break
                validate_stop(method,online,profile['horizon'],engine.stop_reason)
                timing={'case':case,'seed':seed,'method':method,'optimization_seconds':time.monotonic()-start,
                        'proposal_seconds':proposal_times,'accept_seconds':accept_times,
                        'includes_validation':False,'includes_shared_initial':False}
                save(output/'timing.json',timing);timings.append(timing)
                best=incumbent(history);refined=None
                if best is not None:
                    refined=ledger.evaluate(problem_for(case,5.),best.candidate,
                           f'{case}/{seed}/{method}/verification/5','incumbent_step_verification')
                    save(output/'verification.json',refined.to_dict());ensure_trusted(refined)
                summary=summary_for(problem,plan,case,seed,method,initial,online,proposals,engine.stop_reason,refined)
                save(output/'summary.json',summary);summaries.append(summary)
                print(f"{case} {seed} {method}: online={len(online)}, nominal={summary['has_feasible']}, "
                      f"verified={summary['step_verification']['verified_usable']}, "
                      f"charged={ledger.count}/{maximum}", flush=True)
    load_plan(folder)
    result=packet(plan,summaries,ledger.count,ledger.roles,timings)
    save(folder/'results.json',result);return result


def audit(folder,write=True):
    folder=Path(folder);plan=load_plan(folder);before=evidence_hashes(folder)
    events=lines(folder/'calls.jsonl');settled={};roles=Counter();ids=set()
    if len(events)%2:raise ValueError('Unsettled call, retain evidence and do not retry')
    for number,(start,end) in enumerate(zip(events[::2],events[1::2]),1):
        call={k:start[k] for k in ('number','tag','role','candidate','contract','new_rcaide_mda')}
        row=DesignRecord.from_dict(end['record'])
        if (start['event']!='started' or end['event']!='settled' or start['number']!=number
            or start['call_id']!=digest(call) or end['call_id']!=start['call_id'] or start['call_id'] in ids
            or start['tag'] in settled or end['tag']!=start['tag'] or start['new_rcaide_mda'] is not False
            or row.candidate!=start['candidate'] or row.observation.design_id!=digest(row.candidate)
            or row.observation.contract_id!=start['contract'] or end['record_digest']!=digest(row.to_dict())
            or not np.isfinite(end['seconds']) or end['seconds']<0):
            raise ValueError('Ledger identity or order changed')
        settled[start['tag']]=(row,start['role']);roles[start['role']]+=1;ids.add(start['call_id'])
    if len(ids)>plan['budget']['maximum_model_calls'] or any(n>plan['budget'].get(role,-1) for role,n in roles.items()):
        raise ValueError('Budget exceeded or unknown role')
    claimed=set();summaries=[];timings=[];checked=0
    def claim(tag,role,problem,candidate):
        if tag not in settled or tag in claimed:raise ValueError('Missing or repeated registered call')
        row,actual=settled[tag]
        if (actual!=role or row.candidate!=candidate or problem.canonical(candidate)!=candidate
            or row.observation.contract_id!=problem.contract_id
            or any(not np.isfinite(v) or v>0 for v in problem.precheck(candidate).values())):
            raise ValueError('Candidate, role or task changed')
        ensure_trusted(row);claimed.add(tag);return row
    for case in CASES:
        problem=problem_for(case)
        for seed in plan['profile']['seeds']:
            initial=[claim(f'{case}/{seed}/initial/{i}','shared_initial',problem,c)
                     for i,c in enumerate(plan['tasks'][case]['initial'][str(seed)]['candidates'])]
            for method in execution_order(plan,case,seed):
                output=folder/'optimization'/case/f'{seed}_{method}'
                if read(output/'manifest.json')!=manifest(plan,case,seed,method,initial):raise ValueError('Manifest changed')
                proposals=lines(output/'proposals.jsonl');saved=read(output/'summary.json');history=list(initial);online=[]
                for i,prop in enumerate(proposals):
                    pool,rejected=pool_for(problem,plan,case,seed,i,history)
                    if (prop['iteration']!=i or prop['pool']!=pool or prop['exact_screen_rejections']!=rejected
                        or prop['prefix_digest']!=digest([r.to_dict() for r in history])):
                        raise ValueError('Proposal or common-pool stream changed')
                    row=claim(f'{case}/{seed}/{method}/online/{i}','online',problem,prop['candidate'])
                    if row.observation.design_id in {r.observation.design_id for r in history}:raise ValueError('Repeated design')
                    validate_decision(problem,plan,method,seed,history,prop)
                    history.append(row);online.append(row);checked+=1
                validate_stop(method,online,plan['profile']['horizon'],saved['stop_reason'])
                if lines(output/'observations.jsonl')!=plain([r.to_dict() for r in online]):raise ValueError('Observations changed')
                best=incumbent(history);refined=None
                if best is not None:
                    refined=claim(f'{case}/{seed}/{method}/verification/5','incumbent_step_verification',
                                  problem_for(case,5.),best.candidate)
                    if read(output/'verification.json')!=plain(refined.to_dict()):raise ValueError('Verification changed')
                elif (output/'verification.json').exists():raise ValueError('Invented incumbent verification')
                audit_representation(problem,BASE,output,initial,online,proposals,seed,method)
                summary=summary_for(problem,plan,case,seed,method,initial,online,proposals,saved['stop_reason'],refined)
                if saved!=summary:raise ValueError('Summary differs from raw evidence')
                summaries.append(summary)
                timing=read(output/'timing.json')
                if ({k:timing[k] for k in ('case','seed','method')}!={'case':case,'seed':seed,'method':method}
                    or timing['includes_validation'] is not False or timing['includes_shared_initial'] is not False
                    or any(len(timing[k])!=len(online) for k in ('proposal_seconds','accept_seconds'))
                    or any(not np.isfinite(x) or x<0 for x in [timing['optimization_seconds'],*timing['proposal_seconds'],*timing['accept_seconds']])):
                    raise ValueError('Invalid timing evidence')
                if sum(timing['proposal_seconds'])+sum(timing['accept_seconds']) > timing['optimization_seconds']+1e-6:
                    raise ValueError('Component timings exceed recorded optimization duration')
                timings.append(timing)
    if claimed!=set(settled) or roles['shared_initial']!=plan['budget']['shared_initial']:
        raise ValueError('Unclaimed or missing calls')
    if read(folder/'results.json')!=packet(plan,summaries,len(ids),roles,timings) or before!=evidence_hashes(folder):
        raise ValueError('Results or evidence changed')
    result={'passed':True,'settled_calls':len(ids),'trial_count':len(summaries),'checked_proposals':checked,
            'source_bindings':len(plan['sources']),'evidence_hashes':before,'algorithm_replay':False,
            'physical_validation':False,'confirmed_superiority':False,
            'numeric_checkpoint_restore':True,'encoder_retraining':False,'GP_refit':False}
    if write:
        if (folder/'audit.json').exists():
            if read(folder/'audit.json')!=result:raise ValueError('Existing audit seal changed')
        else:save(folder/'audit.json',result)
    return result


def report(folder):
    folder=Path(folder)
    if read(folder/'audit.json')!=audit(folder,False):raise ValueError('Fresh unchanged audit required')
    result=read(folder/'results.json');rows=result['summaries']
    text=['# Five-arm structural-corpus closed-loop aircraft pilot',
          '100/140 km are frozen mission settings. The old sparse identifier is not a below-10% claim.',
          'Eight new pilot seeds are development evidence, not independent confirmation. No MDA or scenario risk optimization.',
          '2000 repeated structural draws are unlabeled; not 2000 new designs or simulator calls.',
          '| Case | Method | Nominal feasible/all | Cold success/cold | 5 s verified/all | Conditional median Wh (n) |',
          '|---|---|---:|---:|---:|---:|']
    for item in result['discovery_summary']:
        group=[r for r in rows if r['case']==item['case'] and r['method']==item['method']]
        values=[r['best_feasible'] for r in group if r['has_feasible']]
        med=f'{np.median(values):.6f}' if values else 'NA'
        verified=sum(r['step_verification']['verified_usable'] for r in group)
        text.append(f"| {item['case']} | {item['method']} | {item['feasible_runs']}/{item['runs']} | "
                    f"{item['cold_discovered']}/{item['cold_n']} | {verified}/{len(group)} | {med} ({len(values)}) |")
    text += ['', '0/0 is an empty stratum, not zero probability. Verification never changes the nominal incumbent.',
             'All native stops, missing objectives and unused calls are retained. Paired objective analyses condition on both arms feasible.',
             'Full raw/phase diagnostics, timing and paired tables are in results.json.',
             f"Calls {result['charged_model_calls']}/{result['maximum_model_calls']}; new RCAIDE MDA 0."]
    text += ['', '## Paired warm final objective (left minus right)',
             '| Case | Left | Right | n | Better/tie/worse | Median difference Wh |',
             '|---|---|---|---:|---:|---:|']
    for item in result['warm_pairs']:
        text.append(f"| {item['case']} | {item['left']} | {item['right']} | {item['n']} | "
                    f"{item['better']}/{item['ties']}/{item['worse']} | {item['median_difference']} |")
    text += ['', '## Primary: 5 s verification then fixed-incumbent energy',
             '| Case | Left | Right | All | Both verified | Win/tie/loss/neither | Median relative difference % |',
             '|---|---|---|---:|---:|---:|---:|']
    for item in result['primary_verified_pairs']:
        c=item['counts']
        text.append(f"| {item['case']} | {item['left']} | {item['right']} | {item['n_all']} | "
                    f"{item['n_both_verified']} | {c.get('win',0)}/{c.get('tie',0)}/{c.get('loss',0)}/{c.get('neither_verified',0)} | "
                    f"{item['median_relative_difference_pct']} |")
    text += ['', 'Representation fit counts and cumulative timings are in every summary.',
             'old480 is a sufficient-training control, not matched FLOPs or cumulative update schedule.',
             'Training grammar round trips are diagnostics, not admission/exclusion criteria.',
             'Do not select task-specific winners or automatically authorize another campaign.']
    with (folder/'report.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(text)+'\n')
    return folder/'report.md'


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','audit','report'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',choices=('smoke','pilot'),default='smoke')
    parser.add_argument('--qualification',type=Path)
    parser.add_argument('--corpus-evidence',type=Path)
    parser.add_argument('--approved-call-limit',type=int)
    a=parser.parse_args(argv)
    if a.command=='prepare':
        p=prepare(a.output,a.profile,a.qualification,a.corpus_evidence);print({'profile':p['profile'],'budget':p['budget']})
    elif a.command=='run':
        r=run(a.output,a.approved_call_limit);print({k:r[k] for k in ('charged_model_calls','maximum_model_calls','new_rcaide_mda_calls')})
    elif a.command=='audit':print({k:v for k,v in audit(a.output).items() if k!='evidence_hashes'})
    else:print(report(a.output))


if __name__=='__main__':main()
