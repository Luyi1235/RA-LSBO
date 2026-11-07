"""Fresh online pilot isolating outcome-adaptive continuation in two-step scores."""
import argparse
from pathlib import Path
import json
import subprocess
import time
from recourse_value_protocol import PARENT, ARMS, ADAPTIVE, FIXED, plan, summarize, check_receipt, verify
from recourse_value_core import require
from pr69_evidence_verify import fingerprint
from pr70_protocol import check_transition
from ra_lsbo.experiments.information_value_stage1.io import write_new, append, manifest
from ra_lsbo.experiments.information_value_stage3.runner import locked_source


def run(output, profile, expected_sha):
    root=Path(__file__).resolve().parent;out=Path(output).resolve()
    require(not out.exists() and out!=root and root not in out.parents, 'NEW external output required')
    source=locked_source(root,expected_sha)
    changes=subprocess.check_output(['git','-C',str(root),'diff','--name-status',PARENT,expected_sha],text=True).splitlines()
    require(changes and all(s.startswith('A\t') for s in changes), 'Frozen PR72 source changed')
    source.update(recourse_parent=PARENT,all_pr72_blobs_unchanged=True,no_historical_result_labels_read=True)
    spec=plan(profile)
    out.mkdir(parents=True,exist_ok=False)
    write_new(out/'plan.json',spec);write_new(out/'source_receipt.json',source)
    from ra_lsbo.experiments.information_value_stage1.runner import ChargedEvaluator, NativeAdapter, _legal, _best
    ledger=ChargedEvaluator(out/'evaluations.jsonl');trials=[];start=time.perf_counter()
    try:
        from ra_lsbo.experiments.information_value_stage3.native import FantasyValueBO
        from ra_lsbo.experiments.risk_budget63_local_value_engine import LocalCredibilityValueBO
        from ra_lsbo.experiments.discovery_stages.policies import filter_near
        from recourse_value_engine import RecourseValueBO
        api=NativeAdapter()
        with api.scope():
            for case in spec['cases']:
                for seed in spec['seeds']:
                    problem=api.problem(case);designs=api.initial(problem,case,seed)
                    _legal(problem,designs,[],12)
                    initial,tickets=[],[]
                    for j,c in enumerate(designs):
                        tickets.append(ledger.started)
                        initial.append(ledger.call(problem,c,{'phase':'initial','case':case,'seed':seed,'index':j}))
                    initial_hash=fingerprint([r.to_dict() for r in initial])
                    write_new(out/'shared_initial'/case/(str(seed)+'.json'),
                              {'records':[r.to_dict() for r in initial],'tickets':tickets,'hash':initial_hash})
                    for arm in ARMS:
                        folder=out/'trials'/case/str(seed)/arm;folder.mkdir(parents=True,exist_ok=False)
                        if arm=='soft_gp_eligible':
                            engine=LocalCredibilityValueBO(problem,api.base,seed,folder,variant='soft_gp',initial_count=12,failure_cap=6)
                        elif arm=='cei_myopic':
                            engine=FantasyValueBO(problem,api.base,seed,folder,arm=arm,horizon=spec['steps'])
                        else:
                            engine=RecourseValueBO(problem,api.base,seed,folder,case=case,horizon=spec['steps'],arm=arm)
                        history=list(initial);curve=[_best(history)];path=[];decisions=[];prev=None;checks=0;seconds=0.
                        for i in range(spec['steps']):
                            raw=api.pool(problem,case,seed,i,history);_legal(problem,raw,history,128)
                            protected=filter_near(problem,raw,history)[0]
                            checks+=check_transition(prev,raw,protected,i)
                            prefix=fingerprint([r.to_dict() for r in history]);t0=time.perf_counter()
                            c,meta=engine.propose(history,raw,i)
                            elapsed=time.perf_counter()-t0;seconds+=elapsed
                            require(c in raw and prefix==fingerprint([r.to_dict() for r in history]), 'Proposal/history changed')
                            require(meta['pool']==protected, 'Actual protected candidate pool differs')
                            decision=check_receipt(meta,c,i,spec['steps'],arm)
                            append(folder/'proposals.jsonl',{'iteration':i,'candidate':c,'candidate_hash':fingerprint(c),
                                'history_hash':prefix,'raw_pool':raw,'protected_pool':protected,'pool_hash':fingerprint(raw),
                                'metadata':meta,'proposal_seconds':elapsed,'outcome_unseen':True})
                            prev=meta.get('information_value_decision',{}).get('selected_future_support')
                            ticket=ledger.started
                            rec=ledger.call(problem,c,{'phase':'online','case':case,'seed':seed,'arm':arm,'iteration':i})
                            append(folder/'observations.jsonl',{'iteration':i,'ticket':ticket,'candidate_hash':fingerprint(c),'record':rec.to_dict()})
                            engine.accept(rec);history.append(rec);path.append(fingerprint(c));curve.append(_best(history))
                            decision.update(iteration=i,candidate_hash=fingerprint(c),nonfeasible=int(not rec.usable_feasible))
                            decisions.append(decision)
                        row={'case':case,'seed':seed,'arm':arm,'online_calls':spec['steps'],
                             'initial_hash':initial_hash,'initial_best':curve[0],'final_best':curve[-1],'curve':curve,
                             'path_hash':fingerprint(path),'decisions':decisions,'proposal_seconds':seconds,
                             'future_pool_checks':checks,'nonfeasible':sum(d['nonfeasible'] for d in decisions),
                             'fantasy_head_fits_started':sum(d['fantasy_head_fits_started'] for d in decisions),
                             'fantasy_head_fits_completed':sum(d['fantasy_head_fits_completed'] for d in decisions),
                             'fantasy_fallbacks':sum(d['fantasy_fallback'] for d in decisions),
                             'model_free_calls':sum(not d['forecast_available'] for d in decisions)}
                        write_new(folder/'result.json',row);trials.append(row)
                        append(out/'progress.jsonl',{k:row[k] for k in ('case','seed','arm','online_calls','final_best','nonfeasible','future_pool_checks')})
                        print('RECOURSE',case,seed,arm,'best',row['final_best'],'failed',row['nonfeasible'],flush=True)
                        del engine
        require(ledger.started==ledger.settled==spec['total_calls'],'Actual evaluation bill mismatch')
        report={'schema':spec['schema'],'profile':profile,'code_sha':expected_sha,
                'status':'COMPLETE_RECOURSE_SCORING_PILOT' if profile=='full' else 'COMPLETE_'+profile.upper()+'_TECHNICAL_ONLY',
                'actual_calls_started':ledger.started,'actual_calls_settled':ledger.settled,
                'initial_calls':spec['initial_calls'],'online_calls':spec['online_calls'],
                'aircraft_calls':0,'rcaide_calls':0,'trials':trials,'summary':summarize(trials,spec),
                'elapsed_seconds':time.perf_counter()-start,
                'limits':['Scoring ablation, not a forced two-call physical commitment.',
                          'Nonnegative finite-node premium is a mathematical identity, not empirical information benefit.',
                          'Four-node integration, four-root screen and fixed-kernel fantasies remain approximations.',
                          'Eight seeds per case are exploratory; no cross-case raw objective averaging.']}
        write_new(out/'report.json',report);manifest(out)
        return report
    except BaseException as error:
        import traceback
        write_new(out/'failure.json',{'code_sha':expected_sha,'type':type(error).__name__,'message':str(error),
            'traceback':traceback.format_exc(),'actual_calls_started':ledger.started,'actual_calls_settled':ledger.settled,
            'unknown_started_calls':ledger.started-ledger.settled,'completed_trials':len(trials)})
        manifest(out)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=('run','verify'),default='run')
    p.add_argument('--profile',choices=('ci','smoke','full'),default='smoke')
    p.add_argument('--expected-sha');p.add_argument('--output',type=Path);p.add_argument('--input',type=Path)
    a=p.parse_args()
    if a.task=='verify':
        if a.input is None:p.error('--input required')
        answer=verify(a.input)
    else:
        if a.output is None or a.expected_sha is None:p.error('--output and --expected-sha required')
        r=run(a.output,a.profile,a.expected_sha)
        answer={k:r[k] for k in ('status','code_sha','actual_calls_started','actual_calls_settled','aircraft_calls','rcaide_calls')}
    print(json.dumps(answer,indent=2),flush=True)


if __name__=='__main__':main()
