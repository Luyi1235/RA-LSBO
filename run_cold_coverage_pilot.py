"""Fresh matched cold-only coverage pilot. No historical labels or fantasies."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from cold_coverage_core import PARENT, CONTROL, TREATMENT, ARMS, plan, require, first_feasible, summarize
from cold_coverage_verify import check, verify
from pr69_evidence_verify import fingerprint
from ra_lsbo.experiments.information_value_stage1.io import write_new, append, manifest
from ra_lsbo.experiments.information_value_stage3.runner import locked_source


def run(output, profile, expected_sha):
    root = Path(__file__).resolve().parent; out = Path(output).resolve()
    require(not out.exists() and out != root and root not in out.parents, 'NEW external output required')
    source = locked_source(root, expected_sha)
    changes = subprocess.check_output(['git','-C',str(root),'diff','--name-status',PARENT,expected_sha],text=True).splitlines()
    require(changes and all(s.startswith('A\t') for s in changes), 'Frozen PR73 source changed')
    source.update(cold_coverage_parent=PARENT, all_pr73_blobs_unchanged=True, no_historical_result_labels_read=True)
    spec = plan(profile)
    out.mkdir(parents=True,exist_ok=False)
    write_new(out/'plan.json',spec); write_new(out/'source_receipt.json',source)
    from ra_lsbo.experiments.information_value_stage1.runner import ChargedEvaluator, NativeAdapter, _legal, _best
    ledger = ChargedEvaluator(out/'evaluations.jsonl'); trials = []; started = time.perf_counter()
    try:
        from ra_lsbo.experiments.information_value_stage3.native import FantasyValueBO
        from ra_lsbo.experiments.risk_budget63_local_value_engine import LocalCredibilityValueBO
        from ra_lsbo.experiments.discovery_stages.policies import filter_near
        from cold_coverage_engine import ColdCoverageBO
        api = NativeAdapter()
        with api.scope():
            for case in spec['cases']:
                for seed in spec['seeds']:
                    problem = api.problem(case); designs = api.initial(problem,case,seed)
                    _legal(problem,designs,[],12)
                    initial,tickets = [],[]
                    for j,c in enumerate(designs):
                        tickets.append(ledger.started)
                        initial.append(ledger.call(problem,c,{'phase':'initial','case':case,'seed':seed,'index':j}))
                    ih = fingerprint([r.to_dict() for r in initial])
                    write_new(out/'shared_initial'/case/(str(seed)+'.json'),
                        {'records':[r.to_dict() for r in initial],'tickets':tickets,'hash':ih})
                    for arm in ARMS:
                        folder = out/'trials'/case/str(seed)/arm; folder.mkdir(parents=True,exist_ok=False)
                        if arm == CONTROL:
                            engine = FantasyValueBO(problem,api.base,seed,folder,arm=CONTROL,horizon=spec['steps'])
                        elif arm == TREATMENT:
                            engine = ColdCoverageBO(problem,api.base,seed,folder,horizon=spec['steps'])
                        else:
                            engine = LocalCredibilityValueBO(problem,api.base,seed,folder,variant='soft_gp',initial_count=12,failure_cap=6)
                        history = list(initial); curve = [_best(history)]; path = []; decisions = []; seconds = 0.
                        for i in range(spec['steps']):
                            raw = api.pool(problem,case,seed,i,history); _legal(problem,raw,history,128)
                            protected = filter_near(problem,raw,history)[0]
                            prefix = fingerprint([r.to_dict() for r in history]); t0 = time.perf_counter()
                            c,meta = engine.propose(history,raw,i)
                            elapsed = time.perf_counter()-t0; seconds += elapsed
                            require(c in raw and prefix == fingerprint([r.to_dict() for r in history]) and
                                meta['pool'] == protected, 'Proposal/prefix/support mismatch')
                            decision = check(meta,c,[r.to_dict() for r in history],i,spec['steps'],arm)
                            append(folder/'proposals.jsonl',{'iteration':i,'candidate':c,'candidate_hash':fingerprint(c),
                                'history_hash':prefix,'raw_pool':raw,'protected_pool':protected,'pool_hash':fingerprint(raw),
                                'metadata':meta,'proposal_seconds':elapsed,'outcome_unseen':True})
                            ticket = ledger.started
                            rec = ledger.call(problem,c,{'phase':'online','case':case,'seed':seed,'arm':arm,'iteration':i})
                            append(folder/'observations.jsonl',{'iteration':i,'ticket':ticket,'candidate_hash':fingerprint(c),'record':rec.to_dict()})
                            engine.accept(rec); history.append(rec); path.append(fingerprint(c)); curve.append(_best(history))
                            decision.update(iteration=i,candidate_hash=fingerprint(c),nonfeasible=int(not rec.usable_feasible))
                            decisions.append(decision)
                        row = {'case':case,'seed':seed,'arm':arm,'online_calls':spec['steps'],
                            'initial_hash':ih,'initial_best':curve[0],'final_best':curve[-1],'curve':curve,
                            'first_feasible_call':first_feasible(curve),'path_hash':fingerprint(path),
                            'decisions':decisions,'proposal_seconds':seconds,
                            'model_free_calls':sum(not d['forecast_available'] for d in decisions)}
                        for field, term in (('nonfeasible','nonfeasible'),('scheduled_coverage_calls','scheduled_coverage'),
                            ('degenerate_coverage_calls','degenerate_coverage'),('model_failure_calls','model_failure'),('cold_decisions','cold_before')):
                            row[field] = sum(d[term] for d in decisions)
                        write_new(folder/'result.json',row); trials.append(row)
                        append(out/'progress.jsonl',{k:row[k] for k in ('case','seed','arm','online_calls','initial_best','final_best','first_feasible_call','nonfeasible')})
                        print('COLD-COVERAGE',case,seed,arm,'first',row['first_feasible_call'],'best',row['final_best'],flush=True)
                        del engine
        require(ledger.started == ledger.settled == spec['total_calls'], 'Evaluation bill mismatch')
        report = {'schema':spec['schema'],'profile':profile,'code_sha':expected_sha,
            'status':'COMPLETE_COLD_COVERAGE_PILOT' if profile == 'full' else 'COMPLETE_'+profile.upper()+'_TECHNICAL_ONLY',
            'actual_calls_started':ledger.started,'actual_calls_settled':ledger.settled,
            'initial_calls':spec['initial_calls'],'online_calls':spec['online_calls'],
            'aircraft_calls':0,'rcaide_calls':0,'fantasy_fits':0,'trials':trials,'summary':summarize(trials,spec),
            'elapsed_seconds':time.perf_counter()-started,
            'limits':['Matched diagnostic baseline, not a new risk-aware algorithm or safety claim.',
                'Primary denominator is initial-cold groups, not every seed. Retain all initialized groups.',
                'No cold-group resampling; if rare, report small denominator instead of enlarging posthoc.',
                'Same true calls need not have same GP cost; scheduled coverage can skip a posterior fit.',
                'Positive results do not prove a probability-calibration defect or information value.']}
        write_new(out/'report.json',report); manifest(out)
        return report
    except BaseException as error:
        import traceback
        write_new(out/'failure.json',{'code_sha':expected_sha,'type':type(error).__name__,'message':str(error),
            'traceback':traceback.format_exc(),'actual_calls_started':ledger.started,'actual_calls_settled':ledger.settled,
            'unknown_started_calls':ledger.started-ledger.settled,'completed_trials':len(trials)})
        manifest(out)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=('run','verify'),default='run')
    p.add_argument('--profile',choices=('ci','smoke','full'),default='smoke')
    p.add_argument('--expected-sha'); p.add_argument('--output',type=Path); p.add_argument('--input',type=Path)
    a = p.parse_args()
    if a.task == 'verify':
        if a.input is None: p.error('--input required')
        answer = verify(a.input)
    else:
        if a.output is None or a.expected_sha is None: p.error('--output and --expected-sha required')
        r = run(a.output,a.profile,a.expected_sha)
        answer = {k:r[k] for k in ('status','code_sha','actual_calls_started','actual_calls_settled','aircraft_calls','rcaide_calls','fantasy_fits')}
    print(json.dumps(answer,indent=2),flush=True)


if __name__ == '__main__': main()
