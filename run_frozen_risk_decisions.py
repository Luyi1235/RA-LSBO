"""Read-only Phase A reconciliation and Phase B0 frozen-ranking replay.

Defaults to plan. No new evaluations, fitting, online rollout or model tuning.
Requires the pinned PR75 result worktree, not a copied unversioned directory.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import subprocess
import traceback
from unittest.mock import patch

from decision_value_core import (BUDGETS, EVIDENCE, EVIDENCE_PATH, MODELS,
    NEW_BILL, PARENT, SCHEMA, SIZES, TASKS, aggregate, average, digest,
    fixed_plan, random_reference, rank_candidates, score_prefix)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def check_roots(code, evidence, expected):
    if not re.fullmatch('[0-9a-f]{40}', expected or ''):
        raise ValueError('Exact audit code SHA required')
    if git(code,'rev-parse','HEAD') != expected or git(code,'status','--porcelain'):
        raise ValueError('Wrong or dirty audit source worktree')
    if git(evidence,'rev-parse','HEAD') != EVIDENCE or git(evidence,'status','--porcelain'):
        raise ValueError('Wrong or dirty pinned input evidence worktree')
    if git(code,'diff','--name-only',PARENT,'HEAD','--','risk_learning','ra_lsbo'):
        raise ValueError('Frozen Phase A verifier/model/evaluator changed')


def reconcile(evidence, expected):
    from risk_learning.io import read, file_sha
    from risk_learning import core as original_core
    from risk_learning import study as original_study
    from risk_learning.core import label_counts

    root = evidence / EVIDENCE_PATH
    # Both old truth entry points are patched: even accidental calls fail.
    def forbidden(*args, **kwargs):
        raise RuntimeError('Phase B0 forbids all true evaluations')
    with ExitStack() as stack:
        stack.enter_context(patch.object(original_core,'truth',forbidden))
        stack.enter_context(patch.object(original_study,'truth',forbidden))
        verified = original_study.verify(root, PARENT)
    if (verified['model_rows'] != 360 or verified['failed_model_rows'] != 0 or
            verified['historical_bill']['math_settled'] != 12672):
        raise ValueError('Pinned completed Phase A roster differs')
    spec = read(root/'plan.json')
    if spec['profile'] != 'full':
        raise ValueError('Only the fixed full study may be replayed')
    rows, coverage = [], []
    for unit in spec['units']:
        task, repeat = unit['task'], unit['repeat']
        folder = root/'datasets'/task/('r%d' % repeat)
        train, test = read(folder/'train.json'), read(folder/'test.json')
        ids = [r['candidate_id'] for r in test]
        for n in SIZES:
            coverage.append(dict(task=task, repeat=repeat, n=n,
                training=label_counts([r['observed'] for r in train[:n]]),
                test=label_counts([r['observed'] for r in test]),
                test_boundary_count=sum(abs(r['truth']['region'])<=.25 for r in test)))
            for method in MODELS:
                source = read(root/'fits'/task/('r%d' % repeat)/('n%d' % n)/(method+'.json'))
                row = dict(task=task, repeat=repeat, n=n, method=method,
                    status=source['status'], privileged_training=(method=='privileged_exact'),
                    train_prefix_sha256=source['train_prefix_sha256'],
                    test_set_sha256=source['test_set_sha256'], metrics=None,
                    head_metrics=None, decisions=[], selection_receipt=None)
                if source['status'] != 'OK':
                    rows.append(row)
                    continue
                probs = source['predictions']['probabilities']['joint']
                # Complete ranking and its receipt are formed before scoring labels.
                order = rank_candidates(ids, probs)
                receipt = dict(ordered_candidate_ids=[ids[i] for i in order],
                               ordered_probabilities=[probs[i] for i in order],
                               selector_inputs_sha256=digest([ids,probs]))
                receipt['ranking_sha256'] = digest(receipt)
                row['selection_receipt'] = receipt
                labels = [r['truth']['feasible'] for r in test]
                for k in BUDGETS:
                    decision = score_prefix(order, labels, k)
                    decision.update(
                        selected_ids=[ids[i] for i in order[:k]],
                        selected_probabilities=[probs[i] for i in order[:k]],
                        predicted_mean_probability=average([probs[i] for i in order[:k]]),
                        observed_feasible_fraction=decision['feasible_selected']/k,
                        nominal_train_plus_selection_count=n+k,
                        new_evaluations=0,
                        random_reference=random_reference(len(test),sum(labels),k))
                    row['decisions'].append(decision)
                row['metrics'] = source['metrics']['all']['joint']
                row['head_metrics'] = {h: source['metrics']['all'][h] for h in ('region','resource')}
                rows.append(row)
    summary = aggregate(rows)
    winners = {}
    for task in TASKS:
        for n in SIZES:
            means = {m: summary['%s/n%d/%s' % (task,n,m)]['joint_metrics_mean']['brier'] for m in MODELS}
            valid = [v for v in means.values() if v is not None]
            winners['%s/n%d' % (task,n)] = [m for m,v in means.items() if v is not None and v==min(valid)] if valid else []
    report = dict(schema=SCHEMA, status='COMPLETE_FROZEN_RISK_DECISION_REPLAY',
        audit_code_sha=expected, input_result_sha=EVIDENCE, input_code_sha=PARENT,
        input_manifest_sha256=file_sha(root/'manifest.json'), input_verification=verified,
        source_model_rows=len(rows), replay_decision_rows=sum(len(r['decisions']) for r in rows),
        new_bill=dict(NEW_BILL), summary=summary, mean_brier_winners=winners,
        limits=[
            'Retrospective reuse of development pools, not fresh independent validation.',
            'Fixed risk-only ordering, no objective model, no online posterior updates.',
            'Budgets and n share data; only three repeat units per task, not independent candidate-level trials.',
            'Uniform random reference is an exact finite-pool expectation, not a newly run optimizer.',
            'privileged_exact uses extra TRAINING margins; it is not a true-test-feasibility oracle.',
            'Training n and hypothetical selected k are both stated; no equal-total-cost BO claim.',
            'No safety guarantee or automatic model promotion. Reserved tasks remain unused.'
        ])
    return report, rows, coverage


def validate_output(output, evidence, expected):
    from risk_learning.io import check_manifest, read
    from risk_learning.validation import metric_equal
    check_manifest(output)
    if (output/'failure.json').exists():
        raise ValueError('Failed replay is not completed evidence')
    if read(output/'plan.json') != fixed_plan():
        raise ValueError('Replay protocol changed')
    report, rows, coverage = reconcile(evidence, expected)
    for name, value in [('report.json',report),('decisions.json',rows),('label_coverage.json',coverage)]:
        if not metric_equal(read(output/name), value):
            raise ValueError('Raw-derived replay differs: '+name)
    return dict(status='VERIFIED_FROZEN_RISK_DECISIONS', input_result_sha=EVIDENCE,
                source_model_rows=len(rows), replay_decision_rows=report['replay_decision_rows'],
                new_bill=dict(NEW_BILL))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('plan','run','verify'),default='plan')
    parser.add_argument('--expected-sha')
    parser.add_argument('--evidence-root',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--input',type=Path)
    args=parser.parse_args()
    if args.task=='plan':
        print(json.dumps(fixed_plan(),indent=2,ensure_ascii=False));return
    if args.evidence_root is None:
        parser.error('--evidence-root is required')
    code=Path(__file__).resolve().parent
    evidence=args.evidence_root.resolve()
    check_roots(code,evidence,args.expected_sha)
    if args.task=='verify':
        if args.input is None: parser.error('--input is required')
        result=validate_output(args.input.resolve(),evidence,args.expected_sha)
    else:
        if args.output is None: parser.error('--output is required')
        out=args.output.resolve()
        if out.exists() or out==code or code in out.parents or out==evidence or evidence in out.parents:
            raise ValueError('Fresh output outside both source worktrees required')
        from risk_learning.io import write,seal
        out.mkdir(parents=True,exist_ok=False)
        write(out/'plan.json',fixed_plan())
        try:
            report,rows,coverage=reconcile(evidence,args.expected_sha)
            write(out/'report.json',report)
            write(out/'decisions.json',rows)
            write(out/'label_coverage.json',coverage)
            seal(out)
            result=dict(status=report['status'],source_model_rows=len(rows),
                        replay_decision_rows=report['replay_decision_rows'],new_bill=dict(NEW_BILL))
        except Exception as exc:
            write(out/'failure.json',dict(status='STOPPED_REPLAY_ERROR',error=str(exc),
                traceback=traceback.format_exc(),new_bill=dict(NEW_BILL)))
            seal(out)
            raise
    check_roots(code,evidence,args.expected_sha)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
