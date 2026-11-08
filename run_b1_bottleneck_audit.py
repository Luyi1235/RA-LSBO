"""Read-only, standard-library PR78 diagnostic. Default task only prints plan."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

from pr78_bottleneck_core import (SCHEMA, PARENT, EVIDENCE, EVIDENCE_PATH,
    BUDGETS, NEW_BILL, analyze_state, summarize, require)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def identities(evidence_root, expected):
    require(bool(re.fullmatch('[0-9a-f]{40}', expected or '')), 'Fixed audit SHA required')
    repo = Path(__file__).resolve().parent
    require(git(repo, 'rev-parse', 'HEAD') == expected and not git(repo, 'status', '--porcelain'),
            'Audit worktree SHA or cleanliness differs')
    require(git(evidence_root, 'rev-parse', 'HEAD') == EVIDENCE
            and not git(evidence_root, 'status', '--porcelain'), 'Fixed input worktree differs')
    protected = ('b1_core.py', 'b1_native.py', 'b1_study.py', 'b1_verify.py',
                 'run_objective_risk_value.py', 'check_b1_evidence.py', 'risk_learning', 'ra_lsbo')
    require(not git(repo, 'diff', '--name-only', PARENT, expected, '--', *protected),
            'Original model, evaluator or verifier source changed')
    return dict(audit_code_sha=expected, input_result_sha=EVIDENCE, input_code_sha=PARENT)


def rebuild(evidence_root, expected):
    from risk_learning.io import read, file_sha
    from risk_learning.validation import metric_equal
    from b1_verify import verify
    from risk_learning import core, study
    source = identities(evidence_root, expected)
    root = Path(evidence_root)/EVIDENCE_PATH
    before = file_sha(root/'manifest.json')
    def forbidden(*args, **kwargs):
        raise RuntimeError('Truth evaluation forbidden during read-only audit')
    with patch.object(core, 'truth', forbidden), patch.object(study, 'truth', forbidden):
        checked = verify(root, PARENT)
        spec = read(root/'plan.json')
        require(spec['profile']=='full' and checked['model_states']==144, 'Wrong input profile')
        details = []
        for u in spec['units']:
            name = '%s/r%d' % (u['task'], u['repeat'])
            pool = read(root/'datasets'/name/'pool.json')
            for n in spec['sizes']:
                state = read(root/'states'/name/('n%d.json' % n))
                source_score = read(root/'scores'/name/('n%d.json' % n))
                row = analyze_state(state, pool)
                if row['status']=='OK':
                    for k in BUDGETS:
                        for policy in ('uniform_hash','risk_only','objective_ei','risk_ei'):
                            recomputed = row['by_budget'][str(k)]['outcomes'][policy]
                            old = source_score['outcomes'][policy][str(k)]
                            for field in ('selected_ids','selected_feasible','selected_nonfeasible',
                                          'best_after','improvement','improved'):
                                require(metric_equal(recomputed[field], old[field]),
                                        'Independent original score mismatch: '+field)
                details.append(row)
    require(before == file_sha(root/'manifest.json'), 'Input evidence changed')
    require(not any(name in sys.modules for name in ('numpy','scipy','torch','gpytorch')),
            'Audit imported model dependencies')
    report = dict(schema=SCHEMA, status='COMPLETE_READ_ONLY_B1_BOTTLENECK_AUDIT',
                  **source, input_manifest_sha256=before, historical_verification=checked,
                  model_states=len(details), new_bill=dict(NEW_BILL),
                  summary=summarize(details),
                  limits=['Posthoc analysis on previously inspected development pools.',
                          'Truth substitutions are privileged, not deployable methods.',
                          'Known objective removes both objective error and uncertainty.',
                          'Information swaps and their interaction can be negative; not causal error shares.',
                          'No posterior updates or new online outcomes; do not infer Phase C success.',
                          'All n/k share data; preserve cold, failed and zero-opportunity states.'])
    return report, details


def run(evidence_root, output, expected):
    from risk_learning.io import write, seal
    root = Path(output).resolve()
    require(not root.exists(), 'Fresh write-once output required')
    report, details = rebuild(evidence_root, expected)
    root.mkdir(parents=True)
    write(root/'report.json', report)
    write(root/'states.json', details)
    write(root/'plan.json', plan())
    seal(root)
    return report


def verify_output(evidence_root, output, expected):
    from risk_learning.io import check_manifest, read
    from risk_learning.validation import metric_equal
    root = Path(output)
    require(check_manifest(root)==3, 'Unexpected output file count')
    require(set(read(root/'manifest.json')['files'])=={'report.json','states.json','plan.json'},
            'Unexpected output names')
    report, details = rebuild(evidence_root, expected)
    require(read(root/'plan.json')==plan(), 'Output plan changed')
    require(metric_equal(read(root/'report.json'),report), 'Output report not reproducible')
    require(metric_equal(read(root/'states.json'),details), 'Output states not reproducible')
    return dict(status='VERIFIED_B1_BOTTLENECK_AUDIT', audit_code_sha=expected,
                states=len(details), input_result_sha=EVIDENCE, new_bill=dict(NEW_BILL))


def tamper_checks(evidence_root, output, expected):
    from risk_learning.io import read, write, seal, file_sha
    original_hash = file_sha(Path(output)/'manifest.json')
    rejected = []
    for mode in ('source_identity','report_count','intervention_ranking','historical_score'):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp)/'copy'
            shutil.copytree(output, copy)
            rel = 'report.json' if mode in ('source_identity','report_count') else 'states.json'
            data = read(copy/rel)
            if mode=='source_identity': data['input_result_sha']='0'*40
            elif mode=='report_count': data['model_states']+=1
            else:
                row = next(r for r in data if r['status']=='OK' and r['initial_warm'])
                if mode=='intervention_ranking': row['rankings']['known_both'].reverse()
                else: row['by_budget']['6']['outcomes']['risk_ei']['best_after']+=1.
            (copy/rel).unlink(); write(copy/rel,data)
            (copy/'manifest.json').unlink(); seal(copy)
            try:
                verify_output(evidence_root, copy, expected)
            except ValueError:
                rejected.append(mode)
            else:
                raise AssertionError('Resealed output tampering accepted: '+mode)
    require(original_hash==file_sha(Path(output)/'manifest.json'), 'Original output changed')
    return dict(status='REJECTED_ALL_B1_AUDIT_TAMPERING', checks=rejected, new_bill=dict(NEW_BILL))


def plan():
    return dict(schema=SCHEMA, input_result_sha=EVIDENCE, input_code_sha=PARENT,
                input_path=EVIDENCE_PATH, input_profile='full', model_states=144,
                budgets=list(BUDGETS), analysis='posthoc_same_pool_truth_substitutions',
                primary='Retain original tradeoff n48/k6 warm setting; no new significance claim',
                new_bill=dict(NEW_BILL))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=('plan','run','verify','check'), default='plan')
    p.add_argument('--evidence-root', type=Path)
    p.add_argument('--expected-sha')
    p.add_argument('--output', type=Path)
    p.add_argument('--input', type=Path)
    a=p.parse_args()
    if a.task=='plan':
        result=plan()
    else:
        if a.evidence_root is None or a.expected_sha is None:
            p.error('--evidence-root and --expected-sha required')
        path=a.output if a.task=='run' else a.input
        if path is None: p.error('--output for run; --input for verify/check')
        result={'run':run,'verify':verify_output,'check':tamper_checks}[a.task](a.evidence_root,path,a.expected_sha)
        if a.task=='run':
            result={k:v for k,v in result.items() if k!='summary'}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


if __name__=='__main__':
    main()
