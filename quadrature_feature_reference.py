"""Authenticated original-prefix normalization, not a guessed codebook alias.

The producer must reconstruct the EXACT pre-existing feature ID natively.
Consumers re-hash the complete original payload with authenticated offset/scale,
check local forward values at the UNCHANGED tolerance, and still replay kernels,
forecasts and Q1/Q2. The reference is derived evidence, never a new identity.
No expected-ID assignment, normalization search, evaluator call or VAE training.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import traceback

from quadrature_audit_protocol import (BASE, EVIDENCE, RESULT, ARM, RTOL, ATOL,
    plan, require, close, fingerprint, sha)

SCHEMA = 'pr72_exact_prefix_feature_reference_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def state_key(state):
    return '%s_%s_%02d' % (state['case'], state['seed'], state['iteration'])


def identity_payload(features, history, session):
    from ra_lsbo.structure_corpus import CORPUS_SCHEMA
    return {'schema': CORPUS_SCHEMA, 'prefix': [r.to_dict() for r in history],
            'corpus': session.corpus['identity'], 'settings': asdict(session.settings),
            'encoder': features.state_hash, 'encoder_prefix': digest(session.history),
            'offset': features.offset.tolist(), 'scale': features.scale.tolist()}


def validate_row(row):
    p = row['payload']
    require(set(p) == {'schema','prefix','corpus','settings','encoder','encoder_prefix','offset','scale'},
            'Feature payload roster differs')
    require(p['schema'] == 'ra_lsbo_structure_corpus_2000_v1', 'Wrong feature schema')
    require(digest(p) == row['recorded_objective_identity'] == row['native_objective_identity'],
            'Reference does not reconstruct the EXACT original feature identity')
    require(row['producer_mode'] == 'native_forward_exact_only', 'Reference was not natively reconstructed')
    require(fingerprint(p['prefix']) == row['history_hash'] and len(p['prefix']) == 12+row['iteration'],
            'Reference prefix differs')
    offset, scale, raw = p['offset'], p['scale'], row['prefix_raw']
    require(len(offset) == len(scale) == 10 and len(raw) == len(p['prefix']), 'Wrong feature dimensions')
    require(all(type(v) in (int,float) and math.isfinite(v) for v in offset+scale), 'Invalid normalization')
    require(all(v > 0 for v in scale) and offset[:6] == [0.]*6 and scale[:6] == [1.]*6,
            'Native geometry normalization changed')
    require(all(len(r) == 10 and all(type(v) in (int,float) and math.isfinite(v) for v in r) for r in raw),
            'Invalid raw prefix features')
    # These operations reproduce the original prefix extrema in the same order.
    expected_offset = [0.]*6 + [min(r[j] for r in raw)-.5 for j in range(6,10)]
    expected_scale = [1.]*6 + [max(r[j] for r in raw)+.5-expected_offset[j] for j in range(6,10)]
    require(offset == expected_offset and scale == expected_scale, 'Normalization is not from the original prefix')
    require(row['checkpoint_state'] == p['encoder'] and row['original_rtol'] == RTOL and row['original_atol'] == ATOL,
            'Checkpoint binding or tolerance changed')
    return row


def validate_reference(reference, expected_sha=None):
    require(reference['schema'] == SCHEMA and reference['status'] == 'COMPLETE_EXACT_FEATURE_REFERENCE',
            'Incomplete feature reference')
    require(reference['input_result_sha'] == EVIDENCE and reference['original_code_sha'] == BASE,
            'Reference from wrong experiment')
    if expected_sha is not None:
        require(reference['producer_code_sha'] == expected_sha, 'Reference producer code differs')
    states = plan('full')['states']
    require(len(reference['states']) == len(states), 'Reference must include all 32 frozen states')
    for row, state in zip(reference['states'], states):
        require({k:row[k] for k in state} == state, 'Reference state roster changed')
        validate_row(row)
    bill = reference['bill']
    for name in ('gp_started','gp_completed','gp_errors','blocked_evaluator_attempts',
                 'blocked_vae_training_attempts','new_true_evaluations','new_vae_training','aircraft_calls','rcaide_calls'):
        require(bill[name] == 0, 'Reference capture performed forbidden work')
    return {state_key(row): row for row in reference['states']}


def read_reference(directory, expected_sha=None):
    from ra_lsbo.experiments.information_value_stage1.io import verify
    directory = Path(directory)
    status = verify(directory)
    reference = json.loads((directory/'feature_reference.json').read_text(encoding='utf-8'))
    validate_reference(reference, expected_sha)
    return reference, status


def restore_reference_features(problem, base, seed, history, session, checkpoint, metadata, reference):
    """Keep the native encoder. Restore only cryptographically bound normalization."""
    import numpy as np
    from ra_lsbo.partial_bo.models import PrefixFeatures
    from ra_lsbo.structure_corpus import feature_snapshot
    validate_row(reference)
    native = feature_snapshot(problem, base, history, session)
    risk = PrefixFeatures(problem, base, history, seed, False, 30)
    require(risk.identity == metadata['risk_feature_identity'] == reference['risk_identity'],
            'Explicit-risk feature identity changed')
    require(reference['payload']['prefix'] == [r.to_dict() for r in history]
            and reference['recorded_objective_identity'] == metadata['objective_feature_identity'],
            'Reference belongs to another prefix/objective feature')
    require(reference['checkpoint_state'] == native.state_hash == metadata['corpus_training']['latest']['parameter_state'],
            'Reference checkpoint tensor state differs')
    checkpoint = Path(checkpoint)
    for name, value in reference['checkpoint_files'].items():
        require(sha(checkpoint/name) == value == metadata['corpus_training']['latest']['files'][name],
                'Reference checkpoint file differs: '+name)
    payload = identity_payload(native, history, session)
    for k in set(payload)-{'offset','scale'}:
        require(payload[k] == reference['payload'][k], 'Feature payload changed beyond normalization: '+k)
    native_identity = native.identity
    raw = native._raw([r.candidate for r in history])
    close(raw.tolist(), reference['prefix_raw'], 'native/reference prefix forward')
    close(payload['offset'], reference['payload']['offset'], 'native/reference offset')
    close(payload['scale'], reference['payload']['scale'], 'native/reference scale')
    native.offset = np.array(reference['payload']['offset'], dtype=float)
    native.scale = np.array(reference['payload']['scale'], dtype=float)
    rebuilt = identity_payload(native, history, session)
    require(rebuilt == reference['payload'], 'Reference payload did not reconstruct')
    native.identity = digest(rebuilt)  # computed, NEVER assigned an expected ID
    require(native.identity == metadata['objective_feature_identity'], 'Exact feature identity still differs')
    proof = {'mode': 'authenticated_prefix_normalization', 'reference_state_digest': fingerprint(reference),
        'reference_key': state_key(reference), 'native_objective_identity': native_identity,
        'recorded_objective_identity': metadata['objective_feature_identity'],
        'restored_objective_identity': native.identity, 'risk_identity': risk.identity,
        'encoder_state': native.state_hash, 'original_rtol': RTOL, 'original_atol': ATOL,
        'native_offset': payload['offset'], 'native_scale': payload['scale'],
        'restored_offset': native.offset.tolist(), 'restored_scale': native.scale.tolist(),
        'max_abs_prefix_forward_difference': float(np.max(np.abs(raw-np.asarray(reference['prefix_raw'])))),
        'native_encoder_preserved': True, 'vae_training': False,
        'recorded_identity_assigned_without_reconstruction': False}
    return native, risk, proof


def verify_replay_proof(proof, reference, history):
    validate_row(reference)
    require(history == reference['payload']['prefix'], 'Replay/reference history mismatch')
    require(proof['mode'] == 'authenticated_prefix_normalization'
            and proof['reference_state_digest'] == fingerprint(reference)
            and proof['reference_key'] == state_key(reference), 'Wrong feature reference proof')
    require(proof['restored_objective_identity'] == digest(reference['payload'])
            == proof['recorded_objective_identity'] == reference['recorded_objective_identity'], 'Feature proof hash mismatch')
    require(proof['restored_offset'] == reference['payload']['offset']
            and proof['restored_scale'] == reference['payload']['scale']
            and proof['encoder_state'] == reference['checkpoint_state']
            and proof['risk_identity'] == reference['risk_identity'], 'Feature proof payload mismatch')
    require(proof['native_encoder_preserved'] is True and proof['vae_training'] is False
            and proof['recorded_identity_assigned_without_reconstruction'] is False, 'Unregistered feature replacement')
    require(proof['original_rtol'] == RTOL and proof['original_atol'] == ATOL, 'Replay tolerance changed')
    close(proof['native_offset'], proof['restored_offset'], 'verified offset')
    close(proof['native_scale'], proof['restored_scale'], 'verified scale')


def capture(output, evidence_root, expected_sha):
    from run_pr70_quadrature_diagnostic import locked, git, read, lines, model_only_guard
    from ra_lsbo.experiments.information_value_stage1.io import write_new, manifest
    from pr70_protocol import verify as verify_input
    from quadrature_source_identity import attest_sources, bind_historical_math_contract
    code, ev, out = Path(__file__).resolve().parent, Path(evidence_root).resolve(), Path(output).resolve()
    locked(code, expected_sha); locked(ev, EVIDENCE)
    require(not out.exists() and out != code and code not in out.parents and out != ev and ev not in out.parents,
            'Reference output must be NEW and external')
    git(code, 'merge-base', '--is-ancestor', BASE, expected_sha)
    require(all(x.startswith('A\t') for x in git(code,'diff','--name-status',BASE,expected_sha).splitlines()),
            'Frozen PR70 files changed')
    source = ev/RESULT
    verified = verify_input(source)
    identity = attest_sources(code, read(source/'source_receipt.json'))
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    bill = {'phase':'reference_capture','gp_started':0,'gp_completed':0,'gp_errors':0,'by_phase':{},
            'blocked_evaluator_attempts':0,'blocked_vae_training_attempts':0,
            'new_true_evaluations':0,'new_vae_training':0,'aircraft_calls':0,'rcaide_calls':0}
    try:
        from ra_lsbo.partial_bo.records import DesignRecord
        from ra_lsbo.features import load_base
        from ra_lsbo.structured_encoder import hierarchy_space
        from ra_lsbo.structure_corpus import CorpusRepresentation, native_observations, feature_snapshot
        from ra_lsbo.partial_bo.models import PrefixFeatures
        from ra_lsbo.experiments.core.protocol import BASE as model_base
        from ra_lsbo.experiments.risk_budget63_runner import _case
        with model_only_guard(bill, 0):
            for state in plan('full')['states']:
                case, seed, i = state['case'], state['seed'], state['iteration']
                folder = source/'trials'/case/str(seed)/ARM
                initial = [DesignRecord.from_dict(x) for x in read(source/'shared_initial'/case/(str(seed)+'.json'))['records']]
                history = initial+[DesignRecord.from_dict(x['record']) for x in lines(folder/'observations.jsonl')[:i]]
                proposal = lines(folder/'proposals.jsonl')[i]; meta=proposal['metadata']
                require(fingerprint([r.to_dict() for r in history]) == proposal['history_hash'], 'Wrong capture prefix')
                problem = _case(case)
                binding = bind_historical_math_contract(problem, history, identity, code)
                modules,_ = load_base(model_base)
                space = hierarchy_space(modules, [asdict(v) for v in problem.space.variables])
                checkpoint = folder/'representation'/'iteration_000'
                session = CorpusRepresentation.load_checkpoint(checkpoint, space, native_observations(initial,modules,space))
                require(session.seed == seed and len(session.events) == 1, 'Wrong reference checkpoint')
                obj = feature_snapshot(problem, model_base, history, session)
                risk = PrefixFeatures(problem, model_base, history, seed, False, 30)
                payload = identity_payload(obj, history, session)
                require(digest(payload) == obj.identity == meta['objective_feature_identity'],
                        'Reference producer cannot EXACTLY reconstruct '+state_key(state)+'; no codebook or hash fallback allowed')
                require(risk.identity == meta['risk_feature_identity'], 'Reference risk identity differs')
                files = meta['corpus_training']['latest']['files']
                require(all(sha(checkpoint/n) == v for n,v in files.items()), 'Reference checkpoint bytes differ')
                row = {**state, 'producer_mode':'native_forward_exact_only', 'payload':payload,
                       'recorded_objective_identity':meta['objective_feature_identity'], 'native_objective_identity':obj.identity,
                       'risk_identity':risk.identity, 'history_hash':proposal['history_hash'],
                       'checkpoint_state':obj.state_hash, 'checkpoint_files':files,
                       'prefix_raw':obj._raw([r.candidate for r in history]).tolist(),
                       'contract_identity':binding, 'original_rtol':RTOL, 'original_atol':ATOL}
                validate_row(row); rows.append(row)
                print('EXACT_REFERENCE', state_key(state), obj.identity, flush=True)
        locked(code, expected_sha); locked(ev, EVIDENCE)
        require(verified == verify_input(source), 'Input changed during capture')
        reference = {'schema':SCHEMA, 'status':'COMPLETE_EXACT_FEATURE_REFERENCE', 'producer_code_sha':expected_sha,
                     'input_result_sha':EVIDENCE, 'original_code_sha':BASE, 'input_manifest_sha256':sha(source/'manifest.json'),
                     'identity_source_receipt':identity, 'states':rows, 'bill':bill,
                     'platform':platform.platform(), 'python':platform.python_version()}
        validate_reference(reference, expected_sha)
        write_new(out/'feature_reference.json',reference); manifest(out)
        return {'status':reference['status'], 'states':len(rows), 'bill':bill}
    except BaseException as error:
        write_new(out/'failure.json',{'status':'FAILED_EXACT_FEATURE_REFERENCE', 'code_sha':expected_sha,
                  'completed_states':len(rows),'bill':bill,'traceback':traceback.format_exc()})
        manifest(out)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=('capture','verify'),default='capture')
    p.add_argument('--expected-sha');p.add_argument('--evidence-root',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--input',type=Path)
    a=p.parse_args()
    if a.task=='verify':
        if a.input is None:p.error('--input required')
        ref,status=read_reference(a.input,a.expected_sha)
        result={**status,'status':'VERIFIED_EXACT_FEATURE_REFERENCE','states':len(ref['states'])}
    else:
        if a.output is None or a.evidence_root is None or a.expected_sha is None:p.error('capture requires output, evidence-root, expected-sha')
        result=capture(a.output,a.evidence_root,a.expected_sha)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
