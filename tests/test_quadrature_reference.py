"""Pure reference-authentication regression tests; no evaluator or model fits."""
from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys
import unittest

from quadrature_feature_reference import (SCHEMA, digest, validate_row, validate_reference,
    verify_replay_proof, state_key, json_value)
from quadrature_audit_protocol import BASE, EVIDENCE, RTOL, ATOL, fingerprint, plan


def row_for(state=None):
    state=state or plan('full')['states'][0]
    raw=[[0.,0.,0.,1.,1.,0.,float(i),float(i+1),float(i+2),float(i+3)] for i in range(12+state['iteration'])]
    offset=[0.]*6+[min(r[j] for r in raw)-.5 for j in range(6,10)]
    scale=[1.]*6+[max(r[j] for r in raw)+.5-offset[j] for j in range(6,10)]
    prefix=[{'candidate':{'test':i},'observation':{'test':'fixture'}} for i in range(len(raw))]
    payload={'schema':'ra_lsbo_structure_corpus_2000_v1','prefix':prefix,'corpus':'a'*64,
             'settings':{},'encoder':'b'*64,'encoder_prefix':'c'*64,'offset':offset,'scale':scale}
    return {**state,'payload':payload,'recorded_objective_identity':digest(payload),
            'native_objective_identity':digest(payload),'producer_mode':'native_forward_exact_only',
            'history_hash':fingerprint(prefix),'prefix_raw':raw,'checkpoint_state':'b'*64,
            'risk_identity':'d'*64,'checkpoint_files':{},'original_rtol':RTOL,'original_atol':ATOL}


def reference():
    return {'schema':SCHEMA,'status':'COMPLETE_EXACT_FEATURE_REFERENCE','producer_code_sha':'a'*40,
            'original_code_sha':BASE,'input_result_sha':EVIDENCE,'states':[row_for(s) for s in plan('full')['states']],
            'bill':dict.fromkeys(('gp_started','gp_completed','gp_errors','blocked_evaluator_attempts',
                'blocked_vae_training_attempts','new_true_evaluations','new_vae_training','aircraft_calls','rcaide_calls'),0)}


def proof(row):
    return {'mode':'authenticated_prefix_normalization','reference_state_digest':fingerprint(row),
            'reference_key':state_key(row),'restored_objective_identity':digest(row['payload']),
            'recorded_objective_identity':row['recorded_objective_identity'],
            'restored_offset':row['payload']['offset'],'restored_scale':row['payload']['scale'],
            'native_offset':list(row['payload']['offset']),'native_scale':list(row['payload']['scale']),
            'encoder_state':row['checkpoint_state'],'risk_identity':row['risk_identity'],
            'native_encoder_preserved':True,'vae_training':False,
            'recorded_identity_assigned_without_reconstruction':False,'original_rtol':RTOL,'original_atol':ATOL}


class ExactReferenceTests(unittest.TestCase):
    def test_native_tuple_and_archived_list_share_exact_json_contract(self):
        native=[{'observation':{'constraints':(('region',{'value':-.2}),('resource',{'value':-.1}))}}]
        archived=json_value(native)
        self.assertNotEqual(native,archived)
        self.assertEqual(digest(native),digest(archived))
        self.assertEqual(archived,json_value(native))

    def test_json_boundary_does_not_round_floats_or_accept_nan(self):
        a=1.;b=math.nextafter(a,math.inf)
        self.assertNotEqual(digest(json_value({'x':a})),digest(json_value({'x':b})))
        with self.assertRaises(ValueError):json_value({'x':float('nan')})

    def test_original_payload_rehashes(self):
        r=row_for();self.assertIs(validate_row(r),r)

    def test_tiny_float_change_is_not_equal_hash(self):
        r=row_for();r['payload']['offset'][-1]=math.nextafter(r['payload']['offset'][-1],math.inf)
        with self.assertRaises(ValueError):validate_row(r)

    def test_expected_hash_assignment_does_not_pass(self):
        r=row_for();r['payload']['scale'][-1]+=.001;r['native_objective_identity']=r['recorded_objective_identity']
        with self.assertRaises(ValueError):validate_row(r)

    def test_raw_prefix_extrema_must_match(self):
        r=row_for();r['prefix_raw'][0][-1]-=.01
        with self.assertRaises(ValueError):validate_row(r)

    def test_nonfinite_forward_rejected(self):
        r=row_for();r['prefix_raw'][1][0]=float('nan')
        with self.assertRaises(ValueError):validate_row(r)

    def test_wrong_prefix_count_rejected(self):
        r=row_for();r['iteration']=3
        with self.assertRaises(ValueError):validate_row(r)

    def test_codebook_is_not_native_producer(self):
        r=row_for();r['producer_mode']='sealed_archived_codebook'
        with self.assertRaises(ValueError):validate_row(r)

    def test_checkpoint_tensor_identity_rejected(self):
        r=row_for();r['checkpoint_state']='e'*64
        with self.assertRaises(ValueError):validate_row(r)

    def test_tolerance_is_unchanged(self):
        r=row_for();r['original_rtol']=1e-3
        with self.assertRaises(ValueError):validate_row(r)

    def test_full_reference_roster(self):
        refs=validate_reference(reference(),'a'*40);self.assertEqual(len(refs),32)

    def test_reference_wrong_commit(self):
        with self.assertRaises(ValueError):validate_reference(reference(),'b'*40)

    def test_reference_missing_state(self):
        r=reference();r['states'].pop()
        with self.assertRaises(ValueError):validate_reference(r)

    def test_reference_duplicate_state(self):
        r=reference();r['states'][-1]=r['states'][0]
        with self.assertRaises(ValueError):validate_reference(r)

    def test_capture_forbids_gp(self):
        r=reference();r['bill']['gp_started']=1
        with self.assertRaises(ValueError):validate_reference(r)

    def test_capture_forbids_training(self):
        r=reference();r['bill']['new_vae_training']=1
        with self.assertRaises(ValueError):validate_reference(r)

    def test_valid_proof_accepts_tiny_native_roundoff(self):
        r=row_for();p=proof(r);p['native_offset'][-1]=math.nextafter(p['native_offset'][-1],math.inf)
        verify_replay_proof(p,r,r['payload']['prefix'])

    def test_proof_rejects_large_native_change(self):
        r=row_for();p=proof(r);p['native_offset'][-1]+=.1
        with self.assertRaises(ValueError):verify_replay_proof(p,r,r['payload']['prefix'])

    def test_proof_rejects_another_reference(self):
        r=row_for();p=proof(r);p['reference_state_digest']='0'*64
        with self.assertRaises(ValueError):verify_replay_proof(p,r,r['payload']['prefix'])

    def test_proof_rejects_encoder_replacement(self):
        r=row_for();p=proof(r);p['native_encoder_preserved']=False
        with self.assertRaises(ValueError):verify_replay_proof(p,r,r['payload']['prefix'])

    def test_import_is_stdlib_only(self):
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,'-S','-c',
            "import quadrature_feature_reference,sys; assert not any(x in sys.modules for x in ('numpy','torch','scipy'))"],
            cwd=str(root),capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
