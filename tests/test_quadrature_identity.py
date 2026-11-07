"""Stdlib-only identity tests; no scientific imports, fits, or evaluator calls."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

from quadrature_source_identity import (
    BASE, SCHEMA, PARENT_PATH, CHILD_PATH, PINNED_BLOBS, HISTORICAL_RAW_SHA256,
    sha256, git_blob_sha, newline_forms, attest_file, parent_inputs,
    contract_from_sources, case_settings, binding_receipt, apply_binding, verify_binding,
)

# Literal constructor excerpt from the frozen HierarchicalBranin source.
CONSTRUCTOR = b'''class HierarchicalBranin:
    def __init__(self, base):
        variables = [
            {"name": "architecture", "kind": "nominal", "default": "basic",
             "values": ["basic", "augmented"]},
            {"name": "module", "kind": "ordinal", "default": 1, "values": [1, 2, 3],
             "active_when": {"architecture": ["augmented"]}},
            {"name": "x", "kind": "continuous", "default": .65, "bounds": [0., 1.]},
            {"name": "y", "kind": "continuous", "default": .2, "bounds": [0., 1.]},
            {"name": "trim", "kind": "continuous", "default": .5, "bounds": [0., 1.],
             "active_when": {"architecture": ["augmented"]}}]
        super().__init__(base, variables, {
            "family": "hierarchical_branin_extension_v1", "ellipse_radii": [.16, .2],
            "feedback": "region violation returns only a sign and skips the objective"})
'''
# Exact contract IDs read from the ORIGINAL Windows CI snapshot histories;
# these are regression expectations, never runtime authority for a binding.
ORIGINAL_IDS = {
    'branin_ordinary': 'd9cd31069148344116e96175a7a5838290af3c7764f6721d01cca558b3a88cb3',
    'branin_sparse': 'c788624cfbec09044e5264a55405a1bfa61b94fec6b170220a6c34f39290d3ab',
    'rosenbrock_ordinary': '57fc7b1b2d52e464b9ad9494f9d7f3a0ff1dc67b767cf8d8c9fb5846a1ff373e',
    'rosenbrock_sparse': 'a152faa67d45761ac0db591ad7ef4a796f55fd7c36da012d6417f23ab15fbbfe',
}


def fixture(case='branin_sparse', same=False):
    """Synthetic attestation for pure binding tests; NOT a runtime source proof."""
    variables, settings = parent_inputs(CONSTRUCTOR)
    files = {}
    for path in PINNED_BLOBS:
        historical = HISTORICAL_RAW_SHA256[path]
        local = historical if same else sha256(path.encode())
        files[path] = {'git_blob_sha1': PINNED_BLOBS[path], 'lf_sha256': local,
                       'crlf_sha256': historical, 'local_raw_sha256': local, 'local_eol': 'LF',
                       'historical_raw_sha256': historical, 'historical_eol': 'CRLF',
                       'newline_only_equivalence_verified': True}
    a = {'schema': SCHEMA, 'base_code_sha': BASE, 'files': files,
         'parent_schema': variables, 'parent_settings': settings,
         'observations_rewritten': False, 'frozen_sources_rewritten': False}
    family, difficulty = case.split('_')
    local, local_settings, _ = contract_from_sources(variables, settings, family, difficulty,
        {p: row['local_raw_sha256'] for p, row in files.items()})
    history = [{'candidate': {'x': .2}, 'observation': {'contract_id': ORIGINAL_IDS[case]}}]
    obj = SimpleNamespace(family=family, difficulty=difficulty, contract_id=local, settings=local_settings)
    return obj, history, a


class SourceIdentityTests(unittest.TestCase):
    def test_lf_and_crlf_forms(self):
        raw = b'a = 1\nb = 2\n'
        self.assertEqual(newline_forms(raw), newline_forms(raw.replace(b'\n', b'\r\n')))
    def test_lone_cr_rejected(self):
        with self.assertRaises(ValueError): newline_forms(b'a\rb\n')
    def test_mixed_line_endings_rejected(self):
        with self.assertRaises(ValueError): newline_forms(b'a\r\nb\n')
    def test_bom_rejected(self):
        with self.assertRaises(ValueError): newline_forms(b'\xef\xbb\xbfa\n')
    def test_empty_rejected(self):
        with self.assertRaises(ValueError): newline_forms(b'')
    def test_anchored_cross_eol_attestation(self):
        raw = b'a = 1\nb = 2\n'; old = raw.replace(b'\n', b'\r\n')
        r = attest_file(raw, raw, sha256(old), git_blob_sha(raw))
        self.assertEqual((r['local_eol'], r['historical_eol']), ('LF', 'CRLF'))
    def test_crlf_checkout_attestation(self):
        raw = b'a = 1\n'; crlf = raw.replace(b'\n', b'\r\n')
        self.assertEqual(attest_file(raw, crlf, sha256(crlf), git_blob_sha(raw))['local_eol'], 'CRLF')
    def test_unknown_historical_sha_rejected(self):
        raw = b'a=1\n'
        with self.assertRaises(ValueError): attest_file(raw, raw, '0'*64, git_blob_sha(raw))
    def test_non_newline_content_change_rejected(self):
        raw = b'a=1\n'
        with self.assertRaises(ValueError): attest_file(raw, b'a=2\n', sha256(raw), git_blob_sha(raw))
    def test_pinned_blob_mismatch_rejected(self):
        raw = b'a=1\n'
        with self.assertRaises(ValueError): attest_file(raw, raw, sha256(raw), '0'*40)
    def test_parent_literals_not_executed(self):
        schema, settings = parent_inputs(CONSTRUCTOR)
        self.assertEqual(len(schema), 5)
        self.assertEqual(settings['ellipse_radii'], [.16, .2])
    def test_nonliteral_constructor_rejected(self):
        tree = ast.parse(CONSTRUCTOR.decode())
        init = tree.body[0].body[0]
        init.body[0].value = ast.Call(func=ast.Name(id='untrusted_function', ctx=ast.Load()), args=[], keywords=[])
        bad = (ast.unparse(ast.fix_missing_locations(tree)) + '\n').encode()
        with self.assertRaises(ValueError): parent_inputs(bad)
    def test_all_four_actual_historical_contract_ids(self):
        schema, settings = parent_inputs(CONSTRUCTOR)
        for case, expected in ORIGINAL_IDS.items():
            with self.subTest(case=case):
                f, d = case.split('_')
                self.assertEqual(contract_from_sources(schema, settings, f, d, HISTORICAL_RAW_SHA256)[0], expected)
    def test_cross_eol_binding_without_record_rewrite(self):
        obj, history, a = fixture(); before = deepcopy(history)
        r = apply_binding(obj, history, a)
        self.assertTrue(r['binding_needed']); self.assertEqual(history, before)
        self.assertEqual(obj.contract_id, ORIGINAL_IDS['branin_sparse'])
    def test_matching_eol_no_binding_needed(self):
        obj, history, a = fixture(same=True)
        self.assertFalse(apply_binding(obj, history, a)['binding_needed'])
    def test_wrong_record_contract_rejected(self):
        obj, history, a = fixture(); history[0]['observation']['contract_id'] = '0'*64
        before = deepcopy(vars(obj))
        with self.assertRaises(ValueError): apply_binding(obj, history, a)
        self.assertEqual(vars(obj), before)
    def test_mixed_contract_history_rejected(self):
        obj, history, a = fixture(); history.append(deepcopy(history[0]))
        history[1]['observation']['contract_id'] = 'f'*64
        with self.assertRaises(ValueError): apply_binding(obj, history, a)
    def test_wrong_native_contract_rejected(self):
        obj, history, a = fixture(); obj.contract_id = '0'*64
        with self.assertRaises(ValueError): apply_binding(obj, history, a)
    def test_changed_case_settings_rejected(self):
        obj, history, a = fixture(); obj.settings['center'][0] += .1
        with self.assertRaises(ValueError): apply_binding(obj, history, a)
    def test_wrong_case_and_empty_prefix_rejected(self):
        obj, history, a = fixture()
        with self.assertRaises(ValueError): binding_receipt('aircraft', 'sparse', history, a)
        with self.assertRaises(ValueError): binding_receipt('branin', 'sparse', [], a)
    def test_receipt_reconstruction_and_tamper_rejection(self):
        obj, history, a = fixture(); proof = apply_binding(obj, history, a)
        verify_binding(proof, 'branin', 'sparse', history, a)
        proof['historical_contract_id'] = '0'*64
        with self.assertRaises(ValueError): verify_binding(proof, 'branin', 'sparse', history, a)
    def test_tampered_source_attestation_rejected(self):
        obj, history, a = fixture(); a['files'][PARENT_PATH]['historical_raw_sha256'] = '0'*64
        with self.assertRaises(ValueError): apply_binding(obj, history, a)
    def test_pure_import_needs_no_scientific_libraries(self):
        root = Path(__file__).resolve().parents[1]
        p = subprocess.run([sys.executable, '-S', '-c',
            "import quadrature_source_identity,sys; assert not any(n in sys.modules for n in ('numpy','torch','scipy'))"],
            cwd=root, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
    def test_only_instance_is_changed(self):
        obj, history, a = fixture(); other, _, _ = fixture(); before = deepcopy(vars(other))
        apply_binding(obj, history, a)
        self.assertEqual(vars(other), before)


if __name__ == '__main__': unittest.main()
