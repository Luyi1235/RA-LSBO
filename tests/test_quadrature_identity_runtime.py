"""Native constructor/source attestation tests; no evaluator or GP/VAE fitting.

The real checkpoint/forecast/Q2 integration is additionally run by the CI smoke.
No dependency skips, source rewrite, or fake scientific modules are allowed.
"""
from copy import deepcopy
from pathlib import Path
import unittest

from quadrature_source_identity import (
    BASE, HISTORICAL_RAW_SHA256, attest_sources, bind_historical_math_contract,
    binding_receipt, contract_from_sources,
)
from test_quadrature_identity import ORIGINAL_IDS
from ra_lsbo.experiments.risk_budget63_runner import _case

ROOT = Path(__file__).resolve().parents[1]


class NativeIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.attestation = attest_sources(ROOT, {
            'head_sha': BASE, 'python_sources_sha256': dict(HISTORICAL_RAW_SHA256)})

    def test_all_native_case_contracts_match_reconstructed_local_sources(self):
        a = self.attestation
        sources = {p: r['local_raw_sha256'] for p, r in a['files'].items()}
        for case in ORIGINAL_IDS:
            p = _case(case); f, d = case.split('_')
            expected, settings, _ = contract_from_sources(a['parent_schema'], a['parent_settings'], f, d, sources)
            self.assertEqual(p.contract_id, expected)
            self.assertEqual(p.settings, settings)

    def test_actual_historical_source_hashes_reproduce_all_four_recorded_ids(self):
        a = self.attestation
        for case, expected in ORIGINAL_IDS.items():
            f, d = case.split('_')
            found, _, _ = contract_from_sources(a['parent_schema'], a['parent_settings'], f, d, HISTORICAL_RAW_SHA256)
            self.assertEqual(found, expected)

    def test_native_binding_keeps_records_and_other_instances_unchanged(self):
        for case, expected in ORIGINAL_IDS.items():
            p, other = _case(case), _case(case)
            old_id = other.contract_id
            # Identity-only fixture; never fed to a model or evaluator.
            prefix = [{'observation': {'contract_id': expected}, 'candidate': {'identity_only': True}}]
            before = deepcopy(prefix)
            proof = bind_historical_math_contract(p, prefix, self.attestation, ROOT)
            self.assertEqual(p.contract_id, expected)
            self.assertEqual(prefix, before)
            self.assertEqual(other.contract_id, old_id)
            self.assertEqual(proof, binding_receipt(p.family, p.difficulty, prefix, self.attestation))

    def test_invalid_native_prefix_is_rejected_before_binding(self):
        p = _case('branin_sparse'); old = p.contract_id
        with self.assertRaises(ValueError):
            bind_historical_math_contract(p, [{'observation': {'contract_id': '0'*64}}], self.attestation, ROOT)
        self.assertEqual(p.contract_id, old)


if __name__ == '__main__': unittest.main()
