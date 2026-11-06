import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from pr71_source_identity import matching_bytes, historical_source_bytes, FILES
from pr71_protocol import PARENT


class SourceIdentityTests(unittest.TestCase):
    def test_exact_and_crlf_roundtrip(self):
        raw = b'a\nb\n'
        crlf = b'a\r\nb\r\n'
        h = lambda b: hashlib.sha256(b).hexdigest()
        self.assertEqual(matching_bytes(raw, h(raw)), raw)
        self.assertEqual(matching_bytes(raw, h(crlf)), crlf)
        self.assertEqual(matching_bytes(crlf, h(raw)), raw)

    def test_real_source_edit_rejected(self):
        with self.assertRaises(ValueError):
            matching_bytes(b'x=2\n', hashlib.sha256(b'x=1\r\n').hexdigest())

    def test_scope_no_file_mutation_or_unrelated_read_change(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            saved = {'head_sha': PARENT, 'python_sources_sha256': {}}
            for name in FILES:
                p = root/name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b'x=1\n')
                saved['python_sources_sha256'][name] = hashlib.sha256(b'x=1\r\n').hexdigest()
            other = root/'evidence.json'
            other.write_bytes(b'{}\n')
            with historical_source_bytes(root, saved) as receipt:
                self.assertEqual((root/FILES[0]).read_bytes(), b'x=1\r\n')
                self.assertEqual(other.read_bytes(), b'{}\n')
                self.assertTrue(receipt[FILES[0]]['newline_representation_adapted'])
            self.assertEqual((root/FILES[0]).read_bytes(), b'x=1\n')

    def test_foreign_source_receipt_rejected(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                with historical_source_bytes(d, {'head_sha': '0'*40}):
                    pass


if __name__ == '__main__':
    unittest.main()
