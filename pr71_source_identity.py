"""Historical source-byte identity for cross-platform DIAGNOSTIC construction only.

The frozen evaluator contracts hash raw source bytes. A clean Windows checkout
can have CRLF while Linux has LF. We do not edit source/records or waive their
hashes: reconstruct only an LF/CRLF representation that EXACTLY matches the
sealed original source receipt, then let the unchanged constructors hash it.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

FILES = ('ra_lsbo/partial_bo/problems.py', 'ra_lsbo/experiments/core/problems.py')


def matching_bytes(raw, expected_sha):
    lf = raw.replace(b'\r\n', b'\n')
    for candidate in (raw, lf, lf.replace(b'\n', b'\r\n')):
        if hashlib.sha256(candidate).hexdigest() == expected_sha:
            return candidate
    raise ValueError('Saved source hash is not an exact LF/CRLF equivalent of frozen source')


@contextmanager
def historical_source_bytes(code_root, saved_receipt):
    code_root = Path(code_root).resolve()
    from pr71_protocol import PARENT
    if saved_receipt.get('head_sha') != PARENT:
        raise ValueError('Wrong original source receipt')
    originals, receipt = {}, {}
    read_bytes = Path.read_bytes
    for name in FILES:
        path = (code_root/name).resolve()
        raw = read_bytes(path)
        expected = saved_receipt['python_sources_sha256'][name]
        historic = matching_bytes(raw, expected)
        originals[path] = historic
        receipt[name] = {'live_raw_sha256': hashlib.sha256(raw).hexdigest(),
                         'historic_raw_sha256': expected,
                         'newline_representation_adapted': raw != historic}
    def read(path):
        key = path.resolve()
        return originals[key] if key in originals else read_bytes(path)
    # Scoped to construction, never the scoring loop or loading evidence data.
    with patch.object(Path, 'read_bytes', read):
        yield receipt
