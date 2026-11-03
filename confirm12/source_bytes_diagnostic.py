"""Read-only source-byte diagnostics; never changes a frozen identity."""
import hashlib
from pathlib import Path


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def inspect(path, expected_sha):
    raw = Path(path).read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    crlf = lf.replace(b'\n', b'\r\n')
    actual, lf_sha, crlf_sha = _sha(raw), _sha(lf), _sha(crlf)
    return {'actual_raw_sha256': actual, 'expected_raw_sha256': expected_sha,
            'raw_match': actual == expected_sha,
            'lf_form_sha256': lf_sha, 'crlf_form_sha256': crlf_sha,
            'expected_matches_lf_form': expected_sha == lf_sha,
            'expected_matches_crlf_form': expected_sha == crlf_sha,
            'line_ending_hypothesis_only': actual != expected_sha and expected_sha in (lf_sha, crlf_sha),
            'historical_identity_replaced': False, 'authorizes_execution': False}
