import hashlib

from confirm12.source_bytes_diagnostic import inspect


def test_line_endings_are_diagnosed_without_changing_file_or_authorization(tmp_path):
    path = tmp_path / 'source.py'
    raw = b'first\r\nsecond\r\n'
    path.write_bytes(raw)
    expected = hashlib.sha256(b'first\nsecond\n').hexdigest()
    result = inspect(path, expected)
    assert not result['raw_match'] and result['expected_matches_lf_form']
    assert result['line_ending_hypothesis_only']
    assert not result['historical_identity_replaced'] and not result['authorizes_execution']
    assert path.read_bytes() == raw


def test_unrelated_content_is_not_attributed_to_line_endings(tmp_path):
    path = tmp_path / 'source.py'
    path.write_bytes(b'first\r\nchanged\r\n')
    result = inspect(path, hashlib.sha256(b'first\nsecond\n').hexdigest())
    assert not result['raw_match'] and not result['line_ending_hypothesis_only']
