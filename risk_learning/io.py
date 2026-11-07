"""Write-once evidence, canonical manifests, fixed-source checks and call journals."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

from .core import PARENT


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def append(path, value):
    with Path(path).open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
        f.flush()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(root):
    root = Path(root)
    files = {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p.name != 'manifest.json'}
    write(root/'manifest.json', dict(schema='sha256_manifest_v1', files=files))


def check_manifest(root):
    root = Path(root).resolve()
    manifest = read(root/'manifest.json')
    if manifest.get('schema') != 'sha256_manifest_v1' or not isinstance(manifest.get('files'), dict):
        raise ValueError('Invalid manifest')
    actual = set()
    for p in root.rglob('*'):
        if p.is_symlink():
            raise ValueError('Symlinks forbidden in evidence')
        if p.is_file() and p != root/'manifest.json':
            actual.add(p.relative_to(root).as_posix())
    if actual != set(manifest['files']):
        raise ValueError('Missing or extra evidence files')
    for rel, sha in manifest['files'].items():
        p = (root/rel).resolve()
        if root not in p.parents or file_sha(p) != sha:
            raise ValueError('Path escape or SHA256 mismatch: '+rel)
    return len(actual)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def source_receipt(root, expected):
    if not re.fullmatch(r'[0-9a-f]{40}', expected or ''):
        raise ValueError('Exact 40-character expected SHA required')
    head = git(root, 'rev-parse', 'HEAD')
    if head != expected or git(root, 'status', '--porcelain'):
        raise ValueError('Wrong fixed SHA or dirty worktree')
    if git(root, 'diff', '--name-only', PARENT, head, '--', 'ra_lsbo'):
        raise ValueError('Historical model/evaluator source must remain unchanged')
    return dict(code_sha=head, parent_sha=PARENT, historical_ra_lsbo_tree_unchanged=True,
                schema='learnability_source_v1')
