"""PR72 R1: attest newline-only source variants before binding historical IDs.

This is an offline replay adapter, not a migration of evaluator contracts.
Historical records, their hashes, and frozen ra_lsbo sources are never changed.
We reconstruct BOTH hash layers from pinned Git blobs and the original source
receipt. An observed contract ID alone is never authority for a replacement.
Only uniform LF/CRLF byte representations of the exact pinned blobs qualify.
"""
import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess

BASE = '275aac2c5d5d6a8c8f6278c0d556de3de5867f74'
PARENT_PATH = 'ra_lsbo/partial_bo/problems.py'
CHILD_PATH = 'ra_lsbo/experiments/core/problems.py'
PINNED_BLOBS = {
    PARENT_PATH: '5bf5a398f79bb359b06e05983b9353fff6f09898',
    CHILD_PATH: 'd335388bf9f5450306266484e86495f91c99d1f9',
}
# These are ORIGINAL RAW hashes, not new canonical identities. Their authority
# is the PR70 source_receipt.json inside the already verified input manifest.
HISTORICAL_RAW_SHA256 = {
    PARENT_PATH: '0e8b18f2536a42ca480f86d73c69335508f3eaea628a3fcab4937b9f62271e6f',
    CHILD_PATH: '3344a794ccecd541ef4f0bfd8fbee7ceb91f5462a39131c142423fa90f6a65b9',
}
SCHEMA = 'pr72_pinned_lf_crlf_identity_v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def native_digest(value):
    # EXACT frozen ra_lsbo.budget.digest serialization, not compact JSON.
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def git_blob_sha(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def newline_forms(raw):
    require(isinstance(raw, bytes) and raw, 'Expected nonempty source bytes')
    require(not raw.startswith(b'\xef\xbb\xbf'), 'BOM is not a newline-only change')
    lf = raw.replace(b'\r\n', b'\n')
    require(b'\r' not in lf, 'Lone CR is not an accepted source representation')
    forms = {'LF': lf, 'CRLF': lf.replace(b'\n', b'\r\n')}
    require(b'\n' in lf and raw in forms.values(), 'Mixed source newlines are not accepted')
    return forms


def attest_file(pinned_blob, local_bytes, historical_sha256, expected_blob_sha):
    require(git_blob_sha(pinned_blob) == expected_blob_sha, 'Pinned Git source blob differs')
    forms = newline_forms(pinned_blob)
    local_eol = [name for name, raw in forms.items() if local_bytes == raw]
    historical_eol = [name for name, raw in forms.items() if sha256(raw) == historical_sha256]
    require(len(local_eol) == 1, 'Runtime source differs by more than uniform LF/CRLF')
    require(len(historical_eol) == 1, 'Original source hash is not a pinned LF/CRLF variant')
    return {
        'git_blob_sha1': expected_blob_sha,
        'lf_sha256': sha256(forms['LF']), 'crlf_sha256': sha256(forms['CRLF']),
        'local_raw_sha256': sha256(local_bytes), 'local_eol': local_eol[0],
        'historical_raw_sha256': historical_sha256, 'historical_eol': historical_eol[0],
        'newline_only_equivalence_verified': True,
    }


def parent_inputs(pinned_source):
    """Read actual constructor literals from pinned source, without exec/eval."""
    module = ast.parse(newline_forms(pinned_source)['LF'].decode('utf-8'))
    classes = [n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'HierarchicalBranin']
    require(len(classes) == 1, 'Pinned HierarchicalBranin class is ambiguous')
    inits = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == '__init__']
    require(len(inits) == 1, 'Pinned constructor is ambiguous')
    definitions = [n.value for n in inits[0].body if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == 'variables']
    calls = []
    for statement in inits[0].body:
        node = statement.value if isinstance(statement, ast.Expr) else None
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == '__init__' and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Name) and node.func.value.func.id == 'super'):
            calls.append(node)
    require(len(definitions) == len(calls) == 1 and len(calls[0].args) == 3
            and not calls[0].keywords and isinstance(calls[0].args[1], ast.Name)
            and calls[0].args[1].id == 'variables', 'Pinned constructor shape differs')
    variables = ast.literal_eval(definitions[0])
    settings = ast.literal_eval(calls[0].args[2])
    require(isinstance(variables, list) and variables and isinstance(settings, dict),
            'Pinned constructor literals have wrong types')
    return variables, settings


def case_settings(parent_settings, family, difficulty, source_sha):
    require(family in ('branin', 'rosenbrock') and difficulty in ('ordinary', 'sparse'),
            'Only the four pinned mathematical cases are accepted')
    # Reproduce the declared, immutable MathematicalCase constructor. Before
    # binding, the result must ALSO match the live unmodified constructor.
    return {**parent_settings, 'family': 'paper_%s_extension_v1' % family,
            'difficulty': difficulty,
            'ellipse_radii': [.42, .40] if difficulty == 'ordinary' else [.16, .20],
            'center': [.68, .25] if family == 'branin' else [.72, .70],
            'source': source_sha, 'upstream_replica': False}


def contract_from_sources(variables, parent_settings, family, difficulty, source_hashes):
    parent_id = native_digest({'schema': variables, 'settings': parent_settings,
                              'source_sha256': source_hashes[PARENT_PATH]})
    settings = case_settings(parent_settings, family, difficulty, source_hashes[CHILD_PATH])
    return native_digest({'parent': parent_id, 'settings': settings}), settings, parent_id


def attest_sources(code_root, original_receipt):
    code_root = Path(code_root).resolve()
    require(original_receipt.get('head_sha') == BASE, 'Original source receipt has wrong code SHA')
    sources = original_receipt.get('python_sources_sha256', {})
    require(all(sources.get(path) == expected for path, expected in HISTORICAL_RAW_SHA256.items()),
            'Original PR70 source hashes differ from the reviewed evidence')
    files, blobs = {}, {}
    for path, expected_blob in PINNED_BLOBS.items():
        p = code_root / path
        require(not p.is_symlink() and p.resolve() == p and p.is_file(), 'Unexpected runtime source path')
        raw = subprocess.check_output(['git', '-C', str(code_root), 'show', BASE + ':' + path])
        blobs[path] = raw
        files[path] = attest_file(raw, p.read_bytes(), sources[path], expected_blob)
    variables, settings = parent_inputs(blobs[PARENT_PATH])
    result = {'schema': SCHEMA, 'base_code_sha': BASE, 'files': files,
              'parent_schema': variables, 'parent_settings': settings,
              'historical_source_receipt_digest': native_digest(original_receipt),
              'observations_rewritten': False, 'frozen_sources_rewritten': False}
    validate_attestation(result)
    return result


def validate_attestation(attestation):
    require(attestation.get('schema') == SCHEMA and attestation.get('base_code_sha') == BASE,
            'Wrong source-identity attestation')
    require(set(attestation['files']) == set(PINNED_BLOBS), 'Wrong contract-source roster')
    for path, blob in PINNED_BLOBS.items():
        row = attestation['files'][path]
        require(row['git_blob_sha1'] == blob and row['newline_only_equivalence_verified'] is True,
                'Source equivalence not established')
        require(row['local_eol'] in ('LF', 'CRLF') and row['historical_eol'] in ('LF', 'CRLF'),
                'Unknown source newline representation')
        require(row['local_raw_sha256'] == row[row['local_eol'].lower() + '_sha256']
                and row['historical_raw_sha256'] == row[row['historical_eol'].lower() + '_sha256']
                == HISTORICAL_RAW_SHA256[path], 'Source-byte attestation mismatch')
    require(attestation['observations_rewritten'] is False
            and attestation['frozen_sources_rewritten'] is False, 'Historical source/data rewrite is forbidden')


def record_packets(history):
    return [deepcopy(r if isinstance(r, dict) else r.to_dict()) for r in history]


def binding_receipt(family, difficulty, history, attestation):
    validate_attestation(attestation)
    packets = record_packets(history)
    require(bool(packets), 'Historical binding requires a nonempty observed prefix')
    variables, parent_settings = attestation['parent_schema'], attestation['parent_settings']
    files = attestation['files']
    local_hashes = {p: f['local_raw_sha256'] for p, f in files.items()}
    historical_hashes = {p: f['historical_raw_sha256'] for p, f in files.items()}
    local_id, local_settings, local_parent = contract_from_sources(
        variables, parent_settings, family, difficulty, local_hashes)
    historical_id, historical_settings, historical_parent = contract_from_sources(
        variables, parent_settings, family, difficulty, historical_hashes)
    require(all(r['observation']['contract_id'] == historical_id for r in packets),
            'Observed contract does not match the independently reconstructed historical contract')
    return {'schema': SCHEMA, 'family': family, 'difficulty': difficulty,
            'local_contract_id': local_id, 'historical_contract_id': historical_id,
            'local_parent_contract_id': local_parent, 'historical_parent_contract_id': historical_parent,
            'local_settings': local_settings, 'historical_settings': historical_settings,
            'prefix_native_digest': native_digest(packets), 'prefix_count': len(packets),
            'source_attestation_digest': native_digest(attestation),
            'binding_needed': local_id != historical_id,
            'observations_rewritten': False, 'source_files_rewritten': False,
            'acceptance_rule': 'pinned_source_byte_equivalence_AND_recomputed_contract_AND_all_prefix_records'}


def apply_binding(problem, history, attestation):
    """Configure only this temporary replay problem AFTER every proof succeeds."""
    proof = binding_receipt(problem.family, problem.difficulty, history, attestation)
    require(problem.contract_id == proof['local_contract_id'], 'Native constructor contract differs')
    require(problem.settings == proof['local_settings'], 'Native constructor settings differ')
    before = native_digest(record_packets(history))
    # Not copied from a record: this ID was independently derived from two
    # pinned source hashes, literal constructor inputs, and checked settings.
    problem.contract_id = proof['historical_contract_id']
    problem.settings = deepcopy(proof['historical_settings'])
    require(native_digest(record_packets(history)) == before, 'Historical observations mutated')
    return proof


def bind_historical_math_contract(problem, history, attestation, code_root):
    """Strict production boundary: only exact pinned native classes/paths."""
    from ra_lsbo.experiments.core.problems import MathematicalCase
    from ra_lsbo.partial_bo.problems import DevelopmentProblem, HierarchicalBranin
    root = Path(code_root).resolve()
    require(type(problem) is MathematicalCase, 'Only native MathematicalCase instances may be bound')
    for cls, path in ((MathematicalCase, CHILD_PATH), (DevelopmentProblem, PARENT_PATH),
                      (HierarchicalBranin, PARENT_PATH)):
        actual = Path(inspect.getfile(cls)).resolve()
        require(actual == root / path and sha256(actual.read_bytes()) ==
                attestation['files'][path]['local_raw_sha256'], 'Loaded source is not the attested module')
    return apply_binding(problem, history, attestation)


def verify_binding(proof, family, difficulty, history, attestation):
    require(proof == binding_receipt(family, difficulty, history, attestation),
            'Stored source/contract binding does not reconcile')
