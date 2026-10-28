"""Offline source/asset identity checks; never import RCAIDE or old results."""

import ast
import hashlib
import io
import json
from pathlib import Path
import tokenize


def setup_fingerprint(path):
    source = Path(path).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "setup")
    # Token hashing avoids Python-version-dependent AST fields.
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
               tokenize.DEDENT, tokenize.ENDMARKER}
    tokens = [(tokenize.tok_name[t.type], t.string) for t in tokenize.generate_tokens(io.StringIO(ast.get_source_segment(source, node)).readline)
              if t.type not in ignored]
    return hashlib.sha256(json.dumps(tokens).encode("utf-8")).hexdigest()


def verify_snapshot():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "model_provenance.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        path = root / "model" / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Bundled v49 source/asset identity mismatch: " + name)
    if setup_fingerprint(root / "runtime.py") != manifest["setup_token_sha256"]:
        raise RuntimeError("Model initialization no longer matches v49 Optimize.setup")
    return manifest


if __name__ == "__main__":
    manifest = verify_snapshot()
    print("PASS: %d original source/assets; setup tokens match v49" % len(manifest["files_sha256"]))
