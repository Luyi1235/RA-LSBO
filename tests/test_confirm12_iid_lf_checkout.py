"""Require raw byte identity of historical 600-candidate q_A sampler on fresh checkout."""
import hashlib
import json
from pathlib import Path

from ra_lsbo.experiments.iid_qualification import ROOT


def test_qualification_source_has_exact_registered_LF_bytes():
    plan_path=ROOT/"results/iid-qualification-preflight-20261103/plan.json"
    plan=json.loads(plan_path.read_text(encoding="utf8"))
    name="ra_lsbo/experiments/iid_qualification.py"
    source=ROOT/name
    raw=source.read_bytes()
    assert b"\r\n" not in raw
    assert hashlib.sha256(raw).hexdigest()==plan["source_sha256"][name]
    assert "text eol=lf" in (ROOT/".gitattributes").read_text(encoding="utf8")
