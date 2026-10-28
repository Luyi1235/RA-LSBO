"""Read one completed/stopped LSBO run and export an immutable data snapshot."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from .legacy import normalize_evaluation
from .records import HIDDEN_MODES, SCHEMA_VERSION
from .training import training_rows


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_json(data):
    def reject_constant(value):
        raise ValueError("Nonstandard JSON constant: " + value)
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(data, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def build_snapshot(source):
    source = Path(source).resolve()
    paths = {name: source / name for name in ("manifest.json", "evaluations.jsonl", "summary.json")}
    contents = {name: path.read_bytes() for name, path in paths.items()}
    manifest, summary = (load_json(contents[name]) for name in ("manifest.json", "summary.json"))
    if summary.get("status") not in ("complete", "stopped_with_error"):
        raise ValueError("Refusing a running or unrecognized run; use a terminal snapshot")
    required = manifest["required_constraints"]
    analytical = manifest.get("analytical_constraints", [])
    if not manifest.get("problem_config") or not manifest.get("design_variables"):
        raise ValueError("Missing evaluator configuration or variable schema")
    names = [v["name"] for v in manifest["design_variables"]]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate variable definitions")
    if any(v["kind"] not in ("continuous", "ordinal", "nominal") for v in manifest["design_variables"]):
        raise ValueError("Unsupported variable kind")
    rows = []
    previous = 0
    for line in contents["evaluations.jsonl"].splitlines():
        if not line.strip():
            continue
        raw = load_json(line)
        index = raw["evaluation"]
        if type(index) is not int or index <= previous:
            raise ValueError("Evaluation indices must be strictly increasing integers")
        previous = index
        candidate = raw["candidate"]
        for kind in ("continuous", "discrete"):
            expected = {v["name"] for v in manifest["design_variables"]
                        if (v["kind"] == "continuous") == (kind == "continuous")}
            if set(candidate[kind]) != expected:
                raise ValueError("Candidate differs from manifest variable schema")
        risk = normalize_evaluation(raw["result"], required, analytical)
        if raw.get("feasible") is not None and risk.feasible_label is not None:
            if raw["feasible"] != risk.feasible_label:
                risk.issues.append("legacy_feasibility_disagrees")
        rows.append({
            "evaluation": index, "candidate": candidate,
            "phase": raw.get("phase"), "proposal": raw.get("proposal", {}),
            "risk": risk.to_dict(),
        })
    if not rows or summary.get("evaluations") != len(rows):
        raise ValueError("Empty journal or summary/journal count mismatch")
    hashes = {name: digest(data) for name, data in contents.items()}
    if any(digest(paths[name].read_bytes()) != sha for name, sha in hashes.items()):
        raise RuntimeError("Source changed during audit; no snapshot accepted")

    # Include saved evaluator hashes rather than today's mutable source code.
    contract = {name: manifest.get(name) for name in (
        "backend", "runtime_postprocess", "energy_model", "problem_config",
        "required_constraints", "analytical_constraints", "constraint_policy",
        "design_variables", "dependencies", "python",
    )}
    contract["evaluator_source_sha256"] = {
        name.replace("\\", "/"): sha for name, sha in manifest.get("source_sha256", {}).items()
        if name.replace("\\", "/").startswith("problems/")
    }
    contract_id = digest(json.dumps(contract, sort_keys=True, allow_nan=False).encode())
    for row in rows:
        row["source_journal_sha256"] = hashes["evaluations.jsonl"]
        row["evaluator_contract_sha256"] = contract_id
    training = training_rows(rows)
    constraints = {
        name: {
            "observed": sum(row["risk"]["constraint_observed"][name] for row in rows),
            "violated": sum(row["risk"]["constraint_observed"][name]
                            and row["risk"]["constraints"][name] > 0 for row in rows),
            "analytical": name in analytical,
        } for name in required
    }
    modes = {}
    for name in HIDDEN_MODES:
        labels = [row["risk"]["hidden_labels"][name] for row in rows]
        modes[name] = {"positive": labels.count(1), "negative": labels.count(0),
                       "unknown": labels.count(None),
                       "two_classes_observed": 0 in labels and 1 in labels}
    report = {
        "schema_version": SCHEMA_VERSION,
        "mode": "offline_data_audit_not_an_optimization_experiment",
        "source_run": str(source), "source_status": summary["status"], "source_sha256": hashes,
        "evaluator_contract_sha256": contract_id,
        "evaluator_hashes_present": bool(contract["evaluator_source_sha256"]),
        "converter_source_sha256": {
            path.name: digest(path.read_bytes()) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "evaluations": len(rows),
        "states": dict(Counter(row["risk"]["state"] for row in rows)),
        "valid_objectives": len(training["objective"]),
        "feasible_count": sum(row["risk"]["feasible_label"] is True for row in rows),
        "constraints": constraints, "hidden_modes": modes,
        "issues": dict(Counter(issue for row in rows for issue in row["risk"]["issues"])),
        "warnings": [
            "No models fitted, calibration performed or new MDA evaluations made.",
            "A terminal solver failure is not proof of physical impossibility.",
            "Hidden mode negatives describe successful terminal evaluations, not transient trials.",
            "Head sample counts do not establish calibration reliability.",
            "Do not pool runs without checking evaluator and solver identity.",
        ],
    }
    return rows, training, report, contract


def export_snapshot(source, output_root):
    source, root = Path(source).resolve(), Path(output_root).resolve()
    if root == source or source in root.parents:
        raise ValueError("Output must be outside the source run")
    rows, training, report, contract = build_snapshot(source)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / ("audit_" + stamp + "_" + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    report["created_utc"] = datetime.now(timezone.utc).isoformat()
    (destination / "observations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    for name, data in (("report.json", report), ("training_rows.json", training),
                       ("evaluator_contract.json", contract)):
        (destination / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    return destination, report
