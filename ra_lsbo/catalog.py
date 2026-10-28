"""Inventory terminal histories without pooling incompatible or copied runs."""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from .audit import build_snapshot, digest, load_json


def catalog_runs(roots):
    entries, unique, groups = [], {}, defaultdict(list)
    paths = sorted({p.parent.resolve() for root in roots
                    for p in Path(root).resolve().rglob("manifest.json")
                    if (p.parent / "evaluations.jsonl").is_file()})
    for source in paths:
        entry = {"source_run": str(source)}
        try:
            rows, _, audit, contract = build_snapshot(source)
            manifest = load_json((source / "manifest.json").read_bytes())
            if digest((source / "manifest.json").read_bytes()) != audit["source_sha256"]["manifest.json"]:
                raise RuntimeError("Manifest changed during inventory")
            # Ignore summary timestamps and path-dependent metadata for copy identity.
            identity = digest(json.dumps({
                "contract": audit["evaluator_contract_sha256"],
                "journal": audit["source_sha256"]["evaluations.jsonl"],
            }, sort_keys=True).encode())
            entry.update(
                status="accepted", source_status=audit["source_status"],
                evaluations=len(rows), states=audit["states"], issues=audit["issues"],
                feasible_count=audit["feasible_count"], valid_objectives=audit["valid_objectives"],
                method=manifest.get("method", "lsbo"),
                seed=manifest.get("settings", {}).get("seed"),
                initial=manifest.get("settings", {}).get("initial"),
                source_sha256=audit["source_sha256"],
                evaluator_contract_sha256=audit["evaluator_contract_sha256"],
                evaluator_hashes_present=audit["evaluator_hashes_present"],
                continuous_variables=sum(v["kind"] == "continuous" for v in contract["design_variables"]),
                discrete_variables=sum(v["kind"] != "continuous" for v in contract["design_variables"]),
                required_constraints=contract["required_constraints"],
                problem_config=contract["problem_config"],
                hidden_modes=audit["hidden_modes"], dataset_identity=identity,
            )
            if identity in unique:
                entry.update(status="duplicate", duplicate_of=unique[identity])
            else:
                unique[identity] = str(source)
                groups[audit["evaluator_contract_sha256"]].append(str(source))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            entry.update(status="excluded", reason=f"{type(error).__name__}: {error}")
        entries.append(entry)
    return {
        "mode": "read_only_history_inventory", "created_utc": datetime.now(timezone.utc).isoformat(),
        "roots": [str(Path(root).resolve()) for root in roots],
        "entries": entries, "compatible_groups": dict(groups),
        "unique_terminal_runs": len(unique),
        "unique_evaluations": sum(e["evaluations"] for e in entries if e["status"] == "accepted"),
        "notes": [
            "Copied datasets are not independent replicates.",
            "Same evaluator contract permits comparison, not an assumption of identical sampling distributions.",
            "Running, malformed and changing sources are excluded; their files are never edited.",
            "No model fitting or MDA is performed by this inventory.",
        ],
    }


def export_catalog(roots, output):
    roots = [Path(root).resolve() for root in roots]
    output = Path(output).resolve()
    if any(output == root or root in output.parents for root in roots):
        raise ValueError("Inventory output must be outside source roots")
    report = catalog_runs(roots)
    destination = output / ("catalog_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                            + "_" + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return destination, report
