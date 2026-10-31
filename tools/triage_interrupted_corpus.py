"""Extract the exact interrupted aircraft trial, without importing any evaluator.

Usage: python tools/triage_interrupted_corpus.py ARCHIVE OUTPUT
Read-only on ARCHIVE; all derived evidence is written into fresh OUTPUT.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile

EXPECTED = "aa7f7b618eb126bcb0f7fa67c886664f235a7432ad43c269ce833262c05bd2e9"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(archive, target):
    archive, target = Path(archive), Path(target)
    if sha256(archive) != EXPECTED:
        raise ValueError("Interrupted release archive SHA256 mismatch")
    if target.exists():
        raise FileExistsError("Refuse to overwrite an existing diagnostic folder")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP entries")
        files = [n for n in names if not n.endswith("/")]
        members = [n for n in files if "/optimization/aircraft_sparse/9936_old30/" in "/" + n]
        if not members:
            raise ValueError("Could not find exact target trial")
        ledgers = [n for n in files if n.endswith("/calls.jsonl")]
        pilot_ledgers = []
        for candidate in ledgers:
            events = [json.loads(line) for line in z.read(candidate).splitlines() if line.strip()]
            started = sum(x.get("event") == "started" for x in events)
            settled = sum(x.get("event") == "settled" for x in events)
            if (started, settled) == (2212, 2212):
                pilot_ledgers.append((candidate, events))
        if len(pilot_ledgers) != 1:
            raise ValueError("Could not identify unique pilot ledger by sealed event counts")
        ledger_path, calls = pilot_ledgers[0]
        members = [n for n in members if n.endswith((".jsonl", ".json", ".txt", ".log"))]
        trial = {}
        for name in members:
            tail = name.split("/optimization/aircraft_sparse/9936_old30/", 1)[1]
            trial[tail] = z.read(name)
        required = {"manifest.json", "proposals.jsonl", "observations.jsonl"}
        if not required.issubset(trial):
            raise ValueError("Missing required original trial evidence")
        proposals = [json.loads(s) for s in trial["proposals.jsonl"].splitlines() if s.strip()]
        observations = [json.loads(s) for s in trial["observations.jsonl"].splitlines() if s.strip()]
        if len(proposals) != 29 or len(observations) != 29:
            raise ValueError("Unexpected online record counts; do not infer a missing proposal")
        # Pilot ledger was uniquely selected above; smoke is excluded.
        counts = {"started": 0, "settled": 0}
        role_counts = {}
        related = []
        for event in calls:
            label = event.get("event")
            if label in counts:
                counts[label] += 1
            if label == "started":
                role = event.get("role", "")
                role_counts[role] = role_counts.get(role, 0) + 1
            tag_parts = str(event.get("tag", "")).split("/")
            if ({"aircraft_sparse", "9936", "old30", "online"}.issubset(set(tag_parts))
                    and event.get("role") == "online"):
                related.append(event)
        if counts != {"started": 2212, "settled": 2212}:
            raise ValueError("Pilot ledger count mismatch")
        if len(related) != 58:
            raise ValueError(f"Expected 29 matched starts/settlements for target, got {len(related)}")
        latest = observations[-1]
        status = latest.get("observation", {}).get("status")
        prefix_28 = proposals[-1].get("prefix_digest")
        summary = {
            "schema": "interrupted_corpus_triage_v1",
            "original_archive_sha256": EXPECTED,
            "archive_entry_count": len(files),
            "trial_entries": sorted(trial),
            "ledger_counts": counts,
            "ledger_started_roles": role_counts,
            "trial_proposals": len(proposals),
            "trial_observations": len(observations),
            "last_online_observation": latest,
            "last_successful_proposal_prefix_digest": prefix_28,
            "first_failed_iteration": 29,
            "failure_proposal_record_present": any(p.get("iteration") == 29 for p in proposals),
            "target_related_ledger_events": len(related),
            "last_status": status,
            "no_evaluator_or_optimizer_used": True,
        }
        target.mkdir(parents=True)
        (target / "trial").mkdir()
        for name, data in trial.items():
            out = target / "trial" / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
        (target / "related_ledger_events.json").write_text(
            json.dumps(related, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (target / "triage.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if sha256(archive) != EXPECTED:
        raise ValueError("Original archive changed")
    print(json.dumps({k: v for k, v in summary.items()
        if k not in {"last_online_observation"}}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: triage_interrupted_corpus.py ARCHIVE NEW_OUTPUT_DIRECTORY")
    main(sys.argv[1], sys.argv[2])
