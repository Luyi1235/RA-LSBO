"""Independent raw H2 P1 sealed-smoke review; standard library, zero model calls.

Read the Release ZIP and the checked-in readonly summaries. Verify hashes,
source snapshots, call identities, budgets and H2 causal event summaries.
Do not import any VAE, GP, optimizer, physical evaluator or simulation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

RELEASE_URL = ("https://github.com/Luyi1235/RA-LSBO/releases/download/"
               "h2-p1-windows-smoke-20261102/"
               "RA-LSBO-H2-P1-smoke-evidence-20261102_01.zip")
EXPECTED_ZIP = "76fa4f10b853d070bcc66e73d4d4a5f72aa5d036709311ffa9c75a368ca8ca1d"
EXPECTED_CODE = "7ca420613ccb50e138dc3c9d4292ac8e863c4ecf"
FOLDER = "RA-LSBO-H2-P0-smoke-plan-20261102_01/"
SMOKE_METHODS = {"random", "onehot_stable", "corpus_stable",
                 "hybrid_onehot", "hybrid_guard", "h2_causal"}
SMOKE_CASES = {"aircraft_ordinary", "aircraft_sparse"}
EXPECTED_PLAN_HASH = "82cbf9149add6f34097bc8d2cd49ec7732bab757918b86c1ef6e1ad1e386413a"
EXPECTED_AUDIT_HASH = "343807dc6cc3cd3cab707df6d5c4ad6213ca39175423897572027ee82367225f"


def fail(message):
    raise ValueError(message)


def require(condition, message):
    if not condition:
        fail(message)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_object(value):
    return digest_bytes(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def parse_manifest(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        require(len(parts) == 2 and len(parts[0]) == 64,
                "Malformed original SHA256SUMS line")
        identity, filename = parts
        require(filename not in result, "Duplicate manifest filename")
        result[filename] = identity.lower()
    return result


def read_zip(url):
    headers = {"User-Agent": "RA-LSBO-H2-independent-smoke-audit"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                timeout=90) as stream:
        raw = stream.read()
    require(digest_bytes(raw) == EXPECTED_ZIP,
            "Release ZIP bytes differ from frozen SHA256")
    return raw


def bytes_in_zip(source, manifest):
    import io
    files = {}
    unlisted = []
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            name = entry.filename
            path = PurePosixPath(name)
            require(not path.is_absolute() and ".." not in path.parts
                    and not name.startswith("/") and "\\" not in name,
                    "Unsafe archive filename")
            require(name not in files, "Duplicate archive entry")
            require(entry.file_size < 20_000_000, "Unregistered oversized member")
            body = archive.read(entry)
            require(len(body) == entry.file_size, "ZIP member length mismatch")
            actual = digest_bytes(body)
            if name in manifest:
                require(actual == manifest[name], "Member SHA mismatch: " + name)
            else:
                unlisted.append(name)
            files[name] = body
    require(len(files) >= 250, "Unexpectedly small sealed archive")
    # A checksum list cannot include its own digest without a self-reference.
    # The packaging copy is checked separately, byte-for-byte, in main().
    allowed = "RA-LSBO-H2-P1-packaging-20261102_01/SHA256SUMS.txt"
    require(unlisted == [allowed],
            "Unexpected unlisted ZIP members: " + repr(unlisted[:6]))
    return files


def data(files, relative):
    key = FOLDER + relative
    require(key in files, "Missing original ZIP member: " + key)
    return files[key]


def obj(files, relative):
    return json.loads(data(files, relative))


def check_records(files, raw_plan, audit, result):
    ledger = [json.loads(line) for line in data(files, "calls.jsonl").splitlines()
              if line.strip()]
    require(len(ledger) == 86, "Expected 43 started and settled ledger pairs")
    counts, seen_ids, seen_tags = Counter(), set(), set()
    for index, (start, end) in enumerate(zip(ledger[::2], ledger[1::2]), 1):
        require(start["event"] == "started" and end["event"] == "settled",
                "Non-paired ledger events")
        require(start["number"] == index,
                "Ledger start numbering changed")
        # Settled records are keyed by call_id/tag; they do not repeat the number.
        identity = {key: start[key] for key in
                    ("number", "tag", "role", "candidate", "contract", "new_rcaide_mda")}
        call_id = digest_object(identity)
        require(start["call_id"] == call_id and end["call_id"] == call_id,
                "Mismatched or fabricated call identity")
        require(start["tag"] == end["tag"] and start["tag"] not in seen_tags,
                "Duplicate or mismatched call tag")
        require(call_id not in seen_ids, "Duplicate call ID")
        require(start["new_rcaide_mda"] is False, "Unexpected RCAIDE charge")
        require(digest_object(end["record"]) == end["record_digest"],
                "Settled observation digest differs")
        require(end["record"]["candidate"] == start["candidate"],
                "Response candidate differs from started candidate")
        require(end["seconds"] >= 0, "Invalid settled duration")
        counts[start["role"]] += 1
        seen_tags.add(start["tag"])
        seen_ids.add(call_id)
    expected = {"shared_initial": 12, "online": 24,
                "incumbent_step_verification": 7}
    require(dict(counts) == expected and result["charged_by_role"] == expected,
            "Role totals mismatch")
    require(result["charged_model_calls"] == len(seen_ids) == audit["settled_calls"] == 43,
            "Charged calls disagree")
    require(result["maximum_model_calls"] == raw_plan["budget"]["maximum_model_calls"] == 48,
            "Registered budget changed")
    require(result["unused_calls_not_reassigned"] == 5 and
            result["new_rcaide_mda_calls"] == 0,
            "Budget remainder or RCAIDE charge incorrect")
    return dict(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--zip-file", type=Path)
    args = parser.parse_args()
    branch = args.results
    original_sha = parse_manifest(branch / "SHA256SUMS.txt")
    raw = args.zip_file.read_bytes() if args.zip_file else read_zip(RELEASE_URL)
    require(digest_bytes(raw) == EXPECTED_ZIP, "Release ZIP SHA mismatch")
    files = bytes_in_zip(raw, original_sha)
    package_manifest = "RA-LSBO-H2-P1-packaging-20261102_01/SHA256SUMS.txt"
    internal = set(files[package_manifest].decode("utf-8").splitlines())
    checked_in = set((branch / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines())
    require(internal == checked_in,
            "Embedded checksum manifest differs: inner_only=" +
            repr(sorted(internal-checked_in)[:5]) +
            " checked_in_only=" + repr(sorted(checked_in-internal)[:5]))

    plan_bytes = data(files, "plan.json")
    audit_bytes = data(files, "audit.json")
    require(digest_bytes(plan_bytes) == EXPECTED_PLAN_HASH, "Plan SHA changed")
    require(digest_bytes(audit_bytes) == EXPECTED_AUDIT_HASH, "Audit SHA changed")
    plan, audit, result = [obj(files, n) for n in ("plan.json", "audit.json", "results.json")]
    for name, actual in (("plan.json", plan), ("audit.json", audit),
                         ("results.json", result)):
        expected = json.loads((branch / name).read_text(encoding="utf-8"))
        require(actual == expected, "Published PR result mismatches ZIP: " + name)

    require(plan["source_commit"] == EXPECTED_CODE, "Wrong optimizer commit")
    require(plan["profile"]["name"] == "smoke" and
            plan["profile"]["seeds"] == [11061] and
            plan["profile"]["horizon"] == 2, "Unexpected smoke scope")
    require(set(plan["arms"]) == SMOKE_METHODS and
            set(plan["cases"]) == SMOKE_CASES, "Experiment roster changed")
    require(plan["budget"]["maximum_rcaide_mda_calls"] == 0, "RCAIDE budget changed")
    require(len(plan["sources"]) == 182, "Source path count changed")
    for sha in set(plan["sources"].values()):
        key = FOLDER + "source_snapshot/" + sha + ".source"
        require(key in files and digest_bytes(files[key]) == sha,
                "Source snapshot is missing or corrupted")
    require(len(set(plan["sources"].values())) == 181,
            "Unique content-addressed source count changed")

    registration_bytes = data(files, "registration.json")
    registration = json.loads(registration_bytes)
    require(registration["plan_sha256"] == EXPECTED_PLAN_HASH and
            registration["maximum_model_calls"] == 48, "Registration changed")

    require(audit["passed"] is True and
            audit["checked_proposals"] == 24 and
            audit["trial_count"] == 12 and
            audit["source_bindings"] == 182 and
            audit["H2_controller_replayed"] is True and
            audit["algorithm_replay"] is False and
            audit["physical_validation"] is False,
            "Saved audit boundary is inconsistent")
    require(len(audit["evidence_hashes"]) >= 90, "Evidence hash coverage reduced")
    for relative, expected_sha in audit["evidence_hashes"].items():
        require(digest_bytes(data(files, relative)) == expected_sha,
                "Independent raw evidence hash mismatch: " + relative)
    roles = check_records(files, plan, audit, result)

    receipt = json.loads((branch / "smoke_receipt.json").read_text(encoding="utf-8"))
    require(receipt["source_commit"] == EXPECTED_CODE and
            receipt["audit_sha256"] == digest_bytes(audit_bytes) and
            receipt["registration_sha256"] == digest_bytes(registration_bytes) and
            receipt["evidence_hashes"] == audit["evidence_hashes"] and
            receipt["smoke_no_feasible_trial_count"] == 5 and
            receipt["smoke_numerical_gate_only"] is True,
            "Smoke receipt does not match sealed raw evidence")

    rows = result["summaries"]
    require(len(rows) == 12 and
            {(row["case"], row["method"]) for row in rows} ==
            {(case, arm) for case in SMOKE_CASES for arm in SMOKE_METHODS},
            "Missing or repeated arm/case result")
    require(all(row["online_calls"] == 2 and not row["stopped"] and
                row["stop_reason"] is None for row in rows),
            "Unregistered early stop or missing online proposal")
    verified = [row for row in rows if row["step_verification"]["status"] == "evaluated"]
    absent = [row for row in rows if row["step_verification"]["status"] ==
              "no_nominal_feasible_incumbent"]
    require(len(verified) == 7 and len(absent) == 5 and
            all(row["step_verification"]["verified_usable"] and
                row["step_verification"]["screen"]["status"] == "passed_screen"
                for row in verified), "Numerical verification mismatch")
    require(all(row["case"] == "aircraft_sparse" for row in absent),
            "Unexpected sparse/no-incumbent labeling")
    require(all(row["model_failure_coverage_calls"] == 0 for row in rows),
            "Unexpected unreported surrogate fallback")
    H2 = {row["case"]: row for row in rows if row["method"] == "h2_causal"}
    require(set(H2) == SMOKE_CASES, "Missing H2 arm")
    require(H2["aircraft_ordinary"]["H2_controller_replay"]["scored_events"] == 2 and
            H2["aircraft_sparse"]["H2_controller_replay"]["scored_events"] == 1 and
            all(row["H2_controller_replay"]["settled_events"] == 2 and
                row["H2_controller_replay"]["final_state"]["onehot_weight"] == 0.5
                for row in H2.values()), "H2 causal warmup summary mismatch")

    output = {"schema": "h2_p1_external_zip_readonly_audit_v1",
              "release_zip_sha256": digest_bytes(raw),
              "source_commit": EXPECTED_CODE,
              "raw_zip_file_count": len(files),
              "manifest_entries": len(original_sha),
              "original_audit_evidence_hashes_checked": len(audit["evidence_hashes"]),
              "source_path_bindings": len(plan["sources"]),
              "distinct_source_blobs_checked": len(set(plan["sources"].values())),
              "settled_calls": 43, "roles": roles,
              "trials": len(rows), "online_proposals": 24,
              "verified_5s": len(verified), "no_nominal_feasible": len(absent),
              "H2_ordinary_scored": 2, "H2_sparse_scored": 1,
              "new_model_calls_by_auditor": 0,
              "new_rcaide_calls_by_auditor": 0,
              "algorithm_replayed": False,
              "simulation_executed": False,
              "performance_advantage_confirmed": False,
              "passed": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, sort_keys=True, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
