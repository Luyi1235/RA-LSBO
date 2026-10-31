"""Bounded D29 SOC GP postmortem. Never calls an evaluator or BO.propose().

Two explicit commands:
  python tools/diagnose_failed_soc_prefix.py plan --archive ARCHIVE --output NEW_DIR
  python tools/diagnose_failed_soc_prefix.py run --archive ARCHIVE --output NEW_DIR

Plan checks immutable data and records exact eight GP attempts. Run fits exactly
D28 and D29 old30 feature encoders, checks D28 original identity, and tries the
eight D29 SOC kernels once. No retries, no parameter changes or online calls.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

import numpy as np

# A direct python tools/... invocation otherwise places tools/, not the checkout,
# on sys.path. This is diagnostic-only import setup, not an evaluator change.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.partial_bo.models import PrefixFeatures, LearnedMixedHead
from ra_lsbo.partial_refinement.posterior import group_divisors
from ra_lsbo.experiments.core.protocol import BASE, ROOT, sha
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
from ra_lsbo.budget import digest

ARCHIVE_SHA = "aa7f7b618eb126bcb0f7fa67c886664f235a7432ad43c269ce833262c05bd2e9"
SOURCE_SHA = "fc1ac773f5bd9e60abd2ca4b56bd971a707735e2"
CASE, SEED, ARM = "aircraft_sparse", 9936, "old30"
EXPECTED_CONSTRAINTS = ("stall", "root_bending", "soc", "current", "voltage", "power")
GRID_C = (0.05, 0.15, 0.4, 1.0)
GRID_D = (0.5, 1.5)


def sanitize(value):
    """Serializable diagnostics: keep NaN/Inf as explicit tagged evidence."""
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"nonfinite": "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")}
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported diagnostic type {type(value).__name__}")


def write(path, object_):
    data = json.dumps(sanitize(object_), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf8") as handle:
        handle.write(data)


def sha256_stream(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_input(archive):
    if sha256_stream(archive) != ARCHIVE_SHA:
        raise ValueError("Archive digest does not match the interrupted release")
    with zipfile.ZipFile(archive) as z:
        candidates = [n for n in z.namelist() if "/optimization/aircraft_sparse/9936_old30/" in "/" + n]
        def read(name):
            matched = [n for n in candidates if n.endswith("/" + name)]
            if len(matched) != 1:
                raise ValueError(f"Expected one historical {name}, found {len(matched)}")
            return z.read(matched[0])
        manifest = json.loads(read("manifest.json"))
        proposals = [json.loads(x) for x in read("proposals.jsonl").splitlines() if x.strip()]
        online = [json.loads(x) for x in read("observations.jsonl").splitlines() if x.strip()]
    if (manifest["case"], manifest["seed"], manifest["method"]) != (CASE, SEED, ARM):
        raise ValueError("The historical trial identity is wrong")
    if len(proposals) != 29 or len(online) != 29 or len(manifest["initial"]) != 12:
        raise ValueError("The expected history is incomplete")
    if [r["iteration"] for r in proposals] != list(range(29)):
        raise ValueError("Proposal order differs")
    history = [DesignRecord.from_dict(x) for x in manifest["initial"] + online]
    if any(history[12+i].candidate != proposals[i]["candidate"] for i in range(29)):
        raise ValueError("Candidate and settled observation mismatch")
    settings = manifest["specification"]["settings"]
    if (tuple(settings["continuous_grid"]) != GRID_C
            or tuple(settings["discrete_grid"]) != GRID_D
            or settings["average_kernel"] or settings["encoder_epochs"] != 30):
        raise ValueError("Unregistered GP or VAE parameters")
    d28 = digest([r.to_dict() for r in history[:40]])
    if proposals[28]["prefix_digest"] != d28:
        raise ValueError("Last successful proposal does not match D28 prefix")
    return manifest, proposals, history


def diagnostic_plan(archive):
    manifest, proposals, history = load_input(archive)
    problem = problem_for(CASE)
    if tuple(n for n in problem.required_constraints
             if n not in problem.analytical_constraints) != EXPECTED_CONSTRAINTS:
        raise ValueError("Nonanalytic constraint order changed")
    if any(x.observation.contract_id != problem.contract_id for x in history):
        raise ValueError("Physical contract differs")
    if not proposals[28]["metadata"].get("feature_identity"):
        raise ValueError("No original D28 feature identity to verify")
    attempts = [{"head": "soc", "continuous_length": c, "discrete_length": d}
                for c in GRID_C for d in GRID_D]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_SHA, commit],
                              cwd=ROOT, capture_output=True)
    if ancestor.returncode:
        raise ValueError("Diagnostic branch must descend from fixed execution commit")
    result = {
        "schema": "aircraft_corpus_soc_D29_diagnostic_v1",
        "source_commit": SOURCE_SHA, "diagnostic_commit": commit,
        "archive_sha256": ARCHIVE_SHA,
        "last_success_prefix": proposals[28]["prefix_digest"],
        "last_success_feature_identity": proposals[28]["metadata"]["feature_identity"],
        "case": CASE, "seed": SEED, "arm": ARM, "failed_iteration": 29,
        "history_D28_count": 40, "history_D29_count": 41,
        "encoder_fits_limit": 2, "gp_candidate_attempt_limit": 8,
        "gp_head": "soc", "soc_seed": SEED + problem.required_constraints.index("soc") + 1,
        "ep_parameters": {"max_sweeps": 60, "damping": .7, "tolerance": 1e-8},
        "attempts": attempts, "no_evaluator_or_BO_calls": True,
        "old_interruption_lock_untouched": True,
    }
    return result


def run(archive, output):
    output = Path(output)
    registered = json.loads((output / "plan.json").read_text(encoding="utf8"))
    fresh = diagnostic_plan(archive)
    if registered != fresh:
        raise ValueError("Diagnostic plan or code/source inputs have changed")
    # This separate lock belongs ONLY to the new diagnosis; an interrupted run
    # cannot silently resume and completed candidates cannot silently rerun.
    descriptor = os.open(str(output / "diagnostic_execution.lock"),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    manifest, proposals, history = load_input(archive)
    problem = problem_for(CASE)
    import ra_lsbo.partial_bo.models as base_models
    original_gp = base_models.EvidenceGP
    summaries = []
    for prefix in (28, 29):
        features = PrefixFeatures(problem, BASE, history[:12 + prefix], SEED, True, 30)
        feature_id = features.identity
        if prefix == 28 and feature_id != registered["last_success_feature_identity"]:
            write(output / "identity_mismatch.json",
                  {"expected": registered["last_success_feature_identity"],
                   "actual": feature_id, "status": "stopped_before_GP"})
            raise RuntimeError("D28 feature mismatch: GP fitting not authorized")
        summaries.append({"prefix": prefix, "feature_identity": feature_id})
        if prefix == 29:
            inputs = features.transform([row.candidate for row in history])
    write(output / "features.json", summaries)
    packets = [r.observation for r in history]
    observations = [packet.constraint("soc") for packet in packets]
    counts = {kind: sum(x.kind == kind for x in observations)
              for kind in ("exact", "interval", "missing")}
    if counts != {"exact": 17, "interval": 24, "missing": 0}:
        raise ValueError("D29 SOC observation types changed")

    state = {"context": None, "root": []}
    class TracingEvidenceGP(original_gp):
        def __init__(self, *args, **kwargs):
            try:
                super().__init__(*args, **kwargs)
            except Exception as exc:
                state["root"].append({"phase": "posterior_construction",
                    "error_type": type(exc).__name__, "error_message": str(exc),
                    "traceback": traceback.format_exc()})
                raise

        def log_evidence(self):
            try:
                return super().log_evidence()
            except Exception as exc:
                state["root"].append({"phase": "log_evidence",
                    "error_type": type(exc).__name__, "error_message": str(exc),
                    "traceback": traceback.format_exc()})
                raise

    journal = output / "candidate_attempts.jsonl"
    # Log a row even when the underlying algorithm catches and transforms errors;
    # tag all NaN/Inf without corrupting the physical data or changing inference.
    try:
        base_models.EvidenceGP = TracingEvidenceGP
        with journal.open("x", encoding="utf8") as stream:
            for index, attempt in enumerate(registered["attempts"]):
                c, d = attempt["continuous_length"], attempt["discrete_length"]
                divisor = group_divisors(inputs.shape[1], len(problem.space.continuous), c, d)
                state["root"] = []
                result = {**attempt, "candidate_index": index, "counts": counts,
                          "raw_D29_feature_identity": summaries[-1]["feature_identity"],
                          "actual_length_scale": 1.0, "success": False}
                try:
                    h = LearnedMixedHead(inputs / divisor, observations,
                        seed=registered["soc_seed"],
                        length_grid=(1 / math.sqrt(inputs.shape[1]),))
                    result.update(success=True, selected=h.selected,
                        ep_diagnostics=h.model.diagnostics)
                except Exception as exc:
                    result.update(error_type=type(exc).__name__, error_message=str(exc),
                                  error_traceback=traceback.format_exc())
                result["inner_errors"] = state["root"]
                stream.write(json.dumps(sanitize(result), sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        base_models.EvidenceGP = original_gp
    write(output / "result.json", {
        "scope": "D29 SOC only, no prior candidate rerun",
        "GP_attempt_count": len(registered["attempts"]),
        "observation_counts": counts,
        "encoder_fit_count": 2,
        "source_altered": False,
        "original_interruption_resumed": False,
        "input_archive_sha256_after": sha256_stream(archive),
    })
    if sha256_stream(archive) != ARCHIVE_SHA:
        raise ValueError("Original archive mutated")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        if args.output.exists():
            raise FileExistsError("Use a new diagnostic output, never overwrite")
        spec = diagnostic_plan(args.archive)
        args.output.mkdir(parents=True)
        write(args.output / "plan.json", spec)
        print("Registered two feature fits + 8 D29 SOC kernel attempts; no evaluator calls")
    else:
        run(args.archive, args.output)
        print("D29 SOC diagnostic complete; original experiment unchanged")


if __name__ == "__main__":
    main()
