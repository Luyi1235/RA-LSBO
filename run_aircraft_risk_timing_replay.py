"""Replay completion forecasts with censoring from a sealed PR113 ledger."""
import argparse
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

from aircraft_risk_active import configuration_brier, learning_summary
from aircraft_risk_discovery import label
from ra_lsbo.risk_core.observations import OutputObservation
from run_aircraft_risk_cold_start import discovery_outcome, file_hash, read, source_commit, verify_evidence


INPUT_CODE = "87ec0dc4cdf662a7f41604d24a7d0dff22415e0e"
INPUT_RESULT = "99e868125d2dde224b6203899382a3ddd77152e5"
INPUT_MANIFESTS = {
    "smoke": "b682bb9edda420cf2947ebbc946a9cc355598bf3c29dc7ed011ca37b4fec37ff",
    "nominal": "bef537675c41055c8d68b18ae260849b8f2618bf71cab6b5acdefe2053b82d2f",
    "headwind": "1fa3d4ae54597bd9017536d7261a9026f4f9e6d4fbe1d4bbcf83dc124fbd356e",
}
PATHS = ("risk_balanced_active", "risk_coverage")
CONTRASTS = (("timing_interval", "timing_sign"), ("timing_interval", "original_probit"),
             ("timing_sign", "original_probit"))
MARGIN_SCALE = .25
TIMING_TOLERANCE_S = 1e-7


def timing_feedback(record, settings, *, use_interval):
    """Translate trustworthy completion records into intervals for a normalized deficit."""
    d = record["diagnostics"]
    packet = record["observation"]
    reason = d.get("termination_reason")
    if reason not in {"complete", "physical_soc_reserve_abort"}:
        raise ValueError("The sealed replay only supports reserve exhaustion and full completion")
    complete = reason == "complete"
    if (packet["status"] != "completed" or d.get("numerical_failure")
            or d.get("mission_completed") is not complete
            or not d.get("trace") or settings["full_trajectory"]):
        raise ValueError("Incompatible completion timing evidence")

    speed = record["candidate"]["continuous"]["cruise_airspeed"]
    wind = settings["scenario"]["headwind_m_s"]
    climb = settings["climb_height_m"]
    ground_speed = speed-wind
    if not math.isfinite(ground_speed) or ground_speed <= 0:
        raise ValueError("Positive ground speed required")
    scheduled = climb/2. + settings["nominal_cruise_distance_m"]/ground_speed + climb/1.5 + settings["reserve_duration_s"]
    last = d["trace"][-1]
    elapsed = last["elapsed_s"]
    if not math.isfinite(elapsed) or not 0 < elapsed <= scheduled+TIMING_TOLERANCE_S:
        raise ValueError("Recorded elapsed time falls outside the flight schedule")
    threshold = settings["soc_reserve_requirement"]
    if complete:
        if (not math.isclose(elapsed, scheduled, rel_tol=0., abs_tol=TIMING_TOLERANCE_S)
                or last["stage"] != "reserve" or not (last["soc"] >= threshold)):
            raise ValueError("Completion lacks a full reserve-qualified trace")
        return OutputObservation("interval", packet["evidence_id"]+":timing", upper=0.)
    if not (last["soc"] < threshold):
        raise ValueError("Reserve stop has no threshold crossing")

    evidence = packet["evidence_id"]+":timing"
    if not use_interval:
        return OutputObservation("interval", evidence, lower=0., lower_inclusive=False)
    step = settings["integration_step_s"]
    lower = max(0., (scheduled-elapsed)/scheduled)
    upper = min(1., (scheduled-elapsed+step)/scheduled)
    return OutputObservation("interval", evidence, lower=lower, upper=upper,
                             lower_inclusive=lower > 0.)


def paired_scores(rows, checkpoints):
    return {
        path: {
            head: learning_summary(
                [{**r, "head": "usable_factored"} for r in rows
                 if r["path"] == path and r["head"] == head],
                checkpoints, contrasts=CONTRASTS)
            for head in ("usable_factored", "mission")
        }
        for path in PATHS
    }


def replay(source, output, profile, expected):
    from scipy.special import ndtr
    import numpy as np
    import torch

    from ra_lsbo.experiments.core.problems import AircraftCase, FlightScenario
    from ra_lsbo.experiments.core.protocol import BASE
    from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import StableGroupedHead
    from ra_lsbo.partial_bo.models import PrefixFeatures
    from ra_lsbo.partial_bo.records import DesignRecord

    code = source_commit(expected, clean=True)
    if file_hash(source/"manifest.json") != INPUT_MANIFESTS[profile]:
        raise ValueError("Replay input is not the frozen PR113 manifest")
    report, _, _ = verify_evidence(source, INPUT_CODE)
    spec, contract = read(source/"plan.json"), read(source/"contract.json")
    settings = contract["settings"]
    torch.set_num_threads(1)
    problem = AircraftCase(BASE, "sparse", scenario=FlightScenario(**settings["scenario"]),
                           integration_step=spec["step_seconds"])
    # Archive IDs include byte hashes of Python sources. Git line endings
    # differ across hosts even when the mathematical evaluator is unchanged.
    replay_settings = dict(problem.settings)
    sources = {"source": Path(inspect.getfile(AircraftCase)),
               "battery_source_sha256": problem.battery_source}
    for key, path in sources.items():
        lf = path.read_bytes().replace(b"\r\n", b"\n")
        variants = (lf, lf.replace(b"\n", b"\r\n"))
        if settings[key] not in {hashlib.sha256(data).hexdigest() for data in variants}:
            raise ValueError("Replay evaluator source differs from PR113")
        replay_settings[key] = settings[key]
    if replay_settings != settings:
        raise ValueError("Replay evaluator settings differ from PR113")
    feature_problem = SimpleNamespace(space=problem.space, contract_id=contract["id"])
    with (source/"calls.jsonl").open(encoding="utf-8") as stream:
        calls = {row["tag"]: row["record"] for row in map(json.loads, stream)
                 if row["event"] == "settled"}
    rows = []
    output.mkdir()
    receipt = {"input_result_commit": INPUT_RESULT, "input_code_sha": INPUT_CODE,
               "input_manifest_sha256": INPUT_MANIFESTS[profile], "code_sha": code, "profile": profile}
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    for seed in spec["seeds"]:
        unit = source/str(seed)
        initial = [calls[f"{seed}/initial/{i}"] for i in range(spec["initial"])]
        tests = [calls[f"{seed}/heldout/{i}"] for i in range(spec["heldout"])]
        heldout = [row["candidate"] for row in tests]
        outcomes = list(map(discovery_outcome, tests))
        labels = {head: [label(outcome, head) for outcome in outcomes]
                  for head in ("mission", "usable_factored")}
        features = PrefixFeatures(feature_problem, BASE, list(map(DesignRecord.from_dict, initial)),
                                  seed, latent=False)
        query = features.transform(heldout)
        for path in PATHS:
            history = initial + [calls[f"{seed}/{path}/{i}"] for i in range(spec["initial"], spec["total"])]
            inputs = features.transform([row["candidate"] for row in history])
            observations = {arm: [timing_feedback(row, settings, use_interval=arm == "timing_interval")
                                  for row in history]
                            for arm in ("timing_sign", "timing_interval")}
            with (unit/f"{path}.jsonl").open(encoding="utf-8") as stream:
                predictions = {row["n"]: row["probabilities"] for row in map(json.loads, stream)
                               if row["event"] == "prediction"}
            for n in spec["learning_checkpoints"]:
                old = predictions[n]
                forecasts = {"original_probit": old["mission"]}
                for arm, evidence in observations.items():
                    gp = StableGroupedHead(inputs[:n], evidence[:n], seed=seed, name=arm,
                                           continuous_count=len(problem.space.continuous),
                                           reference_scale=MARGIN_SCALE)
                    pred = gp.predict(query)
                    forecasts[arm] = ndtr(-pred["mean"]/np.sqrt(np.maximum(pred["variance"], 1e-14)))
                for arm, completion in forecasts.items():
                    usable = np.asarray(completion)*old["feasible_given_mission"]
                    for head, probabilities in (("mission", completion), ("usable_factored", usable)):
                        brier, groups = configuration_brier(zip(
                            heldout, labels[head], np.asarray(probabilities).tolist()))
                        rows.append({"path": path, "seed": seed, "n": n, "arm": arm, "head": head,
                                     "configuration_brier": brier, "scored_configurations": groups})
        print(f"{profile} {seed} replay complete", flush=True)
    summary = paired_scores(rows, spec["learning_checkpoints"])
    with (output/"metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/"analysis.json").write_text(json.dumps({
        "status": "COMPLETE_CENSORED_TIMING_REPLAY", **receipt,
        "archived_aircraft_calls": report["calls"], "new_aircraft_calls": 0, "new_rcaide_mda_calls": 0,
        "comparisons": summary,
    }, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="PR113 profile directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=INPUT_MANIFESTS, required=True)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    replay(args.input, args.output, args.profile, args.expected_sha)


if __name__ == "__main__":
    main()
