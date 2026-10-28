"""Run a fresh mathematical comparison through preparation, audit and reporting."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ra_lsbo.experiments.comparison.runner import audit, prepare, report, run


ROOT = Path(__file__).resolve().parents[3]
METHODS = ("ra_lsbo", "random", "logcei", "scbo_adapted", "sego_utb_adapted", "nsga2_adapted")
EXPECTED_BUDGET = {"smoke": 112, "pilot": 7716}


def validate_plan(plan, profile):
    settings = plan["profile"]
    if (plan["schema"] != "ra_lsbo_external_comparison_v2"
            or plan["suite"] != "mathematical"
            or tuple(plan["arms"]) != METHODS
            or settings["name"] != profile
            or tuple(settings["initial_policies"]) != ("random",)
            or plan["budget"]["maximum_model_calls"] != EXPECTED_BUDGET[profile]
            or plan["budget"]["maximum_rcaide_mda_calls"] != 0):
        raise ValueError("Registered comparison does not match the expected protocol and budget")
    if profile == "pilot" and (tuple(settings["cases"]) != ("branin_sparse", "rosenbrock_sparse")
            or tuple(settings["seeds"]) != tuple(range(9701, 9721))
            or settings["initial"] != 12 or settings["horizon"] != 30):
        raise ValueError("Registered pilot cases, seeds or per-trial budget differ")


def execute(profile="pilot", output=None):
    if profile not in EXPECTED_BUDGET:
        raise ValueError("Unsupported profile")
    if output is None:
        name = f"math_v2_{profile}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        output = ROOT / "runs" / name
    output = Path(output).resolve()
    print(f"Output: {output}", flush=True)
    print(f"[1/4] Preparing {profile} comparison", flush=True)
    plan = prepare(output, profile, "mathematical")
    validate_plan(plan, profile)
    print(f"Maximum model calls: {EXPECTED_BUDGET[profile]}; RCAIDE MDA calls: 0", flush=True)
    print("[2/4] Running comparison", flush=True)
    result = run(output, EXPECTED_BUDGET[profile])
    print(f"Settled model calls: {result['charged_model_calls']}", flush=True)
    print("[3/4] Auditing results", flush=True)
    audit_result = audit(output)
    if not audit_result.get("passed"):
        raise RuntimeError("Audit did not pass; report will not be generated")
    print(f"Audit passed: {audit_result['checked_proposals']} proposals checked", flush=True)
    print("[4/4] Generating report", flush=True)
    report_path = report(output)
    print(f"Report: {report_path}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(EXPECTED_BUDGET), default="pilot")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    return execute(args.profile, args.output)


if __name__ == "__main__":
    main()
