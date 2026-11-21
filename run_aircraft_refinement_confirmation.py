"""Run and aggregate the independent PR153 confirmation over sealed seed shards."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from scipy.stats import binomtest

from aircraft_multi_feasible import AUDIT_CHECKPOINTS, AUDIT_SIZE, PROFILE as MULTI_PROFILE, TARGET, compare_accumulation
from aircraft_structured_risk import (CONFIRMATION_PROFILE, EXTERNAL_PROFILE, LATENT_DECISION, OPTIMIZATION_PROFILE,
                                      PROFILE as PAPER_PROFILE, STRUCTURED_DECISION, STRUCTURED_GREEDY,
                                      STRUCTURED_SIGN_GREEDY)
from ra_lsbo.experiments.core.runner import save
from run_aircraft_external_full_budget import ADAPTATIONS, ONLINE_LOGCEI, SCENARIOS, compare, effect, feasibility_first_counts
from run_aircraft_feedback_replication import paired
from run_aircraft_native_risk_lsbo import (
    COLD_ACTIVE, COLD_ARMS, COLD_CONTRASTS, ENERGY_ARMS, ENERGY_CONTRASTS, ENERGY_GREEDY, ENERGY_LOOKAHEAD,
    EXTERNAL, INITIAL, ORIGINAL, REFINEMENT_CONTRASTS, plan, run, verify,
)
from run_aircraft_refinement_fork import ARMS as FORK_ARMS, FORK_FEASIBLE, PROFILE as FORK_PROFILE, run_fork, verify_fork
from run_aircraft_risk_cold_start import source_commit


ROOT = Path(__file__).resolve().parent
PROFILES = ("smoke", "confirmation", "cold-smoke", "cold-pilot", "energy-pilot", FORK_PROFILE, MULTI_PROFILE,
            PAPER_PROFILE, CONFIRMATION_PROFILE, OPTIMIZATION_PROFILE, EXTERNAL_PROFILE)
PRIMARY = {"scenario": "headwind", "cohort": "all", "budget": 24,
           "contrast": "ra_refine_interval_minus_ra_interval_greedy",
           "endpoint": "feasibility_first_energy_order"}
COLD_PRIMARY = {"scenario": "headwind", "cohort": "cold", "budget": 48,
                "contrast": "ra_refine_interval_cold_active_minus_ra_refine_interval",
                "endpoint": "capped_time_to_energy_target"}
ENERGY_PRIMARY = {"scenario": "headwind", "cohort": "cold", "budget": 48,
                  "contrast": f"{ENERGY_LOOKAHEAD}_minus_{ENERGY_GREEDY}",
                  "endpoint": "feasibility_first_terminal_order"}
FORK_PRIMARY = {"scenario": "headwind", "cohort": "cold_forked", "budget": 48,
                "contrast": f"{FORK_ARMS[1]}_minus_{FORK_ARMS[0]}",
                "endpoint": "feasibility_first_terminal_order"}
MULTI_CONTROL = "ra_refine_interval"
MULTI_PRIMARY = {"scenario": "headwind", "cohort": "cold", "budget": 48,
                 "contrast": f"{COLD_ACTIVE}_minus_{MULTI_CONTROL}",
                 "endpoint": "capped_target_call_to_three_usable_designs"}
PAPER_BUDGET = 32
PAPER_ARMS = (MULTI_CONTROL, COLD_ACTIVE, LATENT_DECISION, STRUCTURED_GREEDY, STRUCTURED_DECISION)
PAPER_CONTRASTS = (
    (STRUCTURED_DECISION, MULTI_CONTROL), (STRUCTURED_GREEDY, MULTI_CONTROL),
    (LATENT_DECISION, MULTI_CONTROL), (STRUCTURED_DECISION, STRUCTURED_GREEDY),
    (STRUCTURED_DECISION, LATENT_DECISION), (STRUCTURED_DECISION, COLD_ACTIVE),
)
PAPER_PRIMARY = {**MULTI_PRIMARY, "budget": PAPER_BUDGET,
                 "contrast": f"{STRUCTURED_DECISION}_minus_{MULTI_CONTROL}"}
CONFIRMATION_ARMS = (ORIGINAL, MULTI_CONTROL, STRUCTURED_SIGN_GREEDY, STRUCTURED_GREEDY, STRUCTURED_DECISION)
CONFIRMATION_CONTRASTS = (
    (STRUCTURED_GREEDY, MULTI_CONTROL), (STRUCTURED_GREEDY, STRUCTURED_SIGN_GREEDY),
    (STRUCTURED_GREEDY, ORIGINAL), (STRUCTURED_DECISION, STRUCTURED_GREEDY),
)
CONFIRMATION_PRIMARY = {**PAPER_PRIMARY, "contrast": f"{STRUCTURED_GREEDY}_minus_{MULTI_CONTROL}"}
OPTIMIZATION_ARMS = (ORIGINAL, MULTI_CONTROL, STRUCTURED_GREEDY)
OPTIMIZATION_CONTRASTS = ((STRUCTURED_GREEDY, ORIGINAL), (STRUCTURED_GREEDY, MULTI_CONTROL))
OPTIMIZATION_ENERGY_TIE_WH = 1.
OPTIMIZATION_PRIMARY = {**PAPER_PRIMARY, "contrast": f"{STRUCTURED_GREEDY}_minus_{ORIGINAL}",
                        "endpoint": "capped_time_to_energy_target",
                        "co_primary_endpoint": "feasibility_first_terminal_order"}
EXTERNAL_ARMS = (STRUCTURED_GREEDY, EXTERNAL, ONLINE_LOGCEI)
EXTERNAL_CONTRASTS = ((STRUCTURED_GREEDY, EXTERNAL), (STRUCTURED_GREEDY, ONLINE_LOGCEI))
EXTERNAL_PRIMARY = {**OPTIMIZATION_PRIMARY, "contrast": f"{STRUCTURED_GREEDY}_minus_{EXTERNAL}"}
THREAD_LIMITS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMBA_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
WORKER_ENTRY = (
    "import sys; sys.path.insert(0, sys.argv.pop(1)); "
    "from run_aircraft_refinement_confirmation import main; main()"
)


def seed_registry(profile):
    if profile in ("smoke", "cold-smoke"):
        return {scenario: plan(scenario, "smoke", "refinement")["seeds"] for scenario in SCENARIOS}
    base, size = {"cold-pilot": (29400100, 3), "energy-pilot": (29430100, 6),
                  FORK_PROFILE: (29460100, 6), MULTI_PROFILE: (29560100, 12),
                  PAPER_PROFILE: (29620100, 24), CONFIRMATION_PROFILE: (29700100, 24),
                  OPTIMIZATION_PROFILE: (29760100, 24), EXTERNAL_PROFILE: (29820100, 24)}.get(profile, (29300100, 24))
    return {scenario: list(range(base+10_000*i, base+10_000*i+100*size, 100))
            for i, scenario in enumerate(SCENARIOS)}


def shard_plan(profile, scenario, seed):
    if seed not in seed_registry(profile)[scenario]:
        raise ValueError("Seed is outside the registered cohort")
    spec = plan(scenario, "smoke" if profile in ("smoke", "cold-smoke") else "full-pilot", "refinement")
    spec.update(name=f"{profile}-refinement-{scenario}", analysis_role=profile, seeds=[seed])
    if profile in (OPTIMIZATION_PROFILE, EXTERNAL_PROFILE):
        arms, contrasts, primary = ((OPTIMIZATION_ARMS, OPTIMIZATION_CONTRASTS, OPTIMIZATION_PRIMARY)
                                    if profile == OPTIMIZATION_PROFILE else
                                    (EXTERNAL_ARMS, EXTERNAL_CONTRASTS, EXTERNAL_PRIMARY))
        spec.update(arms=list(arms), contrasts=[list(pair) for pair in contrasts],
                    total_budget=PAPER_BUDGET, horizon=PAPER_BUDGET-INITIAL, quality_budgets=[16, 24, 32],
                    calls_per_seed=INITIAL+len(arms)*(PAPER_BUDGET-INITIAL),
                    usable_acquisition_target=TARGET, primary=primary,
                    energy_order_tie_Wh=OPTIMIZATION_ENERGY_TIE_WH,
                    post_feasible_policy="risk-aware arms use interval-informed refinement after three usable designs")
        if profile == EXTERNAL_PROFILE:
            spec.update(logcei_global_candidates=512, logcei_local_candidates=128,
                        logcei_adapter=ADAPTATIONS["logcei"],
                        pool_policy="common initial designs and exact screen; each optimizer uses its own proposal search",
                        post_feasible_policy="structured RA-LSBO switches after three usable designs; "
                                             "LogCEI uses EI after its first and ArchSBO follows its native search")
    elif profile in (MULTI_PROFILE, PAPER_PROFILE, CONFIRMATION_PROFILE):
        if profile in (PAPER_PROFILE, CONFIRMATION_PROFILE):
            arms, contrasts, primary = ((PAPER_ARMS, PAPER_CONTRASTS, PAPER_PRIMARY) if profile == PAPER_PROFILE else
                                        (CONFIRMATION_ARMS, CONFIRMATION_CONTRASTS, CONFIRMATION_PRIMARY))
            spec.update(total_budget=PAPER_BUDGET, horizon=PAPER_BUDGET-INITIAL, quality_budgets=[16, 24, 32])
        else:
            arms, contrasts, primary = (MULTI_CONTROL, COLD_ACTIVE), ((COLD_ACTIVE, MULTI_CONTROL),), MULTI_PRIMARY
        post_policy = ("risk-aware arms use interval-informed refinement after three usable designs"
                       if profile == CONFIRMATION_PROFILE else
                       "common interval-informed LSBO refinement after three usable designs")
        spec.update(arms=list(arms), contrasts=[list(pair) for pair in contrasts],
                    calls_per_seed=INITIAL+len(arms)*spec["horizon"]+AUDIT_SIZE, usable_acquisition_target=TARGET,
                    completion_audit_size=AUDIT_SIZE, completion_audit_checkpoints=list(AUDIT_CHECKPOINTS),
                    post_feasible_policy=post_policy,
                    primary=primary)
    elif profile == FORK_PROFILE:
        spec.update(arms=list(FORK_ARMS), contrasts=[list(reversed(FORK_ARMS))],
                    calls_per_seed=INITIAL+len(FORK_ARMS)*spec["horizon"],
                    fork_feasible=FORK_FEASIBLE, shared_policy="native interval RA-LSBO",
                    post_feasible_policy="same native risk acquisition with exact-plus-censored / exact-only correction",
                    quality_budgets=[24, 32, 48], primary=FORK_PRIMARY)
    elif profile.startswith("cold-") or profile == "energy-pilot":
        arms, contrasts, primary = ((ENERGY_ARMS, ENERGY_CONTRASTS, ENERGY_PRIMARY) if profile == "energy-pilot"
                                    else (COLD_ARMS, COLD_CONTRASTS, COLD_PRIMARY))
        spec.update(arms=list(arms), contrasts=[list(pair) for pair in contrasts],
                    calls_per_seed=INITIAL+len(arms)*spec["horizon"],
                    post_feasible_policy="common interval-informed risk refinement within native LSBO",
                    primary=primary)
    return spec


def registration(profile):
    scenarios = seed_registry(profile)
    primary = {"cold-smoke": COLD_PRIMARY, "cold-pilot": COLD_PRIMARY,
               "energy-pilot": ENERGY_PRIMARY, FORK_PROFILE: FORK_PRIMARY,
               MULTI_PROFILE: MULTI_PRIMARY, PAPER_PROFILE: PAPER_PRIMARY,
               CONFIRMATION_PROFILE: CONFIRMATION_PRIMARY, OPTIMIZATION_PROFILE: OPTIMIZATION_PRIMARY,
               EXTERNAL_PROFILE: EXTERNAL_PRIMARY}.get(profile, PRIMARY)
    budget_field = "maximum_calls_per_seed" if profile == FORK_PROFILE else "calls_per_seed"
    registration = {"profile": profile, "primary": primary, "scenarios": {
        scenario: {"seeds": seeds, budget_field: shard_plan(profile, scenario, seeds[0])["calls_per_seed"]}
        for scenario, seeds in scenarios.items()}}
    if profile in (MULTI_PROFILE, PAPER_PROFILE, CONFIRMATION_PROFILE):
        registration.update(initial_evaluations=INITIAL, usable_acquisition_target=TARGET,
                            heldout_evaluations_per_seed=AUDIT_SIZE,
                            completion_audit_checkpoints=list(AUDIT_CHECKPOINTS))
    if profile in (PAPER_PROFILE, CONFIRMATION_PROFILE, OPTIMIZATION_PROFILE, EXTERNAL_PROFILE):
        spec = shard_plan(profile, SCENARIOS[0], scenarios[SCENARIOS[0]][0])
        registration.update(arms=spec["arms"], contrasts=spec["contrasts"])
    if profile in (OPTIMIZATION_PROFILE, EXTERNAL_PROFILE):
        registration.update(initial_evaluations=INITIAL, usable_acquisition_target=TARGET,
                            quality_budgets=spec["quality_budgets"], energy_order_tie_Wh=OPTIMIZATION_ENERGY_TIE_WH,
                            energy_targets_Wh={scenario: shard_plan(profile, scenario, seeds[0])["energy_target_Wh"]
                                               for scenario, seeds in scenarios.items()})
    if profile == EXTERNAL_PROFILE:
        registration.update({key: spec[key] for key in ("logcei_global_candidates", "logcei_local_candidates", "logcei_adapter")})
    return registration


def launch(root, profile, sha, jobs):
    registry = seed_registry(profile)
    root.mkdir(parents=True, exist_ok=True)
    log_root = root/"logs"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    log_root.mkdir(parents=True)
    env = {**os.environ, **dict.fromkeys(THREAD_LIMITS, "1"), "PYTHONUNBUFFERED": "1"}

    def worker(scenario, seed):
        folder = root/scenario/str(seed)
        if (folder/"seal.json").is_file():
            return "sealed"
        if folder.exists():
            return "partial directory preserved"
        command = [sys.executable, "-c", WORKER_ENTRY, str(ROOT), "--task", "seed",
                   "--profile", profile, "--scenario", scenario, "--seed", str(seed),
                   "--expected-sha", sha, "--output", str(folder)]
        log = log_root/f"{scenario}-{seed}.log"
        with log.open("xb") as stream:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, check=False)
        if process.returncode or not (folder/"seal.json").is_file():
            return f"failed (exit {process.returncode}, see {log})"
        return "completed"

    tasks = [(scenario, seed) for group in zip(*(registry[s] for s in SCENARIOS))
             for scenario, seed in zip(SCENARIOS, group)]
    failed = []
    with ThreadPoolExecutor(max_workers=min(jobs, len(tasks))) as executor:
        pending = {executor.submit(worker, scenario, seed): (scenario, seed) for scenario, seed in tasks}
        for future in as_completed(pending):
            scenario, seed = pending[future]
            status = future.result()
            print(scenario, seed, status, flush=True)
            if status not in ("completed", "sealed"):
                failed.append(f"{scenario}/{seed}")
    if failed:
        raise RuntimeError("Unsealed registered tasks: "+", ".join(sorted(failed)))


def quality_contrast(rows, left, right, budget, *, tie_Wh=0.):
    pairs = [(row["arms"][left]["quality_by_budget"][str(budget)],
              row["arms"][right]["quality_by_budget"][str(budget)]) for row in rows]
    order = feasibility_first_counts(pairs, tie_Wh=tie_Wh)
    wins, losses = order["wins"], order["losses"]
    return {
        "feasibility_first_order": {
            **order,
            "exact_sign_p_two_sided": float(binomtest(wins, wins+losses).pvalue) if wins+losses else None,
        },
        "energy_Wh_delta_both_usable": paired([a-b for a, b in pairs if a is not None and b is not None]),
    }


def target_contrast(rows, left, right, budget):
    times = [(row["arms"][left]["first_energy_target"], row["arms"][right]["first_energy_target"])
             for row in rows]
    return {
        "attainment_rate_delta": effect([int(a is not None)-int(b is not None) for a, b in times]),
        "capped_time_delta": paired([(a if a is not None else budget+1)-(b if b is not None else budget+1)
                                     for a, b in times]),
    }


def cohort_summary(rows, spec):
    tie_Wh = spec.get("energy_order_tie_Wh", 0.)
    return {
        "n": len(rows),
        "arms": {
            arm: {
                "found_usable": sum(row["arms"][arm]["first_usable"] is not None for row in rows),
                "hit_energy_target": sum(row["arms"][arm]["first_energy_target"] is not None for row in rows),
                "refinement_risk_calls": sum(row["arms"][arm]["refinement_risk_calls"] for row in rows),
                **({key: sum(row["arms"][arm][key] for row in rows)
                    for key in ("energy_policy_calls", "energy_policy_fallbacks", "energy_bounds_only_calls")}
                   if arm in (ENERGY_GREEDY, ENERGY_LOOKAHEAD) else {}),
                **({"energy_adaptive_calls": sum(row["arms"][arm]["energy_adaptive_calls"] for row in rows)}
                   if arm == ENERGY_LOOKAHEAD else {}),
                **({key: sum(row["arms"][arm][key] for row in rows)
                    for key in ("energy_update_calls", "energy_update_fallbacks")}
                   if spec["analysis_role"] == FORK_PROFILE else {}),
                "feasible_at_checkpoints": {
                    str(b): sum(row["arms"][arm]["quality_by_budget"][str(b)] is not None for row in rows)
                    for b in spec["quality_budgets"]},
            } for arm in spec["arms"]},
        "contrasts": {
            f"{a}_minus_{b}": {
                **compare(rows, a, b, tie_Wh=tie_Wh),
                "quality_by_budget": {str(t): quality_contrast(rows, a, b, t, tie_Wh=tie_Wh)
                                      for t in spec["quality_budgets"]},
                "energy_target": target_contrast(rows, a, b, spec["total_budget"]),
            }
            for a, b in spec.get("contrasts", REFINEMENT_CONTRASTS)},
    }


def analyze(root, profile, sha):
    data = {}
    versions = None
    for scenario, seeds in seed_registry(profile).items():
        rows = []
        for seed in seeds:
            spec = shard_plan(profile, scenario, seed)
            verify_one = verify_fork if profile == FORK_PROFILE else verify
            row = verify_one(root/scenario/str(seed), spec, sha)
            if versions is None:
                versions = row["versions"]
            if row["versions"] != versions:
                raise ValueError("All registered shards must use the same model runtime")
            rows.append(row)
        data[scenario] = {
            "energy_target_Wh": spec["energy_target_Wh"],
            "summary": {name: cohort_summary(group, spec) for name, group in (
                ("all", rows), ("cold", [row for row in rows if row["cold"]]),
                ("initially_usable", [row for row in rows if not row["cold"]]))},
            "rows": rows,
        }
        if profile == FORK_PROFILE:
            forked = [row for row in rows if row["fork_evaluation"] is not None]
            cold_forked = [row for row in forked if row["cold"]]
            pending = [row for row in cold_forked if not row["shared_target_reached"]]
            data[scenario]["summary"].update(forked=cohort_summary(forked, spec),
                                             cold_forked=cohort_summary(cold_forked, spec),
                                             cold_forked_target_pending=cohort_summary(pending, spec))
            # Subtract the common prefix from both arms and cap a miss at remaining budget + 1.
            post_fork_delta = []
            for row in pending:
                times = [row["arms"][arm]["first_energy_target"] or spec["total_budget"]+1 for arm in FORK_ARMS]
                post_fork_delta.append(times[1]-times[0])
            data[scenario]["post_fork_target_time_delta_cold_target_pending"] = paired(post_fork_delta)
        if profile == MULTI_PROFILE:
            data[scenario]["accumulation"] = {
                name: compare_accumulation(group, COLD_ACTIVE, MULTI_CONTROL, AUDIT_CHECKPOINTS)
                for name, group in (("all", rows), ("cold", [row for row in rows if row["cold"]]))
            }
        if profile in (PAPER_PROFILE, CONFIRMATION_PROFILE, OPTIMIZATION_PROFILE, EXTERNAL_PROFILE):
            cohorts = (("all", rows), ("cold", [row for row in rows if row["cold"]]),
                       ("below_initial_target", [row for row in rows if row["initial_usable"] < TARGET]))
            accumulation = {}
            for name, group in cohorts:
                contrasts = {f"{a}_minus_{b}": compare_accumulation(group, a, b, spec.get("completion_audit_checkpoints", ()))
                             for a, b in spec["contrasts"]}
                accumulation[name] = {"n": len(group), "contrasts": contrasts}
                if profile == PAPER_PROFILE:
                    interaction = [sum(sign*row["arms"][arm]["accumulation"]["capped_target_call"]
                                       for arm, sign in ((STRUCTURED_DECISION, 1), (STRUCTURED_GREEDY, -1),
                                                         (LATENT_DECISION, -1), (MULTI_CONTROL, 1))) for row in group]
                    accumulation[name]["structure_by_lookahead_time_interaction"] = paired(interaction)
            data[scenario]["accumulation"] = accumulation
    if profile == EXTERNAL_PROFILE:
        status = "VERIFIED_STRUCTURED_RISK_EXTERNAL"
    elif profile == OPTIMIZATION_PROFILE:
        status = "VERIFIED_STRUCTURED_RISK_OPTIMIZATION"
    elif profile == CONFIRMATION_PROFILE:
        status = "VERIFIED_STRUCTURED_RISK_CONFIRMATION"
    elif profile == PAPER_PROFILE:
        status = "VERIFIED_STRUCTURED_RISK_PAPER"
    elif profile == MULTI_PROFILE:
        status = "VERIFIED_MULTI_FEASIBLE_COLD_START"
    elif profile == FORK_PROFILE:
        status = "VERIFIED_CENSORED_ENERGY_REFINEMENT_FORK"
    elif profile == "energy-pilot":
        status = "VERIFIED_COLD_ENERGY_VALUE_PILOT"
    elif profile.startswith("cold-"):
        status = "VERIFIED_COLD_ACQUISITION_TO_ENERGY"
    else:
        status = "VERIFIED_RISK_AWARE_LSBO_CONFIRMATION"
    return {"status": status, "source_sha": sha,
            "plan": registration(profile), "versions": versions, "scenarios": data,
            "paid_calls": sum(row["paid_calls"] for section in data.values() for row in section["rows"]),
            "rcaide_mda_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plan", "run", "seed", "verify"), required=True)
    parser.add_argument("--profile", choices=PROFILES, default="confirmation")
    parser.add_argument("--expected-sha")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.task == "plan":
        print(json.dumps(registration(args.profile), indent=2))
        return
    if not args.expected_sha or args.output is None:
        parser.error("--expected-sha and --output are required")
    output = args.output.resolve()
    if ROOT in (output, *output.parents):
        parser.error("--output must be outside the code worktree")
    sha = source_commit(args.expected_sha, clean=True)
    if args.task == "run":
        if not 1 <= args.jobs <= 12:
            parser.error("Use between 1 and 12 independent workers")
        launch(output, args.profile, sha, args.jobs)
    elif args.task == "seed":
        if args.scenario is None or args.seed is None or output.exists():
            parser.error("Supply a registered scenario and seed and a new output directory")
        spec = shard_plan(args.profile, args.scenario, args.seed)
        runner = run_fork if args.profile == FORK_PROFILE else run
        runner(spec, args.seed, sha, output)
    else:
        if args.input is None or args.input.resolve() in (output, *output.parents):
            parser.error("Provide a sealed input root and write its report outside it")
        result = analyze(args.input.resolve(), args.profile, sha)
        save(output, result)
        print(result["status"], args.profile, result["paid_calls"], flush=True)


if __name__ == "__main__":
    main()
