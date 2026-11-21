"""Compare early-termination feedback within LSBO, with ArchSBO as an external control."""

import argparse
from functools import partial
import importlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr

from aircraft_multi_feasible import accumulation_metrics, acquisition_stage, completion_audit
from aircraft_structured_risk import (
    COMPACT_PROFILES, DECISION_ARMS, STRUCTURED_ARMS, STRUCTURED_SIGN_GREEDY, HierarchicalTimingHead,
    choose_completion_with_lookahead, with_configuration,
)
from aircraft_timing_energy_value import energy_evidence, target_choice, target_marks
from aircraft_timing_information import score_information, supports_timing
from aircraft_timing_interval_value import deficit_edges
from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.experiments.core.runner import Ledger, save
from ra_lsbo.risk_core.completion_refinement import IntervalCompletionProbability, RiskWeightedRefinement
from run_aircraft_coverage_factorial import FixedTimingHead
from run_aircraft_external_full_budget import (
    SCENARIOS, UsableAircraft, compare, discovery_outcome, evidence_hashes, passes_exact_screen,
    plan as full_budget_plan, read, read_ledger, rosters, structural_margins, trajectory_metrics,
)
from run_aircraft_ra_lsbo_comparison import FullBudgetLSBO
from run_aircraft_risk_cold_start import file_hash, select, source_commit


INITIAL = 8
CANDIDATES = 64
ORIGINAL = "lsbo_native"
EXTERNAL = "archsbo_screened"
ARMS = (ORIGINAL, "ra_sign_greedy", "ra_interval_greedy", "ra_interval_active", EXTERNAL)
CONTRASTS = (("ra_interval_greedy", "ra_sign_greedy"), ("ra_interval_active", "ra_interval_greedy"),
             ("ra_interval_greedy", ORIGINAL), ("ra_interval_greedy", EXTERNAL))
PROFILES = ("smoke", "full-pilot")
STUDIES = ("discovery", "refinement")
REFINEMENT_ARMS = (ORIGINAL, "ra_interval_greedy", "ra_refine_sign", "ra_refine_interval", EXTERNAL)
REFINEMENT_CONTRASTS = (
    ("ra_refine_interval", "ra_interval_greedy"), ("ra_refine_interval", "ra_refine_sign"),
    ("ra_refine_interval", ORIGINAL), ("ra_refine_interval", EXTERNAL),
)
COLD_SIGN = "ra_refine_interval_cold_sign"
COLD_ACTIVE = "ra_refine_interval_cold_active"
COLD_ARMS = (COLD_SIGN, "ra_refine_interval", COLD_ACTIVE)
COLD_CONTRASTS = (("ra_refine_interval", COLD_SIGN), (COLD_ACTIVE, "ra_refine_interval"))
ENERGY_GREEDY = "ra_refine_interval_cold_energy_greedy"
ENERGY_LOOKAHEAD = "ra_refine_interval_cold_energy_lookahead"
ENERGY_POLICIES = {ENERGY_GREEDY: "energy_greedy", ENERGY_LOOKAHEAD: "energy_lookahead"}
ENERGY_ARMS = ("ra_refine_interval", COLD_ACTIVE, *ENERGY_POLICIES)
ENERGY_CONTRASTS = ((ENERGY_LOOKAHEAD, ENERGY_GREEDY), (ENERGY_LOOKAHEAD, COLD_ACTIVE),
                    (ENERGY_GREEDY, "ra_refine_interval"))
REFINEMENT_FEEDBACK = {"ra_refine_sign": "sign", **dict.fromkeys(
    (*COLD_ARMS, *ENERGY_POLICIES, *STRUCTURED_ARMS, *DECISION_ARMS), "interval")}
QUALITY_BUDGETS = (16, 24, 32, 48)
ENERGY_TARGET_WH = {"payload": 38000., "headwind": 40000.}


def plan(scenario, profile, study="discovery"):
    offset = 0 if scenario == "payload" else 10_000
    seed, budget = (29060100+offset, 12) if profile == "smoke" else (29200100+offset, 48)
    base = full_budget_plan(scenario, "full-pilot")
    del base["pooled_arms"], base["logcei_adapter"]
    spec = {**base, "name": f"{profile}-{scenario}", "analysis_role": profile, "seeds": [seed],
            "initial_sizes": [INITIAL], "total_budget": budget, "horizon": budget-INITIAL,
            "calls_per_seed": INITIAL+len(ARMS)*(budget-INITIAL), "arms": list(ARMS),
            "native_gp_steps": 45, "risk_candidate_count": CANDIDATES,
            "pool_policy": "LSBO grammar-decoded plus native mixed proposals in the risk-aware cold stage",
            "post_feasible_policy": "all LSBO arms continue through their own native LSBO engine",
            "primary": {"scenario": "headwind", "initial": INITIAL, "cohort": "cold",
                        "contrast": "ra_interval_greedy_minus_ra_sign_greedy",
                        "endpoint": "first_usable_total_call_capped_at_budget_plus_one"}}
    if study == "refinement":
        seed, budget = (29060100+offset, 16) if profile == "smoke" else (29220100+offset, 48)
        spec.update(study=study, name=f"{profile}-refinement-{scenario}", seeds=[seed],
                    arms=list(REFINEMENT_ARMS), total_budget=budget, horizon=budget-INITIAL,
                    calls_per_seed=INITIAL+len(REFINEMENT_ARMS)*(budget-INITIAL),
                    energy_target_Wh=ENERGY_TARGET_WH[scenario],
                    energy_target_basis="rounded reduced-model landmarks from the earlier PR151 pilot",
                    quality_budgets=[b for b in QUALITY_BUDGETS if b <= budget],
                    post_feasible_policy="matched native LSBO; refinement variants replace the completion term in the acquisition",
                    primary={"scenario": "headwind", "budget": 24, "cohort": "all",
                             "contrast": "ra_refine_interval_minus_ra_interval_greedy",
                             "endpoint": "feasibility_first_terminal_order"})
    return spec


def cold_policy(arm):
    feedback = "sign" if arm in ("ra_sign_greedy", COLD_SIGN, STRUCTURED_SIGN_GREEDY) else "interval"
    if arm in DECISION_ARMS:
        policy = "completion_lookahead"
    else:
        policy = ENERGY_POLICIES.get(arm, "information" if arm in ("ra_interval_active", COLD_ACTIVE) else "greedy")
    return feedback, policy


def engineering_screen(problem, point):
    return {**problem.precheck(point), **structural_margins(problem, point)}


def choose_cold(head, inputs, trusted, query, candidates, settings, policy, energy_target, seed):
    if policy == "greedy":
        moments = head.predict(query)
        p = ndtr(-moments["mean"]/np.sqrt(np.maximum(moments["variance"], 1e-14)))
        if not np.isfinite(p).all():
            raise FloatingPointError("Invalid LSBO completion posterior")
        index = select(candidates, p)
        return index, {"p_complete": float(p[index]), "expected_brier_reduction": None}

    edges = [deficit_edges(point, settings) for point in candidates]
    if policy == "completion_lookahead":
        return choose_completion_with_lookahead(head, query, candidates, edges)
    receipt = {}
    if policy in ENERGY_POLICIES.values():
        evidence = [energy_evidence(record) for record in trusted]
        mask = [observation is not None for observation in evidence]
        observed = [observation for observation in evidence if observation is not None]
        receipt["energy_observations"] = len(observed)
        receipt["energy_lower_bounds"] = sum(energy.kind == "interval" for energy, _ in observed)
        try:
            marks = target_marks(inputs[mask]/head.divisor, query/head.divisor, observed, energy_target, seed)
            if marks is not None and not np.isfinite(marks).all():
                raise FloatingPointError("Invalid energy target marks")
            if marks is not None and (len(query) == 1 or np.ptp(marks) > 1e-12):
                index, scores = target_choice(head, query, candidates, edges, marks,
                                              lookahead=policy == "energy_lookahead")
                return index, {**receipt, **scores, "expected_brier_reduction": None}
            receipt["energy_fallback"] = "no_energy_evidence" if marks is None else "constant_target_marks"
        except (np.linalg.LinAlgError, FloatingPointError, RuntimeError):
            receipt["energy_fallback"] = "invalid_energy_posterior"

    p, _, eligible, gains = score_information(head, query, candidates, edges)
    pick = select([candidates[i] for i in eligible], gains)
    index = int(eligible[pick])
    return index, {**receipt, "p_complete": float(p[index]), "expected_brier_reduction": float(gains[pick])}


class RiskAwareLSBO(FullBudgetLSBO):
    def __init__(self, *args, arm, **kwargs):
        super().__init__(*args, **kwargs)
        self.arm = arm

    def _refinement_energy(self, surrogate, inputs, trusted, head, state):
        return surrogate

    def _propose(self, history):
        if not acquisition_stage(history, self.profile.get("usable_acquisition_target", 1)):
            if self.arm in REFINEMENT_FEEDBACK:
                return self.engine.propose(selection=partial(self._refinement_proposal, history=history))
            return self.engine.propose()
        return self.engine.propose(selection=partial(self._cold_proposal, history=history))

    def _refinement_proposal(self, native, phase, *, history):
        trusted = [record for record in history if supports_timing(record)]
        state = {"mode": "native_recovery_or_exploration"}

        def risk_surrogate(baseline, representation):
            feedback = REFINEMENT_FEEDBACK[self.arm]
            if not trusted:
                state.update(mode="native_fallback", reason="no_supported_completion_feedback")
                return baseline
            inputs = representation.features([
                self.problem.candidate_class(**record.candidate) for record in trusted]).numpy()
            try:
                head = FixedTimingHead(self.problem, inputs, trusted, self.seed, use_interval=feedback == "interval")
            except (np.linalg.LinAlgError, FloatingPointError, RuntimeError):
                state.update(mode="native_fallback", reason="invalid_completion_posterior")
                return baseline
            completion = IntervalCompletionProbability(head)
            state.update(mode="risk_weighted_native_search", feedback=feedback, timing_observations=len(trusted),
                         representation_version=representation.version)
            return self._refinement_energy(RiskWeightedRefinement(baseline, completion),
                                           inputs, trusted, head, state)

        candidate, metadata = native.propose(phase, refine_surrogate=risk_surrogate)
        return candidate, {**metadata, "refinement_risk": state}

    def _cold_proposal(self, native, phase, *, history):
        trusted = [record for record in history if supports_timing(record)]
        if not trusted:
            candidate, info = native.propose(phase)
            return candidate, {**info, "risk_fallback": "no_supported_completion_feedback"}
        rep = native.representation
        if rep.version == 0 or len(native.observations)-native.last_representation_update >= native.settings.update_every:
            epochs = native.settings.pretrain_epochs if rep.version == 0 else native.settings.update_epochs
            rep.fit(native.observations, epochs)
            native.last_representation_update = len(native.observations)
            torch.save(rep.checkpoint(), native.output/"representation.pt")
        pool, decoded = self._native_cold_pool(native, phase)
        if not pool:
            candidate, info = native.propose(phase)
            return candidate, {**info, "risk_fallback": "no_screened_cold_candidates"}
        to_dict = [candidate.to_dict() for candidate in pool]
        inputs = rep.features([self.problem.candidate_class(**record.candidate) for record in trusted]).numpy()
        query = rep.features(pool).numpy()
        feedback, policy = cold_policy(self.arm)
        head_type = FixedTimingHead
        model_receipt = {}
        if self.arm in STRUCTURED_ARMS:
            inputs = with_configuration(inputs, [record.candidate for record in trusted])
            query = with_configuration(query, to_dict)
            head_type = HierarchicalTimingHead
            model_receipt["timing_kernel"] = "hierarchical_configuration"
        last_decision = policy == "completion_lookahead" and len(history)+1 == self.profile["total_budget"]
        if last_decision:
            model_receipt["last_call_greedy"] = True
        try:
            head = head_type(self.problem, inputs, trusted, self.seed, use_interval=feedback == "interval")
            index, scores = choose_cold(head, inputs, trusted, query, to_dict, self.problem.settings,
                                        "greedy" if last_decision else policy,
                                        self.profile.get("energy_target_Wh"), self.seed)
        except (np.linalg.LinAlgError, FloatingPointError, RuntimeError):
            candidate, info = native.propose(phase)
            return candidate, {**info, "risk_fallback": "invalid_completion_posterior"}
        return pool[index], {"source": "lsbo_latent_completion_risk", "feedback": feedback,
                             "policy": policy, **scores, **model_receipt,
                             "candidate_count": len(pool), "candidate_list_sha256": digest(to_dict),
                             "latent_decoded_candidates": decoded,
                             "representation_version": rep.version, "timing_observations": len(trusted)}

    def _native_cold_pool(self, native, phase):
        factory = importlib.import_module(type(native).__module__).mixed_candidate_pool
        rep, rng, space = native.representation, native.rng, native.space
        target = native.coverage_target(phase)
        limit = self.profile["risk_candidate_count"]
        latents = np.vstack([rep.latent(row.candidate.discrete).numpy() for row in native.observations])
        low, high = latents.min(axis=0)-.5, latents.max(axis=0)+.5
        candidates, keys = [], set()
        for _ in range(limit):
            if len(candidates) >= limit//2:
                break
            discrete = rep.decode(rng.uniform(low, high), rng, stochastic=True)
            if target and any(discrete[name] != value for name, value in target.items()):
                continue
            point = space.sample(rng, discrete)
            key = space.key(point)
            if key not in keys and native.accepted(point):
                candidates.append(point)
                keys.add(key)
        return candidates+factory(native, limit-len(candidates), target, keys), len(candidates)


def seed_evidence(folder, spec):
    if spec.get("analysis_role") in COMPACT_PROFILES:
        files = ["plan.json", "calls.jsonl", *(arm+".json" for arm in spec["arms"])]
        return {name: file_hash(folder/name) for name in files}
    return evidence_hashes(folder)


def run(spec, seed, sha, output):
    if EXTERNAL in spec["arms"]:
        from run_aircraft_archsbo_screened import ARCHSBO_VERSION, screened_trajectory

        if version("sb-arch-opt") != ARCHSBO_VERSION:
            raise ValueError("Unexpected screened ArchSBO version")
    torch.set_num_threads(1)
    problem = UsableAircraft(spec)
    initial, pools = rosters(problem, spec, seed)
    initial = initial[:INITIAL]
    output.mkdir(parents=True)
    packages = ("botorch", "gpytorch", "numpy", "scipy")
    if EXTERNAL in spec["arms"] or (spec.get("analysis_role") not in COMPACT_PROFILES
                                   and "completion_audit_size" not in spec):
        packages += ("sb-arch-opt", "pymoo", "smt", "scikit-learn", "numba")
    versions = {name: version(name) for name in packages}
    versions["torch_runtime"] = str(torch.__version__)
    save(output/"plan.json", {"spec": spec, "seed": seed, "source_sha": sha, "contract": problem.contract_id,
                             "versions": versions})
    ledger = Ledger(output, spec["calls_per_seed"])

    def pay(point, tag, role):
        if not passes_exact_screen(problem, point):
            raise ValueError("The candidate failed the common engineering screen")
        return ledger.evaluate(problem, point, tag, role)

    prefix = [pay(point, f"initial/{i}", "initial") for i, point in enumerate(initial, 1)]
    for arm in spec["arms"]:
        def acquire(point, step):
            return pay(point, f"{arm}/{step}", "online")
        if arm == EXTERNAL:
            trace = screened_trajectory(problem, prefix, seed, spec["horizon"], acquire)["steps"]
        else:
            torch.manual_seed(seed)
            factory = FullBudgetLSBO if arm == ORIGINAL else partial(RiskAwareLSBO, arm=arm)
            engine = factory(problem, BASE, seed, spec, output/arm, precheck=partial(engineering_screen, problem))
            history, trace = list(prefix), []
            for step in range(1, spec["horizon"]+1):
                point, info = engine.propose(history, [], step-1)
                record = acquire(point, step)
                engine.accept(record)
                trace.append({"step": step, "candidate_id": digest(point),
                              "usable": record.usable_feasible, "outcome": discovery_outcome(record.to_dict()),
                              "proposal": info["original_proposal"], "lsbo_phase": info["original_phase"]})
                history.append(record)
        save(output/(arm+".json"), trace)
        print(spec["name"], seed, arm, len(trace), ledger.count, flush=True)
    for index, point in enumerate(pools["covered"][:spec.get("completion_audit_size", 0)], 1):
        pay(point, f"risk_audit/{index}", "heldout")
    if ledger.count != spec["calls_per_seed"]:
        raise ValueError("Evaluation budget differs from the registered plan")
    save(output/"seal.json", seed_evidence(output, spec))


def verify(folder, spec, sha):
    seed = spec["seeds"][0]
    problem = UsableAircraft(spec)
    header = read(folder/"plan.json")
    if (header["spec"] != spec or header["seed"] != seed or header["source_sha"] != sha
            or header["contract"] != problem.contract_id or read(folder/"seal.json") != seed_evidence(folder, spec)):
        raise ValueError("The run does not match the registered sealed plan")
    calls = read_ledger(folder, problem.contract_id)
    if len(calls) != spec["calls_per_seed"]:
        raise ValueError("Unexpected physical evaluation count")
    position = 0

    def settled(tag, role):
        nonlocal position
        position += 1
        number, record, actual_role = calls[tag]
        if (number, actual_role) != (position, role) or not passes_exact_screen(problem, record.candidate):
            raise ValueError("A paid call failed ledger order or engineering checks")
        return record

    prefix = [settled(f"initial/{i}", "initial") for i in range(1, INITIAL+1)]
    initial, pools = rosters(problem, spec, seed)
    if [record.candidate for record in prefix] != initial[:INITIAL]:
        raise ValueError("Unexpected shared initialization")
    arms, candidate_lists, histories = {}, {}, {}
    usable_target = spec.get("usable_acquisition_target", 1)
    refinement = spec.get("study") == "refinement"
    for arm in spec["arms"]:
        history = list(prefix)
        seen = {record.observation.design_id for record in prefix}
        trace = read(folder/(arm+".json"))
        if len(trace) != spec["horizon"]:
            raise ValueError("An arm did not spend its registered budget")
        risk_calls, fallbacks, refinement_calls = 0, 0, 0
        energy_calls, energy_fallbacks, bounds_only_calls, adaptive_calls = 0, 0, 0, 0
        for i, step in enumerate(trace, 1):
            record = settled(f"{arm}/{i}", "online")
            key = record.observation.design_id
            if key in seen or (
                step["step"], step["candidate_id"], step["outcome"], step["usable"]
            ) != (i, key, discovery_outcome(record.to_dict()), record.usable_feasible):
                raise ValueError("A proposal does not match its settled observation")
            if arm == EXTERNAL:
                if (step["paid_calls"], step["candidate"]) != (i, record.candidate):
                    raise ValueError("ArchSBO paid proposal does not match its ledger")
            else:
                proposal = step["proposal"]
                if proposal["source"] == "lsbo_latent_completion_risk":
                    if arm in (ORIGINAL, EXTERNAL) or not acquisition_stage(history, usable_target):
                        raise ValueError("A cold completion policy used an unregistered phase")
                    feedback, policy = cold_policy(arm)
                    if (proposal["feedback"], proposal["policy"]) != (feedback, policy):
                        raise ValueError("The receipt differs from the registered cold-start policy")
                    if (proposal.get("timing_kernel") == "hierarchical_configuration") != (arm in STRUCTURED_ARMS):
                        raise ValueError("The receipt differs from the registered timing kernel")
                    if arm in DECISION_ARMS:
                        if len(history)+1 == spec["total_budget"]:
                            if proposal.get("last_call_greedy") is not True:
                                raise ValueError("Last call should use immediate completion probability")
                        elif not (0. <= proposal["p_complete"] <= proposal["fixed_second_completion"] + 1e-12
                                  and proposal["fixed_second_completion"]
                                  <= proposal["adaptive_two_call_completion"] + 1e-12
                                  and proposal["adaptive_two_call_completion"] <= 1.):
                            raise ValueError("Invalid adaptive two-call completion receipt")
                    prefix_key = digest([old.observation.design_id for old in history])
                    list_key = proposal["candidate_list_sha256"]
                    if candidate_lists.setdefault(prefix_key, list_key) != list_key:
                        raise ValueError("Feedback ablations generated different lists for the same history")
                    if arm in ENERGY_POLICIES:
                        evidence = [energy_evidence(old) for old in history if supports_timing(old)]
                        count = sum(item is not None for item in evidence)
                        bounds = sum(item is not None and item[0].kind == "interval" for item in evidence)
                        if (proposal["energy_observations"], proposal["energy_lower_bounds"]) != (count, bounds):
                            raise ValueError("The cold acquisition used an inconsistent energy history")
                        fallback = proposal.get("energy_fallback")
                        if fallback is None:
                            probability = proposal["target_probability"]
                            if not count or not 0. <= probability <= proposal["p_complete"] <= 1.:
                                raise ValueError("Invalid conditional energy target receipt")
                            if arm == ENERGY_LOOKAHEAD and proposal["candidate_count"] > 1 and not (
                                probability <= proposal["predicted_fixed_follow_up_target"] + 1e-12
                                and proposal["predicted_fixed_follow_up_target"]
                                <= proposal["predicted_two_call_target"] + 1e-12
                                and proposal["predicted_two_call_target"] <= 1.
                            ):
                                raise ValueError("Invalid two-call energy target receipt")
                            energy_calls += 1
                            bounds_only_calls += int(bounds == count)
                            if arm == ENERGY_LOOKAHEAD and proposal["candidate_count"] > 1:
                                adaptive_calls += int(proposal["predicted_two_call_target"] >
                                                      proposal["predicted_fixed_follow_up_target"] + 1e-12)
                        elif fallback not in ({"constant_target_marks", "invalid_energy_posterior"} if count
                                              else {"no_energy_evidence"}):
                            raise ValueError("Invalid energy acquisition fallback")
                        else:
                            energy_fallbacks += 1
                    risk_calls += 1
                fallbacks += int("risk_fallback" in proposal)
                if "refinement_risk" in proposal:
                    if arm not in REFINEMENT_FEEDBACK or acquisition_stage(history, usable_target):
                        raise ValueError("Risk refinement applied outside the registered phase")
                    receipt = proposal["refinement_risk"]
                    if receipt["mode"] == "risk_weighted_native_search":
                        if receipt["feedback"] != REFINEMENT_FEEDBACK[arm]:
                            raise ValueError("Incorrect refinement feedback")
                        refinement_calls += 1
                if usable_target > 1 and arm != ORIGINAL:
                    if acquisition_stage(history, usable_target):
                        if proposal["source"] != "lsbo_latent_completion_risk" and "risk_fallback" not in proposal:
                            raise ValueError("The accumulation policy is missing from the receipt")
                    elif "refinement_risk" not in proposal:
                        raise ValueError("The shared refinement policy is missing from the receipt")
            seen.add(key)
            history.append(record)
        arms[arm] = {**trajectory_metrics(history, spec["total_budget"]),
                     "cold_risk_calls": risk_calls, "cold_risk_fallbacks": fallbacks}
        histories[arm] = history
        if usable_target > 1:
            arms[arm]["accumulation"] = accumulation_metrics(history, INITIAL, usable_target)
        if arm in ENERGY_POLICIES:
            arms[arm].update(energy_policy_calls=energy_calls, energy_policy_fallbacks=energy_fallbacks,
                             energy_bounds_only_calls=bounds_only_calls)
        if arm == ENERGY_LOOKAHEAD:
            arms[arm]["energy_adaptive_calls"] = adaptive_calls
        if refinement:
            target_hit = next((i for i, record in enumerate(history, 1) if record.usable_feasible
                               and record.observation.objective.value <= spec["energy_target_Wh"]), None)
            arms[arm].update(refinement_risk_calls=refinement_calls, first_energy_target=target_hit,
                             quality_by_budget={str(b): trajectory_metrics(history[:b], b)["best_usable_Wh"]
                                                for b in spec["quality_budgets"]})
    initial_usable = sum(record.usable_feasible for record in prefix)
    row = {"arms": arms, "cold": initial_usable == 0}
    if usable_target > 1:
        row["initial_usable"] = initial_usable
    if "completion_audit_size" in spec:
        panel = pools["covered"][:spec["completion_audit_size"]]
        heldout = [settled(f"risk_audit/{i}", "heldout") for i in range(1, len(panel)+1)]
        if [record.candidate for record in heldout] != panel:
            raise ValueError("Unexpected held-out candidate panel")
        audited_histories = {arm: history for arm, history in histories.items() if arm != ORIGINAL}
        row["completion_audit"] = completion_audit(problem, seed, audited_histories, heldout,
                                                  spec["completion_audit_checkpoints"], INITIAL)
    if position != len(calls):
        raise ValueError("The run contains unaccounted evaluations")
    status = "VERIFIED_RISK_AWARE_LSBO_REFINEMENT" if refinement else "VERIFIED_NATIVE_RISK_LSBO"
    contrasts = spec.get("contrasts", REFINEMENT_CONTRASTS if refinement else CONTRASTS)
    tie_Wh = spec.get("energy_order_tie_Wh", 0.)
    report = {"status": status, "source_sha": sha, "spec": spec, "seed": seed,
              "paid_calls": position, "rcaide_mda_calls": 0, **row, "versions": header["versions"],
              "contrasts": {f"{a}_minus_{b}": compare([row], a, b, tie_Wh=tie_Wh) for a, b in contrasts}}
    if refinement:
        report["quality_contrasts"] = {}
        for budget in map(str, spec["quality_budgets"]):
            state = {"arms": {arm: {**item, "best_usable_Wh": item["quality_by_budget"][budget]}
                              for arm, item in arms.items()}}
            report["quality_contrasts"][budget] = {}
            for left, right in contrasts:
                paired = compare([state], left, right, tie_Wh=tie_Wh)
                report["quality_contrasts"][budget][f"{left}_minus_{right}"] = {
                    name: paired[name] for name in ("energy_Wh_delta_both_usable",
                                                   "feasibility_first_terminal_order")}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plan", "run", "verify"), required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--study", choices=STUDIES, default="discovery")
    parser.add_argument("--expected-sha")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = plan(args.scenario, args.profile, args.study)
    if args.task == "plan":
        print(json.dumps(spec, indent=2))
        return
    if not args.expected_sha or args.output is None:
        parser.error("--expected-sha and --output are required")
    output = args.output.resolve()
    if output.exists() or Path(__file__).resolve().parent in (output, *output.parents):
        parser.error("Use a new output outside the code worktree")
    sha = source_commit(args.expected_sha, clean=True)
    if args.task == "run":
        run(spec, spec["seeds"][0], sha, output)
    else:
        if args.input is None or args.input.resolve() in (output, *output.parents):
            parser.error("Provide --input and write outside the sealed run")
        report = verify(args.input, spec, sha)
        save(output, report)
        print(report["status"], spec["name"], report["paid_calls"], flush=True)


if __name__ == "__main__":
    main()
