import unittest

from run_aircraft_ra_lsbo_comparison import ARMS, compare_calibration, comparisons, plan


class ComparisonProtocolTests(unittest.TestCase):
    def test_frozen_budget_includes_original_and_two_risk_baselines(self):
        spec = plan("comparison-headwind")
        self.assertEqual(spec["seeds"], plan("comparison-payload")["seeds"])
        self.assertEqual(sum(spec["budget"].values()), 3744)
        self.assertTrue({"original_lsbo", "logcei", "scbo_adapted"}.issubset(ARMS))

    def test_early_native_stop_preserves_censoring_and_missing_energy(self):
        base = {"seed": 7, "initial_kind": "cold"}
        records = [
            {**base, "method": arm, "first_feasible_online_call": 3 if arm == "ra_separate" else None,
             "best_feasible": 100. if arm == "ra_separate" else None}
            for arm in ARMS
        ]
        output = comparisons(records, plan("comparison-headwind"))
        native = next(row for row in output if row["comparator"] == "original_lsbo" and row["cohort"] == "cold")
        self.assertEqual(native["capped_first_call_delta"]["mean"], -18.)
        self.assertEqual(native["discovery_by_8_delta"]["mean"], 1.)
        self.assertIsNone(native["energy_delta_Wh"]["mean"])

    def test_calibration_keeps_seed_pair_and_counts_failed_fit(self):
        rows = [{"seed": 8, "n": n, "method": arm, "completion": .25,
                 "usable": None if n == 8 and arm == "ra_joint_kernel" else .125}
                for n in (4, 8) for arm in ("ra_joint_kernel", "ra_separate")]
        result = compare_calibration(rows)
        self.assertEqual(result["completion"]["seed_deltas"], [[8, 0.]])
        self.assertEqual(result["usable"]["incomplete_seeds"], 1)


class HandoffProtocolTests(unittest.TestCase):
    def test_frozen_budget_and_backend_switch(self):
        from types import SimpleNamespace
        from run_aircraft_ra_lsbo_comparison import active_backend, FULL_NATIVE

        spec = plan("handoff-headwind")
        cold, warm = [SimpleNamespace(usable_feasible=usable) for usable in (False, True)]
        self.assertEqual(sum(spec["budget"].values()), 2976)
        self.assertEqual(spec["seeds"], plan("handoff-payload")["seeds"])
        self.assertTrue(set(spec["seeds"]).isdisjoint(plan("comparison-headwind")["seeds"]))
        self.assertEqual(active_backend("ra_to_lsbo", [cold]), "ra_separate")
        self.assertEqual(active_backend("ra_to_lsbo", [cold, warm]), FULL_NATIVE)
        self.assertEqual(active_backend("logcei_to_lsbo", [warm]), FULL_NATIVE)

    def test_prefix_audit_rejects_pre_handoff_changes(self):
        from types import SimpleNamespace
        from run_aircraft_ra_lsbo_comparison import check_handoff_paths, FULL_NATIVE

        a = SimpleNamespace(usable_feasible=False, design="a")
        b = SimpleNamespace(usable_feasible=False, design="b")
        winner = SimpleNamespace(usable_feasible=True, design="winner")
        paths = {"original_lsbo": [a], FULL_NATIVE: [a, b], "ra_separate": [b, winner, a],
                 "ra_to_lsbo": [b, winner, b], "logcei": [a, b], "logcei_to_lsbo": [a, b]}
        check_handoff_paths([a], paths)
        with self.assertRaisesRegex(ValueError, "declared cold"):
            check_handoff_paths([a], {**paths, "ra_to_lsbo": [a, winner, b]})
        with self.assertRaisesRegex(ValueError, "pre-stop"):
            check_handoff_paths([a], {**paths, FULL_NATIVE: [b, a]})


class TrainingAblationTests(unittest.TestCase):
    def test_training_plan_freezes_matched_primary_and_fresh_seeds(self):
        spec = plan("training-headwind")
        self.assertEqual(spec["arms"], ["ra_to_lsbo", "ra_joint_to_lsbo", "logcei_to_lsbo"])
        self.assertEqual(spec["primary"], {"reference": "ra_to_lsbo", "comparator": "ra_joint_to_lsbo",
                                           "cohort": "cold", "metric": "capped_first_call_delta"})
        self.assertEqual(spec["seeds"], plan("training-payload")["seeds"])
        for prior in ("comparison-headwind", "handoff-headwind", "training-smoke"):
            self.assertTrue(set(spec["seeds"]).isdisjoint(plan(prior)["seeds"]))
        self.assertEqual(sum(spec["budget"].values()), 2560)

    def test_shared_handoff_uses_shared_kernel_only_before_discovery(self):
        from types import SimpleNamespace
        from run_aircraft_ra_lsbo_comparison import active_backend, FULL_NATIVE, training_proposal_kind

        cold, warm = [SimpleNamespace(usable_feasible=v) for v in (False, True)]
        self.assertEqual(active_backend("ra_joint_to_lsbo", [cold]), "ra_joint_kernel")
        self.assertEqual(active_backend("ra_joint_to_lsbo", [cold, warm]), FULL_NATIVE)
        decision = {"mode": "feasible_discovery", "selected_kernel": 4}
        self.assertEqual(training_proposal_kind("ra_joint_kernel", decision), "model")
        with self.assertRaisesRegex(ValueError, "missing"):
            training_proposal_kind("ra_joint_kernel", {**decision, "selected_kernel": None})
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            training_proposal_kind("ra_separate", decision)

    def test_three_way_initial_warm_trajectories_are_equal(self):
        from types import SimpleNamespace
        from run_aircraft_ra_lsbo_comparison import check_handoff_paths

        initial = [SimpleNamespace(usable_feasible=True)]
        spec = plan("training-headwind")
        paths = {arm: ["common-proposal"] for arm in spec["arms"]}
        check_handoff_paths(initial, paths)
        with self.assertRaisesRegex(ValueError, "initial-warm"):
            check_handoff_paths(initial, {**paths, "ra_joint_to_lsbo": ["other-proposal"]})


class BoundaryAblationTests(unittest.TestCase):
    def test_formal_pair_and_fresh_seeds(self):
        from dataclasses import asdict
        from pathlib import Path
        from run_aircraft_ra_lsbo_comparison import BenchmarkAircraft, engine_for, ra_settings

        spec = plan("boundary-headwind")
        self.assertEqual(sum(spec["budget"].values()), 1760)
        self.assertEqual(spec["seeds"], plan("boundary-payload")["seeds"])
        self.assertTrue(set(spec["seeds"]).isdisjoint(plan("training-headwind")["seeds"]))
        self.assertEqual(spec["arms"], ["ra_to_lsbo", "ra_feasibility_to_lsbo"])
        self.assertEqual(spec["primary"]["comparator"], "ra_feasibility_to_lsbo")
        problem = BenchmarkAircraft(spec)
        args = problem, spec["seeds"][0]
        left, right = (engine_for(*args, arm, spec, Path(".")).optimizer for arm in spec["arms"])
        self.assertEqual(asdict(left.settings), asdict(ra_settings(spec)))
        self.assertEqual(asdict(right.settings), {**asdict(left.settings), "boundary_fraction": 0.})

    def test_score_replacement_keeps_gate_and_checks_coverage(self):
        import numpy as np
        from ra_lsbo.soc_revision.acquisition import choose_acquisition
        from run_aircraft_ra_lsbo_comparison import boundary_proposal_kind
        from types import SimpleNamespace

        a = plan("boundary-headwind")["boundary_ablation"]
        options = dict(horizon=20, initial_kappa=1., final_kappa=1., completion_threshold=.5)
        prediction = {"completion": {"mean": np.array([-2., 0.]), "variance": np.array([.01, .01])},
                      "constraints": {}, "objective": None}
        i, info = choose_acquisition(prediction, None, 4, SimpleNamespace(**options, boundary_fraction=.2), True)
        j, feasible = choose_acquisition(prediction, None, 4, SimpleNamespace(**options, boundary_fraction=0.), True)
        self.assertEqual(info["eligible"], feasible["eligible"])
        self.assertEqual((i, j), (1, 0))
        self.assertEqual(boundary_proposal_kind("ra_to_lsbo", 4, info, a), "budgeted_boundary_information")
        self.assertEqual(boundary_proposal_kind("ra_feasibility_to_lsbo", 4, feasible, a),
                         "information_replaced_by_feasibility")
        with self.assertRaisesRegex(ValueError, "coverage schedule"):
            boundary_proposal_kind("ra_to_lsbo", 5, feasible, a)

    def test_preintervention_prefix_cannot_change(self):
        from types import SimpleNamespace
        from run_aircraft_ra_lsbo_comparison import check_boundary_prefix

        same = SimpleNamespace(usable_feasible=False)
        different = SimpleNamespace(usable_feasible=False, another_design=True)
        ref = [same]*6
        paths = {"ra_to_lsbo": ref, "ra_feasibility_to_lsbo": [same]*4+[different, same]}
        check_boundary_prefix(paths, .2)
        with self.assertRaisesRegex(ValueError, "first intervention"):
            check_boundary_prefix({**paths, "ra_feasibility_to_lsbo": [different]+[same]*5}, .2)


class StopEvidenceAblationTests(unittest.TestCase):
    @staticmethod
    def history():
        from ra_lsbo.budget import digest
        from ra_lsbo.partial_bo.records import DesignRecord
        from ra_lsbo.risk_core.observations import OutputObservation, RunObservation

        rows = []
        for i, status in enumerate(("hidden_failure", "completed", "unknown")):
            candidate = {"i": i}
            packet = RunObservation(
                digest(candidate), "contract", f"run-{i}", status,
                OutputObservation("interval", "energy", lower=30.),
                (("span", OutputObservation("exact", "span", value=-1.)),
                 ("stress", OutputObservation("exact", "stress", value=-.5)),
                 ("soc", OutputObservation("interval", "soc", lower=.1)),
                 ("missing", OutputObservation("missing", "missing", reason="absent"))),
                failure_cause="physical_soc_reserve_abort" if status == "hidden_failure" else None)
            rows.append(DesignRecord(candidate, packet, {"trial": i}))
        return rows

    def test_only_known_stops_lose_observed_nonanalytical_targets(self):
        from run_aircraft_ra_lsbo_comparison import BenchmarkAircraft, mask_stopped_constraints

        raw = self.history()
        masked = mask_stopped_constraints(raw, BenchmarkAircraft.analytical_constraints)
        self.assertEqual(masked[1:], raw[1:])
        self.assertEqual(masked[0].candidate, raw[0].candidate)
        self.assertEqual(masked[0].diagnostics, raw[0].diagnostics)
        a, b = masked[0].observation, raw[0].observation
        self.assertEqual(a.completion_label(), b.completion_label())
        self.assertEqual(a.objective, b.objective)
        self.assertEqual(a.constraint("span"), b.constraint("span"))
        self.assertEqual(a.constraint("missing"), b.constraint("missing"))
        self.assertEqual(b.constraint("stress").kind, "exact")
        self.assertEqual(b.constraint("soc").kind, "interval")
        self.assertEqual(a.constraint("stress").kind, "missing")
        self.assertEqual(a.constraint("soc").kind, "missing")
        self.assertEqual(a.constraint("soc").evidence_id, b.constraint("soc").evidence_id)

    def test_gp_features_see_raw_but_constraint_supervision_is_masked(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from run_aircraft_ra_lsbo_comparison import StopStatusRA

        model = StopStatusRA.__new__(StopStatusRA)
        model.problem = SimpleNamespace(analytical_constraints=("span", "packing"))
        model.base, model.session, model.seed, model.settings, model.allocation = None, None, 42, None, "onehot_risk"
        raw = self.history()
        module = "ra_lsbo.experiments.simplified_aircraft.dualview_optimizer"
        with patch(f"{module}.feature_snapshot") as features, patch(f"{module}.PrefixFeatures") as explicit, \
                patch(f"{module}.HeterogeneousRiskPosterior") as posterior:
            model._build_posterior(raw)
            self.assertIs(features.call_args.args[2], raw)
            self.assertIs(explicit.call_args.args[2], raw)
            self.assertEqual(posterior.call_args.args[1][0].observation.constraint("soc").kind, "missing")

    def test_frozen_fresh_pair_and_training_receipts(self):
        from ra_lsbo.budget import digest
        from run_aircraft_ra_lsbo_comparison import BenchmarkAircraft, mask_stopped_constraints, stop_evidence_proposal_kind

        spec = plan("stop-evidence-headwind")
        self.assertEqual(spec["seeds"], plan("stop-evidence-payload")["seeds"])
        self.assertTrue(set(spec["seeds"]).isdisjoint(plan("boundary-headwind")["seeds"]))
        self.assertEqual(spec["primary"]["comparator"], "ra_stop_status_to_lsbo")
        self.assertEqual(sum(spec["budget"].values()), 1760)
        raw = self.history()
        masked = mask_stopped_constraints(raw, BenchmarkAircraft.analytical_constraints)
        p = {"mode": "feasible_discovery", "model_history_digest": digest([r.to_dict() for r in masked])}
        self.assertEqual(stop_evidence_proposal_kind("ra_stop_status", p, raw), "fit_with_masked_stop_evidence")
        with self.assertRaisesRegex(ValueError, "frozen stop-evidence"):
            stop_evidence_proposal_kind("ra_separate", p, raw)

    def test_sign_view_coarsens_exact_and_interval_targets_only_at_known_stops(self):
        from dataclasses import replace
        from ra_lsbo.risk_core.observations import OutputObservation
        from run_aircraft_ra_lsbo_comparison import BenchmarkAircraft, mask_stopped_constraints

        raw = self.history()
        packet = raw[0].observation
        extra = (
            ("unresolved", OutputObservation("interval", "unresolved", lower=-.2, upper=.3)),
            ("zero", OutputObservation("exact", "zero", value=0.)),
            ("inclusive_lower_zero", OutputObservation("interval", "inclusive_lower_zero", lower=0.)),
        )
        raw[0] = replace(raw[0], observation=replace(packet, constraints=packet.constraints+extra))
        signed = mask_stopped_constraints(raw, BenchmarkAircraft.analytical_constraints, keep_signs=True)
        observed = signed[0].observation
        self.assertEqual(signed[1:], raw[1:])
        self.assertEqual(observed.objective, packet.objective)
        self.assertEqual(observed.completion_label(), 0)
        self.assertEqual(observed.constraint("span"), packet.constraint("span"))
        self.assertEqual(observed.constraint("missing"), packet.constraint("missing"))
        self.assertEqual(observed.constraint("stress"), OutputObservation("interval", "stress", upper=0.))
        self.assertEqual(observed.constraint("zero"), OutputObservation("interval", "zero", upper=0.))
        self.assertEqual(observed.constraint("soc"),
                         OutputObservation("interval", "soc", lower=0., lower_inclusive=False))
        self.assertEqual(observed.constraint("unresolved").kind, "missing")
        self.assertEqual(observed.constraint("inclusive_lower_zero").kind, "missing")

    def test_sign_gp_receipt_and_exposure_distinguish_resolved_targets(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from ra_lsbo.budget import digest
        from run_aircraft_ra_lsbo_comparison import BenchmarkAircraft, StopSignRA, stop_evidence_proposal_kind

        raw = self.history()
        model = StopSignRA.__new__(StopSignRA)
        model.problem = SimpleNamespace(analytical_constraints=BenchmarkAircraft.analytical_constraints)
        model.base, model.session, model.seed, model.settings, model.allocation = None, None, 42, None, "onehot_risk"
        module = "ra_lsbo.experiments.simplified_aircraft.dualview_optimizer"
        with patch(f"{module}.feature_snapshot") as features, patch(f"{module}.PrefixFeatures") as explicit, \
                patch(f"{module}.HeterogeneousRiskPosterior") as posterior:
            model._build_posterior(raw)
            self.assertIs(features.call_args.args[2], raw)
            self.assertIs(explicit.call_args.args[2], raw)
            view = posterior.call_args.args[1]
        p = {"mode": "feasible_discovery", "model_history_digest": digest([r.to_dict() for r in view])}
        self.assertEqual(stop_evidence_proposal_kind("ra_stop_sign", p, raw), "fit_with_signed_stop_evidence")
        with self.assertRaisesRegex(ValueError, "frozen stop-evidence"):
            stop_evidence_proposal_kind("ra_stop_status", p, raw)

    def test_sign_plan_freezes_paired_fresh_seeds_and_budgets(self):
        from run_aircraft_ra_lsbo_comparison import PROFILES

        spec = plan("stop-signs-headwind")
        self.assertEqual(spec["primary"], {"reference": "ra_stop_sign_to_lsbo",
                                           "comparator": "ra_stop_status_to_lsbo",
                                           "cohort": "cold", "metric": "capped_first_call_delta"})
        self.assertEqual(spec["arms"], ["ra_stop_sign_to_lsbo", "ra_stop_status_to_lsbo", "ra_to_lsbo"])
        self.assertEqual(spec["seeds"], plan("stop-signs-payload")["seeds"])
        for prior in PROFILES:
            if not prior.startswith("stop-signs-"):
                self.assertTrue(set(spec["seeds"]).isdisjoint(plan(prior)["seeds"]))
        self.assertEqual(sum(spec["budget"].values()), 2560)
        self.assertEqual(sum(plan("stop-signs-smoke")["budget"].values()), 44)
