import unittest

from aircraft_risk_protocol import plan
from aircraft_risk_termination import coarsen
from analyze_aircraft_termination_factorial import compare_discovery
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.risk_core.observations import OutputObservation, RunObservation
from ra_lsbo.budget import digest
from run_aircraft_risk_cold_start import discovery_outcome


def record(stall, root_bending, soc, *, numerical=False):
    candidate = {"continuous": {"wing_area": 13.}, "discrete": {}}
    names = {"span": -.1, "stall": stall, "root_bending": root_bending, "soc": soc}
    constraints = tuple((name, OutputObservation("exact", name, value=value)) for name, value in sorted(names.items()))
    packet = RunObservation(digest(candidate), "native", "observed", "completed",
                            OutputObservation("exact", "energy", value=8000.), constraints)
    return DesignRecord(candidate, packet, {"mission_completed": True, "numerical_failure": numerical})


class TerminationFeedbackTests(unittest.TestCase):
    def test_structural_stop_keeps_only_a_valid_signed_trigger_and_analytical_evidence(self):
        native = record(.2, .1, -.02)
        view = coarsen(native, {"span"}, "structural")
        self.assertEqual(discovery_outcome(native.to_dict()), "violation")
        self.assertEqual(discovery_outcome(view.to_dict()), "abort")
        self.assertEqual(view.observation.contract_id, "structural")
        self.assertEqual(view.diagnostics["controlled_trigger"], "stall")
        self.assertEqual(view.observation.objective.kind, "missing")
        for head in ("root_bending", "soc"):
            self.assertEqual(view.observation.constraint(head).kind, "missing")
        self.assertIs(view.observation.constraint("stall").constraint_satisfaction(), False)
        self.assertEqual(view.observation.constraint("stall").lower, 0.)
        self.assertEqual(view.observation.constraint("span"), native.observation.constraint("span"))

    def test_usable_and_untrusted_labels_are_preserved(self):
        for native, outcome in ((record(-.1, -.1, -.02), "usable"),
                                (record(.2, -.1, -.02, numerical=True), "unknown")):
            view = coarsen(native, {"span"}, "structural")
            self.assertEqual(discovery_outcome(view.to_dict()), outcome)
            self.assertEqual(view.observation.objective, native.observation.objective)
            self.assertEqual(view.diagnostics, native.diagnostics)

    def test_interaction_uses_only_paired_initially_unsolved_seeds(self):
        native = [{"seed": 1, "initial_usable": 0, "capped_calls_delta": 2},
                  {"seed": 2, "initial_usable": 1, "capped_calls_delta": 0}]
        structural = [{**row, "capped_calls_delta": -3 if row["seed"] == 1 else 0} for row in native]
        self.assertEqual(compare_discovery(native, structural)["mean_interaction_delta_calls"], -5)
        with self.assertRaisesRegex(ValueError, "initial usable"):
            compare_discovery(native, [{**structural[0], "initial_usable": 1}, structural[1]])

    def test_registered_profiles_have_identical_budgets(self):
        native, structural = (plan("termination-"+policy) for policy in ("native", "structural"))
        self.assertEqual(native["seeds"], structural["seeds"])
        self.assertEqual(native["budget"], {"initial": 96, "online": 960, "heldout": 768})
        self.assertEqual(native["budget"], structural["budget"])
        self.assertEqual((native["termination_feedback"], structural["termination_feedback"]),
                         ("native", "structural-first"))
        self.assertTrue(set(native["seeds"]).isdisjoint(plan("feedback-baseline-headwind")["seeds"]))
