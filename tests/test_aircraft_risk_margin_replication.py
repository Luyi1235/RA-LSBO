import unittest

from aircraft_risk_protocol import plan
from aircraft_risk_termination import coarsen
from analyze_aircraft_margin_replication import PRIMARY, check_observation_pair, paired_interval
from test_aircraft_risk_termination import record


class MarginReplicationTests(unittest.TestCase):
    def test_registered_replication_uses_new_seeds_and_common_payload(self):
        limited, enriched = (plan("margin-replication-"+kind) for kind in ("trigger", "margins"))
        self.assertEqual(limited["seeds"], enriched["seeds"])
        self.assertEqual(limited["payload_delta_kg"], 24.)
        self.assertEqual(enriched["headwind_m_s"], 0.)
        self.assertEqual(limited["budget"], {"initial": 128, "online": 1280, "heldout": 1024})
        self.assertEqual(limited["budget"], enriched["budget"])
        self.assertTrue(set(limited["seeds"]).isdisjoint(plan("termination-margins")["seeds"]))

    def test_view_pair_checks_measurements_and_censoring(self):
        native = record(.1, -.09, -.01)
        trigger = coarsen(native, {"span"}, "trigger").to_dict()
        enriched = coarsen(native, {"span"}, "rich", keep_margins=True).to_dict()
        check_observation_pair(trigger, enriched, {"span"}, "trigger")
        trigger["observation"]["objective"]["reason"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "differs beyond"):
            check_observation_pair(trigger, enriched, {"span"}, "trigger")

    def test_paired_interval_uses_only_primary_unsolved_pairs(self):
        a, b = PRIMARY
        rows = [{"arm": a, "baseline": b, "capped_calls_delta": value, "initial_usable": solved}
                for value, solved in ((-3, 0), (-3, 0), (99, 1))]
        self.assertEqual(paired_interval(rows, draws=100), [-3., -3.])
        self.assertIsNone(paired_interval(rows[:1], draws=100))
