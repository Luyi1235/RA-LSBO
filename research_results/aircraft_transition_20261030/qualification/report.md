# Reduced-aircraft qualification (not an optimization comparison)

Panel draws are independent of optimization. Anchors/probes are not part of random feasibility rates.

Unknown-conservative envelopes are not safety certificates. Partial convergence is inconclusive.

No automatic pilot authorization. Review labels, terminations, branches and numerical sensitivity.


## aircraft_ordinary

{
  "branches": [
    {
      "discrete": {
        "parallel_strings": 40,
        "prop_layout": "distributed",
        "propeller_count": 4
      },
      "n": 10,
      "unresolved": 0,
      "usable_feasible": 5
    },
    {
      "discrete": {
        "parallel_strings": 40,
        "prop_layout": "wingtip",
        "propeller_count": 2
      },
      "n": 13,
      "unresolved": 0,
      "usable_feasible": 8
    },
    {
      "discrete": {
        "parallel_strings": 32,
        "prop_layout": "wingtip",
        "propeller_count": 2
      },
      "n": 10,
      "unresolved": 0,
      "usable_feasible": 3
    },
    {
      "discrete": {
        "parallel_strings": 32,
        "prop_layout": "distributed",
        "propeller_count": 4
      },
      "n": 8,
      "unresolved": 0,
      "usable_feasible": 5
    },
    {
      "discrete": {
        "parallel_strings": 48,
        "prop_layout": "distributed",
        "propeller_count": 2
      },
      "n": 12,
      "unresolved": 0,
      "usable_feasible": 5
    },
    {
      "discrete": {
        "parallel_strings": 40,
        "prop_layout": "distributed",
        "propeller_count": 6
      },
      "n": 7,
      "unresolved": 0,
      "usable_feasible": 3
    },
    {
      "discrete": {
        "parallel_strings": 48,
        "prop_layout": "distributed",
        "propeller_count": 6
      },
      "n": 7,
      "unresolved": 0,
      "usable_feasible": 4
    },
    {
      "discrete": {
        "parallel_strings": 32,
        "prop_layout": "distributed",
        "propeller_count": 2
      },
      "n": 8,
      "unresolved": 0,
      "usable_feasible": 1
    },
    {
      "discrete": {
        "parallel_strings": 40,
        "prop_layout": "distributed",
        "propeller_count": 2
      },
      "n": 17,
      "unresolved": 0,
      "usable_feasible": 12
    },
    {
      "discrete": {
        "parallel_strings": 48,
        "prop_layout": "distributed",
        "propeller_count": 4
      },
      "n": 8,
      "unresolved": 0,
      "usable_feasible": 3
    },
    {
      "discrete": {
        "parallel_strings": 32,
        "prop_layout": "distributed",
        "propeller_count": 6
      },
      "n": 12,
      "unresolved": 0,
      "usable_feasible": 7
    },
    {
      "discrete": {
        "parallel_strings": 48,
        "prop_layout": "wingtip",
        "propeller_count": 2
      },
      "n": 16,
      "unresolved": 0,
      "usable_feasible": 9
    }
  ],
  "counts": {
    "known_infeasible": 63,
    "unresolved": 0,
    "usable_feasible": 65
  },
  "intervals_are_not_safety_guarantees": true,
  "n": 128,
  "numerical_failures": 0,
  "output_kinds": {
    "current:exact": 102,
    "current:missing": 26,
    "objective:exact": 102,
    "objective:interval": 26,
    "packing:exact": 128,
    "power:exact": 102,
    "power:interval": 26,
    "root_bending:exact": 128,
    "soc:exact": 102,
    "soc:interval": 26,
    "span:exact": 128,
    "stall:exact": 128,
    "voltage:exact": 102,
    "voltage:missing": 26
  },
  "physical_feasibility_partial_identification": [
    0.5078125,
    0.5078125
  ],
  "sparse_below_10pct_supported": false,
  "terminations": {
    "complete": 102,
    "physical_soc_reserve_abort": 26
  },
  "unknown_conservative_binomial_envelope_95": [
    0.41801413935064136,
    0.5972412381415354
  ],
  "usable_binomial_95": [
    0.41801413935064136,
    0.5972412381415354
  ],
  "usable_fraction": 0.5078125
}

Integration screens: {'passed_screen': 10, 'inconclusive_partial_outputs': 4}


## aircraft_sparse

{
  "branches": [
    {
      "discrete": {
        "parallel_strings": 40,
        "prop_layout": "distributed",
        "propeller_count": 4
      },
      "n": 10,
      "unresolved": 0,
      "usable_feasible": 1
    },
    {
      "discrete": {
        "parallel_strings": 40,
        "prop_layout": "wingtip",
        "propeller_count": 2
      },
      "n": 13,
      "unresolved": 0,
      "usable_feasible": 1
    },
    {
      "discrete": {
        "parallel_strings": 32,
        "prop_layout": "wingtip",
        "propeller_count": 2
      },
      "n": 10,
      "unresolved": 0,
      "usable_feasible": 0
    },
    {
      "discrete": {
        "parallel_strings": 32,
        "prop_layout": "distributed",
        "propeller_count": 4
      },
      "n": 8,
      "unresolved": 0,
      "usable_feasible": 0
    },
    {
      "discrete": {
        "parallel_strings": 48,
        "prop_layout": "distributed",
        "propeller_count": 2
      },
      "n": 12,
      "unresolved": 0,
      "usable_feasible": 3
    },
    {
      "discrete": {
        "parallel_strings": 40,
        "prop_layout": "distributed",
        "propeller_count": 6
      },
      "n": 7,
      "unresolved": 0,
      "usable_feasible": 1
    },
    {
      "discrete": {
        "parallel_strings": 48,
        "prop_layout": "distributed",
        "propeller_count": 6
      },
      "n": 7,
      "unresolved": 0,
      "usable_feasible": 3
    },
    {
      "discrete": {
        "parallel_strings": 32,
        "prop_layout": "distributed",
        "propeller_count": 2
      },
      "n": 8,
      "unresolved": 0,
      "usable_feasible": 0
    },
    {
      "discrete": {
        "parallel_strings": 40,
        "prop_layout": "distributed",
        "propeller_count": 2
      },
      "n": 17,
      "unresolved": 0,
      "usable_feasible": 1
    },
    {
      "discrete": {
        "parallel_strings": 48,
        "prop_layout": "distributed",
        "propeller_count": 4
      },
      "n": 8,
      "unresolved": 0,
      "usable_feasible": 2
    },
    {
      "discrete": {
        "parallel_strings": 32,
        "prop_layout": "distributed",
        "propeller_count": 6
      },
      "n": 12,
      "unresolved": 0,
      "usable_feasible": 0
    },
    {
      "discrete": {
        "parallel_strings": 48,
        "prop_layout": "wingtip",
        "propeller_count": 2
      },
      "n": 16,
      "unresolved": 0,
      "usable_feasible": 7
    }
  ],
  "counts": {
    "known_infeasible": 109,
    "unresolved": 0,
    "usable_feasible": 19
  },
  "intervals_are_not_safety_guarantees": true,
  "n": 128,
  "numerical_failures": 0,
  "output_kinds": {
    "current:exact": 41,
    "current:missing": 87,
    "objective:exact": 41,
    "objective:interval": 87,
    "packing:exact": 128,
    "power:exact": 41,
    "power:interval": 87,
    "root_bending:exact": 128,
    "soc:exact": 41,
    "soc:interval": 87,
    "span:exact": 128,
    "stall:exact": 128,
    "voltage:exact": 41,
    "voltage:missing": 87
  },
  "physical_feasibility_partial_identification": [
    0.1484375,
    0.1484375
  ],
  "sparse_below_10pct_supported": false,
  "terminations": {
    "complete": 41,
    "physical_soc_reserve_abort": 87
  },
  "unknown_conservative_binomial_envelope_95": [
    0.09179817376238814,
    0.2220598608384989
  ],
  "usable_binomial_95": [
    0.09179817376238814,
    0.2220598608384989
  ],
  "usable_fraction": 0.1484375
}

Integration screens: {'inconclusive_partial_outputs': 12, 'passed_screen': 2}

Calls: 316/316; new RCAIDE MDA: 0.
