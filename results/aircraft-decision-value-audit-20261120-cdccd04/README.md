# PR146 decision-value audit

Code SHA: `cdccd04f40f734b85ae30b7ed9af1360f70bae84`; input PR135 result SHA: `3b6141131ec790a829f5267f285910a7b0a6f077`.

All eight runs returned `VERIFIED_DECISION_VALUE_AUDIT`, 96 histories each, 0 new evaluator calls, 0 RCAIDE/MDA.

| run | paired | activated | hits decision | hits interval greedy | hits usable greedy | hit differences | gate reasons |
|---|---:|---:|---:|---:|---:|---:|---|
| pr135-headwind-n16-derated | 40 | 3 | 7 | 7 | 7 | 0 | {'gain_cannot_offset_cost': 32, 'activated': 3, 'no_additional_decision_information': 5} |
| pr135-headwind-n16-nominal | 23 | 3 | 8 | 8 | 8 | 0 | {'gain_cannot_offset_cost': 19, 'activated': 3, 'no_additional_decision_information': 1} |
| pr135-headwind-n8-derated | 73 | 9 | 6 | 6 | 6 | 0 | {'gain_cannot_offset_cost': 55, 'no_additional_decision_information': 9, 'activated': 9} |
| pr135-headwind-n8-nominal | 58 | 8 | 9 | 11 | 11 | 2 | {'gain_cannot_offset_cost': 43, 'no_additional_decision_information': 7, 'activated': 8} |
| pr135-payload-n16-derated | 63 | 1 | 6 | 4 | 6 | 2 | {'gain_cannot_offset_cost': 42, 'no_additional_decision_information': 20, 'activated': 1} |
| pr135-payload-n16-nominal | 4 | 0 | 1 | 1 | 1 | 0 | {'gain_cannot_offset_cost': 4} |
| pr135-payload-n8-derated | 84 | 3 | 9 | 8 | 8 | 1 | {'no_additional_decision_information': 20, 'gain_cannot_offset_cost': 61, 'activated': 3} |
| pr135-payload-n8-nominal | 30 | 1 | 11 | 11 | 11 | 0 | {'gain_cannot_offset_cost': 22, 'no_additional_decision_information': 7, 'activated': 1} |

## Reading

- The gate activates rarely (1-9 histories per configuration). Most paired histories are rejected because the expected gain cannot offset the greedy-first cost, or because the query adds no additional decision information.
- Hits are nearly identical between the decision gate and interval greedy; only isolated histories differ.
- This is a read-only replay; it does not establish a policy improvement.
