# PR122 — Nested initial allocation under a fixed 48-evaluation budget

Fixed code SHA: `ae4b9be8ef8fa09b7b45e4be7d40d8956edb59ab`

## Design

- 8 paired seeds per condition; total budget 48 per policy.
- Nested initial sizes 4, 8, 16, 32; each policy gets `48 - n0` online evaluations.
- Three methods: `lsbo` (native LSBO handoff after first feasible), `greedy` (usable-probability greedy), `lookahead` (two-step abort/violation lookahead).
- Cold cohort is the primary; all arms share the same nested initial sequence and candidate pool.
- 0 RCAIDE/MDA calls.

## Acceptance

| Profile | Calls | Verify |
|---|---:|---|
| `smoke` | 96 | `VERIFIED_INITIAL_BUDGET` |
| `pilot-headwind-48` | 3,808 | `VERIFIED_INITIAL_BUDGET` |
| `pilot-payload-48` | 3,808 | `VERIFIED_INITIAL_BUDGET` |

Total 7,712 calls. 27 tests pass.

## Same method: increasing the initial sample

`first Δ` is reference minus control in capped calls to first feasible; negative = smaller initial arm earlier. `W/L/T` is terminal energy (win = reference lower).

### Headwind

| pair | first Δ | 95% CI | W/L/T | energy Δ Wh (n) |
|---|---:|---|---:|---:|
| lsbo 4→8 | -0.125 | [-2.500, +2.503] | 6/2/0 | -484.5 (8) |
| lsbo 8→16 | -3.000 | [-6.000, -1.000] | 5/3/0 | -159.3 (8) |
| lsbo 16→32 | -2.375 | [-7.000, +2.250] | 6/2/0 | -114.3 (8) |
| lsbo 4→32 | -5.500 | [-13.750, +2.000] | 5/3/0 | -758.1 (8) |
| greedy 4→8 | -1.750 | [-6.750, +2.625] | 6/2/0 | -712.1 (8) |
| greedy 8→16 | +0.000 | [-5.000, +4.500] | 5/3/0 | +82.5 (8) |
| greedy 16→32 | -4.125 | [-8.250, -0.875] | 7/1/0 | -685.1 (8) |
| greedy 4→32 | -5.875 | [-12.625, -0.125] | 7/1/0 | -1314.6 (8) |
| lookahead 4→8 | -1.375 | [-5.125, +2.250] | 5/3/0 | -76.9 (8) |
| lookahead 8→16 | -1.750 | [-4.625, +0.750] | 6/2/0 | +19.7 (8) |
| lookahead 16→32 | -3.750 | [-7.500, -1.000] | 7/1/0 | -1236.6 (8) |
| lookahead 4→32 | -6.875 | [-14.625, -0.750] | 7/1/0 | -1293.8 (8) |

### Payload

| pair | first Δ | 95% CI | W/L/T | energy Δ Wh (n) |
|---|---:|---|---:|---:|
| lsbo 4→8 | +2.125 | [-1.625, +7.125] | 7/1/0 | -215.4 (8) |
| lsbo 8→16 | +1.125 | [-1.500, +4.875] | 3/5/0 | +1006.2 (8) |
| lsbo 16→32 | -1.500 | [-4.500, +0.000] | 7/1/0 | -284.1 (8) |
| lsbo 4→32 | +1.750 | [-2.500, +7.125] | 6/2/0 | +506.7 (8) |
| greedy 4→8 | +3.125 | [+0.750, +6.125] | 5/3/0 | -440.4 (8) |
| greedy 8→16 | +0.625 | [-2.250, +4.125] | 5/3/0 | -42.6 (8) |
| greedy 16→32 | -1.750 | [-5.250, +0.000] | 6/2/0 | +126.4 (8) |
| greedy 4→32 | +2.000 | [-4.625, +8.625] | 7/1/0 | -356.6 (8) |
| lookahead 4→8 | +3.875 | [-1.500, +12.375] | 5/3/0 | -234.2 (8) |
| lookahead 8→16 | -0.250 | [-3.375, +2.625] | 5/3/0 | +388.0 (8) |
| lookahead 16→32 | -1.625 | [-4.875, +0.000] | 7/1/0 | -304.5 (8) |
| lookahead 4→32 | +2.000 | [-8.750, +12.500] | 7/1/0 | -150.7 (8) |

## Same initial size: lookahead vs greedy

`first Δ = lookahead − greedy`; negative = lookahead earlier. `W/L/T` is energy (win = lookahead lower).

### Headwind

| initial | first Δ | 95% CI | W/L/T | energy Δ Wh (n) |
|---:|---:|---|---:|---:|
| 4 | -0.500 | [-2.000, +0.625] | 1/4/3 | +561.7 (8) |
| 8 | -0.875 | [-2.250, +0.250] | 2/1/5 | -73.5 (8) |
| 16 | +0.875 | [-0.750, +2.750] | 3/0/5 | -10.7 (8) |
| 32 | +0.500 | [-0.750, +2.250] | 1/2/5 | +540.8 (8) |

### Payload

| initial | first Δ | 95% CI | W/L/T | energy Δ Wh (n) |
|---:|---:|---|---:|---:|
| 4 | +0.000 | [-5.750, +7.250] | 1/3/4 | +205.8 (8) |
| 8 | -0.750 | [-1.750, +0.000] | 2/0/6 | -0.4 (8) |
| 16 | +0.125 | [+0.000, +0.375] | 1/0/7 | -430.9 (8) |
| 32 | +0.000 | [+0.000, +0.000] | 0/0/8 | +0.0 (8) |

## Reading

- No monotone initial-size effect. Most nested comparisons cross zero. In headwind several point estimates favour the smaller initial arm; payload only `greedy 4→8` is clearly significant in the opposite direction (+3.125 calls, CI [+0.75, +6.125]).
- Terminal energy is also mixed; larger initial does not systematically improve the best observed design. This is expected when the initial panel is a random nested prefix and online search is cut by the larger `n0`.
- Lookahead vs greedy has no consistent first-feasible or terminal-energy advantage at any fixed initial size. All first-call intervals cross zero except payload `n0=16` (+0.125 [0.000, +0.375]), a 0.125-call difference with no practical meaning.
- These are 8-seed per-condition estimates; they are descriptive and not a confirmatory test. The reserved 96-budget profiles are not run here.

## Files

- `smoke/`, `pilot-headwind-48/`, `pilot-payload-48/` — sealed ledgers and per-seed evidence
- `analyses/*-analysis.json` — source of the tables above
- `logs/` — tests, run, verify logs, Python version
- `fixed_sha.txt`, `environment.txt`, `SHA256SUMS.txt`
