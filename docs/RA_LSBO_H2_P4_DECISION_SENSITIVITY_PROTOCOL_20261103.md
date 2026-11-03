# H2 P4 — Decision sensitivity diagnosis before any new optimizer experiments

Status: **read-only diagnostic protocol**; 2026-11-03. No new airplane calls, RCAIDE, model refits, default algorithm promotion or future seed selection.

## Source identity and observed reason for analysis

- Frozen H2 P3 source `57e390fc8f6280db9f201940aec14089e81c104c`.
- P3 eight-seed release `h2-p3-windows-pilot-20261103`, zip SHA256 `bd170195201b45942e4fe2c2e04e1e994263bb0a4566a467b28816cdce0473b4`.
- Independent full-archive, ledger and Gaussian exact/interval weight reconstruction: Draft PR #31, GitHub Actions **41562895382**, pass; 800 settled calls, 32 trials, 576 online proposals, 485 audit evidence hashes, no source or data modifications.
- Dynamic weights: 236 settled changes; 220 reflected in later proposal weights. Fixed changes: 0. Historical adaptive-versus-fixed candidate divergence: **210/288 paired online decision slots**; ordinary **110/144**, sparse **100/144**.
- **Important distinction:** after the *first* different candidate, each optimizer has a different training history. The 210 different candidate choices are the **total downstream trajectory effect** of the two policies, **not** 210 separate matched-history one-step causal effects.

P3 dynamic minus fixed nominal feasible energy:

| Task | Paired n | Wins/losses | Mean Wh | Median Wh | Mean without largest absolute-difference seed |
|---|---:|---:|---:|---:|---:|
| 100 km | 8 | 5/3 | -42.460 | -0.558 | -10.366 |
| 140 km | 8 | 4/4 | -262.908 | +0.379 | +9.150 |

140 km seed 12075 contributes **-2167.313 Wh**. The primary development result is mixed/fragile, **not reliable dynamic-weight superiority**. Cold-start sparse seeds 12072, 12075, 12077 both attained feasibility; terminal five-second checks are deterministic numerical screens only.

## Diagnostic questions (do not tune with these data)

1. **Matched prefix and first divergence:** for each of 16 case/seed pairs, verify from raw proposals and observations that both arms share the same initial set, ordered evaluated designs and records up to first difference. Record first differing iteration, actual selected candidate IDs, prior SOC weights, prior score counters, and `mode`, `fallback`, `selected_index`, `pool`, `kappa`, training feature/head identities. Distinguish cases where the candidate pools/forecasts/modes are *not* identical: a weight-only counterfactual selection is **not identified** if any other upstream difference exists. Do not substitute one arm's observation history after divergence.
2. **Candidate ranking and decision relevance:** on *shared-prefix* proposal steps only, inspect the archived full-pool forecasts. If saved fields and the unchanged source acquisition function suffice, recompute candidate acquisition scores at the two registered weights on the **same pool and posterior components** without fitting GP or querying the evaluator. Identify the selection-margin/rank reversals attributable to changing SOC mixture weight, including optimistic support and exact componentwise SOC feasibility probability. Explicitly label when an exact rank counterfactual cannot be reconstructed; never silently use the moment-matched Gaussian as though it were the exact mixture CDF.
3. **Predictive reliability mechanism:** from committed selected SOC Gaussian forecasts and subsequent real SOC observations, independently recompute one-hot/latent exact-density NLL and interval-mass NLL *separately*. Count clipped scores at ±4, stale or missing forecasts, and weight changes that alter the top candidate versus those that merely perturb weight. Describe why selected on-policy observations cannot establish population calibration.
4. **Cost vs. achieved outcome:** compare first feasible call in cold starts, incumbent energy versus online iteration, median/mean per-task paired energy, 5-second numerical verification, fit and proposal time, and the seed-12075 leave-one-out sensitivity. Preserve every seed and infeasible observation. No significance or superiority claim from these eight development seeds.

## Deliverables and stop rules

Create a portable **read-only** analysis script that reads the already sealed P3 ZIP or the extracted **unmodified** archive. It must verify SHA256 before use, require the frozen plan/receipt, and generate:

- `diagnosis.json` with one row per case/seed and precise flags for prefix identity, first divergence, policy weights, candidate pools, forecast equality, decision-margin diagnostics, and unidentifiable comparisons
- `scores_by_type.json` for exact/interval, score clipping and skipped observations; note clustered adaptive data
- `decision_sensitivity.md` summarizing supported findings, negative results, computational overhead and **specific** remaining scientific uncertainties
- Offline CI/tests with **zero** new simplified-aircraft or RCAIDE evaluations; no GP/VAE refit unless explicitly separated as a *new* mathematical diagnostic and labeled as such

Upload small analysis outputs on this **analysis-only branch** and update its Draft PR. Do not alter any P0–P3 code, original ZIP, ledger, audit or Release. Do not rerun any seed, start the next pilot, create H2.5/P5 algorithm changes or claim novelty without a follow-up design review. A negative diagnosis is a valid research result.

Only after this evidence will we choose whether to pursue decision-aware mixture gating, risk-score calibration, a different candidate-selection rule, or to retain the fixed 50/50 policy.
