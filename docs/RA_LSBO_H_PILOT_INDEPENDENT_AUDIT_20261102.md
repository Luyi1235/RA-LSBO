# RA-LSBO-H: independent read-only audit and four-seed paired pilot review

**Date:** 2026-11-02. **Status:** Exploratory development data; frozen algorithm and original Windows evidence not modified. No new model, GP, VAE, acquisition, or RCAIDE evaluations.

## Reproducibility

- Frozen five-arm RA-LSBO-H code: \`b219163fc565dfc32876f38f0bbf5f4e65abbd36\`.
- Original Windows four-seed Release: https://github.com/Luyi1235/RA-LSBO/releases/tag/ra-lsbo-h-pilot-9961-9964-20261102
- Original released ZIP SHA256: \`3d9cb4f684bbc9198c523bb44bf466463c2f85053d87ca1770eca9111b48fed1\`.
- Read-only Linux Actions proof: https://github.com/Luyi1235/RA-LSBO/actions/runs/41373855061
- Independent scripts: \`tools/analyze_dualview_pilot_sealed.py\` and \`tools/analyze_dualview_guard_mechanism.py\` on the separate analysis branch.
- Downloaded 46,591,014-byte ZIP; checked its full SHA-256, read/hash all 641 members, checked 461 member hashes against original audit.json, checked 171 source snapshots, checked registered plan and registration bytes, 856 ordered start/settle pairs, all 40 trial saved summaries against results.json, all 720 online candidate/record pairs, same initial design records for all arms, all 40 successful same-design 5-second checks. The model was not re-executed and the VAE training was not replayed.
- Original 2,212-call interrupted corpus pilot remains separate, not resumed.

## Energy outcomes: four seeds per task

Nominal final best feasible energy (Wh; lower is better). All 40 trials obtained a nominal feasible incumbent and all 40 same-design 5s checks passed. The 5s checks are deterministic numerical step sensitivity, **not independent physical safety validation**.

| Case | Method | Mean Wh | Median Wh | Best Wh | Online usable / 72 |
|---|---|---:|---:|---:|---:|
| 100 km | random | 30,009.20 | 29,301.90 | 28,766.23 | 31 |
| 100 km | onehot_stable | 26,300.73 | 26,292.22 | 26,242.57 | 44 |
| 100 km | corpus_stable | 26,370.75 | 26,344.40 | 26,238.29 | 38 |
| 100 km | hybrid_onehot | 26,356.35 | 26,319.54 | 26,276.28 | 44 |
| 100 km | hybrid_guard | 26,327.91 | 26,327.78 | 26,266.69 | 47 |
| 140 km | random | 43,551.34 | 43,416.65 | 40,166.84 | 7 |
| 140 km | onehot_stable | 34,594.34 | 34,575.37 | 33,333.53 | 41 |
| 140 km | corpus_stable | 35,391.68 | 35,653.98 | 33,343.75 | 38 |
| 140 km | hybrid_onehot | 33,472.72 | 33,429.49 | 33,349.36 | 43 |
| 140 km | hybrid_guard | 34,638.35 | 34,621.75 | 33,358.53 | 41 |

Note that 140 km seed 9962 had **cold initial status** across all arms, the other three 140 km seeds had warm initial status. For 100 km the four initial sets were warm. Do not pool cold/warm and interpret as iid data.

## Preregistered paired ablations

All differences are **left method minus right method** in Wh; negative means the left method obtained lower final energy. Pairing is by the SAME task/seed/12 shared initial records. The optimizer proposals after initialization may differ (on-policy comparison).

| Task | Left minus right | Mean difference Wh | Median Wh | Left lower : Right lower |
|---|---|---:|---:|---:|
| 100 km | hybrid_onehot − corpus_stable | −14.40 | −3.94 | 2 : 2 |
| 100 km | hybrid_guard − hybrid_onehot | −28.44 | −4.80 | 3 : 1 |
| 100 km | hybrid_onehot − onehot_stable | +55.63 | +47.20 | 1 : 3 |
| 140 km | hybrid_onehot − corpus_stable | **−1,918.97** | **−2,127.96** | **3 : 1** |
| 140 km | hybrid_guard − hybrid_onehot | **+1,165.64** | **+1,095.72** | **0 : 4** |
| 140 km | hybrid_onehot − onehot_stable | −1,121.63 | −1,167.20 | 2 : 2 |

140 km individual seed deltas:
- hybrid_onehot − corpus_stable: seed9961 −3425.57; 9962 −2223.69; 9963 +5.61; 9964 −2032.22 Wh.
- hybrid_guard − hybrid_onehot: 9961 +2461.94; 9962 +124.57; 9963 +9.17; 9964 +2066.87 Wh.
- hybrid_onehot − onehot_stable: 9961 −2403.66; 9962 −2350.25; 9963 +15.83; 9964 +251.56 Wh.

Interpretation: 140 km risk-head decoupling is a promising candidate (three of four seeds lower energy relative to uniform corpus latent), but the benefit of latent objective features over onehot is uneven (two of four wins). The additional conservative SOC dual-view rule is associated with *higher* energy in all four 140 km seeds. The 100 km differences across GP arms are much smaller and mixed. Four development seeds support **hypothesis generation**, not a claim of confirmed method superiority.

## SOC risk-gate mechanism audit

Read from stored prediction metadata at the actually selected online design, before seeing the result. This is not a model refit.

- In 100 km hybrid_guard, a latent-SOC view was chosen for 27/72 predicted candidates (37.5%); mean absolute SOC feasibility probability disagreement = 0.0255.
- In 140 km hybrid_guard, latent SOC was chosen for 28/71 predicted candidates (39.4%); mean disagreement = 0.0516.
- Relative to hybrid_onehot, the guard policy chose a different design in **iteration 0** for 140 km seeds 9961, 9963 and 9964. Seed 9962 had its first different choice at iteration 2 (one earlier iteration scheduled coverage/otherwise unfitted). Thus even occasional view switching can change early search trajectories. This does not establish the precise causal mechanism for worsened energy.
- In 140 km, hybrid_guard produced 41 usable online outcomes across 4 seeds versus 43 under hybrid_onehot, despite its more conservative selection heuristic. This is not evidence of a better feasible-return rate in these runs.

## SOC prequential diagnostics: not the same held-out candidates

These are per-arm adaptive selected-point forecasts, NOT a shared fixed test design. Exact SOC density NLL and right-censored interval probability-mass NLL are fundamentally different scores and **must not be averaged together**. The random arm never fits an SOC GP and cannot be scored.

140 km selected-point records:

| Arm | Exact SOC n | Central 95% covered | Exact density NLL mean | Interval SOC n | Interval mass NLL mean |
|---|---:|---:|---:|---:|---:|
| onehot_stable | 56 | 53/56 | −2.325 | 15 | 1.549 |
| corpus_stable | 55 | 49/55 | −1.994 | 16 | 1.845 |
| hybrid_onehot | 60 | 56/60 | −2.104 | 11 | 1.786 |
| hybrid_guard | 61 | 56/61 | −1.876 | 10 | 1.430 |

Coverage can be deceptively close to 95% under adaptive, dependent, different selected trajectories. Differing exact/interval counts and response magnitudes mean these per-arm NLL means cannot be compared as if on a matched supervised test set.

## Next experiment recommendation (NOT authorization)

1. **Do not promote hybrid_guard** to default: its conservative SOC probability heuristic failed the 140 km energy tradeoff in the four-seed pilot; the effect may be partly early trajectory divergence. Retain the implementation/evidence as a negative ablation.
2. Preserve \`hybrid_onehot\` as a *candidate* (not proven winner), alongside \`onehot_stable\` as a competitive low-compute baseline. \`corpus_stable\` is a necessary full-latent control.
3. Before changing any algorithm or running another large pilot, register a zero-truth **cross-policy SOC prediction panel**: replay the saved candidate pools and labels read-only, using strictly prefix-only feature construction and equal-cost paired prediction evaluation. Avoid D29 retrospective leakage and do not use candidate-specific unseen labels in fitting/selection. Do not infer causal benefit from the existing on-policy SOC NLL.
4. For a new independent exploratory study, preregister seeds, task-specific feasibility/power criteria and maximum calls; choose whether to eliminate the guard arm, but never retroactively change this fully completed five-arm pilot.

**No pilot resume, no new aircraft/RCAIDE evaluation, no promotion or main merge occurred in this analysis.**
