# H2 P4: first-divergence decision sensitivity

Frozen P3 archive SHA-256: `bd170195201b45942e4fe2c2e04e1e994263bb0a4566a467b28816cdce0473b4`; 16 case/seed pairs, eight seeds. Zero new aircraft or RCAIDE calls, GP fits, VAE fits, and optimizer runs.

## Matched histories and first divergence

All 16 pairs have identical initial records and ordered observations through the last common candidate. At first divergence, 16/16 have the same ordered novel global pool, but 16/16 have different final augmented pools. All 16/16 have equal saved head diagnostics. Comparing saved forecasts on identical shared candidates, 12/16 have maximum absolute difference at most 1e-10 in every checked field; remaining differences are recorded per field in diagnosis.json.

On the archived dynamic-arm final pool, source acquisition was reproduced and changing only the SOC mixture weight reverses its top model choice in 8/16 first-divergence steps. This is a fixed-pool counterfactual, not a replay of the other arm: local candidate generation was weight-dependent and not rerun. All 16 actual first divergences have different final pools, so direct weight-only attribution of the full decision is not identified. Different choices after first divergence have different training histories and are descriptive trajectory outcomes, not matched one-step effects.

| Case | Seed | First divergence (zero-based) | Dynamic weight | Global pool identical | Shared candidates | Fixed-pool rank reversal |
|---|---:|---:|---:|---|---:|---|
| aircraft_ordinary | 12071 | 4 | 0.484621 | True | 128 | True |
| aircraft_ordinary | 12072 | 4 | 0.604183 | True | 152 | True |
| aircraft_ordinary | 12073 | 4 | 0.702039 | True | 128 | True |
| aircraft_ordinary | 12074 | 4 | 0.515494 | True | 172 | True |
| aircraft_ordinary | 12075 | 4 | 0.493604 | True | 176 | False |
| aircraft_ordinary | 12076 | 4 | 0.419206 | True | 149 | False |
| aircraft_ordinary | 12077 | 4 | 0.491727 | True | 159 | False |
| aircraft_ordinary | 12078 | 4 | 0.507534 | True | 173 | False |
| aircraft_sparse | 12071 | 4 | 0.636766 | True | 139 | True |
| aircraft_sparse | 12072 | 5 | 0.608322 | True | 172 | False |
| aircraft_sparse | 12073 | 5 | 0.465675 | True | 175 | False |
| aircraft_sparse | 12074 | 4 | 0.786312 | True | 140 | True |
| aircraft_sparse | 12075 | 5 | 0.510123 | True | 174 | False |
| aircraft_sparse | 12076 | 4 | 0.566344 | True | 152 | True |
| aircraft_sparse | 12077 | 9 | 0.473652 | True | 176 | False |
| aircraft_sparse | 12078 | 4 | 0.609377 | True | 151 | True |

## Selected SOC likelihood scores

Component NLLs are computed from the committed forecasts and subsequently settled SOC evidence. Exact-density and interval-probability-mass scores are not pooled. Counts are clustered, adaptive, selected on-policy observations; they do not establish population calibration.

| Arm / score type | n | One-hot mean NLL | Latent mean NLL |
|---|---:|---:|---:|
| h2_dynamic/exact_density | 254 | -2.353401 | -2.161391 |
| h2_dynamic/interval_mass | 30 | 1.825449 | 2.509713 |
| h2_fixed50/exact_density | 250 | -2.375958 | -2.209010 |
| h2_fixed50/interval_mass | 34 | 1.721709 | 2.406821 |

h2_dynamic: positive/negative score-difference clipping 4/0; coverage-skipped 4; missing SOC 0; proposal weights at lower/upper bounds 0/0.

h2_fixed50: positive/negative score-difference clipping 4/0; coverage-skipped 4; missing SOC 0; proposal weights at lower/upper bounds 0/0.

## Outcomes and cost

All recorded seed pairs are retained. Energy differences are dynamic minus fixed, conditional on both nominally feasible. The five-second repeat is a numerical screen, not physical validation.

| Case | Mean difference Wh | Median Wh | Mean without largest absolute pair Wh (seed) | Mean without 12075 Wh | Dynamic/fixed mean optimization seconds | Dynamic/fixed mean VAE fit seconds |
|---|---:|---:|---:|---:|---:|---:|
| aircraft_ordinary | -42.460 | -0.558 | -10.366 (12072) | -41.373 | 38.207 / 38.334 | 3.123 / 3.023 |
| aircraft_sparse | -262.908 | 0.379 | 9.150 (12075) | 9.150 | 49.187 / 48.967 | 2.971 / 2.975 |

Sparse seed 12075 contributes -2167.313 Wh; its removal changes the sparse mean to the value above. This is sensitivity analysis, not a license to discard the seed.

| Cold-start case / seed | First feasible online call, dynamic | First feasible online call, fixed |
|---|---:|---:|
| aircraft_sparse / 12072 | 2 | 2 |
| aircraft_sparse / 12075 | 2 | 2 |
| aircraft_sparse / 12077 | 9 | 9 |

All 32/32 same-design five-second numerical checks were usable. VAE fit time is included in optimization time, not additive. GP fit time was not separately recorded and cannot be inferred from proposal time. Full per-iteration incumbent curves and timings are in diagnosis.json.

## Limits

Archived full-pool scores are recomputed with the frozen acquisition, exact componentwise SOC feasibility CDF and moment-based optimistic margins. Scores are reconstructed only for the already saved pool, not the unobserved counterfactual local-search trajectory; coverage overrides are marked unidentifiable. No GP/VAE refit or true evaluator was called. The P3 data are an eight-seed development panel; these results neither prove superiority nor identify a new optimal policy.
