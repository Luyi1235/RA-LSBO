# Propulsion-layout partial pooling for aircraft risk cold start

PR110 and PR111 showed earlier first discovery versus maximin coverage without a corresponding gain on the registered held-out Brier endpoint. In PR111, the staged interleaved policy found 29 usable designs in still air versus 7 for coverage. It visited 162 versus 173 new configurations (within-seed unique counts summed over seeds), and its mean conditional feasibility Brier at 16 calls was 0.25911 versus 0.24513. These observations motivate a controlled test of the surrogate while collecting the same labels. PR111 evidence is frozen at ab06a893dedbb5d44840908d4572fffb1cf62332.

## Transfer across the hierarchy

For each probit risk head, use the existing explicit masked features and grouped Matérn-5/2 kernel, with the same eight kernel length-scale candidates and evidence selection. Compare three zero-mean latent priors with identical unit marginal variance and the original unit probit noise.

| Prior | Global random-intercept variance | Propulsion-layout intercept variance | Local Matérn variance |
| --- | ---: | ---: | ---: |
| Matérn | 0 | 0 | 1 |
| Global | 0.50 | 0 | 0.50 |
| Layout | 0.25 | 0.25 | 0.50 |

The layout intercept adds covariance only when two designs share the propulsion-layout parent, which activates different design variables. Thus, the same-layout and global intercept terms represent two known levels of statistical sharing. The global comparator keeps the local variance and total intercept variance fixed. Variance splits are fixed without fitting them to result files. The model uses no past simulation labels and learns each task independently. Aborted tasks retain completion evidence. They never enter the conditional feasibility likelihood. The existing independent-head product gives usable probability. The active selector uses the joint-usable expected Brier reduction from PR110 with each arm's own GP and conditional label receipt.

| Arm | Prior | Selector |
| --- | --- | --- |
| risk_coverage | Matérn | explicit maximin |
| risk_global_coverage | Global | explicit maximin |
| risk_layout_coverage | Layout | explicit maximin |
| risk_active | Matérn | joint-usable information |
| risk_layout_active | Layout | joint-usable information |

The three coverage models replay the identical maximin decisions and use one charged physical evaluation per step, with independent model fits and forecast journals. The two active models have individually charged paths. The verify step checks that coverage variants reuse the exact same designs and records in order.

## Frozen evaluation

Two 140 km, 20 s integration conditions, 0 and 4 m/s headwind. Both share 24 new seeds, disjoint from the prior formal experiments. Each seed shares four initial evaluations, a pool of 128, a reference panel of 96 and a held-out panel of 64. Each policy uses at most 12 online evaluations. Held-out labels are evaluated after all policy evaluations and never enter fit or selection. This opt-in experiment holds the explicit representation fixed to identify the effect of the risk prior.

The primary paired contrast is risk_layout_coverage minus risk_coverage. For each seed, average the held-out configuration-macro usable Brier at 8, 12 and 16 calls. Report paired bootstrap 95% intervals separately by condition. The global variants isolate layout-specific sharing, risk_layout_active minus risk_active measures the end-to-end change with information selection, and risk_layout_active minus risk_layout_coverage tests acquisition with the new prior. Separately compare each forecast with a Beta(1,1)-smoothed constant usable rate from its own observed prefix under the same configuration-macro scoring. This simple rate diagnostic helps distinguish probability gains available without input structure. Report staged-head metrics and first-discovery results as secondary evidence.

| Profile | Initial + online + held-out | Charged reduced evaluations |
| --- | ---: | ---: |
| hierarchical-prior-smoke | 4 + 9 + 16 | 29 |
| hierarchical-prior-nominal | 96 + 864 + 1536 | 2496 |
| hierarchical-prior-headwind | 96 + 864 + 1536 | 2496 |

Total 5021 reduced-aircraft calls and zero RCAIDE/MDA calls. Use the existing Python 3.9.25 environment. Run the aircraft risk tests, then plan, run, verify and analyze each profile. Each formal discovery_pairs.csv contains 144 seed-contrast rows. Preserve original ledgers and analyses with the frozen code SHA, commands, environment and checksums.
