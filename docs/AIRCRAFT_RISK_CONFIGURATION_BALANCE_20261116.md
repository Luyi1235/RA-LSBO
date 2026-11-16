# Configuration-balanced active learning in a four-sample aircraft risk cold start

PR112 compared alternative risk GP priors on identical coverage data. The layout prior lowered the registered Brier endpoint in still air relative to the original GP (−0.001735), while the headwind difference was +0.000139. Relative to the global-sharing control, the headwind layout-prior difference was +0.000641. On the original GP, joint-usable information selection discovered feasible designs earlier but raised headwind Brier by +0.006763. At 12 calls, the original coverage and active arms had visited a mean of 11.50 and 9.33 configurations in both conditions. At 16 calls in headwind, their configuration-macro signed errors in usable probability were +0.04353 and +0.08744. These are exploratory mechanism diagnostics from the raw PR112 records, frozen at f855dd014dcc8cda94fb98c346280ee25f61c3f9. They motivate a controlled acquisition intervention.

## Query rule

An exact configuration is the full canonical discrete part of the design, including the propulsion branch, active discrete selections and resolved inactive defaults. At decision t, let N_t(g) be the number of charged evaluations in configuration g among the initial and online observations of the current arm. For balanced acquisition, score only candidates whose configuration has minimum N_t(g) among configurations still represented in the available pool. Maximize the underlying acquisition over this eligible set with the existing candidate hash tie-break.

As long as an unvisited configuration has an available candidate, a balanced policy visits one of the unvisited configurations. This follows directly from N_t(g)=0 for those configurations and N_t(g)>0 for visited ones. The next round of visits prioritizes configurations with the fewest calls. Counts include attempts yielding incomplete feedback because they consume the evaluation budget. The rule uses labels only through the underlying information score. It does not require a weight parameter and it cannot guarantee balanced valid labels or balanced points within each continuous subspace.

| Arm | Candidate eligibility | Ranking among eligible candidates |
| --- | --- | --- |
| risk_coverage | all available | original explicit maximin |
| risk_active | all available | PR110 joint-usable expected Brier reduction |
| risk_joint_interleaved | all available | maximin every third call starting with the first online call, joint information otherwise |
| risk_balanced_coverage | smallest configuration count | maximin |
| risk_balanced_active | smallest configuration count | joint information |

All arms use the original staged probit GPs, the existing explicit hierarchical input features and the same conditional label semantics. This isolates the acquisition effect before testing additional prior or representation changes. All policies charge their own online calls. The verifier reconstructs eligible candidates from the sealed prefix and the remaining pool and replays every decision.

## Registered evaluation

The primary comparison is risk_balanced_active minus risk_coverage. For each seed, average the held-out configuration-macro usable Brier scores at 8, 12 and 16 evaluated designs. Report paired bootstrap 95% intervals separately under still air and 4 m/s headwind. Negative differences favour the first arm. Matched comparisons with risk_balanced_coverage, risk_active and risk_joint_interleaved measure the effects of information ranking within the gate, the gate with fixed ranking, and per-query balancing versus fixed interleaving. The final matched contrast risk_balanced_coverage minus risk_coverage measures balancing with maximin ranking. First discovery remains secondary and is evaluated on initial-unsolved seeds with the existing capped call score.

Record mean signed error in usable probability with the same configuration weighting, numbers of evaluated configurations, maximum occupancy of any configuration, and squared error on unseen held-out configurations with its denominator. The unseen subset can differ across arms and supports descriptive diagnosis. Report counts of usable, aborted and completed-but-violating online labels. Positive signed error means overpredicted usability.

Use 140 km reduced missions with a 20 s integration step. Both scenarios share the same 24 fresh seeds, independent of prior formal profiles. For each seed, all arms share four initial calls, 128 candidates, 96 unlabelled reference points and 64 held-out evaluations. The held-out calls run after the online loop. Each arm makes 12 online evaluations in a formal profile.

| Profile | Initial + online + held-out | Charged reduced evaluations |
| --- | ---: | ---: |
| config-balance-smoke | 4 + 15 + 16 | 35 |
| config-balance-nominal | 96 + 1440 + 1536 | 3072 |
| config-balance-headwind | 96 + 1440 + 1536 | 3072 |

Total 6,179 reduced aircraft calls and zero RCAIDE/MDA calls. Use Python 3.9.25, run the aircraft risk tests, and then plan, run, verify and analyze each profile. Each formal discovery_pairs.csv contains 120 rows of seed-level contrasts. Preserve raw ledgers, analyses, code SHA, environment, command logs and checksums.
