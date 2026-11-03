# H2 P6 — Two-stage SOC-weight intervention: local generation versus final selection

Status: pre-registered zero-truth research protocol, 2026-11-03. New simplified-aircraft calls 0, RCAIDE 0, VAE training 0. No algorithm promotion or additional pilot.

## Motivation and frozen evidence

- Original P3 eight-seed source SHA 57e390fc8f6280db9f201940aec14089e81c104c; ZIP SHA256 bd170195201b45942e4fe2c2e04e1e994263bb0a4566a467b28816cdce0473b4, independently audited in PR #31.
- P4 PR #32 established 16/16 equal matched observation histories and ordered global candidate pools at the first different online proposal, but different final locally generated pools. Fixed-archived-pool top model rank reversed in 8/16 cases, without full local replay.
- P5 PR #33 latest SHA 592068514a7cb23103b3f07f6ebb1a6b78a4e875, CI run 41571260193: 34 passing tests. Both original complete proposals, saved augmented pools and SLSQP local reports reproduced twice in 16/16 pairs. Only 2/16 had non-near-tied choices under the primary score threshold (1e-10); both showed differing selected candidates. 14/16 remained not identified. Historical P3 did NOT record all internal SLSQP query points; reproducing saved endpoint/report evidence cannot retroactively establish identical original query traces.
- P3 adaptive versus fixed 50/50 energy remains mixed: 100 km mean -42.46 Wh and median -0.56 Wh (5 wins/3 losses); 140 km mean -262.91 Wh and median +0.38 Wh (4 wins/4 losses), with 140 km mean +9.15 Wh when influential seed 12075 is removed for sensitivity only. These are development, not confirmation data.

## Scientific estimand

In the same P3 historical GP posterior, observations, RNG, branch/radius state and global candidate pool, what is the effect of SOC weight during **local candidate generation**, separately from the weight used during **final ranking**?

For each of 16 previously frozen first-divergence case/seed pairs, perform four single-step controlled cells with two input factors, generation weight wg and final-selection weight ws:

| Cell | wg | ws | Role |
|---|---|---|---|
| DD | recorded dynamic wd | wd | native dynamic reproduction |
| FF | 0.5 | 0.5 | native fixed reproduction |
| DF | wd | 0.5 | dynamic-generated pool, fixed final ranking |
| FD | 0.5 | wd | fixed-generated pool, dynamic final ranking |

There are 64 single-step decisions, 32 historical reproduction controls and 32 crossed interventions, but only 16 independently seeded matched histories. No counterfactual physical response or optimization continuation is known. Candidate ID is categorical: do not compute an additive numerical effect over design IDs.

## Implementation and invalidation rules

1. Base on the updated P5 analysis branch. Validate original P3 ZIP SHA256, all source files, per-trial prefix identities and saved checkpoints before doing anything. Read checkpoint using CorpusRepresentation.load_checkpoint; no VAE fitting. Fit one GP ensemble per original matched history using ONLY prior settled observations (count all kernel candidates and wall-clock GP time), then reuse that exact posterior for all four cells. No future observations, no physical evaluator, no new structural corpus training.
2. Use the original code behavior of DiscoveryBO.propose, RiskRefiningBO._augment, source choose_acquisition, exact two-Gaussian SOC feasibility CDF, moment-based optimistic margins, deterministic random stream, SLSQP solver, near-filter, eligibility, original coverage overrides and branch radii/stalls. Do not edit P3 optimizer or physical sources. A P6-only analysis adapter may change posterior.h2_weight between stages: before preliminary global scoring and all calls inside local augmentation it MUST be wg; immediately after candidate-generation/near-filter and before final posterior scoring it becomes ws. Preserve the original final selection/coverage control flow. Independently reset the same GP weight and optimizer state before every cell.
3. Native DD and FF must each reproduce the original candidate identity, ordered augmented pool, local reports, scores, selected indices, coverage/mode and head diagnostics, twice. If either fails, mark the entire pair not identified; never relax checks to force a result.
4. Use conditional contrasts: generation effect at fixed wd = DD versus FD; generation effect at fixed 0.5 = DF versus FF; final-ranking effect on dynamic-generated pool = DD versus DF; final-ranking effect on fixed-generated pool = FD versus FF. Conditional effects need not agree, and their interactions cannot be added as scalar energy effects. Record changed candidate IDs, distinct pool members, source-of-candidate (global/local), generated branches, solver failures and final selected margin.
5. Near tie handling: primary top-two acquisition gap threshold 1e-10 (P5 rule). Mark pair not identified when any necessary cell is near-tied, has a coverage override not faithfully re-executed, numerical solver instability or native-replay discrepancy. Sensitivity tables at thresholds 1e-12, 1e-11, 1e-10, 1e-9, 1e-8 are descriptive only; do not choose a favorable post-hoc cutoff. Record score scale, normalized gap, exact ties and repeated-run identity.
6. Original SLSQP query traces do not exist. New diagnostic callback logging in an offline run is permitted only if explicitly labeled newly generated, with no claim it reproduces unrecorded historical calls. No jitter used to force a decision.

## Deliverables

Add a self-contained P6 zero-truth analysis and tests to this Draft PR:
- two_stage_cells.json, 16 pairs x 4 cells; original/current identities, native gates, conditional candidate contrasts, score margins, first generated pool divergence, counts and explicit identification categories.
- tie_sensitivity.json with the fixed threshold grid, decision stability, solver and coverage diagnostics, and competing near-tie candidate IDs.
- two_stage_attribution.md with cautious mechanism conclusions, computational costs and the original P3 outcome sensitivity, and a stated go/no-go for designing any new algorithm.
- GitHub CI and Windows tests with patched aircraft/RCAIDE and disabled VAE training; 0 new aircraft evaluations, 0 RCAIDE.

Stop after this diagnostic. Do not design P7, alter original H2 method, relaunch P3, promote default algorithms or claim dynamic SOC weights improve safety/performance. If non-identifiability persists, report it plainly rather than changing thresholds.