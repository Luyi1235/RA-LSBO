# EP stability: numerical audit and leakage-safe SOC forecast protocol

**Date** 2026-11-01. Investigation only; never resume the interrupted original aircraft pilot. Original fixed pilot costs: 2,212 fully settled model evaluations, still incomplete with no final results/audit.

## Evidence and verified claims

- Original program SHA: fc1ac773f5bd9e60abd2ca4b56bd971a707735e2.
- Stable-tail code SHA: b5744736c942eadf0a56f79d811f7e7b8ca3d52b (draft PR #15).
- Original interrupted evidence ZIP SHA: aa7f7b618eb126bcb0f7fa67c886664f235a7432ad43c269ce833262c05bd2e9.
- Fixed stable-tail D29 evidence ZIP SHA: 0787fed8ce9ce59fa568c761e1bfe01337c672ebd74965ec970235cf30e1279f.
- All 8 D29/SOC candidate EP fits converge with opt-in tail quadrature, 25-36 sweeps. This establishes convergence on a single historical input, not posterior accuracy or uncertainty calibration.

## Completed independent historical old30 prequential review (no fits)

Read the previously verified interrupted-pilot triage ZIP (SHA ef08416196ca25cbf713c3b32187e7eff03b73d3534c3478c506a3563c73a22a).
For each saved old30 aircraft_sparse/seed9936 proposal, use ONLY its saved selected-candidate SOC forecast and subsequently settled outcome; ensure matching iteration/candidate/pool[selected_index]. Do not recompute GP, VAE or the flight model. Exclude coverage proposals that explicitly skipped a posterior (do not impute a forecast). Proper scoring:

- Exact observed SOC margin: Gaussian negative log density, physical output units;
- One-sided interval: negative log Gaussian probability mass for the observed bound;
- Never combine exact-density and interval-mass scores in one headline average.

Actual result: 29 online proposals, 2 scheduled coverage/no posterior, 15 exact SOC predictions and 12 interval predictions. Of 15 exact outcomes, only 6 lie inside saved central 90% Gaussian prediction intervals and 7 inside central 95% intervals. Median absolute standardized error 2.201; maximum 56.523. Exact NLL median -0.150, mean 165.213 (dominated by extreme misses); interval mean NLL 3.982, median 1.739. E.g. online indices 22/23/27/28 miss by about -14.12 / -35.49 / -56.52 / -10.30 predictive SDs.

**Interpretation:** historical old30 can be severely overconfident. It is the original old30 model, NOT stable-tail or onehot. This review does NOT show stable-tail improves calibration, nor that all 29 adaptive targets represent independent test replicates. Scoring uses predictions saved BEFORE evaluation. The source and detailed 29-row replay are separately included in the conversation handoff.

## Numerical geometry audit (tool under review, not a tested model fix)

tools/audit_ep_d29_static.py takes the frozen 14-file EP-D29 Release ZIP and an unused output directory. It verifies ZIP digest, per-file hashes, 41 source evidence rows, 8 original kernel identities and saved EP convergence trace. Independently reconstructs the covariance BEFORE EP using the saved D29 feature matrix, six continuous features and original grouped Matérn-5/2 kernel. It records exact-conditioning and interval-conditional covariance eigenvalues/condition numbers and approximate EP-evidence weight concentration. This uses NumPy/SciPy only: **no EP fit, GP fit, VAE fit or model evaluator.**

The tool and synthetic tests are checked into this analysis branch for review. Its full fixed-Release execution has **not** been verified; do not claim covariance metrics or issue a quality pass from its mere presence.

## Next registered assessment — distinct from prior in-sample D29 convergence

A properly leakage-safe forward SOC comparison MUST use a model and encoder trained solely on the prefix *before* the held-out outcome. Never train a D29 encoder (which saw the 41st label) then pretend the 41st observation is a prequential holdout.

An initial *bounded* correctness gate can use D28 (40 observations) to predict the actual 41st SOC outcome at the recorded iteration 28 selected candidate, with:
1. The original old30 D28 SOC predictive mean and variance saved prior to evaluation (NO old30 refit).
2. Opt-in stable-tail D28 GP, with encoder trained only on D28 history, on the original 8 group-kernel candidates, one fit each.
3. Identically sourced hierarchical one-hot D28 GP on that same 40-row history, with the same 8 kernel candidates, one fit each.

Each model must predict the **same exact candidate** under the same chronological input, target and masking rules. The original D28 observed outcome is the held-out test, not GP training data. Main outputs: each head's 8-kernel success count, selected kernel/evidence, proper exact SOC density NLL, variance/mean, calibration error and fit time; use exact data only as observed, and keep one-sided outcomes as intervals in other comparisons.

That is at most 1 new D28 VAE encoder fit, 16 new GP kernel fits, 0 simulator/RCAIDE calls. **This is a specification, NOT an executed or authorized new numerical run.** One holdout cannot establish calibration or general benefit; decide whether to register a broader, seed-matched historical prequential grid only after validating implementation and leakage controls. No model is promoted based on this one point.

If extending to multiple historical prefixes: pre-register seeds, prefixes, eligible outcomes and same subsequent candidate per method before fitting; use original prefix data only; report by case/seed and exact-vs-interval type; keep missing and errors, uncertainty scores and wall time separate. A pooled 80-point convenience sample cannot be treated as 80 independent trials.

## Prohibitions

- Do not run any part of the old pilot or release its locks.
- Do not change original simulator equations or evidence.
- Do not carry D29-trained VAE labels backward into earlier prefixes.
- Do not claim EP convergence proves calibration.
- Do not refit the already tested 8 D29 SOC kernels as a quality comparison.
- Do not rerun RCAIDE, true objectives, acquisition optimisation or full new BO experiments without separate approval.
