# RA-LSBO-H SOC common-prefix panel — independent audit and robustness

Date: November 2, 2026. **Development-only evidence.** Zero new model fitting, optimization, simulator or RCAIDE calls.

## Reproducibility and verification

Original Windows pilot Release ZIP SHA256: 3d9cb4f684bbc9198c523bb44bf466463c2f85053d87ca1770eca9111b48fed1.
New 24-prefix SOC GP panel Release ZIP SHA256: a8b177391b5c6fdbbd2f8a00f60f6f2501191f0c87341762c5daef825b5cbe01.
Fixed original panel execution source commit: 4e864d400b363cdb06821b3872cb2d4c7a039958.

Independent read-only audit CI: https://github.com/Luyi1235/RA-LSBO/actions/runs/41391333688

The standard-library/NumPy/SciPy independent audit re-reads both entire SHA-pinned ZIPs; reconstructs each of the 120 SOC outcome identities from the original 856-call pilot observations (not from panel-provided labels); independently recomputes 120 Gaussian density/interval-mass NLLs, feasibility Brier scores and exact 95% interval indicators from stored GP predictive mean and variance. It checks 87 panel members, 641 original pilot members, 24 chronological prefixes, 48 head results with all eight kernel candidates reporting success, 24 recorded original onehot prediction reproduction flags, and 116 unique prefix-level designs from 120 policy-linked target records.

This does NOT replay GP/EP model fitting, VAE training or historical optimizer selection. File-level SHA covers the entire two sealed release archives.

## Primary descriptive result (same prefix, same candidate, same label)

Each difference is latent SOC NLL minus hierarchical onehot SOC NLL. Positive means onehot is better at that matched heldout target. Exact Gaussian density NLL and censored interval probability-mass NLL are not pooled.

| Task | SOC evidence | n | Mean paired NLL delta | Median delta | Onehot lower NLL |
|---|---|---:|---:|---:|---:|
| 100 km | Exact | 53 | +0.100385 | +0.039549 | 35/53 |
| 100 km | Interval | 7 | -0.346171 | -0.308721 | 2/7 |
| 140 km | Exact | 36 | -0.074094 | -0.008921 | 18/36 |
| 140 km | Interval | 24 | **+0.560331** | **+0.071892** | **16/24** |

Overall, exact n=89: onehot NLL -2.391805, latent NLL -2.361996; onehot 95% hits 87/89, latent 88/89. Interval n=31: onehot NLL 1.279576 and latent NLL 1.635213. Across all 120 resolved event feasibility labels: onehot Brier 0.117659, latent Brier 0.119611.

## High-influence censored point and seed dependence

The 140km interval mean (+0.560331) is substantially affected by one historical target: seed 9963, online iteration 1, corpus_stable target. That target had onehot interval NLL 6.648396 and latent interval NLL 16.532252, paired difference **+9.883856**. It accounts for approximately **73.5% of the total positive net NLL delta sum** among the 24 interval records.

Do not remove it from the primary result. A clearly labeled one-out sensitivity calculation gives mean +0.154961 after deleting just the largest positive paired difference. The 24-point median remains +0.071892; 16/24 intervals favor onehot. Distinct original case/seed clusters number only FOUR, not 24 independent trials.

140km interval mean deltas by original seed:

- 9961: +0.252235 (n=4, 4/4 positive).
- 9962: +0.696944 (n=4, 4/4 positive).
- 9963: +0.943289 (n=9, only 2/9 positive; median -0.111000, mean dominated by +9.883856).
- 9964: +0.165947 (n=7, 6/7 positive).

Thus 4/4 seed means are positive but the seed 9963 median and pairwise wins disagree. These data do not justify a population-level superiority claim.

## Brier score reveals mission dependence

| Mission | onehot Brier | latent Brier | latent - onehot |
|---|---:|---:|---:|
| 100 km (60 target records) | 0.080144 | 0.072818 | -0.007326 |
| 140 km (60 target records) | 0.155174 | 0.166405 | +0.011231 |

100km Brier favors latent, but 140km favors onehot. An interval probability-mass NLL and a Bernoulli Brier assess different probability events. A future-view selection rule cannot condition on the kind of the future SOC observation, which is unknown at the moment of the next proposal.

## Method implications and bounded next research

- Keep stable-tail EP as an opt-in numerical candidate. All 384 GP kernel attempts succeeded in the panel, but successful convergence is not calibration proof.
- Retain hybrid_onehot as a promising 140km energy optimization candidate, not a confirmed winner, alongside efficient onehot_stable and corpus_stable controls.
- Do not promote the existing per-candidate min-feasibility hybrid_guard as a reliable safety mechanism: its 140km exploratory energy performance worsened, while this audit documents mission-dependent SOC predictive reliability.
- A potential new method is an online, prefix-causal risk-head controller that uses only **previously observed** prequential scores to choose or shrink the risk view, with explicit handling of censored likelihood and model-failure states. It must not use the future censor/exact observation kind, must register its clipping/shrinkage rule before evaluation, and must avoid adapting weights using test labels or the already selected 9961-9964 pilot seeds as independent confirmation.
- Prior to any new aircraft expense: synthetic hierarchical/censored benchmark correctness tests, separate independent experimental seeds, fixed comparator arms and predeclared total evaluator/GP/VAE budgets.

The original 856-call finished pilot, 24-prefix GP panel, older interrupted 2,212-call pilot and their locks remain immutable. PRs remain Draft. No confirmation claim or physical aviation validation.