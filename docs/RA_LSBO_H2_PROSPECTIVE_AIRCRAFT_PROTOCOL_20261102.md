# RA-LSBO-H2-M independent six-arm prospective aircraft study — registration only

**Status:** prospective source candidate and preregistration protocol, not
a running aircraft experiment. No new simplified truth calls or RCAIDE calls
are authorized by publishing this file. The original interrupted 2,212-call
pilot, completed 856-call H1 pilot, 384-GP SOC panel and all source hashes,
execution locks and evidence Releases remain unchanged.

## Distinct research question

Can a bounded controller which uses ONLY previously settled paired SOC
likelihood differences improve on a fixed hierarchical-onehot risk head
and the H1 per-candidate minimum-feasibility guard, without sacrificing
feasible discovery or running excessive GP/VAE computation?

The implemented H2-M arm uses:
- the same frozen 2,000-corpus persistent latent objective features;
- hierarchical onehot completion and non-SOC constraint features;
- TWO independent SOC mixed exact+interval EP heads, one onehot and
  one latent, both selected from the 4x2 original kernel grid;
- a Gaussian *mixture* SOC feasibility CDF, with the onehot mixture
  weight bounded in [0.1,0.9], initially 0.5;
- a minimum of 4 past scored selected outcomes, then a past-only
  sigmoid of cumulative clipped paired log-likelihood ratios (eta
  0.35, per-event |delta| <= 4). Future SOC censoring/exact kind is
  never used for next-point selection;
- explicit no-update for scheduled coverage and a full SOC GP failure;
  real physical/infrastructure error still stops the campaign.

The H2-B hard-head Brier controller is **synthetic-only** and is NOT an
arm in this prospective registered aircraft experiment. The H2-M
Gaussian mixture's feasibility probability uses exact component CDFs,
while the original acquisition's optimistic support margins use only
moment-matched Gaussian geometry; this is a known approximation,
not calibrated physical safety.

## Frozen study roster

| Stage | Cases | Seeds | Arms | Initial labels shared per (case,seed) | Online per arm | Max physical calls | RCAIDE |
|---|---:|---|---:|---:|---:|---:|---:|
| Smoke | 2 | 11061 | 6 | 6 | 2 | **48** | 0 |
| Exploratory pilot | 2 | 11062–11065 | 6 | 12 | 18 | **1008** | 0 |

Budget breakdown, fixed before experiments:
- Smoke: 2×1×6 initial=12; 2×1×6×2 online=24; up to
  2×1×6 final 5s checks=12; total <=48.
- Pilot: 2×4×12 initial=96; 2×4×6×18 online=864; up to
  2×4×6 final 5s checks=48; total <=1008.
- Combined new Windows maximum 1056 simplified calls, independent
  of the historical 2,212+856 and earlier Linux H1 test charges.
  If there is no feasible incumbent, do not fabricate a 5s check
  or consume an otherwise unused slot.

Cases are 100km aircraft_ordinary and 140km aircraft_sparse.
The six arms are random, onehot_stable, corpus_stable, hybrid_onehot,
hybrid_guard and h2_causal. All arms use the same initial labels per
case/seed and predetermined candidate RNG stream. Arm order cyclically
rotates based on seed and case, independent of outcomes. Comparison
groups, case strata and exact online horizon are immutable.

Primary development contrasts (left minus right energy Wh, negative
means left lower on both-nominal-feasible seeds):
1. H2-M vs hybrid_onehot (adaptive mixture compared to onehot risk).
2. H2-M vs hybrid_guard (bounded past-score mixture vs worst-view).
3. H2-M vs onehot_stable (objective-latent/value-for-compute).
4. hybrid_onehot vs corpus_stable (fixed risk-view control).

Report feasibility discovery and missing outcomes separately; do not
impute an objective penalty to missing/nonfeasible runs. Each arm has
18 online evaluations and all stopped runs remain failures or
censored outcomes. This is a FOUR-SEED DEVELOPMENT pilot, not a
statistically powered independent confirmation.

Secondary: paired first feasible cost, warm/cold stratification,
same-incumbent 5-second numerical step sensitivity, correct H2 SOC
prequential **two-component mixture density** NLL for exact SOC,
correct two-component interval-mass NLL for censored SOC, exact
mixture central-95% hit, Brier for resolved SOC feasibility,
controller-event integrity and skipped updates, number of successful
EP kernel candidates, VAE encoder checkpoints and complete proposal
wall time (do not double-count encoder time).

## Software and audit gates

Runner module:

    ra_lsbo.experiments.simplified_aircraft.h2_pilot

Commands: prepare, run, audit, report.

The Windows execution must use a freshly created clean detached
worktree at the final H2 fixed commit and the unchanged original
Rcaide_vsp dependencies/contract. Registration preserves full source
snapshots and the reviewed historical 316-call qualification receipt.
The new pilot requires an independently finished and audited
**same-source** H2 smoke; successful H1 smoke is not a substitute.

The runner checks:
- exact commit and clean tracked source worktree;
- original reviewed aircraft equations unchanged;
- 48/1008 complete-call hard caps; all RCAIDE charges zero;
- all started/settled physical calls, unmodified source snapshot hashes;
- common initial design and all online proposal/evaluation identities;
- H2 registered controller snapshot and prior forecast commitment;
- H2 controller replay in chronological order using only saved
  selected forecasts and actual **subsequently** settled SOC evidence;
- proper Gaussian-mixture SOC scores, not moment-matched Gaussian NLL;
- persistent VAE numerical checkpoints and 5s sensitivity evidence.

No hidden policy recovery, silent skip, seed reroll, automatic repair,
resume or deletion of any lock. Real evaluator/unknown outcomes stop
after settlement. A surrogate GP-only failure may take the previously
registered deterministic coverage fallback, which is recorded as
a deviation from GP-guided search, not invented predictions.

## Present research status and limits

Before any 48-call smoke, require:
1. Code-only and synthetic tests in Linux CI, including historical
   RA-LSBO regression, exact-mixture scoring and no future leakage.
2. Read-only Linux preview of the NEW 48-call and 1008-call
   registration plans with no model truth.
3. A newly fixed H2 commit and a new isolated Windows worktree.
4. Explicit user approval for the **48-call** independent H2 smoke.
5. After Windows smoke audit, fresh approval for the **1008-call**
   independent exploratory pilot.

The 2026-11-02 analytic synthetic results do not establish H2
superiority: on some tasks an oracle-like fixed view wins. In a
separately disclosed crossing-view synthetic stress, the old min-P
guard can be excessively conservative and H2 can attain lower
proxy energy but still need not beat the best fixed expert.
These are controlled mathematical constructions, not independent
high-fidelity aircraft results.

Never merge default RA-LSBO behavior or claim confirmed novelty,
calibration, or airworthiness based on this stage.
