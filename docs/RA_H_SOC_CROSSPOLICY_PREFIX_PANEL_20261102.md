# RA-LSBO-H: frozen common-prefix, cross-policy SOC panel

**Registration date:** 2026-11-02. **Status:** prospective GP-only audit of
previously settled data; source pilot and all simulation evaluations are
immutable. This protocol was specified after inspecting the 4-seed
exploratory RA-LSBO-H pilot. It is a *development* model-assessment, not
independent confirmation, causal ablation, or aircraft risk certification.

## Scientific question

The 4-seed, 856-call RA-LSBO-H pilot suggested different 140km performance
of explicit onehot SOC risk and frozen corpus latent SOC risk. Its
per-arm prequential scores were measured on DIFFERENT, adaptively selected
designs, so direct averages cannot establish which representation models
SOC more accurately on the same design.

This experiment fixes both the historical *training data* and the
*future evaluated candidate* for both SOC models. Future target labels
must never enter either GP, the frozen VAE checkpoint or kernel selection.

## Frozen source

- Original 856-call Windows pilot Release:
  https://github.com/Luyi1235/RA-LSBO/releases/tag/ra-lsbo-h-pilot-9961-9964-20261102
- Original ZIP SHA256:
  \`3d9cb4f684bbc9198c523bb44bf466463c2f85053d87ca1770eca9111b48fed1\`.
- Original execution code:
  \`b219163fc565dfc32876f38f0bbf5f4e65abbd36\`.
- Original plan:
  \`ce59583c1fc482406e709225e7ee2496e85b36c3cbc82df9532366f45f239950\`.
- Original registration:
  \`f1d9e8d734ca81d2cb0e8c6ea9ec53fbbb7c118661a076046ea5b44b2dccb271\`.
- This GP-only analysis uses a NEW, explicitly pinned research commit descending
  from the original, a separate output directory, and no old evaluator call ledger.

## Fixed design

**Anchor history:** the previously evaluated \`hybrid_onehot\` arm.
Same 12 initial records per case/seed as all five original arms.
For each mission and original seed 9961,9962,9963,9964, use exactly three
chronological online indices:

| Held-out online index t | GP training records | Latest pretrained encoder checkpoint |
|---:|---:|---:|
| 1 | 12+1=13 | t=0 (trained on 12) |
| 9 | 12+9=21 | t=8 (trained on 20) |
| 17 | 12+17=29 | t=16 (trained on 28) |

Two cases x four seeds x three fixed prefixes = **24 training sets**.
Each checkpoint is from its *own anchored historical prefix*; no future
VAE/GP training and no D29-to-D28 label transfer.

For each fixed anchor prefix, predict the candidates **previously selected
at online index t** by five policies: random, onehot_stable,
corpus_stable, hybrid_onehot, hybrid_guard. These designs were evaluated
after that prefix on their respective historical trajectories. Before
scoring, exclude any target design identity already present in the anchor
training prefix. Exclusion uses ONLY the candidate identity, never the
future label, feasibility or performance. Preserve excluded rows and count.
Up to **120** policy-target records are scored; duplicate target designs
across policies are transparently retained as correlated observations,
not 120 independent test replicates.

### Two SOC posterior variants on each identical prefix

- **Hierarchical onehot:** original \`PrefixFeatures\` with \`latent=False\`,
  no VAE fit. Uses all anchor SOC exact and interval labels up through t−1.
- **Frozen corpus latent:** original \`CorpusRepresentation.load_checkpoint\`
  from the pre-t, source-sealed t=0/8/16 checkpoint; \`feature_snapshot\`
  extends the latent normalization only with anchor prefix designs known
  before t. No encoder training or future label is used.

Both use the same anchor historical SOC evidence, the exact original 4x2
grouped Matérn-5/2 kernel grid:
continuous (.05,.15,.4,1.0), discrete (.5,1.5).
GP settings: stable extreme-tail moments opt-in, 60 EP sweeps, damping .7,
tolerance 1e−8, exact noise SD 1e−3, interval noise zero, jitter 1e−8.
One model selected on the **training prefix approximate log evidence**
only, never on held-out performance.

Each of 24 prefixes fits two heads × eight candidates = at most
**384 GP kernel candidate attempts** in one bounded GP-only run. It
loads 24 frozen encoder checkpoints, performs zero new VAE training,
zero optimizer/acquisition calls, zero simplified aircraft evaluations,
zero RCAIDE/MDA and zero mutation of old evidence.

**Safety gate:** for every anchor selected candidate where the original
pilot saved a prior SOC predictive mean/variance, the newly reconstructed
onehot GP must reproduce that historical forecast within the fixed
numerical tolerance. Check original frozen latent and onehot feature
identities. A mismatch stops the GP-only analysis, preserving its lock,
rather than changing kernel parameters or retroactively tuning.

All-kernel failure for any head is recorded with eight error records;
do not invent predictions. Unexpected schema/label/identity errors stop
the run without retry. Use one output directory, exclusive \`run.lock\`,
and append-only \`prefix_attempts.jsonl\`; no resume or repeated fitting.

## Primary scoring and interpretation

For each eligible target, score **the same exact historical SOC evidence**
against both view predictions. Two strata:

1. Exact final SOC: continuous Gaussian density NLL, standardized residual,
   and central-95% interval hit.
2. Censored SOC interval: negative log Gaussian interval *probability mass*,
   preserving one-sided/open bounds.

Never pool density-NLL and interval-mass-NLL into one headline metric.
Secondary: Brier score for resolved g<=0 labels (mark unknown separately),
finite prediction count, kernel fit success and total model-only time.
Report paired latent-minus-onehot NLL (positive favors onehot) by
task, target policy and *independent seed cluster*, with all exclusions
and nonfinite scores disclosed. An anchor historical trajectory is
not representative of every possible BO policy; four seeds and a
cross-policy convenience panel still do not establish statistical
confirmation or physical safety.

## Execution stages

1. Linux CI: download/hash original 46MB ZIP, verify all 24 historical
   prefixes and 120 target commitments; verify plan JSON roundtrip,
   and explicitly refuse \`run\` from a cross-platform preview BEFORE
   creating a lock or fitting any GP.
2. Original Windows/Rcaide_vsp: fresh detached worktree at the **reviewed
   analysis commit**, same exact original model dependencies and task
   contract; create a fresh directory and run \`plan\` ONCE.
3. Independently verify plan, registered hashes, 24×2×8 limits.
4. Run GP-only \`run\` ONCE, no fallback tuning, retain lock and all errors.
5. After successful completion, \`audit\` for 24 complete prefixes and
   the full 384 attempt count. Seal output+logs with SHA256 and deliver.
6. Do not launch a new 856-call pilot or high-fidelity aircraft evaluation.
   Any new algorithm change needs a separate, explicitly proposed protocol.

Commands in Windows from fixed detached worktree:

\`\`\`powershell
$Py = 'C:\Users\pc\.conda\envs\Rcaide_vsp\python.exe'
$Zip = 'D:\LSBO\RA-LSBO-H-pilot-9961-9964-20261102.zip'
$Out = 'D:\LSBO\RA-LSBO-H-SOC-crosspolicy-20261102_01'
& $Py tools/ra_h_soc_panel_run.py plan --archive $Zip --output $Out
& $Py -u tools/ra_h_soc_panel_run.py run --archive $Zip --output $Out
& $Py tools/ra_h_soc_panel_run.py audit --archive $Zip --output $Out
\`\`\`

The real Windows run must **never** pass \`--cross-platform-check\`.
The three commands have distinct stopping/approval gates; do not run
the next command if its prerequisite fails. Code SHA supplied after CI.
