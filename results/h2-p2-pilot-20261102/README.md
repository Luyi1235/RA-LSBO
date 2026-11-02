# H2 P2 Windows developmental pilot

- Fixed execution code: `7ca420613ccb50e138dc3c9d4292ac8e863c4ecf`, original P0 Windows worktree, unchanged Rcaide_vsp environment.
- Seeds 11062-11065, two aircraft tasks, six methods; all 48 trials and 864 proposals passed the registered audit.
- Total 1,007 settled simplified-aircraft calls: 96 shared initial, 864 online and 47 five-second checks. RCAIDE calls: zero. One trial had no nominal feasible incumbent, so its five-second check was not performed; all 47 performed checks passed their numerical screen.
- Read-only analysis reports 142 H2 scored observations (128 exact density, 14 interval mass) and 118 effective changes in the onehot mixture weight. Effective changes count adjacent audited before-proposal weights and the final replayed weight differing by more than 1e-14; this is not the number of scores or proof of better performance.
- No registered model-failure coverage heads were recorded. The complete per-trial results preserve all seeds, feasibility, objectives, SOC scores, VAE training costs and verification status; see `analysis.json` and the original `results.json`.
- This is a four-seed developmental comparison, not confirmatory evidence, a claim of statistical superiority, or a physical safety validation. No new physical evaluation or GP fitting was used for the read-only analysis.
- Original source snapshot, plan, ledger, proposals, observations, numerical checkpoints, lock, environment-related preflight output, stdout, stderr, exit codes, audit and report are included in the evidence archive. The original P1 smoke and 316-call qualification remain unchanged.
- Release asset `RA-LSBO-H2-P2-pilot-evidence-20261102_02.zip` SHA-256: `5af27b7fcec3a1d6aaf8a9d139f13c2d61f976d4067ae30f062eceb7c0d5f2e0`. All 790 ZIP entries were checked against local files.
