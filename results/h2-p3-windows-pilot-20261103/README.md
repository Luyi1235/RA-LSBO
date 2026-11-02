# H2 P3 Windows developmental pilot

Fixed source: `57e390fc8f6280db9f201940aec14089e81c104c`. Eight seeds 12071-12078, two cases and two arms; 32 trials completed. All 800 simplified-aircraft calls settled and no RCAIDE calls occurred. Native audit passed for 576 proposals and the H2 controller replay. All 32 final nominal designs passed the registered five-second same-design numerical check.

Dynamic one-hot mixture weight changed in 236 settlement transitions (absolute difference greater than 1e-12); 220 changes were present at a subsequent proposal. Fixed 0.5 weights did not change. See `weight-analysis.json` for individual trials and definitions.

The original report contains all seed-paired outcomes. Both arms achieved nominal feasibility in all trials; sparse-case cold starts occurred for three seeds. These eight seeds are a developmental comparison, not independent confirmation or evidence of physical safety. Algorithm training was not replayed by the audit.

Full source snapshots, raw ledger, proposal and observation records, execution lock, model checkpoints, original stdout/stderr, and qualification checks are in the associated independent GitHub Release. ZIP SHA-256: `bd170195201b45942e4fe2c2e04e1e994263bb0a4566a467b28816cdce0473b4`.
