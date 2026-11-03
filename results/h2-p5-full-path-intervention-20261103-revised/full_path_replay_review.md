# H2 P5: frozen-prefix full-path weight intervention

Native two-arm replay passed 16/16 pairs, twice per arm. Full-path identified pairs: 2/16; confirmed changes in final choice: 2/2.
A further 14 pairs have a top-two acquisition gap <= 1e-10 and are conservatively not identified; no failed or near-tied pair is counted as zero effect.
The threshold and absolute/relative comparison tolerances were fixed before fitting. Each prefix was fitted once; 1,152 GP kernel candidates were attempted, with no VAE fit or physical/RCAIDE call.

The original checkpoints and all 186 P3 source identities matched. The replay uses the original DiscoveryBO.propose and RiskRefiningBO._augment, including SLSQP, near filtering and coverage. Archived records contain solver starts, endpoints and status, not every internal SLSQP query; internal per-query path identity cannot be independently verified.

The fixed-pool P4 reversal (8/16) is a distinct estimand. P5 does not predict downstream outcomes of the unchosen candidate, establish safety or demonstrate performance superiority.
P3 sparse mean energy difference was -262.908 Wh (dynamic minus fixed), but +9.150 Wh without seed 12075; this post-divergence sensitivity is descriptive and no seed is removed from P5.

| Case | Seed | Iteration | Native replay | Category | Weight change | Margins |
|---|---:|---:|---|---|---|---|
| aircraft_ordinary | 12071 | 4 | True | not_identified | True | [2.0339285811132868e-13, 6.412495612773128e-05] |
| aircraft_ordinary | 12072 | 4 | True | not_identified | True | [1.5188406088384454e-11, 1.123701132144106e-11] |
| aircraft_ordinary | 12073 | 4 | True | not_identified | True | [1.7918999617450027e-13, 2.692845946228317e-12] |
| aircraft_ordinary | 12074 | 4 | True | not_identified | True | [3.6004532688593827e-13, 6.254996520738132e-13] |
| aircraft_ordinary | 12075 | 4 | True | not_identified | True | [2.1580182085756405e-11, 4.490618987773587e-11] |
| aircraft_ordinary | 12076 | 4 | True | not_identified | True | [1.1532996779806126e-12, 1.0553780072086738e-12] |
| aircraft_ordinary | 12077 | 4 | True | not_identified | True | [8.136439114236893e-05, 4.940825526489334e-12] |
| aircraft_ordinary | 12078 | 4 | True | identified | True | [1.929702422731605e-05, 1.9574213574791166e-05] |
| aircraft_sparse | 12071 | 4 | True | not_identified | True | [1.895317236488836e-11, 3.391353864401481e-11] |
| aircraft_sparse | 12072 | 5 | True | not_identified | True | [3.6841640849161195e-12, 3.674394122299418e-12] |
| aircraft_sparse | 12073 | 5 | True | not_identified | True | [6.295408638834488e-12, 6.439293542825908e-12] |
| aircraft_sparse | 12074 | 4 | True | not_identified | True | [1.2884027178472479e-11, 3.696820627396846e-12] |
| aircraft_sparse | 12075 | 5 | True | identified | True | [3.408904536428281e-08, 3.389235114781286e-08] |
| aircraft_sparse | 12076 | 4 | True | not_identified | True | [1.180500142083929e-12, 4.201305969786517e-12] |
| aircraft_sparse | 12077 | 9 | True | not_identified | True | [2.2306045899256333e-11, 1.1427858659374124e-11] |
| aircraft_sparse | 12078 | 4 | True | not_identified | True | [2.4947821586351893e-12, 0.00023461175232031284] |
