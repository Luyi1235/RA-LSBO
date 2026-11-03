# H2 P6: two-stage SOC-weight attribution

Native DD/FF controls passed twice for 16/16 histories.
At the preregistered 1e-10 gap threshold, 2/16 matched histories are identified; all four cells, repeated-run identity and coverage must pass.

| Conditional contrast | Changed final candidate |
|---|---:|
| generation_at_dynamic | 2/2 |
| generation_at_fixed | 2/2 |
| ranking_on_dynamic_pool | 0/2 |
| ranking_on_fixed_pool | 0/2 |

These are categorical one-step contrasts on matched historical states, not additive energy effects. Unidentified histories are retained in the JSON; threshold sensitivity is descriptive and is not used to select a favorable cutoff.
Offline GP kernel-candidate fits: 1152; total measured fit seconds: 18.818. New aircraft calls, RCAIDE calls, and VAE fits: zero.
Historical SLSQP did not store individual query trajectories; archived starts, endpoints, status, final pools and decisions were checked, while the unrecorded queries cannot be verified.
The P3 140 km mean dynamic-minus-fixed energy was -262.91 Wh, versus +9.15 Wh excluding seed 12075 in a sensitivity analysis; no seed is excluded here. The data do not establish an energy, safety or calibration advantage.
Decision gate: no algorithm promotion or new algorithm design is justified by these exploratory matched-step comparisons alone.

| Case | Seed | Category | Reason | DD/FF/DF/FD score gaps |
|---|---:|---|---|---|
| aircraft_ordinary | 12071 | not_identified | numerical_near_tie_or_single_candidate | {'DD': 2.0339285811132868e-13, 'FF': 6.412495612773128e-05, 'DF': 5.294631399976879e-11, 'FD': 5.3623509361999666e-05} |
| aircraft_ordinary | 12072 | not_identified | numerical_near_tie_or_single_candidate | {'DD': 1.5188406088384454e-11, 'FF': 1.123701132144106e-11, 'DF': 4.525924079956667e-11, 'FD': 1.1667111721180845e-10} |
| aircraft_ordinary | 12073 | not_identified | numerical_near_tie_or_single_candidate | {'DD': 1.7918999617450027e-13, 'FF': 2.692845946228317e-12, 'DF': 3.019822499505942e-05, 'FD': 6.062923899818884e-06} |
| aircraft_ordinary | 12074 | not_identified | numerical_near_tie_or_single_candidate | {'DD': 3.6004532688593827e-13, 'FF': 6.254996520738132e-13, 'DF': 1.705413588126703e-12, 'FD': 2.6829649613091533e-12} |
| aircraft_ordinary | 12075 | not_identified | numerical_near_tie_or_single_candidate | {'DD': 2.1580182085756405e-11, 'FF': 4.490618987773587e-11, 'DF': 2.656963538072432e-11, 'FD': 3.942945969725997e-11} |
| aircraft_ordinary | 12076 | not_identified | numerical_near_tie_or_single_candidate | {'DD': 1.1532996779806126e-12, 'FF': 1.0553780072086738e-12, 'DF': 9.688472246693891e-12, 'FD': 7.43016759230386e-12} |
| aircraft_ordinary | 12077 | not_identified | numerical_near_tie_or_single_candidate | {'DD': 8.136439114236893e-05, 'FF': 4.940825526489334e-12, 'DF': 7.899784628506712e-05, 'FD': 8.811174012635092e-12} |
| aircraft_ordinary | 12078 | identified | None | {'DD': 1.929702422731605e-05, 'FF': 1.9574213574791166e-05, 'DF': 1.9470394039822203e-05, 'FD': 1.9259693725848237e-05} |
| aircraft_sparse | 12071 | not_identified | numerical_near_tie_or_single_candidate | {'DD': 1.895317236488836e-11, 'FF': 3.391353864401481e-11, 'DF': 1.4265433279092576e-10, 'FD': 1.0847034381811227e-10} |
| aircraft_sparse | 12072 | not_identified | numerical_near_tie_or_single_candidate | {'DD': 3.6841640849161195e-12, 'FF': 3.674394122299418e-12, 'DF': 3.6841640849161195e-12, 'FD': 3.674394122299418e-12} |
| aircraft_sparse | 12073 | not_identified | numerical_near_tie_or_single_candidate | {'DD': 6.295408638834488e-12, 'FF': 6.439293542825908e-12, 'DF': 6.298961352513288e-12, 'FD': 6.437517185986508e-12} |
| aircraft_sparse | 12074 | not_identified | numerical_near_tie_or_single_candidate | {'DD': 1.2884027178472479e-11, 'FF': 3.696820627396846e-12, 'DF': 0.0002250351429348152, 'FD': 0.00017329649673869518} |
| aircraft_sparse | 12075 | identified | None | {'DD': 3.408904536428281e-08, 'FF': 3.389235114781286e-08, 'DF': 3.408904536428281e-08, 'FD': 3.389235114781286e-08} |
| aircraft_sparse | 12076 | not_identified | numerical_near_tie_or_single_candidate | {'DD': 1.180500142083929e-12, 'FF': 4.201305969786517e-12, 'DF': 2.107158891817562e-11, 'FD': 2.7424285065080767e-11} |
| aircraft_sparse | 12077 | not_identified | numerical_near_tie_or_single_candidate | {'DD': 2.2306045899256333e-11, 'FF': 1.1427858659374124e-11, 'DF': 3.987943308914055e-11, 'FD': 9.129808020702512e-12} |
| aircraft_sparse | 12078 | not_identified | numerical_near_tie_or_single_candidate | {'DD': 2.4947821586351893e-12, 'FF': 0.00023461175232031284, 'DF': 4.4270920263045355e-10, 'FD': 2.426682188527707e-10} |
