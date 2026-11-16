# Interval-censored reserve timing on PR113 aircraft paths

- Code commit `56bcd913563f71252ed30f377c7d8db6a7782425`
- Archived result commit `99e868125d2dde224b6203899382a3ddd77152e5`
- Python 3.9.25. Zero new aircraft or RCAIDE/MDA calls.
- Smoke / nominal / headwind hold 12 / 864 / 864 metric rows.
- Same paths, engineering-constraint forecasts and held-out labels in every comparison.

Both formal conditions use 24 seeds and the mean configuration-macro Brier score at calls 8, 12 and 16. Negative differences favour the interval-informed forecast. These are retrospective comparisons.

| Path | Endpoint | Comparator for timing_interval | Nominal difference [95% paired interval] | Headwind difference [95% paired interval] |
| --- | --- | --- | ---: | ---: |
| risk_balanced_active | usable_factored | timing_sign | -0.005896 [-0.008538, -0.003289] | -0.002561 [-0.005114, +0.000060] |
| risk_balanced_active | usable_factored | original_probit | -0.004741 [-0.008875, -0.000621] | -0.000341 [-0.004786, +0.004334] |
| risk_balanced_active | mission | timing_sign | -0.026835 [-0.034981, -0.018294] | -0.021408 [-0.029319, -0.013915] |
| risk_balanced_active | mission | original_probit | -0.039237 [-0.046060, -0.032413] | -0.021789 [-0.030856, -0.011939] |
| risk_coverage | usable_factored | timing_sign | -0.005754 [-0.009654, -0.002366] | -0.002758 [-0.005380, -0.000144] |
| risk_coverage | usable_factored | original_probit | -0.003852 [-0.006926, -0.000961] | -0.001382 [-0.004048, +0.001136] |
| risk_coverage | mission | timing_sign | -0.027357 [-0.035468, -0.019294] | -0.018312 [-0.026775, -0.010012] |
| risk_coverage | mission | original_probit | -0.037684 [-0.045842, -0.029673] | -0.023995 [-0.031807, -0.015904] |

The frozen protocol and replay command are on PR #114.
