# Structured RA-LSBO cold-start and optimization evidence

The three batches use independent seeds and the same evaluation budget. Cold-start seeds have no usable design among the eight shared initial evaluations. The end-to-end confirmation designates both optimization endpoints as primary in headwind cold-start seeds versus original LSBO. The external comparison designates the same endpoints and population versus screened ArchSBO. External arms share the engineering screen and evaluation budget, and use their own candidate search.

[Internal target-attainment curves](aircraft-structured-risk-attainment.pdf) · [External target-attainment curves](aircraft-structured-risk-external-attainment.pdf)

## Optimization outcomes

Differences are structured-risk greedy minus the named comparator. Times include the initial evaluations and assign 33 evaluations to a seed that does not reach its prespecified target by 32. Terminal energy means are paired among seeds where both methods found a usable design. Feasibility-first win/loss/tie counts include every seed and use each batch's registered comparison rule. Intervals are the archived 95% paired bootstrap intervals for means. Negative differences favor structured-risk greedy.

| Batch | Scenario (cold pairs) | Comparator | Target hits (structured / comparator) | Δ target evaluations [95% CI] | Δ best feasible energy at 32, kWh [95% CI] | Energy pairs | Terminal W/L/T |
|---|---|---|---|---:|---:|---:|---:|
| Independent confirmation | Headwind (13) | Original LSBO | 12/13 vs. 5/13 | −7.85 [−13.08, −1.85] | −2.79 [−4.34, −1.41] | 12 | 10/3/0 |
| Independent confirmation | Headwind (13) | Interval-risk greedy | 12/13 vs. 9/13 | −5.23 [−10.00, −0.54] | −1.46 [−2.52, −0.55] | 13 | 9/2/2 |
| Independent confirmation | Payload (9) | Original LSBO | 7/9 vs. 5/9 | −6.11 [−11.00, −0.89] | −1.89 [−4.73, +0.33] | 8 | 7/2/0 |
| Independent confirmation | Payload (9) | Interval-risk greedy | 7/9 vs. 8/9 | +0.67 [−3.89, +6.11] | +0.13 [−1.17, +1.37] | 9 | 5/3/1 |
| End-to-end confirmation | Headwind (9) | Original LSBO | 9/9 vs. 6/9 | −3.89 [−9.78, +2.22] | −0.65 [−1.42, +0.02] | 8 | 6/2/1 |
| End-to-end confirmation | Headwind (9) | Interval-risk greedy | 9/9 vs. 6/9 | −1.78 [−9.56, +5.78] | −1.06 [−2.31, −0.01] | 9 | 6/2/1 |
| End-to-end confirmation | Payload (11) | Original LSBO | 10/11 vs. 3/11 | −9.00 [−13.64, −2.64] | −3.84 [−6.07, −1.61] | 11 | 9/2/0 |
| End-to-end confirmation | Payload (11) | Interval-risk greedy | 10/11 vs. 11/11 | +0.09 [−4.36, +5.73] | +0.21 [−0.26, +0.94] | 11 | 7/4/0 |
| External comparison | Headwind (10) | Screened ArchSBO | 10/10 vs. 6/10 | −6.90 [−12.80, −0.50] | −1.30 [−2.00, −0.61] | 10 | 9/1/0 |
| External comparison | Headwind (10) | Adapted LogCEI | 10/10 vs. 5/10 | −9.90 [−13.20, −6.60] | −1.61 [−3.14, −0.47] | 9 | 8/2/0 |
| External comparison | Payload (7) | Screened ArchSBO | 6/7 vs. 4/7 | −6.14 [−11.14, −0.71] | −0.79 [−2.45, +1.55] | 7 | 6/1/0 |
| External comparison | Payload (7) | Adapted LogCEI | 6/7 vs. 4/7 | −6.43 [−12.14, −0.14] | −1.36 [−2.56, −0.37] | 6 | 7/0/0 |

The end-to-end headwind primary contrasts against original LSBO favor the structured arm in point estimates. Both mean intervals include zero. The prespecified terminal order is six wins, two losses, and one tie (exact sign test p=0.289). Payload has favorable target-time and terminal-energy mean intervals against original LSBO. The incremental optimization effect against interval-risk greedy varies by scenario and endpoint.

In the external headwind primary comparison against screened ArchSBO, the mean target-time difference is −6.90 evaluations, with six wins, three losses, and one tie (exact sign test p=0.508). Terminal feasibility-first order gives nine wins and one loss (p=0.0215). In payload, terminal mean energy is lower in point estimate against ArchSBO and the interval includes zero. The adapted LogCEI comparison favors the structured method in both scenarios on both mean endpoints.

## Cold-start evidence

| Scenario | Δ Brier at evaluation 12, independent confirmation | Independent Δ evaluations to third usable | End-to-end Δ evaluations to third usable |
|---|---:|---:|---:|
| Headwind | −0.0364 [−0.0458, −0.0262] | −4.54 [−7.15, −2.23] | −3.89 [−6.22, −1.67] |
| Payload | −0.0327 [−0.0546, −0.0132] | −2.56 [−4.33, −1.00] | −2.27 [−4.09, −0.45] |

These endpoints compare structured-risk with interval-risk greedy. The Brier score uses the independent held-out panel from the independent-confirmation batch.

## Archive correction

The [End-to-end confirmation README](https://github.com/Luyi1235/RA-LSBO/blob/236c3262230215b5644495ade9d576a478457982/results/aircraft-structured-risk-optimization-20261121-c0151e4/README.md) labels the evaluation count to the third usable design as time to the energy target in its cold-start and all-seed summaries.

| Scenario | Cohort | Comparator | README values (third usable) | Correct energy-target Δ evaluations |
|---|---|---|---:|---:|
| Headwind | Cold start | Original LSBO | −5.22 [−10.11, −1.33] | −3.89 [−9.78, +2.22] |
| Headwind | Cold start | Interval-risk greedy | −3.89 [−6.22, −1.67] | −1.78 [−9.56, +5.78] |
| Headwind | All seeds | Original LSBO | −1.79 [−4.08, −0.08] | −2.12 [−5.12, +0.88] |
| Headwind | All seeds | Interval-risk greedy | −1.92 [−3.21, −0.79] | +0.71 [−2.46, +3.75] |
| Payload | Cold start | Original LSBO | −8.00 [−11.55, −4.36] | −9.00 [−13.64, −2.64] |
| Payload | Cold start | Interval-risk greedy | −2.27 [−4.09, −0.45] | +0.09 [−4.36, +5.73] |
| Payload | All seeds | Original LSBO | −3.58 [−5.92, −1.38] | −4.88 [−8.04, −1.54] |
| Payload | All seeds | Interval-risk greedy | −1.25 [−2.25, −0.38] | −1.00 [−3.71, +2.00] |

The [External comparison README](https://github.com/Luyi1235/RA-LSBO/blob/30d63fbc42498166d537e651053181b8d2c859c3/results/aircraft-structured-risk-external-20261121-62750a5/README.md) labels the evaluation count to the third usable design as time to the energy target in its cold-start and all-seed summaries.

| Scenario | Cohort | Comparator | README values (third usable) | Correct energy-target Δ evaluations |
|---|---|---|---:|---:|
| Headwind | Cold start | Screened ArchSBO | −6.50 [−9.80, −3.60] | −6.90 [−12.80, −0.50] |
| Headwind | Cold start | Adapted LogCEI | −11.20 [−15.30, −6.90] | −9.90 [−13.20, −6.60] |
| Headwind | All seeds | Screened ArchSBO | −6.54 [−8.92, −4.29] | −8.08 [−11.54, −4.50] |
| Headwind | All seeds | Adapted LogCEI | −4.75 [−7.71, −2.12] | −5.46 [−8.21, −2.54] |
| Payload | Cold start | Screened ArchSBO | −2.57 [−4.14, −1.00] | −6.14 [−11.14, −0.71] |
| Payload | Cold start | Adapted LogCEI | −10.71 [−16.86, −3.00] | −6.43 [−12.14, −0.14] |
| Payload | All seeds | Screened ArchSBO | −1.71 [−2.58, −0.88] | −2.75 [−5.79, +0.12] |
| Payload | All seeds | Adapted LogCEI | −3.79 [−6.71, −1.08] | −3.83 [−6.92, −0.67] |


## Fixed sources

- [Independent confirmation analysis JSON](https://github.com/Luyi1235/RA-LSBO/blob/ff076555615150a3885611dbf0d288906fca1767/results/aircraft-structured-risk-confirmation-20261121-956b595/ra-lsbo-structured-confirmation-956b595-analysis.json)
- [End-to-end confirmation analysis JSON](https://github.com/Luyi1235/RA-LSBO/blob/236c3262230215b5644495ade9d576a478457982/results/aircraft-structured-risk-optimization-20261121-c0151e4/ra-lsbo-structured-optimization-c0151e4-analysis.json)
- [External comparison analysis JSON](https://github.com/Luyi1235/RA-LSBO/blob/30d63fbc42498166d537e651053181b8d2c859c3/results/aircraft-structured-risk-external-20261121-62750a5/ra-lsbo-structured-external-62750a5-analysis.json)
