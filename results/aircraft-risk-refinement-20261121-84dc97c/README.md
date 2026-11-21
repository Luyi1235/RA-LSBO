# PR153 risk-aware LSBO refinement pilot

Fixed code SHA: `84dc97c3c1794d3e5488f6531b75a046debec996`.
Two independent scenarios, one registered seed each (`29220100` payload, `29230100` headwind);
five arms; 208 reduced-aircraft evaluations per scenario; 0 RCAIDE/MDA.

## Acceptance

| scenario | status | paid_calls | RCAIDE/MDA |
|---|---|---:|---:|
| headwind | `VERIFIED_RISK_AWARE_LSBO_REFINEMENT full-pilot-refinement-headwind 208` | 208 | 0 |
| payload | `VERIFIED_RISK_AWARE_LSBO_REFINEMENT full-pilot-refinement-payload 208` | 208 | 0 |

## Terminal usable energy (Wh)

| arm | headwind best@24 | headwind best@40 | payload best@24 | payload best@40 |
|---|---:|---:|---:|---:|
| `lsbo_native` | 41,110.36 | 38,192.93 | 39,782.05 | 39,726.22 |
| `ra_interval_greedy` | 41,110.36 | 38,192.93 | 39,782.05 | 39,726.22 |
| `ra_refine_sign` | 43,969.99 | 38,193.34 | 36,396.96 | 36,321.63 |
| `ra_refine_interval` | 38,162.77 | 38,162.77 | 36,684.59 | 36,321.68 |

`ra_interval_greedy` is byte-identical to `lsbo_native` in both scenarios.
First usable call is 9 for all four native arms in both scenarios (shared initial prefix).

## Primary: `ra_refine_interval − ra_interval_greedy`

| scenario | best@24 ΔWh | best@40 ΔWh |
|---|---:|---:|
| headwind | -2,947.59 | -30.15 |
| payload | -3,097.45 | -3,404.53 |

Descriptive one-seed pilot only; no paired inference. `archsbo_screened` has no comparable terminal usable energy field in the sealed output; its 40-step usable counts are 21 (headwind) and 26 (payload).
