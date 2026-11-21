# PR151 aircraft refinement handoff pilot

Fixed code SHA: `7e693e2190b44a5b1737028d46cecfe78f43aac2`.
Two seeds per scenario (`29100100/29100200`, `29110100/29110200`), five arms:
`interval_open`, `interval_fixed`, `sign_open`, `sign_fixed`, `archsbo_screened`.

| scenario | status | seeds | paid_calls | RCAIDE/MDA |
|---|---|---:|---:|---:|
| payload | `VERIFIED_AIRCRAFT_REFINEMENT_HANDOFF full-pilot-payload 2 416` | 2 | 416 | 0 |
| headwind | `VERIFIED_AIRCRAFT_REFINEMENT_HANDOFF full-pilot-headwind 2 416` | 2 | 416 | 0 |

Two-seed pilot only; no multi-seed inference. All five arms hit 2/2.
