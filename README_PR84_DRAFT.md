# PR84 — PR83 C2 integrity audit and selected-feedback diagnostics

PR84 is a read-only audit of the archived PR83 result at
`ade354de8b35d20a2bb32b469b04bc7da92aa004`.

It independently recomputes per-arm final feasibility, no-solve, online
infeasible calls, exact objective labels, first-feasible positions and initial
warm/cold groups from the original unit journals. It compares those counts
with the published reports and investigates the `arms_all_final_feasible`
flag discrepancy.

It also evaluates archived predictions only at candidates that were actually
selected and evaluated. These metrics describe policy-selected feedback and
must not be interpreted as unbiased calibration over the candidate pool.

**No new truth evaluations, GP/VAE fits, aircraft or RCAIDE calls.**
See [Issue #84](https://github.com/Luyi1235/RA-LSBO/issues/84) and
[protocol](docs/PR84_C2_READONLY_AUDIT.txt).
