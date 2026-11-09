# Phase C2 — risk-head × objective-head update attribution

C1/C1R showed a directionally positive risk-EI online-update effect on known
tasks, accompanied by increased infeasible evaluations and fewer exact
objective labels. C2 isolates risk-head and objective-head update contributions
with a 2×2 factorial design and two reference arms.

**Workstation-ready at fixed code SHA**
`64ecc3b2f1638ba1e3586be71400ed420579582e`.

Validation run `42610032063` succeeded on the exact SHA:
- Ubuntu/Python 3.9 native integration: success.
- Windows/Python 3.9 native integration: success.
- Python 3.13 cross-version verification of Windows evidence: success.

The CI executed 120 mathematical evaluations for the six-arm integration and
completed model-free verification, factorial analysis and tamper checks.
The workstation must run smoke first and full only after smoke verification.

See [C2 protocol](docs/PHASE_C2_HEAD_FACTORIAL.txt),
[PR #83](https://github.com/Luyi1235/RA-LSBO/pull/83), and
[design Issue #82](https://github.com/Luyi1235/RA-LSBO/issues/82).

PR #83 remains a draft and must not be merged before results are reviewed.
