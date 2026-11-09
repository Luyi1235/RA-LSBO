# Phase C2 — risk-head × objective-head update attribution

C1/C1R showed a directionally positive risk-EI online-update effect on known
tasks, accompanied by increased infeasible evaluations and fewer exact
objective labels. C2 isolates risk-head and objective-head update contributions
with a 2×2 factorial design and two reference arms.

**Workstation handoff:** use only the exact current PR #83 head SHA, and only
after the latest CI run for that same SHA succeeds on Ubuntu/Python 3.9,
Windows/Python 3.9, and Python 3.13 cross-version verification.

The CI gate includes a 120-call six-arm real-GP integration, model-free
verification, read-only factorial analysis and tamper checks. It is technical
validation, not a research result.

See [C2 protocol](docs/PHASE_C2_HEAD_FACTORIAL.txt),
[PR #83](https://github.com/Luyi1235/RA-LSBO/pull/83), and
[design Issue #82](https://github.com/Luyi1235/RA-LSBO/issues/82).

PR #83 remains draft and must not be merged before results are reviewed.
