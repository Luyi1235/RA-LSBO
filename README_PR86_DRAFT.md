# PR86-A — cold-start risk learnability

PR84 established that PR83/C2 had 64/64 initial-warm groups and no cold-start
evidence. PR86-A therefore moves upstream: it measures whether the existing
censored risk GP is learnable from small initial datasets, and how training
sample size, class balance and design-space coverage relate to held-out risk
prediction.

This is an offline supervised diagnostic, **not an online optimizer** and not
a new acquisition algorithm. It uses six existing mathematical tasks, a fixed
192-row labelled training pool, an independent held-out test set, nested
training prefixes (12/24/48/96/192), four sampling orders and two supervision
contracts. All pool and test labels are charged; no hidden test label enters
training. The `privileged_exact` arm is an information upper bound only.

Full budget: 12,672 mathematical evaluations, 720 model rows, at most 1,440 GP
heads and 11,520 kernel constructions. CI and smoke budgets are frozen in the
protocol. No aircraft/RCAIDE/VAE/fantasy calls.

See [Issue #86](https://github.com/Luyi1235/RA-LSBO/issues/86) and
[protocol](docs/PR86_A_COLD_START_LEARNABILITY.txt).

**Do not run this draft until PR86 CI succeeds on the exact code SHA.**
