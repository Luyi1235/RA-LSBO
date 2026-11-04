### Brier / LogLoss (held-out original selected calls)

| task / arm | event | raw B / LL | pooled B / LL | phase B / LL | prefix B / LL |
|---|---|---|---|---|---|
| aircraft_ordinary / full_latent | feasible | 0.1962 / 0.5845 | 0.1891 / 0.5545 | 0.1888 / 0.5529 | 0.1921 / 0.5758 |
| aircraft_ordinary / full_latent | mission | 0.0341 / 0.1165 | 0.0331 / 0.1130 | 0.0306 / 0.1049 | 0.0344 / 0.1174 |
| aircraft_ordinary / ra_lsbo | feasible | 0.1879 / 0.5630 | 0.1789 / 0.5335 | 0.1789 / 0.5327 | 0.1835 / 0.5513 |
| aircraft_ordinary / ra_lsbo | mission | 0.0302 / 0.1068 | 0.0301 / 0.1061 | 0.0284 / 0.1011 | 0.0302 / 0.1069 |
| aircraft_sparse / full_latent | feasible | 0.1999 / 0.5856 | 0.1810 / 0.5378 | 0.1801 / 0.5358 | 0.1948 / 0.5713 |
| aircraft_sparse / full_latent | mission | 0.1542 / 0.5042 | 0.1502 / 0.4755 | 0.1496 / 0.4743 | 0.1511 / 0.4913 |
| aircraft_sparse / ra_lsbo | feasible | 0.1899 / 0.5530 | 0.1805 / 0.5320 | 0.1767 / 0.5242 | 0.1879 / 0.5498 |
| aircraft_sparse / ra_lsbo | mission | 0.1326 / 0.4248 | 0.1322 / 0.4222 | 0.1291 / 0.4163 | 0.1357 / 0.4316 |

### 5-bin ECE / false accepted at p>=0.8

| task / arm | event | raw ECE / FP8 | pooled ECE / FP8 | phase ECE / FP8 | prefix ECE / FP8 |
|---|---|---|---|---|---|
| aircraft_ordinary / full_latent | feasible | 0.0997 / 105 | 0.0525 / 54 | 0.0463 / 49 | 0.0753 / 101 |
| aircraft_ordinary / full_latent | mission | 0.0170 / 13 | 0.0124 / 19 | 0.0083 / 13 | 0.0175 / 13 |
| aircraft_ordinary / ra_lsbo | feasible | 0.0970 / 96 | 0.0518 / 55 | 0.0491 / 49 | 0.0778 / 90 |
| aircraft_ordinary / ra_lsbo | mission | 0.0078 / 12 | 0.0070 / 15 | 0.0145 / 9 | 0.0087 / 12 |
| aircraft_sparse / full_latent | feasible | 0.1262 / 70 | 0.0383 / 38 | 0.0299 / 37 | 0.0979 / 66 |
| aircraft_sparse / full_latent | mission | 0.0837 / 77 | 0.0724 / 51 | 0.0745 / 48 | 0.0728 / 62 |
| aircraft_sparse / ra_lsbo | feasible | 0.0985 / 55 | 0.0508 / 44 | 0.0412 / 45 | 0.0884 / 53 |
| aircraft_sparse / ra_lsbo | mission | 0.0546 / 49 | 0.0457 / 47 | 0.0415 / 43 | 0.0621 / 47 |

### Boundary phase only

| task / arm | event | raw B / ECE / FP8 | pooled B / ECE / FP8 | phase B / ECE / FP8 | prefix B / ECE / FP8 |
|---|---|---|---|---|---|
| aircraft_ordinary / full_latent | feasible | 0.2549 / 0.0798 / 0 | 0.2563 / 0.0880 / 0 | 0.2508 / 0.0536 / 0 | 0.2489 / 0.0418 / 0 |
| aircraft_ordinary / full_latent | mission | 0.1571 / 0.0517 / 12 | 0.1657 / 0.0945 / 18 | 0.1568 / 0.0490 / 12 | 0.1585 / 0.0496 / 12 |
| aircraft_ordinary / ra_lsbo | feasible | 0.2548 / 0.0979 / 0 | 0.2483 / 0.0668 / 0 | 0.2456 / 0.0196 / 0 | 0.2506 / 0.0812 / 0 |
| aircraft_ordinary / ra_lsbo | mission | 0.1432 / 0.0527 / 10 | 0.1459 / 0.0670 / 13 | 0.1398 / 0.0360 / 7 | 0.1421 / 0.0495 / 10 |
| aircraft_sparse / full_latent | feasible | 0.2691 / 0.2646 / 0 | 0.2057 / 0.0834 / 0 | 0.2000 / 0.0369 / 0 | 0.2638 / 0.2251 / 0 |
| aircraft_sparse / full_latent | mission | 0.2805 / 0.2091 / 10 | 0.2402 / 0.0653 / 5 | 0.2381 / 0.0591 / 1 | 0.2705 / 0.1688 / 8 |
| aircraft_sparse / ra_lsbo | feasible | 0.2695 / 0.2610 / 0 | 0.2339 / 0.1808 / 0 | 0.2092 / 0.0899 / 0 | 0.2594 / 0.2309 / 0 |
| aircraft_sparse / ra_lsbo | mission | 0.2489 / 0.1459 / 11 | 0.2437 / 0.1266 / 11 | 0.2295 / 0.0910 / 3 | 0.2484 / 0.1239 / 9 |

### Budget diagnostic: mean predicted debit vs realized non-feasible (18 original calls per seed)

| task / arm | raw debit | prefix debit | realized | B=8 actual / raw pred / prefix pred | raw-within-but-actual-exceeds | prefix-within-but-actual-exceeds |
|---|---:|---:|---:|---|---:|---:|
| aircraft_ordinary/full_latent | 4.68 | 5.14 | 6.44 | 13 / 0 / 3 | 13 | 10 |
| aircraft_ordinary/ra_lsbo | 4.57 | 4.98 | 6.31 | 9 / 0 / 2 | 9 | 7 |
| aircraft_sparse/full_latent | 6.13 | 6.65 | 8.33 | 28 / 7 / 14 | 21 | 14 |
| aircraft_sparse/ra_lsbo | 6.27 | 6.64 | 7.86 | 23 / 6 / 11 | 19 | 13 |
