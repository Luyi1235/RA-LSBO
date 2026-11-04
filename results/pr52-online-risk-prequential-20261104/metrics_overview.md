### aircraft_ordinary/full_latent/all
| proxy | target | n | raw Brier | raw LL | raw ECE | cal Brier | cal LL | cal ECE | base Brier | base LL | base ECE | raw FP@.5 | cal FP@.5 | base FP@.5 | raw FP@.8 | cal FP@.8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| constraint_product | feasible | 1152 | 0.1962 | 0.5845 | 0.0997 | 0.1891 | 0.5545 | 0.0525 | 0.2595 | 0.7136 | 0.1434 | 319/412 | 159/412 | 261/412 | 105 | 54 |
| legacy_completion_product | feasible | 1152 | 0.1885 | 0.5558 | 0.0463 | 0.1868 | 0.5517 | 0.0403 | 0.2595 | 0.7136 | 0.1434 | 211/412 | 167/412 | 261/412 | 72 | 38 |
| record_completion_as_mission | mission | 1152 | 0.0404 | 0.1780 | 0.0342 | 0.0391 | 0.1686 | 0.0071 | 0.0775 | 0.3067 | 0.1711 | 47/47 | 47/47 | 47/47 | 47 | 47 |
| soc_mission | mission | 1152 | 0.0341 | 0.1165 | 0.0170 | 0.0331 | 0.1130 | 0.0124 | 0.0775 | 0.3067 | 0.1711 | 47/47 | 47/47 | 47/47 | 13 | 19 |

### aircraft_ordinary/ra_lsbo/all
| proxy | target | n | raw Brier | raw LL | raw ECE | cal Brier | cal LL | cal ECE | base Brier | base LL | base ECE | raw FP@.5 | cal FP@.5 | base FP@.5 | raw FP@.8 | cal FP@.8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| constraint_product | feasible | 1152 | 0.1879 | 0.5630 | 0.0970 | 0.1789 | 0.5335 | 0.0518 | 0.2601 | 0.7149 | 0.1431 | 323/404 | 152/404 | 263/404 | 96 | 55 |
| legacy_completion_product | feasible | 1152 | 0.1810 | 0.5387 | 0.0424 | 0.1788 | 0.5344 | 0.0279 | 0.2601 | 0.7149 | 0.1431 | 220/404 | 160/404 | 263/404 | 62 | 41 |
| record_completion_as_mission | mission | 1152 | 0.0389 | 0.1737 | 0.0381 | 0.0375 | 0.1628 | 0.0114 | 0.0763 | 0.3035 | 0.1717 | 45/45 | 45/45 | 45/45 | 45 | 45 |
| soc_mission | mission | 1152 | 0.0302 | 0.1068 | 0.0078 | 0.0301 | 0.1061 | 0.0070 | 0.0763 | 0.3035 | 0.1717 | 45/45 | 45/45 | 45/45 | 12 | 15 |

### aircraft_sparse/full_latent/all
| proxy | target | n | raw Brier | raw LL | raw ECE | cal Brier | cal LL | cal ECE | base Brier | base LL | base ECE | raw FP@.5 | cal FP@.5 | base FP@.5 | raw FP@.8 | cal FP@.8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| constraint_product | feasible | 1133 | 0.1999 | 0.5856 | 0.1262 | 0.1810 | 0.5378 | 0.0383 | 0.3233 | 0.8763 | 0.2420 | 343/516 | 126/516 | 14/516 | 70 | 38 |
| legacy_completion_product | feasible | 1133 | 0.1909 | 0.5596 | 0.0821 | 0.1816 | 0.5416 | 0.0317 | 0.3233 | 0.8763 | 0.2420 | 204/516 | 132/516 | 14/516 | 50 | 22 |
| record_completion_as_mission | mission | 1133 | 0.2089 | 0.6877 | 0.1703 | 0.1791 | 0.5411 | 0.0203 | 0.2883 | 0.7795 | 0.3115 | 270/270 | 269/270 | 94/270 | 270 | 40 |
| soc_mission | mission | 1133 | 0.1542 | 0.5042 | 0.0837 | 0.1502 | 0.4755 | 0.0724 | 0.2883 | 0.7795 | 0.3115 | 235/270 | 141/270 | 94/270 | 77 | 51 |

### aircraft_sparse/ra_lsbo/all
| proxy | target | n | raw Brier | raw LL | raw ECE | cal Brier | cal LL | cal ECE | base Brier | base LL | base ECE | raw FP@.5 | cal FP@.5 | base FP@.5 | raw FP@.8 | cal FP@.8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| constraint_product | feasible | 1131 | 0.1899 | 0.5530 | 0.0985 | 0.1805 | 0.5320 | 0.0508 | 0.3308 | 0.8924 | 0.2616 | 305/484 | 145/484 | 21/484 | 55 | 44 |
| legacy_completion_product | feasible | 1131 | 0.1833 | 0.5404 | 0.0562 | 0.1815 | 0.5373 | 0.0542 | 0.3308 | 0.8924 | 0.2616 | 170/484 | 150/484 | 21/484 | 40 | 34 |
| record_completion_as_mission | mission | 1131 | 0.1731 | 0.5706 | 0.1290 | 0.1567 | 0.4893 | 0.0467 | 0.2893 | 0.7812 | 0.3356 | 225/225 | 225/225 | 121/225 | 225 | 183 |
| soc_mission | mission | 1131 | 0.1326 | 0.4248 | 0.0546 | 0.1322 | 0.4222 | 0.0457 | 0.2893 | 0.7812 | 0.3356 | 190/225 | 177/225 | 121/225 | 49 | 47 |

