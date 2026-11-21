# PR155 cold-acquisition pilot (3 seeds/scenario)

Fixed code SHA: `76de181b90e0b4d84324c471db37ada432e14792`.
Common interval-risk refinement; only the cold-start acquisition differs:
`ra_refine_interval_cold_sign`, `ra_refine_interval`, `ra_refine_interval_cold_active`.
3 new seeds per scenario, 768 reduced-aircraft calls, 0 RCAIDE/MDA.

## Acceptance

`VERIFIED_COLD_ACQUISITION_TO_ENERGY cold-pilot 768`; 6 sealed directories
(3 headwind `29410100..29410300`, 3 payload `29400100..29400300`).

## Per-seed terminals

| scenario | arm | first usable | first <=40,000 Wh | best Wh |
|---|---|---:|---:|---:|
| headwind | cold_sign | 13.0 (11/4/24) | 16.7 (14/12/24) | 38,164.3 |
| headwind | interval | 9.3 (11/4/13) | 22.0 (22/12/32) | 38,213.4 |
| headwind | cold_active | 10.0 (12/4/14) | 19.3 (24/12/22) | 38,190.6 |
| payload | cold_sign | 9.0 (3/8/16) | 19.0 (15/26/16) | 36,391.6 |
| payload | interval | 8.3 (3/8/14) | 22.7 (15/26/27) | 36,392.5 |
| payload | cold_active | 6.7 (3/8/9) | 20.7 (15/26/21) | 36,391.6 |

## Paired contrasts (3 seeds; descriptive only)

Headwind:
- `cold_active − interval`: first usable +0.67 calls [0,+1]; terminal energy −22.7 Wh
  [−66.4,0]; energy target −2.67 calls [−10,+2].
- `interval − cold_sign`: first usable −3.67 calls [−11,0]; terminal energy +49.1 Wh
  [+0,+103]; energy target +5.33 calls [0,+8].

Payload:
- `cold_active − interval`: first usable −1.67 calls [−5,0]; terminal energy −0.91 Wh
  [−2.74,0]; energy target −2 calls [−6,0].
- `interval − cold_sign`: first usable −0.67 [−2,0]; terminal energy +0.91 Wh
  [0,+2.74]; energy target +3.67 [0,+11].

All arms hit 3/3 and all reached <=40,000 Wh. Three seeds are far too few for
inference; this pilot only screens whether cold acquisition could matter.
