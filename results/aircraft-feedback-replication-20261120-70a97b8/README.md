# PR144 — Feedback-form replication

Fixed code SHA: `70a97b87e7bcc3b3222f9fa262536b30cba2e72d`

No RCAIDE/MDA calls. Both profiles `VERIFIED_AIRCRAFT_FEEDBACK_REPLICATION`.

## Acceptance and calls

| Profile | initial | online | total |
|---|---:|---:|---:|
| smoke | 16 | 7 | 23 |
| payload | 768 | 846 | 1,614 |

Formal upper bound was 3,072; the payload run stopped at 1,614 after all
cold histories resolved. `rcaide_mda_calls=0`.

## Arms

Five arms: `sign_greedy`, `sign_active`, `interval_greedy`,
`interval_active`, `interval_fit_sign_score`.

Primary interaction:

`feedback_by_policy_interaction = (interval_active − interval_greedy) −
(sign_active − sign_greedy)`

Positive means the active-policy benefit is smaller under interval feedback
than under sign feedback.

## Payload results

Capped first call, 48 seeds; all five arms found **45/48** usable designs.
No arm has a hit-rate advantage.

| Contrast | n=8 mean [95%] | n=16 mean [95%] |
|---|---|---|
| **feedback × policy interaction** | **+1.562 [+0.083, +3.062]** | +0.292 [−0.167, +0.812] |
| sign_active − sign_greedy | **−1.583 [−3.021, −0.250]** | −0.042 [−0.583, +0.438] |
| interval_active − interval_greedy | −0.021 [−1.313, +1.250] | +0.250 [−0.292, +0.812] |
| interval_greedy − sign_greedy | **−2.396 [−4.021, −0.833]** | **−0.958 [−2.000, −0.104]** |
| interval_active − sign_active | −0.833 [−2.000, +0.250] | −0.667 [−1.625, +0.167] |
| interval_fit_sign_score − sign_active | −0.500 [−1.771, +0.792] | −0.667 [−1.625, +0.167] |

Cold (n=8: 42; n=16: 28):

- interaction n=8 **+1.786 [+0.048, +3.453]**; n=16 +0.500 [−0.286, +1.358].
- sign_active − sign_greedy n=8 **−1.810 [−3.429, −0.380]**.
- interval_active − interval_greedy n=8 −0.024 [−1.500, +1.357].
- interval_greedy − sign_greedy n=8 **−2.738 [−4.571, −1.000]**; n=16 **−1.643 [−3.322, −0.179]**.

## Smoke

One cold seed: all five arms 19 calls except `sign_active` 18; only
`sign_active` hit (1). This is a technical smoke, not evidence.

## Interpretation

1. At n=8 there is a **significant positive interaction**: active selection is
   much faster under sign feedback (−1.58 calls) than under interval feedback
   (−0.02). Interval feedback removes the active-policy benefit. At n=16 the
   interaction shrinks to +0.29 and is no longer significant.
2. Raw interval-greedy is faster than sign-greedy at both n=8 and n=16
   (−2.40 / −0.96 calls), but the active variant does not preserve that gain.
3. No arm changes top-one usable hits: all 45/48 payload. The effects are
   discovery timing only.
4. `interval_fit_sign_score` is behaviorally identical to `interval_active`
   (n=8 difference −0.33 with CI crossing 0; n=16 exactly 0), so the separate
   score wrapper adds nothing here.
5. This is a reduced-aircraft replication with a single formal condition;
   no safety or generalization claim.
