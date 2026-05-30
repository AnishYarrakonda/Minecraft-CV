---
name: schmitt-trigger-chatter-fix
description: Pinch chatter (rapid-fire Attack) fixed by widening the hysteresis band, not by lowering the engage threshold.
metadata:
  type: project
---

Early single-threshold pinch detection caused "chatter": when a fingertip hovered near
the trigger distance, CV frame jitter flipped it across the threshold every few frames,
firing rapid KEY_DOWN/KEY_UP — Attack machine-gunned, Jump stuttered.

Fix: a proper **Schmitt trigger** with two thresholds and `T_release > T_engage`
strictly. Engage only when distance < `T_engage`; release only when distance >
`T_release`. The gap between them swallows jitter. The instinct to "fix" misfires by
lowering `T_engage` is wrong — it narrows the effective band and makes chatter worse;
the correct lever is **widening the band** (or adding One-Euro smoothing upstream).

Distances are normalized by hand scale (wrist→middle-MCP span) so thresholds are
unitless ratios, invariant to how far the hand is from the camera.

**Why:** This is the single most important correctness invariant in the project; a test
asserts `T_release > T_engage` for every configured gesture. Collapsing the two
thresholds reintroduces this exact bug. See [[project-overview]].

**How to apply:** Tune the band against a known-jittery recorded clip via the
`frame-analyzer` skill (it reports chatter events). Ring/pinky hotbar gestures are
momentary pulses, not holds, because pinky pinch anatomically drags the ring finger.
