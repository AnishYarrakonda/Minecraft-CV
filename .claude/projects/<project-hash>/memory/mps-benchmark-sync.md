---
name: mps-benchmark-sync
description: MPS is async — benchmarks reported impossibly fast numbers until we added torch.mps.synchronize() before timing.
metadata:
  type: project
---

A backend benchmark reported MPS inference at a fraction of a millisecond — far too
fast to be real. Optimizing against those numbers led nowhere.

Root cause: MPS execution is **asynchronous**. The Python call returns once GPU work is
*queued*, not *completed*. Reading the clock right after measured only queue-submit
time, not actual compute.

Fix in `mcv-bench`: call `torch.mps.synchronize()` immediately before stopping the
timer (and after warmup) so the measured region covers real GPU completion. Also: warm
up ≥20 frames to discard first-call shader-compilation cost, and report p50/p95/p99 +
effective FPS rather than a single mean.

**Why:** An un-synced MPS timer lies low and makes you "optimize" the wrong stage. This
is benchmark-only — do **not** synchronize on the hot path, it stalls the pipeline.
Related: [[pytorch-mps-setup]].

**How to apply:** Any MPS timing = warmup → sync → start clock → work → sync → stop
clock. Pin clip + backend + device in the result label so numbers are comparable.
