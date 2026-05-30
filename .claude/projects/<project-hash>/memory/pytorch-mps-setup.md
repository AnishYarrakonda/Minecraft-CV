---
name: pytorch-mps-setup
description: MPS device selection on Apple Silicon, with mandatory CPU fallback and the MPS_FALLBACK env var.
metadata:
  type: project
---

Setting up the PyTorch hand-detector backend on Apple Silicon (M-series). MPS
(Metal Performance Shaders) gives a real speedup over CPU for inference, but two
things bit us:

1. Some ops the model used weren't implemented on MPS and threw `NotImplementedError`
   mid-pipeline. Fixed by exporting `PYTORCH_ENABLE_MPS_FALLBACK=1` (now in
   `.claude/settings.json` `env`) so unsupported ops fall back to CPU instead of
   crashing. Each fallback is still a perf smell to chase, not a free pass.
2. `float64` tensors silently misbehaved/slowed on Metal — standardized model + inputs
   to `float32`.

**Why:** MPS is an accelerator, never a hard dependency — the controller must run on
any Mac and in CI without a GPU. Device selection lives in one helper
(`select_device()`: mps → cuda → cpu) so no `"mps"` string literals leak across the
codebase. Related: [[mps-benchmark-sync]] (async timing) and the rules file §2.

**How to apply:** Move the model to the device once at startup (never per frame), wrap
inference in `torch.inference_mode()` + `model.eval()`, keep everything `float32`, and
make sure the CPU path is exercised by tests. Treat per-frame CPU fallbacks inside an
MPS graph as latency bugs.
