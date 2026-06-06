---
description: Detect this machine's GPU and (re)generate docs/my-gpu-spec.md so the curriculum targets recalibrate.
---

Calibrate the learner's GPU for this curriculum.

1. Run: `uv run python3 scripts/calibrate_gpu.py`
2. If it fails because CUDA is unavailable, tell the user this curriculum needs an NVIDIA GPU, and stop.
3. Read the generated `docs/my-gpu-spec.md` and summarize for the user: measured memory bandwidth, FP32 and tensor-core ridge points, tensor-core data types, shared-memory budget, and VRAM.
4. Remind them the file is gitignored (personal, not committed) and that every unit's roofline math and "what good looks like" targets now grade against it.

This is portable sugar over `scripts/calibrate_gpu.py` — the script is the engine and works for anyone with no agent at all.
