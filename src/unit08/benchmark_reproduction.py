"""
Unit 08 - Benchmark: your reproduction vs the original vs PyTorch

Goal: measure your reproduction against the production kernel AND a PyTorch reference,
then explain the gap. See docs/unit-08-pick-your-battle.md.

This harness is helper-owned; the kernel under test is student-owned.
"""

from __future__ import annotations

import torch


# === BENCHMARK TODO ===
# - realistic shapes for the op you reproduced
# - median time: yours vs original vs PyTorch
# - throughput: GB/s or TFLOP/s as appropriate
# - correctness: vs original AND vs PyTorch
# - GAP ANALYSIS: itemize WHY the original is faster (vectorization? tiling? fused
#   epilogue? tuned config you didn't match?). "3x slower" is not a result; the
#   named cause is.


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print("Unit 08 benchmark starter loaded.")


if __name__ == "__main__":
    main()
