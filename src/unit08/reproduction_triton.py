"""
Unit 08 - Exercise: Reproduce a production kernel

Goal: reproduce one approachable production kernel (e.g. RMSNorm, fused add+RMSNorm,
RoPE, SwiGLU) from your UNDERSTANDING -- not by copy-paste -- and benchmark it.
Full spec: docs/unit-08-pick-your-battle.md

Student-owned work:
- Implement the kernel here (and/or reproduction_cuda.cu) from the contract.
- Match the original's dtypes, shapes, and edge handling.
- Verify correctness against BOTH the original kernel and a PyTorch reference.
- Itemize and EXPLAIN every performance gap to the original.

AI/helper-owned work allowed:
- benchmark harnesses, correctness checks, build glue, review feedback.

Suggested first target: RMSNorm (Unit 5 with no mean-subtraction): x / rms(x) * gamma.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# === KERNEL TODO ===
# Implement your chosen kernel from its contract. Record source repo/commit in NOTES.md.


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print("Unit 08 starter loaded. Reproduce your chosen production kernel.")


if __name__ == "__main__":
    main()
