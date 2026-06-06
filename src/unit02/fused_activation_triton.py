"""
Unit 02 - Exercise: Fused ReLU / GeLU (Triton)

Goal: y = activation(x + b) fused in one kernel; meet @triton.autotune and the SFU.
Full spec and roofline: docs/unit-02-fused-activation.md

Student-owned work:
- Write the autotuned @triton.jit fused activation kernel here (ReLU and GeLU).
- Write the CUDA scalar + float4 kernels in fused_activation_cuda.cu.
- Keep the intermediate (x + b) in-register; never store it to global memory.

AI/helper-owned work allowed:
- benchmark harnesses, correctness checks, build glue, review feedback.

torch.relu / F.gelu are BASELINES. Match F.gelu's form (erf default vs tanh).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# === TRITON KERNEL TODO ===
# @triton.autotune(
#     configs=[triton.Config({'BLOCK_SIZE': bs}, num_warps=w)
#              for bs in (256, 512, 1024, 2048, 4096) for w in (2, 4, 8)],
#     key=['n_elements'],
# )
# @triton.jit
# def fused_act_kernel(x_ptr, b_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
#     pid     = tl.program_id(0)
#     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask    = offsets < n_elements
#     x = tl.load(x_ptr + offsets, mask=mask)
#     b = tl.load(b_ptr + offsets, mask=mask)
#     t = x + b
#     # y = relu(t)  OR  gelu(t)  -- compute in-register, never store t
#     # tl.store(y_ptr + offsets, y, mask=mask)


# === BENCHMARK TODO ===
# - N = 32 * 1024 * 1024 float32 elements
# - MANDATORY experiment 1: fused vs unfused relu(x + b); confirm ~20/12 speedup
# - MANDATORY experiment 2: ReLU vs GeLU at equal bytes; measure the SFU gap
# - warm up (autotune search runs on first call); effective BW = bytes / s / 1e9
# - verify vs PyTorch (match the GeLU form you implement)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print("Unit 02 starter loaded. Implement the fused activation kernel next.")


if __name__ == "__main__":
    main()
