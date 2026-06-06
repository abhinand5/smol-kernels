"""
Unit 01 - Exercise: Vector Add (Triton)

Goal: c[i] = a[i] + b[i], twice (CUDA + Triton), benchmarked against torch.add.
Full spec and roofline: docs/unit-01-vector-add.md

Student-owned work:
- Write the @triton.jit `add_kernel` in this file.
- Write the CUDA kernel(s) in vector_add_cuda.cu (1-per-thread + grid-stride).
- Own the launch/grid math and be able to predict the bandwidth before you run.

AI/helper-owned work allowed:
- benchmark harnesses, correctness checks, build glue, review feedback.

`a + b` / `torch.add` is the BASELINE, not your kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# === TRITON KERNEL TODO ===
# @triton.jit
# def add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
#     pid     = tl.program_id(0)
#     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask    = offsets < n_elements
#     a = tl.load(a_ptr + offsets, mask=mask)
#     b = tl.load(b_ptr + offsets, mask=mask)
#     tl.store(c_ptr + offsets, a + b, mask=mask)   # <- you write the body
#
# Launch with grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
# Tile sizes to benchmark: 128, 256, 512, 1024, 2048, 4096, 8192


# === BENCHMARK TODO ===
# - N = 32 * 1024 * 1024 float32 elements (a, b, c => 384 MB resident)
# - allocate a, b, c once outside the timed region; warm up before measuring
# - time with triton.testing.do_bench or CUDA events
# - effective bandwidth = 12 * N / seconds / 1e9  (two reads + one write)
# - efficiency vs 264 GB/s; verify max abs error vs (a + b)
# - report CUDA (both variants), Triton, and the torch.add baseline


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print("Unit 01 starter loaded. Implement the add_kernel next.")


if __name__ == "__main__":
    main()
