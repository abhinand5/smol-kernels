"""
Unit 03 - Exercise: Matrix Transpose (Triton)

Goal: B[j, i] = A[i, j] at memcpy bandwidth; coalescing, shared memory, bank conflicts.
Full spec: docs/unit-03-matrix-transpose.md

Student-owned work:
- Write the tiled Triton transpose here (2D offsets, swapped strides on store).
- Write the CUDA naive / tiled / tiled+padded kernels in transpose_cuda.cu.
- Explain each bandwidth jump: naive -> tiled (coalescing) -> padded (bank conflicts).

AI/helper-owned work allowed:
- benchmark harnesses, correctness checks, build glue, review feedback.

A.t().contiguous() is the BASELINE.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# === TRITON KERNEL TODO ===
# @triton.jit
# def transpose_kernel(a_ptr, b_ptr, M, N,
#                      stride_am, stride_an, stride_bm, stride_bn,
#                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
#     pid_m = tl.program_id(0); pid_n = tl.program_id(1)
#     rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
#     # build 2D pointer blocks for A and B (B uses swapped strides), mask ragged edges,
#     # load the A tile, store it transposed into B.


# === BENCHMARK TODO ===
# - 4096 x 4096 float32 plus one non-square, non-divisible case (e.g. 4097 x 2049)
# - effective bandwidth = 2 * M * N * 4 / s / 1e9 (read + write)
# - compare vs 264 GB/s AND vs your Unit 0 memcpy bandwidth
# - verify exact match vs A.t(); benchmark tile shapes 32x32, 64x64, 32x64


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print("Unit 03 starter loaded. Implement the transpose kernel next.")


if __name__ == "__main__":
    main()
