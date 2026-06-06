"""
Unit 04 - Exercise: Softmax (Triton)

Goal: numerically stable row-softmax; parallel reductions; the online-softmax bridge.
Full spec: docs/unit-04-softmax.md

Student-owned work:
- Write the Triton row-softmax here (one program per row; subtract the row max).
- Write the CUDA two-pass + online softmax in softmax_cuda.cu (warp+block reductions).
- The `- max(x)` is mandatory; the stability test exists to catch its absence.

AI/helper-owned work allowed:
- benchmark harnesses, correctness checks, build glue, review feedback.

torch.softmax(x, dim=-1) is the BASELINE.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# === TRITON KERNEL TODO ===
# @triton.jit
# def softmax_kernel(x_ptr, y_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
#     row  = tl.program_id(0)
#     cols = tl.arange(0, BLOCK_SIZE)
#     mask = cols < n_cols
#     x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=-float('inf'))
#     m = tl.max(x, axis=0)            # subtract the max -- mandatory
#     e = tl.exp(x - m)
#     s = tl.sum(e, axis=0)
#     tl.store(y_ptr + row * row_stride + cols, e / s, mask=mask)
#
# Use BLOCK_SIZE = next power of 2 >= n_cols for the single-tile design.


# === BENCHMARK TODO ===
# - rows N in {1024, 4096, 16384}, M = 4096 rows
# - effective bandwidth = 2 * M * N * 4 / s / 1e9
# - MANDATORY stability test: add a large constant (x + 100) and confirm no NaN vs torch
# - verify the online (one-pass) CUDA variant matches the two-pass variant


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print("Unit 04 starter loaded. Implement the softmax kernel next.")


if __name__ == "__main__":
    main()
