"""
Unit 05 - Exercise: LayerNorm (Triton)

Goal: y = (x - mu) / sqrt(var + eps) * gamma + beta; two stats at once; epilogue fusion.
Full spec: docs/unit-05-layernorm.md

Student-owned work:
- Write the Triton LayerNorm here (one program per row; fuse the affine epilogue).
- Write CUDA two-pass + Welford LayerNorm in layernorm_cuda.cu.
- Use a stable variance (two-pass or Welford); naive E[x^2]-E[x]^2 fails the stability test.

AI/helper-owned work allowed:
- benchmark harnesses, correctness checks, build glue, review feedback.

F.layer_norm is the BASELINE.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# === TRITON KERNEL TODO ===
# @triton.jit
# def layernorm_kernel(x_ptr, y_ptr, g_ptr, b_ptr, row_stride, N, eps,
#                      BLOCK: tl.constexpr):
#     row  = tl.program_id(0)
#     cols = tl.arange(0, BLOCK)
#     mask = cols < N
#     x    = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0)
#     mu   = tl.sum(x, axis=0) / N
#     xc   = tl.where(mask, x - mu, 0.0)
#     var  = tl.sum(xc * xc, axis=0) / N
#     rstd = 1.0 / tl.sqrt(var + eps)            # eps INSIDE the sqrt
#     g    = tl.load(g_ptr + cols, mask=mask)
#     b    = tl.load(b_ptr + cols, mask=mask)
#     y    = xc * rstd * g + b                   # fused affine epilogue
#     tl.store(y_ptr + row * row_stride + cols, y, mask=mask)


# === BENCHMARK TODO ===
# - M = 4096 rows, N in {1024, 4096, 16384}, eps = 1e-5
# - effective bandwidth = 2 * M * N * 4 / s / 1e9 (gamma/beta amortized across rows)
# - MANDATORY stability test: large row mean (x + 1e4); naive variance gives wrong/neg var
# - verify Welford and two-pass agree; verify vs F.layer_norm


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print("Unit 05 starter loaded. Implement the layernorm kernel next.")


if __name__ == "__main__":
    main()
