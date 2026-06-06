"""
Unit 06 - Exercise: Tiled GEMM (Triton)

Goal: C = A @ B; cross the roofline ridge via tiling; engage FP16 tensor cores (tl.dot).
Full spec and roofline: docs/unit-06-tiled-gemm.md

Student-owned work:
- Write the autotuned tl.dot GEMM here (FP16 inputs, FP32 accumulate).
- Write CUDA naive -> tiled -> register-blocked (-> wmma stretch) in gemm_cuda.cu.
- Always state which ceiling you grade against (FP32 ~4.55 vs tensor ~36/18 TFLOP/s).

AI/helper-owned work allowed:
- benchmark harnesses, correctness checks, build glue, review feedback.

torch.matmul (cuBLAS) is the BASELINE and north star.
On Turing the tensor-core path is FP16 -- NOT TF32 (Ampere+).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# === TRITON KERNEL TODO ===
# @triton.autotune(configs=[...over BLOCK_M/N/K, num_warps, num_stages...], key=['M','N','K'])
# @triton.jit
# def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
#                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
#                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
#     # one program computes a BLOCK_M x BLOCK_N tile of C
#     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
#     for k0 in range(0, K, BLOCK_K):
#         a = tl.load(a_tile_ptrs, mask=...)   # (BLOCK_M, BLOCK_K) fp16
#         b = tl.load(b_tile_ptrs, mask=...)   # (BLOCK_K, BLOCK_N) fp16
#         acc += tl.dot(a, b)                  # tensor-core MMA, fp32 accumulate
#         # advance pointers by BLOCK_K
#     # mask ragged edges in M, N, K; store acc.to(c dtype)


# === BENCHMARK TODO ===
# - square M=N=K in {1024, 2048, 4096} (4096 fp32 = 64 MB/matrix, fits 6 GB)
# - throughput = 2 * M * N * K / s / 1e12 TFLOP/s
# - grade vs the RELEVANT ceiling AND vs cuBLAS; report which ceiling
# - rel error vs torch.matmul (looser tol for fp16)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print("Unit 06 starter loaded. Implement the GEMM kernel next.")


if __name__ == "__main__":
    main()
