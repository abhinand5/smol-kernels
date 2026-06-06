"""
Unit 07 - Exercise: FlashAttention forward (Triton)

Goal: O = softmax(Q Kt / sqrt(d)) V without materializing the N x N scores. IO-aware.
Full spec: docs/unit-07-flash-attention.md

Student-owned work:
- Write the Triton flash-attention forward here (online softmax over streamed K/V blocks).
- Add a causal flag (skip blocks above the diagonal; mask only the diagonal block).
- In flash_attention_cuda.cu: write the CUDA execution-model EXPLANATION (impl optional).
- Respect the 64 KB SRAM budget -- do NOT copy A100 block sizes.

AI/helper-owned work allowed:
- benchmark harnesses, correctness checks, build glue, review feedback.

F.scaled_dot_product_attention is the BASELINE.
The speedup is LESS HBM TRAFFIC, not fewer FLOPs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# === TRITON KERNEL TODO ===
# @triton.jit
# def flash_attn_fwd(Q, K, V, O, scale, N, d,
#                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
#     # one program owns BLOCK_M rows of Q (one head), resident in SRAM
#     # m = -inf (BLOCK_M,); l = 0; acc = 0 (BLOCK_M, d)
#     # for j in range(0, N, BLOCK_N):
#     #     k = load K_j; v = load V_j
#     #     s = tl.dot(q, tl.trans(k)) * scale          # (BLOCK_M, BLOCK_N)
#     #     # if CAUSAL: mask the diagonal block
#     #     m_new = tl.maximum(m, tl.max(s, axis=1))
#     #     p     = tl.exp(s - m_new[:, None])
#     #     alpha = tl.exp(m - m_new)
#     #     l     = l * alpha + tl.sum(p, axis=1)
#     #     acc   = acc * alpha[:, None] + tl.dot(p, v)  # rescale O in lockstep with l
#     #     m     = m_new
#     # o = acc / l[:, None]   # final normalization (don't forget!)
#     # store O_i = o
#
# Start non-causal and bit-correct vs SDPA, THEN add causal.


# === BENCHMARK TODO ===
# - shapes: sweep N (seq len) for fixed d (e.g. 64); causal on/off
# - throughput ~ 4 * N^2 * d / s / 1e12 TFLOP/s (x0.5 for causal)
# - MANDATORY: memory-footprint sweep -- naive O(N^2) OOMs / thrashes while flash stays O(N)
# - report causal vs non-causal (~2x); verify vs SDPA (fp16 tolerance)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print("Unit 07 starter loaded. Implement the flash-attention kernel next.")


if __name__ == "__main__":
    main()
