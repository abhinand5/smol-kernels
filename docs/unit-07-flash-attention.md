# Unit 07: FlashAttention

> The capstone. Online softmax (Unit 4) plus tiling (Unit 6) plus on-chip fusion (Units 2, 3, 5) combine to kill attention's `O(N²)` memory bottleneck — *without changing the math and without doing fewer FLOPs.* The speedup is pure memory-hierarchy engineering. If you can write this and explain it, you have the skill this whole curriculum exists to build.

Attention:

```text
S = Q @ Kᵀ / sqrt(d)        # scores,   [N, N]
P = softmax(S, axis=-1)     # weights,  [N, N]
O = P @ V                   # output,   [N, d]
```

for `Q, K, V ∈ [N, d]` (one head; `N` = sequence length, `d` = head dimension).

The naive implementation materializes `S` and `P` — two `N×N` matrices — in global memory. For `N = 4096` that is 16M elements **each**, and the kernel is bottlenecked moving them to and from HBM. FlashAttention **never materializes them**: it tiles `Q`, `K`, `V`, streams `K`/`V` blocks through SRAM, and uses **online softmax** to fuse the score computation, the softmax, and the `@V` into one pass that keeps only `O(N)` state. The result is mathematically identical, uses the same FLOPs, and is dramatically faster because it slashes HBM traffic. This is **IO-aware** kernel design.

You implement:

1. Triton FlashAttention (forward) — the real deliverable
2. CUDA: **explain** the equivalent execution model (full implementation is a stretch)

Baseline: `torch.nn.functional.scaled_dot_product_attention`.

Numbers from the [hardware spec sheet](reference-hardware-spec-sheet.md).

---

## 7.1 Why Naive Attention Is Memory-Bound

Count the HBM traffic of the textbook version for sequence `N`, head dim `d`:

```text
S = Q@Kᵀ : write N² scores to HBM
P = softmax(S) : read N², write N²
O = P@V : read N²
```

The `N²` intermediates dominate everything. Attention's FLOPs are `O(N² d)`, its naive memory traffic is `O(N²)` — and for the sequence lengths that matter, the kernel spends its life moving the score matrix to and from HBM, not computing. It is **memory-bound on a matrix that never needed to exist.** Worse, the `N²` memory *footprint* is what caps your context length.

The insight (FlashAttention, Dao et al. 2022): you never need the whole `S` row at once. Softmax can be computed **online** (Unit 4), so you can process `K`/`V` in blocks, updating a running output, and the `N×N` matrices never touch HBM.

---

## 7.2 The Algorithm: Stream K/V, Accumulate Online

For each block of queries `Q_i` (kept resident in SRAM), loop over blocks of keys/values `K_j, V_j`, maintaining three running quantities per query row: the max `m`, the softmax denominator `l`, and the output accumulator `O`.

```text
for each query block Q_i:
    m = -inf  (per row);  l = 0;  O = 0
    for each key/value block (K_j, V_j):
        S_ij = Q_i @ K_jᵀ * scale            # small block, lives in SRAM
        m_new = max(m, rowmax(S_ij))
        P_ij  = exp(S_ij - m_new)             # rescale to the new max
        l     = l * exp(m - m_new) + rowsum(P_ij)   # online denom update
        O     = O * exp(m - m_new) + P_ij @ V_j     # online output update
        m     = m_new
    O = O / l                                  # final normalization
    write O_i to HBM                           # the ONLY N-sized write
```

This is exactly the Unit 4 online-softmax recurrence (`m`, `l` and the `exp(m_old - m_new)` rescale) — extended so the **output accumulator `O` is rescaled in lockstep with the denominator `l`.** Every `K`/`V` block is read **once** from HBM; `S_ij` and `P_ij` are block-sized and stay in SRAM; only `O` (size `N×d`) is written back. HBM traffic drops from `O(N²)` to `O(N d)`.

Where every prior unit shows up:

- **Online softmax (Unit 4):** the `m`/`l` recurrence and the rescale — now driving `O` too.
- **Tiling (Unit 6):** `Q_i @ K_jᵀ` and `P_ij @ V_j` are tiled GEMMs (`tl.dot`).
- **Shared memory / SRAM (Unit 3):** blocks live on-chip; the whole trick is staging.
- **Fusion (Units 2, 5):** score → softmax → `@V` fused into one kernel, no intermediate HBM round-trips.

---

## 7.3 Causal Masking

For autoregressive models, query `i` may not attend to key `j > i`. In the tiled loop you skip `K`/`V` blocks entirely above the diagonal, and apply a triangular mask (`-inf` before the `exp`) only on the **diagonal block** where `i` and `j` overlap. Skipping fully-masked blocks roughly halves the work for causal attention — a real, measurable optimization, not just correctness.

---

## 7.4 The Turing Constraint: SRAM Is the Budget

FlashAttention's block sizes are bounded by on-chip SRAM. From the spec sheet, this GPU has **up to 64 KB shared memory per SM** — small. You must keep `Q_i`, `K_j`, `V_j`, and the `S_ij` scratch simultaneously resident:

```text
SRAM per block ≈ (BLOCK_M + 2*BLOCK_N) * d * dtype_bytes + scratch
```

So `BLOCK_M`, `BLOCK_N`, and `d` are constrained by 64 KB. Use FP16 for the resident tiles (half the bytes), keep `d ≤ 64` comfortable, and tune block sizes down relative to what an A100 (164 KB+) tutorial assumes. **Do not copy A100 block sizes** — they will fail to launch (out of shared memory) or spill. This hardware-awareness *is* the unit.

---

## 7.5 Triton Exercise Spec

Write a Triton FlashAttention **forward** kernel that:

- assigns one program to a query block (`BLOCK_M` rows of `Q`, one head)
- loops over `K`/`V` blocks of size `BLOCK_N`, maintaining `m`, `l`, `O` in registers/SRAM
- uses `tl.dot` for `Q@Kᵀ` and `P@V`, FP32 accumulate
- applies the online-softmax rescale each iteration
- normalizes by `l` and writes `O` once
- supports a `causal` flag (skip + diagonal-mask)
- respects the 64 KB SRAM budget when choosing block sizes

Structural skeleton (shape, not solution):

```python
@triton.jit
def flash_attn_fwd(Q, K, V, O, scale, N, d, BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    # q_block = load Q_i  (BLOCK_M, d), resident in SRAM
    # m = -inf (BLOCK_M,); l = 0; acc = 0 (BLOCK_M, d)
    # for j in range(0, N, BLOCK_N):
    #     k = load K_j; v = load V_j
    #     s = tl.dot(q_block, k.T) * scale          # (BLOCK_M, BLOCK_N)
    #     (apply causal mask on diagonal block)
    #     m_new = tl.maximum(m, tl.max(s, axis=1))
    #     p = tl.exp(s - m_new[:, None])
    #     alpha = tl.exp(m - m_new)
    #     l = l * alpha + tl.sum(p, axis=1)
    #     acc = acc * alpha[:, None] + tl.dot(p, v)
    #     m = m_new
    # o = acc / l[:, None]
    # store O_i = o
```

Start non-causal, get it bit-correct against SDPA, *then* add causal.

---

## 7.6 CUDA Side: Explain the Execution Model

A full hand-written CUDA FlashAttention is genuinely hard (warp-level MMA, careful SRAM management). For this unit you must be able to **explain**, precisely:

- how the Triton program maps to a CUDA block (one query block per block/CTA)
- where `Q_i`, `K_j`, `V_j` live (shared memory / registers) and how the 64 KB budget constrains tiles
- how the two GEMMs would use `wmma` tensor-core fragments
- where `__syncthreads()` barriers sit in the streaming loop
- why the online recurrence avoids the `N×N` HBM round-trips that the naive CUDA version pays

Writing the CUDA kernel is an excellent stretch goal; *explaining* it is the required deliverable, because that maps directly to success criterion #1 of this curriculum.

---

## 7.7 Benchmark Requirements

Report per implementation:

| Field | Meaning |
|---|---|
| implementation | naive (materialized) / Triton flash / PyTorch SDPA |
| shape | `N` (seq) × `d` (head) , `causal` on/off |
| median time | µs |
| throughput | attention TFLOP/s (`≈ 4 * N² * d / s / 1e12`, ×0.5 for causal) |
| HBM traffic | naive `O(N²)` vs flash `O(N d)` — report both |
| peak memory | naive materializes `N²`; flash does not — show the footprint gap |
| correctness | max abs error vs SDPA (fp16 tolerance) |

Mandatory experiments:

1. **Memory footprint vs `N`.** Plot naive attention's peak memory growing as `N²` while flash stays linear. Then push the workload until naive actually OOMs on 6 GB while flash still runs — at single-head fp32 you need a large `N` (the `N×N` scores reach ~1 GB only around `N≈16k`), so either sweep `N` that high or scale `batch × heads` to force the OOM sooner. The point is the *growth curve* (`N²` vs `N`) and the OOM it eventually causes, not a specific small `N`.
2. **Causal vs non-causal** throughput — confirm the ~2× from block skipping.

---

## 7.8 What Good Looks Like

- Matches SDPA to fp16 tolerance, causal and non-causal.
- Pushed far enough (large `N`, or scaled `batch × heads`), naive attention OOMs on the 6 GB card at a point where flash still runs comfortably — the `N²`-vs-`N` memory story, demonstrated rather than asserted.
- Flash throughput a solid fraction of SDPA (PyTorch's SDPA may dispatch its own fused kernel — matching it within a few× on Turing is a real result).
- You can explain *why* it is faster: not fewer FLOPs, but vastly less HBM traffic. If you say "it does less math," you have missed the unit.

---

## 7.9 Common Mistakes

### Rescaling `l` but not `O` (or vice versa)

The denominator `l` **and** the accumulator `O` must both be multiplied by `exp(m_old - m_new)` every time the max updates. Rescale one and not the other and the output is silently wrong.

### Copying A100 block sizes

64 KB SRAM on Turing is the hard limit. Big tutorial block sizes fail to launch or spill here. Size to *your* hardware (§7.4).

### Thinking the speedup is fewer FLOPs

Same FLOPs as naive attention. The win is HBM traffic and memory footprint. Internalize this.

### Materializing `S` or `P` "just to debug"

The moment you write the `N×N` matrix to global memory you have rebuilt naive attention. Debug on tiny `N` with on-chip prints instead.

### Forgetting the final `/ l`

The online loop accumulates an *unnormalized* `O`. Divide by `l` once at the end, after the last block.

### Causal mask on the wrong blocks

Skip blocks fully above the diagonal; mask only the diagonal block. Masking everything is correct but throws away the causal speedup; masking nothing is wrong.

---

## 7.10 Files

```text
src/unit07/
  flash_attention_triton.py   # student: Triton flash-attn forward (+ causal)
  flash_attention_cuda.cu     # student: optional CUDA impl / written explanation
  benchmark_attention.py      # naive vs flash vs SDPA; memory-footprint sweep
```

---

## 7.11 Check Your Understanding

1. Why is naive attention memory-bound, and which intermediates dominate its HBM traffic?
2. What is the memory footprint of naive vs flash attention as a function of `N`?
3. State the online-softmax recurrence FlashAttention uses for `m`, `l`, and `O`.
4. Why must the output accumulator `O` be rescaled, not just the denominator `l`?
5. Does FlashAttention do fewer FLOPs than naive attention? Then why is it faster?
6. Which two `tl.dot` calls are the tiled GEMMs inside the loop?
7. How does the 64 KB SRAM limit constrain `BLOCK_M`, `BLOCK_N`, and `d`?
8. Why does causal attention skip blocks, and roughly what speedup does that give?
9. How would the Triton program map onto a CUDA block, and where would the barriers go?
10. Name the prior unit each ingredient comes from: online softmax, tiled matmul, SRAM staging, fusion.

---

**Next:** Unit 08 — Pick Your Battle: read a real production kernel line by line, then reproduce it
