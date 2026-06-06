# Unit 06: Tiled GEMM

> The unit everything was building toward. For five units you optimized memory traffic because you had no choice — every kernel sat far left of the roofline ridge. GEMM is the first kernel where **tiling raises arithmetic intensity** until you cross the ridge into the compute-bound region — and the ceiling that suddenly matters is no longer your FP32 line but your much higher **tensor-core** line. (Reference GPU figures below; use your own from `docs/my-gpu-spec.md`.)

General matrix multiply:

```text
C[M, N] = A[M, K] @ B[K, N]      C_ij = Σ_k A_ik * B_kj
```

FLOPs: `2 * M * N * K`. This is the kernel where the GPU's compute units finally become the bottleneck — *if* you tile correctly. Naively, GEMM is memory-bound and slow. The art is **reuse**: each element of `A` and `B` participates in many output elements, and tiling keeps it on-chip across all of them.

You implement, twice:

1. CUDA C++ tiled GEMM (shared-memory tiling; tensor cores optional/stretch)
2. Triton GEMM (`tl.dot`, autotuned)

Baseline: `torch.matmul` (cuBLAS).

Numbers from the [hardware spec sheet](reference-hardware-spec-sheet.md).

---

## 6.1 The Central Idea: Tiling Raises Intensity

Do the roofline arithmetic, because it explains *everything* about GEMM.

**Naive GEMM** (each thread computes one `C_ij`, reading a full row of `A` and column of `B` from global memory):

```text
FLOPs per output  = 2K
bytes per output  = read K of A + K of B + write 1 = (2K + 1) * 4 ≈ 8K
intensity         ≈ 2K / 8K = 0.25 FLOP/byte
```

`0.25` — far left of the ridge (17). Naive GEMM is **memory-bound** and wastes the compute units, just like everything before it. The fix is to stop re-reading `A` and `B` from global memory.

**Tiled GEMM:** a block loads a `T×T` tile of `A` and a `T×T` tile of `B` into shared memory, and computes a `T×T` block of `C` by looping over `K` in `T`-sized chunks. Per shared tile-pair you do `2*T³` FLOPs against `2*T²` loads:

```text
intensity ≈ 2T³ / (2T² * 4 bytes) = T / 4 FLOP/byte
```

**Intensity scales with tile size `T`.** This is the whole point. On the reference GPU the FP32 ridge is ~17 (use yours from `docs/my-gpu-spec.md`), so:

```text
T / 4 > 17   ⇒   T > 68      (reference GPU; your crossover scales with your ridge)
```

A `T = 32` tile gives intensity ~8 (still slightly memory-bound). Add **register blocking** — each thread computes a small `R×R` micro-tile of outputs, multiplying the effective tile and reuse — and you push intensity well past the ridge. *That* is how you climb the roofline: tiling and register-blocking convert a memory-bound problem into a compute-bound one by manufacturing reuse.

> This is the single most important performance idea in GPU computing. Memorize the shape of it: **reuse on-chip → higher arithmetic intensity → cross the ridge → compute-bound.**

---

## 6.2 Two Ceilings, and Why Tensor Cores Exist

Once you are compute-bound, *which* compute ceiling applies?

| Path | Peak (reference GPU) | When |
|---|---|---|
| FP32 CUDA cores | ~4.55 TFLOP/s | your hand-written FP32 GEMM |
| FP16 tensor cores | ~36 (fp16-acc) / ~18 (fp32-acc) TFLOP/s | `wmma` / `tl.dot` on FP16 inputs |

That is a **large gap** — multiples, not percentages. A perfectly optimized FP32 GEMM tops out at the FP32 line; the tensor cores do matrix-multiply-accumulate far faster. This gap is the entire reason tensor cores exist and the reason mixed-precision training is universal. Your `docs/my-gpu-spec.md` lists both of *your* ceilings. Your hand-tiled FP32 kernel is the *learning* target; the tensor-core path is the *performance* target.

**Dtype constraint — check `docs/my-gpu-spec.md`:** tensor cores only accept certain data types, and which ones depends on your architecture. On the reference Turing card that is **FP16 only** (no TF32/BF16/FP8); on Ampere you'd use **TF32/BF16**; on Ada/Hopper, **FP8**; on Blackwell, **FP4/FP6**. Use the highest-throughput dtype *your* card supports for `tl.dot` / `wmma`, and **accumulate in FP32** for accuracy. (On the reference card, `torch.matmul` fp16 measures ~18.5 TFLOP/s — not the ~36 fp16-accumulate figure — because GeForce Turing caps fp32-accumulate tensor throughput at half rate. Your measured number is the one that counts.)

---

## 6.3 The Tiled Algorithm (CUDA)

```text
__shared__ float As[T][T], Bs[T][T];
acc = 0
for (k0 = 0; k0 < K; k0 += T):
    As[ty][tx] = A[row][k0 + tx]      // coalesced load of an A tile
    Bs[ty][tx] = B[k0 + ty][col]      // coalesced load of a B tile
    __syncthreads()
    for (k = 0; k < T; ++k)
        acc += As[ty][k] * Bs[k][tx]  // all reads hit fast shared memory
    __syncthreads()
C[row][col] = acc
```

Everything you learned earlier shows up here:

- **Coalescing (Unit 3):** the global tile loads must be coalesced.
- **Shared memory + `__syncthreads()` (Unit 3):** stage tiles on-chip; sync between load and compute, and again before overwriting the tiles next iteration.
- **Bank conflicts (Unit 3):** the inner `As[ty][k]` / `Bs[k][tx]` access pattern can conflict; padding or careful layout applies.

Then **register blocking**: have each thread accumulate an `R×R` block of `C` in registers (e.g. `4×4 = 16` accumulators), loading each shared value once and reusing it across the micro-tile. This is what actually gets you near peak — it raises the compute-to-shared-load ratio the same way tiling raised the compute-to-global ratio.

---

## 6.4 Triton's `tl.dot`

Triton expresses GEMM at the tile level and maps `tl.dot` onto tensor cores automatically:

```python
# pid_m, pid_n select the C tile this program computes
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k0 in range(0, K, BLOCK_K):
    a = tl.load(a_tile_ptrs, mask=...)   # (BLOCK_M, BLOCK_K)
    b = tl.load(b_tile_ptrs, mask=...)   # (BLOCK_K, BLOCK_N)
    acc += tl.dot(a, b)                  # tensor-core MMA, fp32 accumulate
    # advance a/b pointers by BLOCK_K
c = acc.to(c_ptr.dtype.element_ty)
tl.store(c_tile_ptrs, c, mask=...)
```

Key points:

- `tl.dot` on FP16 inputs lowers to tensor-core MMA. Feed it FP16 `a`, `b`; accumulate in FP32.
- `BLOCK_M, BLOCK_N, BLOCK_K`, `num_warps`, `num_stages` are the tuning surface — this is the canonical `@triton.autotune` use case. `num_stages` controls software pipelining (overlapping the next tile's load with this tile's compute).
- This is essentially the official Triton matmul tutorial structure. Study it, then write it yourself.

---

## 6.5 CUDA Exercise Spec

Write, in order, and benchmark each:

1. **Naive** — one thread per `C_ij`, global reads. The slow, memory-bound baseline.
2. **Shared-memory tiled** — the §6.3 algorithm. Big jump.
3. **Register-blocked tiled** — each thread computes an `R×R` micro-tile. Approaches FP32 peak.
4. **(Stretch) Tensor-core** — `wmma::fragment` API, FP16 inputs, FP32 accumulate.

Requirements: handle non-tile-divisible `M`, `N`, `K`; verify against `torch.matmul`; report TFLOP/s as `2*M*N*K / seconds / 1e12`.

Sizes (keep within your VRAM): square `M=N=K ∈ {1024, 2048, 4096}`. At 4096 fp32, each matrix is 64 MB — fine on the 6 GB reference card; scale to yours.

> `torch.matmul` (cuBLAS) is the baseline and your north star. Reaching a respectable fraction of it is the goal.

---

## 6.6 Triton Exercise Spec

Write an autotuned `tl.dot` GEMM that:

- computes a `BLOCK_M × BLOCK_N` tile of `C` per program
- loops over `K` in `BLOCK_K` chunks, accumulating in FP32
- masks ragged edges in all three dimensions
- autotunes over `BLOCK_M/N/K`, `num_warps`, `num_stages`

Run it in **FP16 inputs / FP32 accumulate** to engage tensor cores, and compare against your FP32 CUDA kernels and cuBLAS.

---

## 6.7 Benchmark Requirements

Report per implementation:

| Field | Meaning |
|---|---|
| implementation | CUDA naive / tiled / reg-blocked / (wmma) / Triton fp16 / cuBLAS |
| dtype | fp32 or fp16 in, fp32 accumulate |
| shape | `M×N×K` |
| median time | µs |
| throughput | `2*M*N*K / s / 1e12` TFLOP/s |
| efficiency | vs your **relevant** measured ceiling (FP32 or tensor, `docs/my-gpu-spec.md`) **and** vs cuBLAS |
| correctness | rel error vs `torch.matmul` (looser tol for fp16) |

Always state which ceiling you grade against. An FP32 kernel hitting most of *your* FP32 ceiling is excellent. The *same absolute* TFLOP/s graded against your (much higher) tensor-core ceiling is a failure. Same number, opposite verdict, because the ceiling differs — which is exactly why the calibrator records both.

---

## 6.8 What Good Looks Like

- **Naive:** memory-bound, a small fraction of peak. Expected.
- **Tiled:** several× faster; now compute-bound-ish in FP32.
- **Register-blocked FP32:** a large fraction of your FP32 ceiling (70%+ is strong for hand-written).
- **Triton FP16 / tensor cores:** multiples of the FP32 result, graded against your measured tensor-core ceiling, and within striking distance of cuBLAS (matching cuBLAS is hard; 60–80% is a real achievement).
- You can state, for any result, which ceiling applies and why.

---

## 6.9 Common Mistakes

### Grading against the wrong ceiling

The headline trap. FP32 and tensor-core kernels have different ceilings (often a multiple apart). Always name which one, and read both from `docs/my-gpu-spec.md`.

### Using a dtype your tensor cores don't support

The dtype that engages tensor cores depends on your architecture (TF32/BF16 need Ampere+, FP8 needs Ada/Hopper, FP4 needs Blackwell). On the reference Turing card, only FP16 works — a TF32 `tl.dot` does nothing useful there. Check `docs/my-gpu-spec.md` and use the highest-throughput dtype your card actually supports. (Fix any tutorial that assumes a different architecture.)

### Forgetting the second `__syncthreads()`

You need a barrier after computing on a shared tile, *before* overwriting it with the next `K` chunk. Miss it and a fast thread corrupts the tile for a slow one.

### No register blocking, then wondering why you're at 20% of peak

Shared-memory tiling alone is bandwidth-limited on the shared/register boundary. Register blocking is what reaches peak.

### Accumulating in FP16

Long `K` reductions in FP16 lose accuracy fast. Accumulate in **FP32** even with FP16 inputs.

### Ignoring ragged edges

`M`, `N`, `K` are not always multiples of your tile. Mask/guard all three or corrupt the border.

---

## 6.10 Files

```text
src/unit06/
  gemm_triton.py   # student: autotuned tl.dot GEMM (fp16 in, fp32 acc)
  gemm_cuda.cu     # student: naive -> tiled -> register-blocked (-> wmma stretch)
  benchmark_gemm.py
```

---

## 6.11 Check Your Understanding

1. What is the arithmetic intensity of naive GEMM, and is it memory- or compute-bound?
2. Show that tiled GEMM has intensity ~`T/4`, and find the tile size that crosses the ridge.
3. Why does register blocking raise throughput beyond plain shared-memory tiling?
4. What are the two compute ceilings on this GPU, and what is the gap between them?
5. Which data types do *your* tensor cores support (check `docs/my-gpu-spec.md`), and which would you pick for `tl.dot`?
6. Why must you accumulate in FP32 even with FP16 inputs?
7. What do `num_stages` / software pipelining do in the Triton kernel?
8. Why is the *same* TFLOP/s number a success for FP32 and a failure for tensor cores?
9. Where are the two `__syncthreads()` in the tiled loop, and what does each protect?
10. How does GEMM reuse every concept from Units 2–5?

---

**Next:** Unit 07 — FlashAttention: the capstone, where online softmax + tiling kill the N² memory bottleneck
