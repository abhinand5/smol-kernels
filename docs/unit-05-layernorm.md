# Unit 05: LayerNorm

> Two reductions in one pass, a variance formula that lies in floating point, and an epilogue you fold in for free. LayerNorm is softmax's sibling — same reduction machinery, harder numerics.

LayerNorm normalizes each row to zero mean and unit variance, then applies a learned affine transform:

```text
y = (x - μ) / sqrt(σ² + ε) * γ + β
```

where `μ` and `σ²` are the **mean and variance of the row**, `γ` (scale) and `β` (shift) are learned per-feature vectors, and `ε` guards the division. You already have the reduction toolkit from Unit 4. The new content: computing **two statistics at once** stably (the naive variance formula is numerically dangerous), and **epilogue fusion** — folding the affine transform into the same kernel for free.

You implement, twice:

1. CUDA C++ LayerNorm (fused reduction + affine)
2. Triton LayerNorm

Baseline: `torch.nn.functional.layer_norm`.

Numbers from the [hardware spec sheet](reference-hardware-spec-sheet.md).

---

## 5.1 Two Statistics, One Pass

LayerNorm needs both `μ` and `σ²`. The textbook variance identity tempts you:

```text
σ² = E[x²] - E[x]²          # ONE pass: accumulate Σx and Σx² together
```

It is one pass and it is **numerically treacherous**. When the mean is large relative to the variance, `E[x²]` and `E[x]²` are two big nearly-equal numbers, and subtracting them is **catastrophic cancellation** — you lose most of your significant digits, and `σ²` can even come out *negative*. Do not ship this naively.

Two safer options:

**Two-pass (simple, stable):**
```text
pass 1: μ = (Σ x) / N
pass 2: σ² = (Σ (x - μ)²) / N
```
Stable, easy, but reads the row twice (or once if held on-chip).

**Welford (one pass, stable):** maintain running count, mean, and `M2` (sum of squared deviations):
```text
for each x:
    count += 1
    delta  = x - mean
    mean  += delta / count
    M2    += delta * (x - mean)   # uses the UPDATED mean
variance = M2 / count
```

Welford gets one-pass *and* stability. There is a parallel-merge form (combine two partial `(count, mean, M2)` triples) that lets you use the same warp/block reduction as Unit 4 — you reduce triples instead of scalars. Implement two-pass first for correctness, then Welford to feel the difference.

---

## 5.2 Epilogue Fusion

Once you have `μ` and `σ²`, the normalize-and-affine step is pure elementwise work:

```text
y_i = (x_i - μ) * rstd * γ_i + β_i      where rstd = 1 / sqrt(σ² + ε)
```

The point: **do not launch a second kernel for the affine.** You already have the row (and `μ`, `σ²`) on-chip — apply `γ`/`β` right there before the store. This is the **epilogue**: the tail-end elementwise transform fused into the producer kernel. Same lesson as Unit 2's fusion, now applied to the *output* side of a reduction. A separate affine kernel would re-read `y` and `γ` and `β` from global memory for nothing.

`γ` and `β` are length-`N` and **shared across all `M` rows**, so they stay hot in L2/cache — their traffic is amortized. The dominant traffic is still `read x` + `write y` = `8N` per row → **memory-bound** (intensity is low; `rsqrt` touches the SFU but only once per row, not per element).

---

## 5.3 Backward Pass (Know It Exists)

Training needs `dx`, `dγ`, `dβ`. The backward is a *harder* fused reduction: `dγ` and `dβ` are reductions **across rows** (accumulate over `M`), while `dx` needs two more per-row reductions. You are not required to implement backward to pass this unit, but you should be able to explain why it is the more interesting kernel and why naive implementations serialize on the cross-row `dγ`/`dβ` accumulation (atomics or a two-stage reduction). Production LayerNorm kernels (you will read one in Unit 8) spend most of their cleverness here.

---

## 5.4 CUDA Exercise Spec

Write a CUDA LayerNorm (forward) that:

- assigns one block per row
- computes `μ` and `σ²` with a reduction (two-pass first; then Welford with the parallel-merge reduction)
- computes `rstd = rsqrt(σ² + ε)`
- applies `(x - μ) * rstd * γ + β` as a **fused epilogue** before storing
- handles `N` not a multiple of block size, and the affine vectors `γ`, `β`

Sizes: `M = 4096` rows, `N = 1024, 4096, 16384` features. `ε = 1e-5`.

> `F.layer_norm` is the baseline.

---

## 5.5 Triton Exercise Spec

Write a Triton LayerNorm that:

- uses one program per row
- loads the row, computes mean and variance with `tl.sum` (compute `σ²` as `mean(x²) - mean(x)²` only if you've checked stability for your data range, else two-pass `mean((x-μ)²)`)
- loads `γ`, `β`
- stores the fused affine result

Skeleton (shape, not solution):

```python
@triton.jit
def layernorm_kernel(x_ptr, y_ptr, g_ptr, b_ptr, row_stride, N, eps, BLOCK: tl.constexpr):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    # x    = tl.load(...)
    # mu   = tl.sum(x, axis=0) / N
    # var  = tl.sum((x - mu)*(x - mu), axis=0) / N
    # rstd = 1 / tl.sqrt(var + eps)
    # g, b = tl.load(g_ptr+cols, ...), tl.load(b_ptr+cols, ...)
    # y    = (x - mu) * rstd * g + b
    # tl.store(...)
```

---

## 5.6 Benchmark Requirements

Report per implementation:

| Field | Meaning |
|---|---|
| implementation | CUDA two-pass / CUDA Welford / Triton / PyTorch |
| shape | `M × N` |
| median time | µs |
| effective bandwidth | `2 * M * N * 4 / s / 1e9` (x in, y out; γ/β amortized) |
| efficiency | vs 264 GB/s |
| correctness | max abs error vs `F.layer_norm` |
| stability | passes with large-mean input (e.g. `x + 1e4`) |

Mandatory: a **stability test** with a large row mean. A naive `E[x²] - E[x]²` kernel will produce wrong (or negative) variance here; two-pass and Welford will not.

---

## 5.7 What Good Looks Like

- Matches `F.layer_norm` to fp32 tolerance, including the large-mean stability test.
- Memory-bound bandwidth comparable to softmax (same `8N` traffic shape).
- The fused-epilogue version beats any "normalize then separate affine kernel" approach you might write to compare.
- Welford and two-pass agree; the naive identity diverges on the stability test (demonstrate this on purpose).

---

## 5.8 Common Mistakes

### Naive `E[x²] - E[x]²` variance

Catastrophic cancellation for large means. The stability test catches it. Use two-pass or Welford.

### Forgetting `ε` inside the sqrt

`1/sqrt(σ² + ε)`, not `1/(sqrt(σ²) + ε)`. The epsilon is *inside*.

### Launching a separate affine kernel

Throws away the epilogue-fusion win and re-reads everything from global memory.

### Wrong `γ`/`β` indexing

`γ`, `β` are indexed by **feature (column)**, shared across rows. Index them by `cols`, not by row.

### Welford merge with the wrong formula

The parallel merge of two `(count, mean, M2)` triples has a specific correction term (`delta² * nA*nB/nAB`). Getting it wrong makes Welford silently inaccurate.

---

## 5.9 Files

```text
src/unit05/
  layernorm_triton.py   # student: Triton LayerNorm
  layernorm_cuda.cu     # student: two-pass + Welford CUDA LayerNorm
  benchmark_layernorm.py
```

---

## 5.10 Check Your Understanding

1. What two statistics does LayerNorm compute per row?
2. Why is `σ² = E[x²] - E[x]²` numerically dangerous, and when specifically?
3. Write Welford's update for mean and M2.
4. What is epilogue fusion, and what traffic does it save here?
5. Why do `γ` and `β` contribute little to effective bandwidth despite being read every row?
6. Where does the `ε` go, and what does it protect?
7. Is LayerNorm forward memory- or compute-bound, and what is the dominant traffic?
8. Why is the backward pass a harder kernel than the forward?
9. How does the parallel-merge form of Welford reuse the Unit 4 reduction?
10. How would you demonstrate, with a benchmark, that the naive variance formula is wrong?

---

**Next:** Unit 06 — Tiled GEMM: where tiling finally pushes you past the roofline ridge and tensor cores enter the picture
