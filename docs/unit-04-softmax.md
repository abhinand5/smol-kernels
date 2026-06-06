# Unit 04: Softmax

> The first kernel where threads must *cooperate* to compute one number. Reductions, numerical stability, and a one-pass trick that — three units from now — becomes the beating heart of FlashAttention.

Row-wise softmax over a matrix `X[M, N]`:

```text
softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

That `- max(x)` is not optional decoration — it is the difference between a working kernel and `inf/inf = NaN`. And computing `max` and `Σ` requires a **parallel reduction**: many threads collapsing a row into a single scalar. This is the unit where you learn warp shuffles, block reductions, and the numerical-stability discipline that every normalization kernel needs.

You implement, twice:

1. CUDA C++ row-softmax (warp + block reduction)
2. Triton row-softmax (tile reduction)

Baseline: `torch.softmax(x, dim=-1)`.

Numbers from the [hardware spec sheet](reference-hardware-spec-sheet.md).

---

## 4.1 Numerical Stability Is the Whole Game

The naive definition `exp(x_i) / Σ exp(x_j)` overflows: `exp(89.0)` in fp32 is already `inf`. Once any term is `inf`, the sum is `inf`, and the result is `NaN`. Subtracting the row max fixes it **exactly** (it is algebraically identical) while guaranteeing the largest exponent is `exp(0) = 1`:

```text
exp(x_i - m) / Σ exp(x_j - m)   where m = max(x)
```

Now every `exp` argument is `≤ 0`, so every term is in `(0, 1]`. No overflow, ever. **This subtraction is mandatory.** It is the canonical example of the rule: *a correct formula on paper can be a broken kernel in floating point.*

---

## 4.2 The Three Passes (and Why Traffic Matters)

A row softmax needs three quantities in order: the **max**, the **sum of shifted exps**, and the **normalized output**. The naive structure reads the row from global memory three times:

```text
pass 1: read row -> m = max
pass 2: read row -> s = Σ exp(x - m)
pass 3: read row -> y = exp(x - m) / s   (write row)
```

That is `3` reads + `1` write = `4N` element-traffic per row. If the row fits in shared memory or registers, you read it **once** into on-chip storage and do all three passes there — `1` read + `1` write = `2N`. Same fusion principle as Unit 2: minimize global round-trips. For large rows that don't fit on-chip, you genuinely need multiple global passes — or the one-pass trick in §4.5.

Bytes (fp32, single-read design): `8N` per row. FLOPs: a few per element plus an `exp` (SFU — recall Unit 2). Intensity is low → **memory-bound**, but the `exp` puts pressure on the SFU, so softmax often lands between "pure memory-bound" and "SFU-bound." Measure both régimes.

---

## 4.3 Parallel Reduction: Warp Shuffles

To compute `max` or `sum` across a row, many threads must combine their partial results into one. The fast path on NVIDIA GPUs is the **warp shuffle**: threads in a warp exchange registers directly, no shared memory, no `__syncthreads()`.

```text
// sum-reduce within a warp (32 lanes) — tree reduction
for (offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
// lane 0 now holds the warp's total
```

Two levels for a full block:

```text
1. Each warp reduces its 32 lanes via __shfl_down_sync  -> one partial per warp.
2. Warp leaders write partials to __shared__.
3. __syncthreads(); the first warp reduces the partials -> block result.
```

`max` is the same pattern with `max()` instead of `+`. This **warp-then-block** two-level reduction is the standard CUDA idiom; you will reuse it verbatim in Unit 5.

---

## 4.4 Triton Makes Reductions a One-Liner

Triton hides the shuffle/shared dance. A reduction over a tile axis is a single call:

```python
row = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
m   = tl.max(row, axis=0)
num = tl.exp(row - m)
s   = tl.sum(num, axis=0)
y   = num / s
```

Note `other=-inf` for masked lanes so they never win the `max`. The compiler lowers `tl.max` / `tl.sum` to exactly the warp-then-block reduction you wrote by hand in CUDA. Writing it both ways is the point: CUDA shows you the machinery, Triton shows you the abstraction over the *same* machinery.

One program per row works when `N` fits a tile. For very large `N`, you tile the row and combine partials — which leads directly to the next section.

---

## 4.5 Online Softmax: The Bridge to FlashAttention

What if you can't afford even one full pass to find the max before summing — because the row is streaming in blocks (exactly the situation in attention)? **Online softmax** computes the max and the sum in a **single pass**, rescaling the running sum whenever a new, larger max appears:

```text
m = -inf, s = 0
for each new value x:
    m_new = max(m, x)
    s     = s * exp(m - m_new) + exp(x - m_new)   # rescale old sum to the new max
    m     = m_new
```

The `s * exp(m - m_new)` correction retroactively re-bases the accumulated sum onto the new maximum. This single recurrence — **carry a running max and a running sum, rescale on max updates** — is the core of FlashAttention (Unit 7), where it lets attention stream over K/V blocks without ever materializing the full score row. Implement the two/three-pass version first for correctness; then implement the online version and verify it matches. You will be grateful in Unit 7.

---

## 4.6 CUDA Exercise Spec

Write a CUDA row-softmax that:

- assigns one block per row (a reasonable first design)
- loads the row, computes the **max** via warp+block reduction
- computes `Σ exp(x - max)` via a second reduction
- writes `exp(x - max) / sum`
- subtracts the max — non-negotiable
- handles `N` not a multiple of the block size

Then implement the **online (one-pass)** variant (§4.5) and verify it matches.

Sizes: rows of `N = 1024, 4096, 16384`; `M = 4096` rows.

> `torch.softmax` is the baseline.

---

## 4.7 Triton Exercise Spec

Write a Triton softmax that:

- uses one program per row (start here)
- loads the row with `other=-inf` on masked lanes
- uses `tl.max` / `tl.exp` / `tl.sum`
- subtracts the max before exponentiating

Skeleton (shape, not solution):

```python
@triton.jit
def softmax_kernel(x_ptr, y_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    # x = tl.load(x_ptr + row*row_stride + cols, mask=mask, other=-inf)
    # m = tl.max(x, axis=0); e = tl.exp(x - m); s = tl.sum(e, axis=0)
    # tl.store(y_ptr + row*row_stride + cols, e / s, mask=mask)
```

(Use `BLOCK_SIZE = next_power_of_2(n_cols)` for the single-tile design.)

---

## 4.8 Benchmark Requirements

Report per implementation:

| Field | Meaning |
|---|---|
| implementation | CUDA two-pass / CUDA online / Triton / PyTorch |
| shape | `M × N` |
| median time | µs |
| effective bandwidth | `2 * M * N * 4 / s / 1e9` |
| efficiency | vs 264 GB/s |
| correctness | max abs error vs `torch.softmax` |
| stability | passes with large inputs (e.g. `x += 100`) without NaN |

Mandatory: a **stability test** — add a large constant to the input and confirm your kernel still matches PyTorch (a kernel that forgot the max-subtraction will `NaN` here).

---

## 4.9 What Good Looks Like

- Matches `torch.softmax` to fp32 tolerance, including the large-input stability test.
- Effective bandwidth in the memory-bound régime; if it is far below memcpy, suspect extra global passes or SFU saturation from `exp`.
- The online variant matches the two-pass variant bit-for-bit-close, proving you understand the rescaling.

---

## 4.10 Common Mistakes

### Forgetting the max-subtraction

The single most common softmax bug. Works on small random data, `NaN`s on real logits. The stability test exists to catch it.

### Masked lanes polluting the reduction

A masked-out lane loaded as `0` (or garbage) can win a `max` or inflate a `sum`. Load `-inf` for max-reductions and `0` for sum-reductions on masked lanes.

### Reading the row more times than necessary

Three global passes when the row fits on-chip is wasted bandwidth. Read once, reduce on-chip.

### Bungling the two-level reduction

Warp-reduce, write partials to shared, sync, then reduce partials. Skipping the sync or the second level gives per-warp (not per-block) results.

### Online rescale applied in the wrong order

`s = s * exp(m_old - m_new) + exp(x - m_new)` — rescale the *old* sum, then add the new term, both based at `m_new`. Order and base matter.

---

## 4.11 Files

```text
src/unit04/
  softmax_triton.py   # student: Triton row-softmax
  softmax_cuda.cu     # student: two-pass + online CUDA softmax
  benchmark_softmax.py
```

---

## 4.12 Check Your Understanding

1. Why does naive softmax overflow, and how does subtracting the max fix it *exactly*?
2. What three quantities does a row softmax compute, and in what order?
3. Why can the number of global memory passes dominate softmax runtime?
4. What does `__shfl_down_sync` do, and why is it faster than a shared-memory reduction?
5. Describe the warp-then-block two-level reduction.
6. Why must masked lanes load `-inf` for a max-reduction?
7. Write the online-softmax update for the running max and sum.
8. What does the `s * exp(m_old - m_new)` term correct for?
9. Is softmax memory-bound or compute-bound on this GPU, and what role does the SFU play?
10. How does the online trick enable FlashAttention to avoid materializing the score matrix?

---

**Next:** Unit 05 — LayerNorm: two reductions at once, Welford's stable variance, and epilogue fusion
