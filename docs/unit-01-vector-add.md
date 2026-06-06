# Unit 01: Vector Add

> Your first real kernel. It does one add per element and is still completely memory-bound. Internalize why, because almost every kernel you write for the next month will be too.

In Unit 0 you moved bytes. Now you do the smallest possible amount of arithmetic on them:

```text
c[i] = a[i] + b[i]
```

Two reads, one add, one write. That is the entire kernel. The point is not the add. The point is to own the **launch configuration**, the **index math**, and the **roofline** so completely that you can predict the benchmark before you run it.

You will implement it twice:

1. CUDA C++ `__global__` kernel
2. Triton `@triton.jit` kernel

And you will check both against `torch.add` (or `a + b`), which is the baseline, not your kernel.

---

## 1.1 What Is New Since Unit 0

Unit 0 was a memcpy: one input, one output. Vector add changes three things that matter.

| Change | Unit 0 (copy) | Unit 1 (add) |
|---|---|---|
| Input pointers | one (`x`) | two (`a`, `b`) |
| Arithmetic | none | one add per element |
| Bytes per element (fp32) | 8 (read 4 + write 4) | 12 (read 4 + read 4 + write 4) |
| What you tune | block/tile size | block/tile size **and** work-per-thread |

That extra input pointer is the whole conceptual step in Triton: you now load *two* tiles, operate on them as vectors, and store one. In CUDA it is the first time `idx` indexes more than one array, so a single index bug corrupts your output in an obvious, debuggable way. Good.

---

## 1.2 The Roofline: Why This Is Memory-Bound

This is the most important section in the unit. Do the arithmetic by hand once and you will never again confuse "does math" with "compute-bound".

**Arithmetic intensity** is the ratio of useful work to memory traffic:

```text
intensity = FLOPs / bytes moved
```

For float32 vector add, per element:

```text
FLOPs        = 1   (one add)
bytes moved  = 12  (read a: 4, read b: 4, write c: 4)
intensity    = 1 / 12 ≈ 0.083 FLOP/byte
```

Now place that on the roofline. The worked numbers below are the **reference GPU** (RTX 2060 Max-Q) so the arithmetic is concrete — substitute your own from `docs/my-gpu-spec.md` (run `uv run python3 scripts/calibrate_gpu.py` first; method in the [reference spec sheet](reference-hardware-spec-sheet.md)):

| Quantity | Reference GPU (approx) |
|---|---|
| Peak memory bandwidth | ~264 GB/s |
| Peak FP32 throughput | ~4.55 TFLOP/s |
| Roofline ridge point | ~4550 / 264 ≈ **17 FLOP/byte** |

Your kernel sits at **0.083 FLOP/byte**. The reference ridge is **17**. You are more than two orders of magnitude to the *left* of the ridge, buried deep in the memory-bound region. Your card's ridge will differ — but at 0.083 FLOP/byte, *no* GPU's ridge could make this kernel compute-bound. That robustness is the point.

The consequence, stated as a number you can predict:

```text
max achievable FLOP rate = intensity * peak bandwidth
                         = 0.083 * 264e9
                         ≈ 22 GFLOP/s
```

Your GPU can do ~4550 GFLOP/s of FP32. This kernel will use roughly **0.5%** of that compute. The arithmetic unit is idle, waiting on memory, essentially the entire time. So you do **not** optimize the add. You optimize the memory traffic — and there is no traffic to remove, because every byte is read or written exactly once. That means a correct, coalesced vector add should land at essentially the **same effective bandwidth as your Unit 0 memcpy**. If it does not, you have a bug or a bad launch config.

> The lesson generalizes: most elementwise and reduction kernels are memory-bound. You earn the right to think about FLOPs only when intensity climbs past the ridge — which does not happen until GEMM (Unit 6).

---

## 1.3 Launch Configuration, Owned

In Unit 0 you launched one thread per element and moved on. Here you make the launch math explicit and deliberate, because everything after this unit assumes you have it cold.

### The one-element-per-thread launch

```text
threads_per_block = T          # 128, 256, 512, or 1024
num_blocks        = cdiv(N, T) # ceiling division, never (N + T - 1) / T by hand in Triton
total_threads     = num_blocks * T   # >= N, the tail is masked/guarded
```

Each thread computes a global index and handles exactly one element:

```text
idx = blockIdx.x * blockDim.x + threadIdx.x   # CUDA
if (idx < N) c[idx] = a[idx] + b[idx];
```

This is fine and you should benchmark it first. But it ties your grid size to `N`. The next pattern breaks that tie.

### The grid-stride loop

A grid-stride loop lets a **fixed-size grid** process an array of **any** size by striding through it:

```text
stride = gridDim.x * blockDim.x        # total threads launched
for (idx = global_thread_id; idx < N; idx += stride)
    c[idx] = a[idx] + b[idx];
```

Why you want this in your toolbox:

- **Decouples grid size from N.** You can launch a grid sized to your GPU's SM count (good occupancy) instead of to the data.
- **Each thread does more work**, amortizing launch and index-setup overhead across multiple elements.
- **One kernel handles every N**, including N larger than `2^31` without re-tuning the grid.
- It is the canonical CUDA idiom. You will see it in real codebases constantly.

Benchmark both. On a pure memory-bound copy/add, the grid-stride loop often matches one-element-per-thread; the win shows up more on smaller or oddly-sized inputs and as a habit that pays off later. Measure, do not assume.

> **Index type trap:** `blockIdx.x * blockDim.x + threadIdx.x` is computed in `int` by default. For N near or above `2^31` (≈ 2.1 billion elements) this overflows. Use `size_t` or `int64_t` for the index when N can be large. At our default N = 32M this is safe, but write the code as if it will not be.

---

## 1.4 CUDA Exercise Spec

Write a CUDA implementation that:

- accepts input pointers `a`, `b`, output pointer `c`, and element count `n`
- computes the global index from `blockIdx`, `blockDim`, `threadIdx`
- guards out-of-bounds access with an explicit `if (idx < n)`
- computes `c[idx] = a[idx] + b[idx]`
- is launched from a Python harness (via a PyTorch C++/CUDA extension or `load_inline`) or a standalone C++ harness

Then write a **second** version using a grid-stride loop, and benchmark both.

Block sizes to test, minimum:

```text
128, 256, 512, 1024
```

Rules:

- Use a ceiling-division helper for `num_blocks`. Do not hide the launch math from yourself.
- `a + b` in PyTorch is the **baseline**, not your kernel. Do not measure it as your own work.
- Keep `a`, `b`, `c` the same dtype, shape, and device.

---

## 1.5 Triton Exercise Spec

Write a Triton kernel `add_kernel` that:

- accepts `a_ptr`, `b_ptr`, `c_ptr`, `n_elements`, and a `BLOCK_SIZE` constexpr
- computes `pid = tl.program_id(0)`
- builds consecutive offsets: `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`
- builds a tail mask: `mask = offsets < n_elements`
- loads `a` and `b` tiles with the mask
- adds them as vectors
- stores the result to `c` with the mask

Launch with `grid = (triton.cdiv(n_elements, BLOCK_SIZE),)`.

Tile sizes to benchmark:

```text
128, 256, 512, 1024, 2048, 4096, 8192
```

The structural skeleton (this is shape, not solution — you write the body):

```python
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # offsets = ...
    # mask    = ...
    # a = tl.load(...)
    # b = tl.load(...)
    # tl.store(c_ptr + offsets, a + b, mask=mask)
```

> Once this works, try `@triton.autotune` over the block sizes. You will meet autotuning properly in Unit 2; previewing it here is free.

---

## 1.6 Benchmark Requirements

Use a size that fits comfortably in your VRAM (the reference card has 6 GB; scale `N` to yours). Vector add needs **three** arrays, not two:

```text
N = 32 * 1024 * 1024 float32 elements
a = 128 MB, b = 128 MB, c = 128 MB   -> 384 MB resident
traffic per call = 12 * N bytes = 384 MB moved
```

For each implementation, report:

| Field | Meaning |
|---|---|
| implementation | CUDA 1-per-thread / CUDA grid-stride / Triton / PyTorch baseline |
| block or tile size | threads per block or `BLOCK_SIZE` |
| median time | microseconds |
| effective bandwidth | `12 * N / seconds / 1e9` GB/s |
| efficiency | measured bandwidth / your measured BW (`docs/my-gpu-spec.md`) |
| correctness | max abs error vs `a + b` |

Rules (same hygiene as Unit 0, do not regress):

- Allocate `a`, `b`, `c` once, outside the timed region.
- Warm up before measuring.
- Time with CUDA events or `triton.testing.do_bench`, not naive wall clock.
- Verify correctness after every configuration.
- Print device name, dtype, and N.

---

## 1.7 What Good Looks Like

Predict before you run, then confront the numbers:

- Your best vector add should reach roughly the **same effective GB/s as your Unit 0 memcpy** (give or take ~10%). Both are memory-bound and move every byte once.
- Tiny block/tile sizes underperform (not enough warps in flight to hide latency).
- Mid sizes (256–1024 threads, 1024–4096 tile) are usually best.
- Huge tiles do not keep helping and can hurt occupancy.
- PyTorch's `a + b` is a strong, tuned baseline. Matching it within ~10–20% is a win at this stage. Beating it is not the goal yet.

If your add is dramatically slower than your copy, suspect: a missing/incorrect mask, a non-coalesced index, accidental host-device copies inside the timed region, or measuring allocation.

---

## 1.8 Common Mistakes

### Counting the bytes wrong

Vector add moves **12** bytes per fp32 element, not 8. Two reads plus one write. Get this wrong and your reported GB/s is off by 50%.

### Believing "more math" means "compute-bound"

It does not. One add at intensity 0.083 is still memory-bound by two orders of magnitude (see 1.2). Adding arithmetic does not move you off the memory roofline until the intensity crosses ~19 FLOP/byte.

### Index overflow on large N

`int` index math overflows past ~2.1B elements. Use `size_t`/`int64_t` for indices in code meant to be general.

### Wrong grid-stride stride

The stride must be `gridDim.x * blockDim.x` (total threads), not `blockDim.x`. A wrong stride either skips elements or re-processes them.

### Aliasing assumptions

Do not assume `c` can safely alias `a` or `b` unless you have reasoned about it. For a pure elementwise add it happens to be safe, but make that a decision, not an accident.

### Timing async work without synchronization

Still true, still the most common benchmark bug. GPU launches are asynchronous; synchronize or use CUDA events / `do_bench`.

---

## 1.9 Files

Suggested layout:

```text
src/unit01/
  vector_add_triton.py     # student-written Triton kernel + harness
  vector_add_cuda.cu       # student-written CUDA kernel(s): 1-per-thread + grid-stride
  benchmark_add.py         # optional shared Python benchmark harness
```

The student writes every kernel body. Harnesses, correctness checks, and build glue may be assisted.

---

## 1.10 Check Your Understanding

Before Unit 2, answer these without looking:

1. How many bytes does float32 vector add move per element, and why?
2. What is the arithmetic intensity of vector add, and where does it sit relative to your roofline ridge point (`docs/my-gpu-spec.md`)?
3. Given your measured bandwidth, what is the maximum FLOP/s this kernel can achieve, and what fraction of your peak compute is that?
4. Write the global thread index expression in CUDA.
5. What does a grid-stride loop buy you over one-element-per-thread?
6. What must the stride in a grid-stride loop equal, and why?
7. When does `int` index math become a bug, and what do you use instead?
8. Why should a correct vector add reach roughly the same bandwidth as a memcpy?
9. In Triton, how do you load two input tiles and why do both loads need the same mask?
10. Why is `a + b` in PyTorch a baseline rather than your kernel?

---

**Next:** Unit 02 — Fused ReLU / GeLU: elementwise fusion, autotuning, and why fusion is a *memory* optimization
