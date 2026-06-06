# Unit 02: Fused ReLU / GeLU

> Fusion is a memory optimization wearing a compute costume. The win is never the math you do — it is the global-memory round-trips you *don't*.

In Unit 1 you did one add per element and stayed memory-bound. Now you do an **activation** — `ReLU` and `GeLU` — and learn the single most important elementwise optimization in ML kernels: **fusion**. You will also meet `@triton.autotune` properly, and discover the **special function unit (SFU)** as a second bottleneck that hides behind "memory-bound."

You implement, twice:

1. CUDA C++ activation kernel(s)
2. Triton activation kernel(s) with autotuning

Baseline: `torch.relu`, `torch.nn.functional.gelu`.

Numbers come from the [hardware spec sheet](reference-hardware-spec-sheet.md).

---

## 2.1 The Functions

**ReLU** — trivial, branch-free with a max:

```text
relu(x) = max(x, 0)
```

**GeLU** — the activation in most transformers. Two common forms:

```text
exact:  gelu(x) = x * Φ(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
tanh:   gelu(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ))
```

ReLU is ~1 FLOP and a compare. GeLU is ~10–15 FLOPs **plus a transcendental** (`erf` or `tanh`). That difference is the whole lesson of §2.4.

---

## 2.2 Fusion: Why It Is a Memory Win

Activations rarely appear alone. The real pattern is `y = activation(x + bias)` or `y = activation(linear(x))`. Consider `y = relu(x + b)` done two ways.

**Unfused** — two kernels, each making a full round-trip to global memory:

```text
kernel 1 (add):   read x (4) + read b (4) + write t (4) = 12 bytes/elem
kernel 2 (relu):  read t (4) + write y (4)              =  8 bytes/elem
total traffic                                            = 20 bytes/elem
```

**Fused** — one kernel, the intermediate `t` never leaves registers:

```text
fused: read x (4) + read b (4) + write y (4) = 12 bytes/elem
```

You moved **12 instead of 20 bytes** — a 40% traffic cut — for the *exact same arithmetic*. On a memory-bound kernel, traffic *is* runtime. That is fusion: you delete global-memory round-trips by keeping intermediates on-chip.

> The general principle, which you will use through FlashAttention: **every elementwise op you can fold into the producer or consumer kernel is one fewer trip to VRAM.** ML kernel performance is mostly a fight to minimize global-memory traffic. Fusion is your primary weapon.

---

## 2.3 The Roofline (Still Left of the Ridge)

Plain activation, single input and output: `8` bytes/element (read 4, write 4).

| Function | FLOPs/elem (approx) | Intensity | vs ridge (~17, reference GPU) |
|---|---|---|---|
| ReLU | ~1 | ~0.13 FLOP/byte | memory-bound (~130× left) |
| GeLU (tanh) | ~12 + 1 transcendental | ~1.5 FLOP/byte | memory-bound (~11× left) |

Both are memory-bound on any GPU (check your ridge in `docs/my-gpu-spec.md`). But notice GeLU moved an order of magnitude *toward* the ridge — the first time in this curriculum that arithmetic intensity is not negligible. It is still well below the ridge, so memory traffic should still dominate... except for one thing.

---

## 2.4 The SFU: When "Memory-Bound" Lies

`exp`, `tanh`, `erf`, `sin`, `rsqrt` do **not** run on the regular FP32 ALUs. They run on the **Special Function Unit (SFU)**, of which each SM has only a few — roughly **1 SFU op per 4 (or more) FP32 ops** of throughput.

Experiment to run: benchmark ReLU and GeLU at the same `N`. They move the **identical** 8 bytes/element, so a naive roofline predicts identical runtime. Now interpret what you measure — and *both* outcomes teach the lesson:

- **GeLU measurably slower** → the SFU work did not fully hide under the memory traffic. You are watching a *second* throughput ceiling (transcendentals on the SFU) become the bottleneck.
- **GeLU ≈ ReLU** → on this card the `tanh`/`exp` work (a few hundred µs) overlapped with the ~1 ms of memory traffic at this `N`, so memory latency hid it. The SFU pressure is real but not *exposed* here.

Whether the gap appears depends on how much SFU work overlaps with memory — push `N` down (less memory to hide behind) or use the heavier `erf` form to expose it. The durable lesson: the roofline's "compute ceiling" assumes cheap ALU FLOPs; transcendentals run on the scarce SFU and form a separate limit that may or may not surface. When a memory-bound kernel is slower than its byte count predicts, suspect the SFU.

> Practical knobs: prefer the `tanh` GeLU approximation over `erf` if accuracy allows; use fast intrinsics (`__expf`, `tl.exp` which already maps to the fast path) when precision permits; and know that `erf`-based GeLU is the most SFU-heavy option.

---

## 2.5 Autotuning (Triton)

Block size, number of warps, and pipelining stages interact with occupancy in ways you cannot reason about precisely. So you **search**. `@triton.autotune` runs a set of configs once and caches the winner per input shape:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': bs}, num_warps=w)
        for bs in (256, 512, 1024, 2048, 4096)
        for w in (2, 4, 8)
    ],
    key=['n_elements'],   # re-tune when this changes
)
@triton.jit
def activation_kernel(...):
    ...
```

What to internalize:

- `key=[...]` lists the args that, when changed, trigger a re-tune. Get this wrong and you either re-tune constantly (slow) or use a stale config (also slow).
- The first call pays the search cost. Warm up before benchmarking or you measure autotuning.
- `num_warps` controls threads-per-program (warps × 32). It trades occupancy against per-program work.

---

## 2.6 CUDA Exercise Spec

Write CUDA kernels that:

- implement `relu(x)` and a fused `relu(x + b)` (or `gelu(x + b)`)
- compute the global index, guard bounds
- keep the intermediate in a register — **do not** write it to global memory between add and activation
- launch from a Python harness (`load_inline` or an extension)

Then, as an optimization, write a **vectorized** variant using `float4` loads/stores so each thread processes 4 contiguous elements per instruction. Compare its bandwidth to the scalar version — vectorized memory instructions can raise achieved bandwidth on memory-bound kernels.

Block sizes to test: `128, 256, 512, 1024`. Implement both ReLU and a GeLU variant so you can observe the SFU gap (§2.4).

> `torch.relu` / `F.gelu` are baselines, not your kernels.

---

## 2.7 Triton Exercise Spec

Write an autotuned Triton kernel that:

- accepts `x_ptr`, (optional) `b_ptr`, `y_ptr`, `n_elements`, `BLOCK_SIZE: tl.constexpr`
- builds masked offsets exactly as in Unit 1
- loads `x` (and `b`), computes the fused activation in-register, stores `y`
- is wrapped in `@triton.autotune` over block size and `num_warps`

Structural skeleton (shape, not solution):

```python
@triton.jit
def fused_act_kernel(x_ptr, b_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # offsets, mask = ...
    # x = tl.load(x_ptr + offsets, mask=mask)
    # b = tl.load(b_ptr + offsets, mask=mask)
    # t = x + b
    # y = <relu or gelu of t>   # compute in-register; never store t
    # tl.store(y_ptr + offsets, y, mask=mask)
```

Implement both ReLU and GeLU (`tanh` approximation is fine; `tl.tanh`, `tl.exp` available).

---

## 2.8 Benchmark Requirements

```text
N = 32 * 1024 * 1024 float32 elements
```

Report, per implementation:

| Field | Meaning |
|---|---|
| implementation | CUDA scalar / CUDA float4 / Triton (autotuned) / PyTorch |
| function | ReLU / GeLU |
| variant | unfused (2 kernels) / fused (1 kernel) |
| median time | µs |
| effective bandwidth | `bytes_moved / s / 1e9` (use the *correct* byte count per variant) |
| efficiency | vs your measured bandwidth (`docs/my-gpu-spec.md`) |
| correctness | max abs error vs PyTorch |

Mandatory experiments:

1. **Fused vs unfused** for `relu(x + b)`. Confirm the fused version is faster and that the speedup tracks the traffic ratio (≈ 20/12 ≈ 1.67×).
2. **ReLU vs GeLU** at identical N and byte count. Measure the SFU gap.

---

## 2.9 What Good Looks Like

- Plain ReLU (8 bytes/elem) approaches your Unit 0 memcpy bandwidth — same régime.
- Fused `relu(x+b)` beats unfused by roughly the traffic ratio, not by magic.
- If GeLU is slower than ReLU at equal bytes, you can *name why* (SFU); if it isn't, you can explain that too (the SFU work hid under memory). Either way you reason, not shrug.
- `float4` vectorization closes some of the gap to PyTorch on the scalar version.

---

## 2.10 Common Mistakes

### Writing the intermediate to global memory

If you store `x + b` to a temp buffer and read it back, you have un-fused your own kernel and thrown away the entire point of this unit.

### Reporting bandwidth with the wrong byte count

Fused and unfused move different numbers of bytes. Compare them by **time**, and compute each one's GB/s with its own correct traffic.

### Blaming memory for a GeLU slowdown

If GeLU is slower than ReLU at the *same* byte count, the difference is the SFU, not memory — don't "optimize memory" to fix a compute-unit bottleneck. (And if there's no slowdown, don't claim the SFU is free; it was hidden, not absent.)

### Measuring the autotune search

The first autotuned call runs every config. Warm up, then measure.

### erf vs tanh mismatch with the baseline

PyTorch `F.gelu` defaults to the **exact (erf)** form; `F.gelu(approximate='tanh')` is the tanh form. Match the baseline you compare against or your error check will fail for a non-bug reason.

---

## 2.11 Files

```text
src/unit02/
  fused_activation_triton.py   # student: autotuned Triton ReLU/GeLU
  fused_activation_cuda.cu     # student: scalar + float4 CUDA kernels
  benchmark_activation.py      # optional shared harness
```

---

## 2.12 Check Your Understanding

1. Why is fusion a *memory* optimization rather than a compute one?
2. For `y = relu(x + b)`, how many bytes/element does the fused version move vs the unfused?
3. What is the expected fused-vs-unfused speedup, and where does the number come from?
4. What is the arithmetic intensity of plain ReLU, and is it memory- or compute-bound here?
5. Under what condition is GeLU slower than ReLU despite equal bytes, and when might they measure equal?
6. What is the SFU and roughly what is its throughput relative to FP32 ALUs?
7. What does `key=[...]` control in `@triton.autotune`?
8. Why must you warm up before benchmarking an autotuned kernel?
9. What does `float4` vectorization buy a memory-bound kernel?
10. Which GeLU form does `torch.nn.functional.gelu` use by default, and why does that matter for your correctness check?

---

**Next:** Unit 03 — Matrix Transpose: shared memory, coalescing, and the bank-conflict you have to pad away
