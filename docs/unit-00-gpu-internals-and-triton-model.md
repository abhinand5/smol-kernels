# Unit 00: GPU Internals + Triton Mental Model

> "Triton thinks in tiles, not threads."

## 0.1 The SIMT Execution Model

GPUs are **massively parallel SIMT** (Single Instruction, Multiple Thread) machines.

**Key facts:**
- Thousands of threads run simultaneously on a single GPU
- Threads are grouped into **warps** (32 threads on NVIDIA GPUs)
- All threads in a warp execute the **same instruction** in lockstep
- If threads in a warp take different branches → **warp divergence** → both branches execute serially → slow

**Triton implications:**
- You don't explicitly control warps, but be aware of them
- `tl.where(condition, a, b)` is safe — it evaluates both branches and selects
- Avoid divergent `if/else` inside loops when possible

---

## 0.2 Memory Hierarchy — The Single Most Important Concept

This diagram determines whether every kernel you write is fast or slow:

```
           Register File (~256 KB / SM)
           ╱  ~0 cycles, infinite bandwidth
          ╱
      Shared Memory (48-164 KB / SM)
      ╱  ~20-30 cycles, ~200 TB/s aggregate
     ╱
   L2 Cache (~40 MB on A100)
   ╱  ~200 cycles
  ╱
Global Memory / HBM (GBs)
  ~400-600 cycles, ~900 GB/s (A100)
```

| Level | Size | Latency | Scope |
|-------|------|---------|-------|
| Registers | ~256 KB / SM | ~0 cycles | Per thread |
| Shared Memory | 48-164 KB / SM | ~20-30 cycles | Per block |
| L2 Cache | ~40 MB | ~200 cycles | All SMs |
| HBM (Global) | GBs | ~400-600 cycles | All SMs |

**The golden rule: Global memory is 20-30x slower than shared memory.**

Why tiling exists:
1. Load a chunk of data from slow global memory → fast shared memory
2. Compute on the chunk (many operations per byte loaded)
3. Write results back to global memory
4. Repeat for the next chunk

If a kernel is memory-bound (most are), you win or lose based on how well you manage this hierarchy.

---

## 0.3 Coalesced Memory Access

Global memory is accessed through **128-byte cache lines** (or 32-byte sectors).

- **Coalesced:** 32 threads in a warp access 32 consecutive 4-byte floats → **1 memory transaction** (128 bytes) → optimal
- **Strided:** 32 threads access every Nth element → multiple transactions → up to 32x slower
- **Random:** Chaos → worst case

**Triton implications:**
- `tl.load(ptr + tl.arange(0, BLOCK_SIZE))` → **coalesced** (consecutive addresses)
- `tl.load(ptr + tl.arange(0, BLOCK_SIZE) * stride)` → **strided** (potential problem)
- Always think: "What pattern does my memory access make?"

---

## 0.4 Occupancy

**Occupancy** = (active warps per SM) / (max warps per SM)

An SM can run many warps simultaneously, but resources are limited:
- **Registers:** Each block takes some registers. More registers per block → fewer blocks.
- **Shared memory:** Each block uses some SRAM. More SRAM per block → fewer blocks.
- **Max warps per SM:** Fixed by hardware (64 warps on Turing)

**Low occupancy** → SM can't hide memory latency as well → slower.
**High occupancy** → more warps to swap in while others wait for memory → faster.

**The tradeoff:** Larger `BLOCK_SIZE` usually means more registers/SRAM per block, which reduces occupancy. There's a sweet spot — this is why `@triton.autotune` exists.

> Note: On modern GPUs with high memory bandwidth, occupancy matters less than you'd think. But understanding it is essential for diagnosing performance.

---

## 0.5 The Triton Mental Model — Tiles, Not Threads

### The Critical Shift

If you know CUDA, **forget `threadIdx.x`**. You won't use it.

| Concept | CUDA | Triton |
|---------|------|--------|
| Grid | `gridDim.x` blocks | `grid` tuple in `kernel[grid](...)` |
| Block index | `blockIdx.x` | `tl.program_id(0)` |
| Thread index | `threadIdx.x` | **You don't have one** |
| Block size | `blockDim.x` threads | `BLOCK_SIZE` — a tile of elements |
| Data per program | Each thread loads 1 element | Each program loads a **tile** of BLOCK_SIZE elements |
| Thread management | Manual | Automatic — Triton decomposes your tile into threads |

### The Three Essential APIs

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(
    x_ptr,        # Pointer to input tensor
    y_ptr,        # Pointer to output tensor  
    n_elements: tl.constexpr,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # TILE SIZE (compile-time constant)
):
    # 1. WHICH BLOCK AM I?
    pid = tl.program_id(0)          # Like blockIdx.x

    # 2. WHICH ELEMENTS DO I OWN?
    offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE  # A range of indices

    # 3. LOAD A TILE (with bounds masking)
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # (operate on the tile — vectorized)
    y = x * 2.0

    # 4. STORE A TILE
    tl.store(y_ptr + offsets, y, mask=offsets < n_elements)
```

### Calling from Python

```python
def call_kernel(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    my_kernel[grid](x, y, n, BLOCK_SIZE=1024)
    return y
```

**Key details:**
- `grid` is a function that takes `meta` (autotune params) and returns `(grid_x, grid_y, grid_z)`
- `triton.cdiv(a, b)` = `(a + b - 1) // b` — ceiling division
- `BLOCK_SIZE` is `tl.constexpr` — must be known at compile time
- `mask=...` handles the last block where `offsets` may exceed `n_elements`
- `other=0.0` is the fill value for out-of-bounds elements

---

## 0.6 Hands-On: Bandwidth Benchmark

Let's make the memory hierarchy real by measuring it.

**Setup:** Create a temporary directory for this exercise, then delete it after.

```bash
mkdir -p ~/dev/cuda-learning/src/
```

**Your task:** Write a Triton kernel that:
1. Takes an input tensor `x` and an output tensor `y`
2. Copies `x` into `y` element-wise (a `memcpy`)
3. Benchmark it for various `BLOCK_SIZE` values

We'll measure bandwidth (GB/s) and see how close we get to the theoretical peak.

### Theoretical Peak for RTX 2060 Max-Q

- Memory clock: 11 GHz (GDDR6 effective)
- Bus width: 192-bit
- **Peak bandwidth:** 11e9 × 192 / 8 = **264 GB/s**

**File:** `src/unit00/bandwidth.py`

### Expected result pattern:
- Small `BLOCK_SIZE` (< 256): Low bandwidth (low efficiency, more overhead)
- Medium `BLOCK_SIZE` (512-2048): High bandwidth (near peak for a simple copy)
- Large `BLOCK_SIZE` (> 4096): Slightly lower (register pressure, fewer blocks)

---

## Check Your Understanding

Before moving to Unit 1, you should be able to answer:

1. Why is shared memory faster than global memory? (physical reason)
2. What makes a memory access pattern "coalesced"?
3. What is a warp and what happens during warp divergence?
4. What does `tl.program_id(0)` correspond to in CUDA?
5. Why does `BLOCK_SIZE` affect both occupancy and performance?
6. What is `tl.constexpr` and why must it be compile-time constant?
7. What happens if we remove the `mask` from `tl.load`?

---

**Next:** Unit 01 — Vector Add: Your First Real Kernel
