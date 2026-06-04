# Unit 00: GPU Internals + CUDA/Triton Kernel Model

> CUDA thinks in grids, blocks, warps, and threads. Triton thinks in programs and tiles. The hardware is the same.

This unit makes the memory hierarchy real by implementing the same bandwidth benchmark twice:

1. CUDA C++ memcpy kernel
2. Triton memcpy kernel

The goal is not cleverness. The goal is to move bytes correctly, measure bandwidth, and understand what the machine is doing.

---

## 0.1 The GPU Execution Model

NVIDIA GPUs execute many lightweight threads using **SIMT**: Single Instruction, Multiple Threads.

Key facts:

- Threads are grouped into **warps** of 32 threads.
- A warp executes one instruction stream in lockstep.
- If threads in the same warp branch differently, the warp serializes paths. This is **warp divergence**.
- Threads are grouped into **blocks**.
- Blocks are scheduled onto **SMs**: streaming multiprocessors.
- Blocks do not assume ordering with other blocks.

CUDA exposes this directly:

```text
grid -> blocks -> threads
```

Triton hides explicit threads and exposes:

```text
grid -> programs -> vectorized tile operations
```

But Triton programs still compile down to GPU execution using warps, registers, memory instructions, and scheduling rules.

---

## 0.2 CUDA vs Triton Mental Model

| Concept | CUDA C++ | Triton |
|---|---|---|
| Kernel marker | `__global__` | `@triton.jit` |
| Program/block id | `blockIdx.x` | `tl.program_id(0)` |
| Thread id | `threadIdx.x` | hidden |
| Block size | `blockDim.x` threads | `BLOCK_SIZE` tile elements |
| Grid size | launch config `<<<grid, block>>>` | `kernel[grid](...)` |
| Bounds check | explicit `if (idx < n)` | mask on `tl.load` / `tl.store` |
| Memory access | one or more elements per thread | vector of offsets per program |
| Ceiling division | usually helper macro/function | `triton.cdiv(a, b)` |

CUDA teaches you what the GPU is doing.

Triton teaches you how modern ML kernels are often written: tile first, thread mapping second.

You need both.

---

## 0.3 Memory Hierarchy

This is the performance hierarchy you will keep coming back to:

```text
Registers       fastest, per thread, tiny
Shared memory   fast, per block, explicitly managed in CUDA
L1 / L2 cache   hardware managed
Global memory   VRAM, large, high bandwidth, high latency
Host memory     CPU RAM, accessed over PCIe when copied or mapped
```

For the local RTX 2060 Max-Q:

| Item | Value |
|---|---|
| Compute capability | 7.5 |
| VRAM | 6 GB |
| Theoretical memory bandwidth | about 264 GB/s |
| Max shared memory per block | 64 KB hardware limit, lower defaults may apply |
| Tensor cores | FP16/TF32 era features; no FP8 |

The golden rule:

> Most beginner kernels are memory-bound. If memory access is bad, nothing else matters.

---

## 0.4 Coalesced Memory Access

A warp is happiest when its threads access consecutive memory.

Good pattern:

```text
thread 0 -> x[0]
thread 1 -> x[1]
thread 2 -> x[2]
...
thread 31 -> x[31]
```

Bad pattern:

```text
thread 0 -> x[0]
thread 1 -> x[1024]
thread 2 -> x[2048]
...
```

In CUDA, coalescing comes from how each thread computes `idx`.

In Triton, coalescing comes from offsets like:

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

Consecutive offsets are good. Random or strided offsets need justification.

---

## 0.5 Occupancy, Briefly

**Occupancy** is how many warps are active on an SM relative to the maximum.

Limited by:

- registers per thread
- shared memory per block
- threads per block
- hardware limits

Low occupancy can fail to hide memory latency.

High occupancy is not automatically fast.

The practical rule for now:

> Try several block/tile sizes and measure.

---

## 0.6 Unit Exercise: Bandwidth Benchmark

Write two kernels that copy one tensor into another:

```text
y[i] = x[i]
```

Implement both:

1. CUDA C++ kernel
2. Triton kernel

Then benchmark both for multiple block/tile sizes.

### Why memcpy?

Because it has almost no math. Performance mostly reflects memory throughput and launch overhead.

For float32 copy:

```text
bytes moved = N * 4 bytes read + N * 4 bytes written
            = N * 8 bytes
```

Bandwidth:

```text
GB/s = bytes_moved / seconds / 1e9
```

---

## 0.7 CUDA Exercise Spec

Create a CUDA implementation that:

- accepts input pointer `x`, output pointer `y`, and element count `n`
- computes global index from block and thread IDs
- guards out-of-bounds access
- copies one or more elements per thread
- is launched from a Python benchmark harness or standalone C++ harness

You should understand every part of the launch configuration:

```text
threads_per_block
num_blocks
```

Use a helper for ceiling division. Do not hide the launch math from yourself yet.

Minimum block sizes to test:

```text
128, 256, 512, 1024
```

Optional later experiment:

- one element per thread
- multiple elements per thread using a grid-stride loop

Do not use `cudaMemcpy` as your measured kernel. `cudaMemcpy` is the library baseline, not your kernel.

---

## 0.8 Triton Exercise Spec

Create a Triton implementation that:

- accepts `x_ptr`, `y_ptr`, `n_elements`, and `BLOCK_SIZE`
- uses `tl.program_id(0)`
- creates consecutive offsets with `tl.arange`
- uses a mask for the tail
- loads from `x_ptr`
- stores to `y_ptr`

Benchmark at least:

```text
128, 256, 512, 1024, 2048, 4096, 8192
```

Use `triton.cdiv(n, BLOCK_SIZE)` for grid size.

---

## 0.9 Benchmark Requirements

Use a tensor size that fits comfortably in 6 GB VRAM.

Recommended starting point:

```text
N = 32 * 1024 * 1024 float32 elements
input  = 128 MB
output = 128 MB
traffic per copy = 256 MB
```

For each implementation, report:

| Field | Meaning |
|---|---|
| implementation | CUDA / Triton / PyTorch baseline |
| block or tile size | threads per block or `BLOCK_SIZE` |
| median time | microseconds or milliseconds |
| bandwidth | GB/s |
| efficiency | measured bandwidth / 264 GB/s |
| correctness | max error vs input |

Benchmark rules:

- warm up before measuring
- synchronize before/after timing if using wall clock
- prefer CUDA events or `triton.testing.do_bench`
- verify correctness after every run
- print device name and dtype

---

## 0.10 Expected Pattern

You should usually see:

- very small block/tile sizes: worse bandwidth
- medium sizes: best bandwidth
- huge sizes: not necessarily better
- PyTorch/library copy may be strong because it uses tuned paths

Do not chase peak bandwidth yet. First chase correctness and clean measurement.

---

## 0.11 Common Mistakes

### Forgetting bounds checks

Last block/tile may run past `n`.

CUDA uses an `if` guard.

Triton uses `mask=`.

### Timing async GPU work incorrectly

GPU launches are asynchronous. If you time without synchronization, your timing is garbage.

### Measuring allocation

Do not include `torch.empty_like`, random tensor creation, or compilation time in the measured region.

### Comparing different amounts of work

Make sure CUDA, Triton, and PyTorch copy the same `N`, dtype, and device tensors.

### Trusting one run

Use warmup and median over repeated measurements.

---

## 0.12 Files

Suggested files:

```text
src/unit00/
  bandwidth_triton.py      # student-written Triton kernel + benchmark
  bandwidth_cuda.cu        # student-written CUDA kernel
  benchmark_bandwidth.py   # optional shared Python benchmark harness
```

The current starter file is:

```text
src/unit00/bandwidth.py
```

It may be kept as the Triton starter or split into the suggested files above.

---

## 0.13 Check Your Understanding

Before Unit 1, answer these without looking:

1. What is a CUDA block?
2. What is a warp?
3. What is warp divergence?
4. What does `blockIdx.x * blockDim.x + threadIdx.x` compute?
5. What does `tl.program_id(0)` correspond to conceptually?
6. Why do CUDA kernels need bounds checks?
7. Why do Triton loads/stores need masks?
8. What is coalesced memory access?
9. Why is naive timing often wrong for GPU kernels?
10. Why is bandwidth counted as read bytes plus written bytes?

---

**Next:** Unit 01 — Vector Add in CUDA and Triton
