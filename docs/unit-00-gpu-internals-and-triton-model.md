# Unit 00: GPU Foundations — The Machine, the Model, and Your First Benchmark

> Before you write a fast kernel, you have to know three things cold: what machine you are programming, how it executes your code, and how to tell whether you made it faster. This unit installs all three, then makes you prove it by measuring memory bandwidth two different ways.

This is the foundation everything else stands on. It has two halves:

1. **The machine and the model** — what a GPU is, how SIMT execution works, the memory hierarchy, and the one mental model (the roofline) that explains *why* every later kernel is the way it is.
2. **The first exercise** — implement a memcpy-style bandwidth benchmark in CUDA and in Triton, and measure how close to peak memory bandwidth you can get.

The goal is not cleverness. The goal is to move bytes correctly, measure honestly, and understand what the machine is doing — with no hand-waving.

---

## 0.1 What You Need Installed

Required local baseline:

- Linux machine with an NVIDIA GPU
- NVIDIA driver visible through `nvidia-smi`
- CUDA toolkit visible through `nvcc --version`
- Python environment managed by `uv`
- PyTorch with CUDA support, and Triton

On this repo's machine: CachyOS / Arch, RTX 2060 Max-Q (compute capability 7.5), 6 GB VRAM, CUDA 13.2, Python 3.14, PyTorch 2.12, Triton 3.7. The full performance spec — clocks, TFLOP/s, tensor-core data types, bandwidth, ridge points — lives in the [hardware spec sheet](reference-hardware-spec-sheet.md), the single source of truth this curriculum grades against.

Check your machine before going further:

```bash
nvidia-smi
nvcc --version
uv run python3 - <<'PY'
import torch, triton
print("torch", torch.__version__)
print("triton", triton.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))
PY
```

---

## 0.2 Two Kinds of "Thread"

The word **thread** means different things in OS land and GPU land, and conflating them is the first conceptual trap.

| Concept | CPU / OS thread | GPU / CUDA thread |
|---|---|---|
| Owner | OS scheduler | GPU hardware scheduler |
| Typical count | tens to thousands | thousands to millions |
| Execution | independent instruction stream | SIMT — groups execute together |
| Memory | process virtual address space | global / shared / register spaces |
| Synchronization | mutexes, condition vars, atomics | barriers, atomics, warp/block rules |
| Main cost | context switches, cache pollution, locking | memory access pattern, occupancy, divergence |

A CPU thread is a heavyweight OS-managed context (registers, stack, program counter, scheduling state). A CUDA thread is a lightweight hardware execution lane inside a hierarchy:

```text
grid -> blocks -> warps -> threads
```

Triton hides explicit thread indexing and asks you to think in **programs/tiles**, but the generated code still runs on GPU threads and warps underneath. You will see both views in §0.4–0.5.

---

## 0.3 The OS Boundary, and Why Tiny Kernels Are Slow

A normal program runs in **user mode**; the OS kernel and the GPU driver run in **kernel mode**. Allocating memory, launching work, and talking to the hardware all cross that boundary, and crossing it is not free.

Consequences you must respect:

- Kernel **launches have overhead** (driver/runtime cost before a single byte moves).
- System and driver calls cost real time.
- Benchmarking a *tiny* kernel can accidentally measure launch overhead instead of GPU throughput.

> GPUs are fast at bulk parallel work. They are **not** magic for tiny work. A kernel over 1,000 elements may be dominated by launch overhead; the same kernel over 32M elements is dominated by memory bandwidth. This is why the bandwidth exercise uses a large `N`.

---

## 0.4 The GPU Execution Model: SIMT

NVIDIA GPUs execute many lightweight threads using **SIMT**: Single Instruction, Multiple Threads.

- Threads are grouped into **warps** of 32. A warp executes one instruction stream in lockstep.
- If threads in a warp branch differently, the warp serializes the paths — **warp divergence**, a performance cost.
- Threads are grouped into **blocks**; blocks are scheduled onto **SMs** (streaming multiprocessors).
- Blocks make **no ordering assumptions** about other blocks.

CUDA exposes this directly:

```text
grid -> blocks -> threads
```

Triton hides explicit threads and exposes:

```text
grid -> programs -> vectorized tile operations
```

But a Triton program still compiles down to warps, registers, memory instructions, and scheduling rules. Same hardware, two abstractions.

---

## 0.5 CUDA vs Triton Mental Model

| Concept | CUDA C++ | Triton |
|---|---|---|
| Kernel marker | `__global__` | `@triton.jit` |
| Program / block id | `blockIdx.x` | `tl.program_id(0)` |
| Thread id | `threadIdx.x` | hidden |
| Block size | `blockDim.x` threads | `BLOCK_SIZE` tile elements |
| Grid size | launch config `<<<grid, block>>>` | `kernel[grid](...)` |
| Bounds check | explicit `if (idx < n)` | mask on `tl.load` / `tl.store` |
| Memory access | one or more elements per thread | a vector of offsets per program |
| Ceiling division | a helper macro/function | `triton.cdiv(a, b)` |

CUDA teaches you what the GPU is doing. Triton teaches you how modern ML kernels are often written: **tile first, thread mapping second.** You need both — that is the entire premise of this curriculum.

---

## 0.6 The Memory Hierarchy

This is the performance hierarchy you will return to in every unit:

```text
Registers       fastest, per thread, tiny
Shared memory   fast, per block, explicitly managed in CUDA (the compiler manages it in Triton)
L1 / L2 cache   hardware managed
Global memory   VRAM, large, high bandwidth, high latency
Host memory     CPU RAM, reached over PCIe when copied or mapped
```

GPU programming adds memory *categories* the CPU world doesn't foreground:

| Memory | Where | Notes |
|---|---|---|
| Host pageable | CPU RAM | normal allocations |
| Host pinned | CPU RAM | page-locked; faster async H2D/D2H copies |
| Global | VRAM | large, high bandwidth, high latency |
| Shared | on-chip, per block | small, fast, programmer-managed in CUDA |
| Registers | on-chip, per thread | fastest, limited |

For this repo, assume tensors live on the GPU unless stated otherwise (`x = torch.randn(1024, device="cuda")`). The headline numbers for the local card (from the [spec sheet](reference-hardware-spec-sheet.md)): ~264 GB/s theoretical bandwidth, up to 64 KB shared memory per SM, 6 GB VRAM.

The golden rule of this entire curriculum:

> Most kernels you will write are **memory-bound**. If memory access is bad, nothing else matters. §0.9 makes that statement precise.

---

## 0.7 Coalesced Memory Access

A warp is happiest when its 32 threads access **consecutive** memory — the hardware services them in one (or few) cache-line transactions.

```text
good:  thread 0 -> x[0], thread 1 -> x[1], ..., thread 31 -> x[31]
bad:   thread 0 -> x[0], thread 1 -> x[1024], thread 2 -> x[2048], ...
```

In CUDA, coalescing comes from how each thread computes its index. In Triton, it comes from building consecutive offsets:

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

Consecutive offsets are good. Strided or random offsets need a justification. You will *feel* the cost of getting this wrong in Unit 3 (transpose), where a naive uncoalesced write throttles the whole kernel.

---

## 0.8 Occupancy, Briefly

**Occupancy** is the ratio of active warps on an SM to the hardware maximum. It is limited by registers per thread, shared memory per block, threads per block, and hardware caps.

- Low occupancy can fail to **hide memory latency** (too few warps to swap in while others wait on memory).
- High occupancy is **not** automatically fast.

The practical rule for now: **try several block/tile sizes and measure.** You will do exactly this in the exercise.

---

## 0.9 The Roofline: The Mental Model Behind Every Unit

This is the single idea that ties the whole curriculum together. Every unit's "roofline" section is an instance of it.

**Arithmetic intensity** is the ratio of useful work to memory traffic:

```text
intensity = FLOPs / bytes moved   (FLOP/byte)
```

A kernel is **memory-bound** if its intensity is below the **ridge point**, and **compute-bound** above it:

```text
ridge point = peak compute / peak bandwidth
            ≈ 4550 GFLOP/s / 264 GB/s
            ≈ 17 FLOP/byte        (FP32, from the spec sheet)
```

The mental picture: plot achievable throughput against intensity. To the left of the ridge, you are climbing the slanted "bandwidth roof" — performance is capped by memory and the only thing that helps is **moving fewer bytes**. To the right, you hit the flat "compute roof" — now the math units are the limit.

Where the units live:

- **This unit (memcpy):** intensity ≈ 0 FLOP/byte — the *purest* memory-bound kernel. Pure traffic, no math. The perfect calibration tool.
- **Units 1–5** (vector add, activation, transpose, softmax, layernorm): all far left of 17 — memory-bound. You optimize traffic.
- **Unit 6 (GEMM):** the first kernel whose tiling pushes intensity *past* 17 into compute-bound — and the ceiling that matters jumps from the 4.55 TFLOP/s FP32 line to the ~36 TFLOP/s tensor-core line. That jump is why tensor cores exist.
- **Unit 7 (FlashAttention):** the art of *reducing traffic* (not FLOPs) to move a kernel rightward off the memory roof.

> Internalize this now: you do not "optimize a kernel" in the abstract. You first locate it on the roofline, which tells you whether your enemy is bytes or FLOPs. For almost everything you write early on, it is bytes.

---

## 0.10 Correctness Under Concurrency

Speed is worthless if the answer is wrong, and parallelism makes "wrong" easy. The no-nonsense version you need now:

- **Race condition** — two execution contexts touch the same data, at least one writes, and ordering is not controlled. The result is nondeterministic garbage.
- **Synchronization** — forcing ordering or mutual exclusion. CPU: mutex, condition variable, join, atomics. GPU: block barrier (`__syncthreads()`), atomics, and *careful ownership* — one thread/program writes one output region.
- **Deadlock** — everyone waits forever. Rare in early kernels, real once synchronization gets complex.
- **Performance trap** — synchronization fixes correctness but can destroy performance. Use the least of it that is correct.

The cleanest defense early on is **ownership**: design kernels so each thread/program owns a disjoint slice of the output and no two writers collide. The bandwidth exercise is deliberately embarrassingly parallel — every element is independent — so you can focus on the machine, not on races.

---

## 0.11 The Benchmarking Mindset

A benchmark is not "time this function." A benchmark always specifies: **hardware, input size, dtype, warmup iterations, measured iterations, a correctness check, and a metric.** Drop any one and the number is meaningless.

Metrics by kernel type:

| Kernel type | Metric |
|---|---|
| copy / vector add / activation | GB/s |
| matrix multiply | TFLOP/s |
| softmax / layernorm | GB/s and latency |
| attention | TFLOP/s, GB/s, latency, memory footprint |

The cardinal GPU-timing rule: **GPU launches are asynchronous.** If you time with a naive wall clock and no synchronization, you are timing the *launch*, not the *work*. Use CUDA events or `triton.testing.do_bench`, which handle synchronization and warmup correctly. This single mistake invalidates more beginner benchmarks than anything else — and the exercise below exists partly to drill it out of you.

---

## 0.12 Unit Exercise: Bandwidth Benchmark

Write two kernels that copy one tensor into another:

```text
y[i] = x[i]
```

Implement both a **CUDA C++** kernel and a **Triton** kernel, then benchmark each across several block/tile sizes.

**Why memcpy?** Because it has almost no math (intensity ≈ 0 — see §0.9), so performance reflects almost purely **memory throughput** and **launch overhead**. It is the cleanest possible instrument for measuring how close to the 264 GB/s ceiling your code can get, and it calibrates your intuition for every memory-bound kernel that follows.

For a float32 copy:

```text
bytes moved = N * 4 (read) + N * 4 (write) = N * 8 bytes
GB/s        = bytes_moved / seconds / 1e9
```

---

## 0.13 CUDA Exercise Spec

Create a CUDA implementation that:

- accepts input pointer `x`, output pointer `y`, and element count `n`
- computes the global index from block and thread IDs
- guards out-of-bounds access
- copies one or more elements per thread
- is launched from a Python harness (e.g. `load_inline`) or a standalone C++ harness

You should understand every part of the launch configuration:

```text
threads_per_block
num_blocks   = ceil(n / threads_per_block)
```

Use a helper for ceiling division — do not hide the launch math from yourself yet. Minimum block sizes to test: `128, 256, 512, 1024`.

Optional later experiment: one element per thread vs. multiple elements per thread via a **grid-stride loop** (you will meet this properly in Unit 1).

> `cudaMemcpy` is the library baseline, **not** your measured kernel.

---

## 0.14 Triton Exercise Spec

Create a Triton kernel `memcpy_kernel` that:

- accepts `x_ptr`, `y_ptr`, `n_elements`, and `BLOCK_SIZE`
- computes `pid = tl.program_id(0)`
- builds consecutive offsets with `tl.arange`
- masks the tail: `mask = offsets < n_elements`
- loads from `x_ptr` and stores to `y_ptr`, both masked

Launch with `grid = (triton.cdiv(n_elements, BLOCK_SIZE),)`. Benchmark at least `128, 256, 512, 1024, 2048, 4096, 8192`.

---

## 0.15 Benchmark Requirements

Use a tensor size that fits comfortably in 6 GB:

```text
N = 32 * 1024 * 1024 float32 elements
input  = 128 MB,  output = 128 MB,  traffic per copy = 256 MB
```

For each implementation, report:

| Field | Meaning |
|---|---|
| implementation | CUDA / Triton / PyTorch baseline |
| block or tile size | threads per block or `BLOCK_SIZE` |
| median time | µs |
| bandwidth | `N * 8 / seconds / 1e9` GB/s |
| efficiency | measured bandwidth / 264 GB/s |
| correctness | max error vs input |

Benchmark hygiene (from §0.11, non-negotiable):

- allocate `x`, `y` once, outside the timed region
- warm up before measuring
- prefer CUDA events or `triton.testing.do_bench`; if using a wall clock, synchronize before/after
- verify correctness after every run
- print device name and dtype

---

## 0.16 What Good Looks Like

Predict before you run:

- **Tiny** block/tile sizes: worse bandwidth (too little work to hide latency / launch overhead dominates).
- **Medium** sizes (256–1024 threads, 1024–4096 tile): usually best.
- **Huge** sizes: not necessarily better; can hurt occupancy.
- The **PyTorch / library copy** is a strong, tuned baseline — matching it is a real result.

A correct, coalesced copy should reach a healthy fraction of 264 GB/s. Whatever fraction you hit becomes your **practical bandwidth ceiling** — the number Units 1–5 are graded against, because none of them can move bytes faster than your memcpy does. Do not chase peak yet; chase correctness and clean measurement first.

---

## 0.17 Common Mistakes

### Forgetting bounds checks

The last block/tile runs past `n`. CUDA needs an `if` guard; Triton needs a `mask=`.

### Timing async GPU work incorrectly

The headline benchmark bug. GPU launches are asynchronous — time without synchronization and your number is garbage. Use CUDA events or `do_bench`.

### Measuring allocation

Do not include `torch.empty_like`, random tensor creation, or compilation in the timed region.

### Comparing different amounts of work

CUDA, Triton, and PyTorch must copy the same `N`, dtype, and device tensors.

### Trusting one run

Warm up, then take a median over repeated measurements.

---

## 0.18 Files

```text
src/unit00/
  bandwidth.py           # current starter (Triton kernel + benchmark)
  bandwidth_cuda.cu      # student-written CUDA kernel
  bandwidth_triton.py    # optional: split the Triton kernel out
  benchmark_bandwidth.py # optional: shared Python benchmark harness
```

The student writes the kernels. Harnesses, correctness checks, and build glue may be assisted.

---

## 0.19 Check Your Understanding

Before Unit 1, answer these without looking:

1. What is the difference between a CPU thread and a CUDA thread?
2. What is a warp, and what is warp divergence?
3. Why can a tiny GPU workload be slower than the same work on a CPU?
4. What does `blockIdx.x * blockDim.x + threadIdx.x` compute, and what does `tl.program_id(0)` correspond to conceptually?
5. Why do CUDA kernels need bounds checks and Triton loads/stores need masks?
6. What is coalesced memory access, and why does it matter?
7. What is arithmetic intensity, and what is this GPU's roofline ridge point?
8. Why is a memcpy the purest memory-bound kernel?
9. Why is naive wall-clock timing usually wrong for GPU kernels?
10. Why is copy bandwidth counted as read bytes *plus* written bytes?
11. What is a race condition, and what is the simplest way to avoid one in an elementwise kernel?
12. What seven things must every benchmark specify?

---

**Next:** Unit 01 — Vector Add: your first real kernel, and why one add per element is still memory-bound
