# Init 00: Hardware + OS Prerequisites

> Before writing GPU kernels, know what machine you are actually programming.

This is not a full OS course. This is the minimum hardware/OS foundation needed to reason about CUDA and Triton kernels without hand-waving.

---

## 0.1 What You Need Installed

### Required local baseline

- Linux machine with NVIDIA GPU
- NVIDIA driver visible through `nvidia-smi`
- CUDA toolkit visible through `nvcc --version`
- Python environment managed by `uv`
- PyTorch with CUDA support
- Triton

On this repo's local machine:

| Item | Value |
|---|---|
| OS | CachyOS / Arch Linux |
| Kernel | Linux 6.18 LTS series |
| GPU | RTX 2060 Max-Q |
| Compute capability | 7.5 |
| VRAM | 6 GB |
| Driver | 595.71.05 |
| CUDA toolkit | 13.2 |
| Python | 3.14.4 |
| PyTorch | 2.12 CUDA build |
| Triton | 3.7 |

Check your machine:

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

## 0.2 CPU Threads vs GPU Threads

The word **thread** means different things in OS land and GPU land.

| Concept | CPU / OS | GPU / CUDA |
|---|---|---|
| Thread owner | OS scheduler | GPU hardware scheduler |
| Typical count | Tens to thousands | Thousands to millions |
| Execution | Independent instruction stream | SIMT groups execute together |
| Memory | Process virtual address space | Global/shared/register memory spaces |
| Synchronization | mutexes, condition variables, atomics | barriers, atomics, warp/block rules |
| Main cost | context switches, cache pollution, locking | memory access pattern, occupancy, divergence |

A CPU thread is an OS-managed execution context: registers, stack, program counter, scheduling state.

A CUDA thread is a lightweight GPU execution lane inside a larger hierarchy:

```text
grid -> blocks -> warps -> threads
```

Triton hides explicit GPU thread indexing and asks you to think in **programs/tiles**, but the generated code still runs on GPU threads and warps underneath.

---

## 0.3 The OS Boundary

A normal program runs in **user mode**. The OS kernel runs in **kernel mode**.

Why you care:

- Allocating memory, launching work, loading drivers, and talking to hardware cross OS/driver boundaries.
- System calls and driver calls are not free.
- Kernel launches have overhead.
- Benchmarking tiny kernels can accidentally measure launch overhead instead of GPU throughput.

Simple rule:

> GPUs are fast at bulk parallel work. They are not magic for tiny work.

---

## 0.4 Process, Address Space, and Memory

A process owns a virtual address space. CPU threads inside that process share it.

GPU programming adds more memory categories:

| Memory | Where | Notes |
|---|---|---|
| CPU pageable memory | host RAM | normal Python/C++ allocations |
| CPU pinned memory | host RAM | page-locked; faster async copies |
| GPU global memory | VRAM | large, high bandwidth, high latency |
| GPU shared memory | on-chip per block | small, fast, programmer-managed in CUDA |
| Registers | on-chip per thread | fastest, limited |
| Caches | on-chip | behavior depends on access pattern |

For this repo, assume tensors live on GPU unless explicitly stated.

```python
x = torch.randn(1024, device="cuda")
```

---

## 0.5 Concurrency Basics You Need

You only need the no-nonsense version for now.

### Race condition

Two execution contexts touch the same data, at least one writes, and ordering is not controlled.

### Synchronization

A way to force ordering or mutual exclusion.

CPU examples:

- mutex
- condition variable
- join
- atomic operation

GPU examples:

- block barrier: `__syncthreads()` in CUDA
- atomic operations
- careful ownership: one thread/program writes one output region

### Deadlock

Everyone waits forever. Less common in early GPU kernels, but possible once synchronization gets complex.

### Performance trap

Synchronization fixes correctness but can destroy performance.

---

## 0.6 Benchmarking Mindset

A benchmark is not just "time this function".

Always specify:

- hardware
- input size
- dtype
- warmup iterations
- measured iterations
- correctness check
- metric

Common metrics:

| Kernel type | Metric |
|---|---|
| copy / vector add / activation | GB/s |
| matrix multiply | TFLOP/s |
| softmax / layernorm | GB/s and latency |
| attention | TFLOP/s, GB/s, latency, memory footprint |

For GPU timing, use CUDA events or `triton.testing.do_bench`. Do not trust naive wall-clock timing unless you synchronize correctly.

---

## 0.7 What You Should Know Before Unit 0

You should be able to answer:

1. What is the difference between a CPU thread and a CUDA thread?
2. What does the OS manage for a process?
3. Why does crossing into the driver/kernel have overhead?
4. Why can tiny GPU workloads be slower than CPU workloads?
5. What is a race condition?
6. What is the difference between host memory and device memory?
7. Why must benchmarks include warmup and synchronization?

If these are fuzzy, that is fine. Keep this page open while doing Unit 0.
