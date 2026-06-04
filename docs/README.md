<img src="../assets/featured.png" alt="smol-kernels" width="100%">

# GPU Programming with CUDA + Triton

A structured, hands-on curriculum for learning GPU kernel programming from scratch.

The rule is simple: **learn the same idea twice** — once in CUDA C++ and once in Triton.

CUDA teaches the hardware model explicitly. Triton teaches the modern tile/program model used in ML kernel work.

**Hardware baseline:** RTX 2060 Max-Q (Turing SM 7.5, 6 GB VRAM). Can scale to cloud GPUs later.

**Tools:** CUDA C++ + Triton 3.x + PyTorch 2.x. Python is used for harnesses, correctness checks, and benchmarks.

---

## Learning Units

| # | Unit | CUDA focus | Triton focus | Deliverable |
|---|------|------------|--------------|-------------|
| Init 00 | Hardware + OS Prerequisites | CPU threads, OS boundary, driver/toolchain | Python/CUDA environment sanity | [Prerequisite checklist](init-00-hardware-os-prerequisites.md) |
| 0 | GPU Internals + Kernel Mental Model | grids, blocks, threads, warps, global memory | programs, tiles, masks, `tl.arange` | [Bandwidth benchmark](unit-00-gpu-internals-and-triton-model.md) |
| 1 | Vector Add | first `__global__` kernel | first `@triton.jit` kernel | Correct + benchmarked kernels |
| 2 | Fused ReLU / GeLU | elementwise fusion | autotuned block sizes | Fused activation kernels |
| 3 | Matrix Transpose | shared memory tiling, bank conflicts | tiled pointer arithmetic | Tiled transpose kernels |
| 4 | Softmax | reductions and numerical stability | row-wise tile reductions | Stable softmax kernels |
| 5 | LayerNorm | fused mean/variance | fused reduction + epilogue | Norm kernels |
| 6 | Tiled GEMM | thread blocks, shared memory, tensor cores | block matmul, `tl.dot` | GEMM kernels |
| 7 | FlashAttention | online softmax, SRAM tiling | production-style attention tiling | Attention kernels |
| 8 | Pick Your Battle | read CUDA-style production kernels | reproduce vLLM / FlashInfer-style kernel | Production kernel review + reproduction |

---

## Structure

```text
docs/           — Learning materials, one markdown file per unit
src/unit00/     — Bandwidth benchmark exercises
src/unit01/     — Vector add exercises
src/unit02/     — Fused activation exercises
...             — One subdirectory per unit
```

Each unit should eventually include:

- A doc explaining the concept and exercise spec
- A CUDA implementation written by the student
- A Triton implementation written by the student
- A PyTorch reference for correctness checking
- A benchmark comparing CUDA, Triton, and PyTorch where appropriate

---

## Ground Rules

- The student writes kernel code.
- AI may write docs, specs, skeletons, reviews, and benchmark harnesses.
- Use `uv run python3 ...` for Python commands.
- Keep tensor sizes reasonable for 6 GB VRAM.
- Prefer measured results over intuition.
- Commit between units.

---

## References

- [Triton Documentation](https://triton-lang.org/)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
