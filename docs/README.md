<img src="../assets/featured.png" alt="smol-kernels" width="100%">

# GPU Programming with CUDA + Triton

A structured, hands-on curriculum for learning GPU kernel programming from scratch.

The rule is simple: **learn the same idea twice** — once in CUDA C++ and once in Triton.

CUDA teaches the hardware model explicitly. Triton teaches the modern tile/program model used in ML kernel work.

**Works on any CUDA GPU.** Step one is to calibrate — `uv run python3 scripts/calibrate_gpu.py` measures your card and writes `docs/my-gpu-spec.md`, and every roofline target recalibrates to it. The units use an RTX 2060 Max-Q for concrete worked examples; your numbers come from your own spec.

**Tools:** CUDA C++ + Triton 3.x + PyTorch 2.x. Python is used for harnesses, correctness checks, and benchmarks.

---

## Learning Units

| # | Unit | CUDA focus | Triton focus | Deliverable |
|---|------|------------|--------------|-------------|
| 0 | GPU Foundations + Bandwidth | the machine, SIMT, memory hierarchy, roofline, coalescing | programs, tiles, masks, `tl.arange` | [GPU foundations + bandwidth benchmark](unit-00-gpu-internals-and-triton-model.md) |
| 1 | Vector Add | first `__global__` kernel | first `@triton.jit` kernel | [Vector add](unit-01-vector-add.md) |
| 2 | Fused ReLU / GeLU | elementwise fusion | autotuned block sizes | [Fused activation](unit-02-fused-activation.md) |
| 3 | Matrix Transpose | shared memory tiling, bank conflicts | tiled pointer arithmetic | [Matrix transpose](unit-03-matrix-transpose.md) |
| 4 | Softmax | reductions and numerical stability | row-wise tile reductions | [Softmax](unit-04-softmax.md) |
| 5 | LayerNorm | fused mean/variance | fused reduction + epilogue | [LayerNorm](unit-05-layernorm.md) |
| 6 | Tiled GEMM | thread blocks, shared memory, tensor cores | block matmul, `tl.dot` | [Tiled GEMM](unit-06-tiled-gemm.md) |
| 7 | FlashAttention | online softmax, SRAM tiling | production-style attention tiling | [FlashAttention](unit-07-flash-attention.md) |
| 8 | Pick Your Battle | read CUDA-style production kernels | reproduce vLLM / FlashInfer-style kernel | [Pick your battle](unit-08-pick-your-battle.md) |

> **Hardware:** the curriculum is GPU-agnostic. Run `uv run python3 scripts/calibrate_gpu.py` to generate your personal [spec sheet](reference-hardware-spec-sheet.md) (`docs/my-gpu-spec.md`); roofline math and targets recalibrate to your card. Switching GPUs? Just re-run it.

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
- Keep tensor sizes within your VRAM (see `docs/my-gpu-spec.md` after calibrating).
- Prefer measured results over intuition.
- Commit between units.

---

## References

- [Triton Documentation](https://triton-lang.org/)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
