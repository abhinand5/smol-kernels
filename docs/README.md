# GPU Programming with Triton

A structured, hands-on curriculum for learning GPU kernel programming from scratch.

**Hardware baseline:** RTX 2060 Max-Q (Turing SM 7.5, 6 GB VRAM). Can scale to cloud GPUs (A100/H100).

**Tools:** Triton 3.x + PyTorch 2.x. No raw CUDA C.

---

## Learning Units

| # | Unit | Skill | Deliverable |
|---|------|-------|-------------|
| 0 | GPU Internals + Triton Mental Model | SIMT, memory hierarchy, coalescing, occupancy, tiles vs threads | [Bandwidth benchmark](unit-00-gpu-internals-and-triton-model.md) |
| 1 | Vector Add | `@triton.jit`, `tl.load`/`tl.store`, program IDs, masking | Correct + benchmarked kernel |
| 2 | Fused ReLU / GeLU | Element-wise fusion, `@triton.autotune` | Autotuned activation kernel |
| 3 | Matrix Transpose | Shared memory tiling, bank conflicts | Tiled transpose kernel |
| 4 | Softmax | Two-pass reduction, numerical stability | Stable softmax kernel |
| 5 | LayerNorm | Fused mean+variance, epilogue fusion | Norm kernel |
| 6 | Tiled GEMM | Canonical tiling, tensor core hints | GEMM kernel |
| 7 | FlashAttention | Online softmax, tiling for SRAM | Attention kernel |
| 8 | Pick Your Battle | Reproduce a vLLM / FlashInfer kernel | Production-ready kernel |

---

## Structure

```
docs/           — Learning materials, one markdown file per unit
src/unit00/     — Bandwidth benchmark
src/unit01/     — Vector add
src/unit02/     — Fused activation
...             — One subdirectory per unit
```

Each unit includes:
- A doc explaining the concept and exercise spec
- A kernel implementation with benchmarks
- A PyTorch reference for correctness checking

---

## References

- [Triton Documentation](https://triton-lang.org/)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
