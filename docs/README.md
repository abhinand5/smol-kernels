# CUDA / Triton Mastery — Frontier AI Lab Track

**Goal:** Reach a level where you can write, optimize, and land custom GPU kernels at a frontier AI lab (FAIR, OpenAI, DeepMind, Anthropic, etc.)

**Hardware baseline:** RTX 2060 Max-Q (Turing SM 7.5, 6 GB VRAM). Can scale to cloud GPUs (A100/H100).

**Tools:** Triton 3.x + PyTorch 2.x. No raw CUDA C.

---

## Learning Units

| # | Unit | Skill | Deliverable |
|---|------|-------|-------------|
| 0 | GPU Internals + Triton Mental Model | SIMT, memory hierarchy, coalescing, occupancy, tiles vs threads | Bandwidth benchmark |
| 1 | Vector Add | `@triton.jit`, `tl.load`/`tl.store`, program IDs, masking | Correct + benchmarked kernel |
| 2 | Fused ReLU / GeLU | Element-wise fusion, `@triton.autotune` | Autotuned activation kernel |
| 3 | Matrix Transpose | Shared memory tiling, bank conflicts | Tiled transpose kernel |
| 4 | Softmax | Two-pass reduction, numerical stability | Stable softmax kernel |
| 5 | LayerNorm | Fused mean+variance, epilogue fusion | Norm kernel |
| 6 | Tiled GEMM | Canonical tiling, tensor core hints | GEMM kernel |
| 7 | FlashAttention | Online softmax, tiling for SRAM | Attention kernel |
| 8 | Pick Your Battle | Reproduce a vLLM / FlashInfer kernel | Production-ready kernel |

---

## How We Work

1. **I write the docs.** Each unit lives in `docs/unit-NN-*.md` with concept explanations, references, and the exercise spec.
2. **You write the code.** Kernels go in `src/kernels/` alongside unit tests and benchmarks.
3. **I review.** We run your kernel, compare to a PyTorch reference, and inspect the generated PTX.
4. **You own it.** Don't move on until you can explain every line of your own code.

---

## References

- [Triton Documentation](https://triton-lang.org/)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
