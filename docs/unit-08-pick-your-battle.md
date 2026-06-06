# Unit 08: Pick Your Battle

> No new primitive to learn — every tool is now in your hands. This unit is about *transfer*: opening a real production kernel from a real inference engine, explaining every line, and reproducing one well enough to benchmark against the original. This is the job.

Units 0–7 built the vocabulary: grids and warps, coalescing, shared memory and bank conflicts, reductions, numerical stability, tiling and the roofline, tensor cores, online softmax, IO-aware fusion. Unit 8 proves you can *read* — that you can open a kernel you did not write, in a codebase that does not hold your hand, and account for every decision in it.

This unit has two deliverables:

1. A **line-by-line annotation** of one production kernel.
2. A **reproduction**: your own implementation of that kernel (or a faithful subset), benchmarked against the original and against PyTorch.

Numbers from the [hardware spec sheet](reference-hardware-spec-sheet.md).

---

## 8.1 Where Production Kernels Live

These are the codebases that the frontier-lab roles actually touch:

| Project | What to read there | Language |
|---|---|---|
| **vLLM** (`csrc/`) | paged attention, fused RMSNorm+residual, RoPE, fused MoE, dequant kernels | CUDA C++ |
| **FlashInfer** | prefill/decode attention, paged KV, sampling kernels | CUDA C++ / templates |
| **Triton tutorials & kernels** | matmul, fused-attention, layernorm, dropout | Triton |
| **liger-kernel** | fused Triton kernels for training (RMSNorm, SwiGLU, cross-entropy) | Triton |
| **TensorRT-LLM** (`cpp/`) | heavily-templated production attention/GEMM | CUDA C++ |
| **PyTorch** (`aten/src/ATen/native/cuda/`) | reference fused kernels for everything | CUDA C++ |

Match the kernel to your level. Triton kernels in `liger-kernel` or the Triton tutorials are the most readable; vLLM's CUDA `csrc` is a step up; TensorRT-LLM's templated CUDA is the deep end.

---

## 8.2 Suggested Targets by Ambition

**Approachable (recommended first reproduction):**

- **RMSNorm** (vLLM `csrc/layernorm_kernels.cu` or liger's Triton RMSNorm). It is Unit 5 with one fewer statistic (no mean-subtraction; just `x / rms(x) * γ`). You can reproduce this *today*.
- **Fused add + RMSNorm** (residual + norm in one kernel) — Unit 2 fusion applied to Unit 5. A clean, high-value real kernel.
- **RoPE** (rotary position embedding) — elementwise with a twist; great coalescing/indexing practice.

**Stretch:**

- **Paged attention** (vLLM) — FlashAttention (Unit 7) with a block-table indirection for the KV cache. Read it fully; reproduce the decode path on a single sequence.
- **Fused SwiGLU** (liger) — gated activation fusion.
- **A Triton autotuned matmul with epilogue fusion** — Unit 6 plus a fused bias/activation epilogue.

**Deep end:**

- A FlashInfer or TensorRT-LLM attention kernel — read and annotate; reproduction optional.

Pick **one** approachable kernel to reproduce fully. Read one stretch kernel for annotation even if you don't reproduce it.

---

## 8.3 How to Dissect a Kernel (Methodology)

Reading a production kernel is a skill with a method. For any kernel, answer these in order:

1. **Signature & contract.** What are the inputs, outputs, dtypes, shapes, strides? What does it compute mathematically? Find the PyTorch/numpy equivalent first.
2. **Launch configuration.** What is the grid? What does one block/program own — a row, a tile, a token, a KV block? This tells you the parallelization strategy.
3. **Memory hierarchy.** What goes to shared memory / SRAM and why? What stays in registers? What is the global-traffic pattern, and is it coalesced?
4. **Reductions & sync.** Where are the reductions (warp shuffles, `tl.sum`)? Where are the barriers, and what race does each prevent?
5. **Numerics.** Where are the stability tricks (max-subtraction, FP32 accumulate, `ε`, rescaling)? Why each one?
6. **The roofline.** Is it memory- or compute-bound? Compute the intensity. Which ceiling does it target?
7. **The tuning surface.** What is autotuned / templated? What hardware assumptions are baked in (SRAM size, warp count, SM count)?
8. **The clever bit.** Every production kernel has one non-obvious trick. Name it. (Online softmax? Block-table indirection? Vectorized loads? Bank-conflict padding? Persistent kernels?)

If you can answer all eight for your chosen kernel, you understand it.

---

## 8.4 Reproduction Spec

For your chosen approachable kernel:

- Implement it yourself (CUDA, Triton, or both) from your *understanding*, not by copy-paste.
- Match the original's contract: same dtypes, shapes, edge handling.
- Verify correctness against the original kernel **and** a PyTorch reference.
- Benchmark against both. Report the gap and **explain it** — where does the production kernel's extra performance come from (vectorization? better tiling? a fused epilogue? a tuned config you didn't match?).
- Identify at least one optimization in the original that you did *not* reproduce, and explain what it buys.

The goal is not to beat the production kernel. It is to reproduce it competently and **account for every difference** in the numbers.

---

## 8.5 Annotation Deliverable

Produce a written annotation (in `src/unit08/NOTES.md`) of one kernel that:

- links the exact source file + commit/line range
- answers all eight §8.3 questions
- walks the kernel in execution order, explaining every non-trivial line
- names the clever bit explicitly
- notes the hardware assumptions and how they'd change on this 2060 vs the kernel's intended target (often an A100/H100)

This document is the proof of success criterion #2: *open any production kernel and explain every line.*

---

## 8.6 Benchmark Requirements

Report for your reproduction:

| Field | Meaning |
|---|---|
| kernel | what you reproduced |
| source | repo + file + commit |
| your impl | CUDA / Triton |
| shape(s) | realistic sizes for the op |
| median time | µs, you vs original vs PyTorch |
| throughput | GB/s or TFLOP/s as appropriate to the op |
| correctness | vs original and vs PyTorch |
| gap analysis | *why* the original is faster, itemized |

---

## 8.7 What Good Looks Like

- Your reproduction is correct against both references.
- You are within a sensible factor of the production kernel — and every remaining gap has a named cause, not a shrug.
- Your annotation answers all eight dissection questions and names the clever bit.
- You can hold a conversation about the kernel's design trade-offs and how they'd shift on different hardware.

That is the bar for the role this curriculum targets.

---

## 8.8 Common Mistakes

### Copy-pasting instead of understanding

Reproducing by transcription teaches nothing. Implement from the *contract and your understanding*, then compare.

### Picking a kernel beyond your level for reproduction

Annotate the deep-end kernel; reproduce the approachable one. A finished RMSNorm beats an abandoned paged-attention.

### Reporting the gap without explaining it

"Mine is 3× slower" is not a result. "Mine is 3× slower because the original uses `float4` vectorized loads and a tuned `BLOCK_K` I didn't match" is.

### Ignoring the hardware mismatch

Production kernels target A100/H100. Their block sizes and SRAM assumptions may be wrong for your 64 KB Turing card. Note where you had to adapt.

### Skipping the PyTorch correctness check

Always anchor to a framework reference, not just the production kernel (which could have its own assumptions you're misusing).

---

## 8.9 Files

```text
src/unit08/
  NOTES.md                  # student: the line-by-line annotation (§8.5)
  reproduction_triton.py    # student: your reproduction (and/or .cu)
  reproduction_cuda.cu      # optional
  benchmark_reproduction.py # you vs original vs PyTorch
```

---

## 8.10 Check Your Understanding

For the kernel you chose:

1. What is its mathematical contract, and what is the PyTorch equivalent?
2. What does one block/program own, and what is the grid?
3. What lives in shared memory / SRAM, and what is the global access pattern?
4. Where are the reductions and barriers, and what race does each barrier prevent?
5. What numerical-stability techniques does it use, and why each?
6. Is it memory- or compute-bound? What is its arithmetic intensity?
7. What is autotuned or templated, and what hardware assumptions are baked in?
8. What is the one clever, non-obvious trick?
9. Where is your reproduction slower, and exactly why?
10. How would this kernel change if retargeted from an A100 to this 2060 Max-Q?

---

## 8.11 Where to Go From Here

You have, twice over, built every core GPU kernel pattern from memcpy to FlashAttention, learned to read production code, and grounded every result in the roofline. From here:

- Implement the FlashAttention **backward** pass (much harder than forward).
- Write a real CUDA FlashAttention with `wmma` tensor cores.
- Reproduce a paged-attention decode kernel end to end.
- Move to a cloud A100/H100 (update the [spec sheet](reference-hardware-spec-sheet.md)), and learn what changes when SRAM, TF32/BF16/FP8 tensor cores, and async copy (`cp.async`, TMA) enter the picture.
- Read the CUTLASS / CuTe abstractions and the latest FlashAttention/FlashInfer source.

The curriculum ends; the craft does not.

---

**End of curriculum.** Back to the [overview](README.md).
