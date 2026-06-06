# Reference: RTX 2060 Max-Q Spec Sheet

> One source of truth for every number this curriculum grades against. When a unit says "the ridge point" or "peak bandwidth," it means the values on this page. If you move to a cloud GPU, copy this file, change the numbers, and re-derive the ridge.

This is the **target machine**: a power-limited mobile Turing GPU. Every roofline, every "what good looks like" target, and every byte/FLOP estimate in the units is computed from the table below.

---

## R.1 The Numbers

| Quantity | Value | Notes |
|---|---|---|
| Architecture | Turing | |
| Compute capability | **7.5** (SM 7.5) | |
| CUDA cores | 1920 | 30 SMs × 64 |
| SM count | 30 | |
| Boost clock | ~1185 MHz | thermal/power dependent — *yours will vary* |
| **Peak FP32** | **~4.55 TFLOP/s** | `2 × 1920 × 1.185e9` |
| Peak FP16 (packed, non-tensor) | ~9.1 TFLOP/s | 2× FP32 rate |
| Tensor cores | 240 | Turing 2nd-gen |
| **Tensor-core FP16 (FP16 accumulate)** | **~36 TFLOP/s** | nominal; the GEMM/attention ceiling |
| Tensor-core FP16 (FP32 accumulate) | ~18 TFLOP/s | GeForce halves FP32-accumulate |
| Tensor-core data types | **FP16, INT8, INT4, INT1** | see R.3 |
| VRAM | 6 GB GDDR6 | keep working sets small |
| **Theoretical memory bandwidth** | **~264 GB/s** | the bandwidth ceiling |
| L2 cache | 3 MB | |
| Shared memory / SM | up to 64 KB | Turing: 64 KB unified L1/shared, configurable |
| Max threads / block | 1024 | |
| Warp size | 32 | |
| TDP | 65 W | this is *why* it is slower than a desktop 2060 |

---

## R.2 The Two Rooflines You Will Use

A kernel is **memory-bound** if its arithmetic intensity (FLOPs per byte of global-memory traffic) is below the ridge point, and **compute-bound** above it.

```text
ridge point = peak compute / peak bandwidth
```

This GPU has **two** compute ceilings, so two ridges:

| Path | Peak compute | Ridge point | Used in |
|---|---|---|---|
| FP32 (CUDA cores) | ~4.55 TFLOP/s | **~17 FLOP/byte** | Units 1–5 (all memory-bound) |
| FP16 (tensor cores) | ~36 TFLOP/s | **~136 FLOP/byte** | Units 6–7 (GEMM, attention) |

Units 1 through 5 sit far to the *left* of 17 — they are memory-bound, and you optimize traffic. Unit 6 (GEMM) is the first kernel whose tiling pushes intensity *past* the ridge into the compute-bound region, and at that point the ceiling that matters jumps from 4.55 to ~36 TFLOP/s — which is the entire reason tensor cores exist.

---

## R.3 What the Tensor Cores Can and Cannot Do

This is a frequent source of wrong tutorials. On **Turing SM 7.5**:

| Data type | Tensor core support | Notes |
|---|---|---|
| FP16 | ✅ | input FP16, accumulate FP16 or FP32 |
| INT8 / INT4 / INT1 | ✅ | quantized inference |
| **TF32** | ❌ | **Ampere SM 8.0+ only** |
| **BF16** | ❌ | Ampere+ |
| **FP8** | ❌ | Hopper / Ada |
| FP64 | ❌ | A100-class only |

So in Unit 6 and Unit 7 the tensor-core path on this machine is **FP16**. Any tutorial telling you to use TF32 (`tl.dot` with `tf32` mode, `wmma` TF32 fragments) is targeting Ampere and will not engage tensor cores here.

---

## R.4 Practical Ceilings (Measure Your Own)

The table above is *theoretical*. Real achievable numbers are lower and clock-dependent. Establish your machine's **practical** ceilings empirically and grade against those:

| Ceiling | How to measure | Used as target for |
|---|---|---|
| Achievable copy bandwidth | Unit 0 memcpy benchmark | Units 1–5 GB/s targets |
| Achievable GEMM throughput | a cuBLAS FP16 GEMM (or PyTorch `@` in fp16) | Unit 6 TFLOP/s target |
| Achievable attention throughput | PyTorch SDPA / FlashAttention reference | Unit 7 target |

> Rule: theoretical peaks tell you *which régime* you are in (memory- vs compute-bound). Measured peaks tell you *how close to done* you are. Use both.

---

**Referenced by:** every unit. Update this file (not the units) if hardware changes.
