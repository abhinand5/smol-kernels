# Reference: Hardware Spec Sheet & Calibration

> This curriculum is **hardware-independent**. Every roofline, bandwidth ceiling, and
> "what good looks like" target is relative to *your* GPU — not to whatever card the
> author happened to own. Step one of the course is to **calibrate**: measure your
> machine and generate your personal spec sheet. The lessons grade against that.

## R.1 Calibrate Your GPU (do this first)

```bash
uv run python3 scripts/calibrate_gpu.py
```

This writes **`docs/my-gpu-spec.md`** (gitignored — it is yours, not committed). It works on any CUDA GPU because it **measures** rather than guesses:

- **Achievable memory bandwidth** — a large device-to-device copy, timed with CUDA events.
- **Achievable FP32 GEMM throughput** — `torch.matmul` in fp32, timed.
- **Achievable tensor-core GEMM throughput** — `torch.matmul` in fp16, timed.
- **Your two roofline ridge points** — derived from the measurements above.
- **Hardware facts** — compute capability, SM count, VRAM, shared-memory budget, and your tensor cores' supported data types (from compute capability).

It uses only library ops (`torch.matmul`, tensor copy), **never your kernels**, so you can run it on day one before implementing anything. Re-run it whenever you switch GPUs (e.g. local → cloud A100/H100). When a unit says "your ridge point" or "your measured bandwidth," it means the numbers in `docs/my-gpu-spec.md`.

---

## R.2 The Roofline Method (universal)

A kernel's **arithmetic intensity** is its useful work per byte of global-memory traffic:

```text
intensity = FLOPs / bytes moved        (FLOP/byte)
```

A kernel is **memory-bound** below the **ridge point** and **compute-bound** above it:

```text
ridge point = peak compute / peak bandwidth
```

Every GPU has **two** ridges, because it has two compute ceilings — the general FP32 cores and the much faster tensor cores:

| Path | Ceiling | Ridge | Used in |
|---|---|---|---|
| FP32 (CUDA cores) | your measured FP32 GEMM | FP32 GEMM ÷ bandwidth | Units 1–5 (memory-bound) |
| Tensor cores | your measured fp16/bf16 GEMM | tensor GEMM ÷ bandwidth | Units 6–7 |

The **conclusion that almost every early kernel is memory-bound holds on every GPU** — only the exact crossover moves. Units 1–5 sit far left of your FP32 ridge. Unit 6 (GEMM) is the first kernel whose tiling pushes intensity past the ridge into compute-bound, where the tensor-core ceiling takes over.

---

## R.3 Tensor-Core Data Types by Architecture

The one fact the calibrator cannot *measure* is which data types your tensor cores accept — it derives this from your compute capability. This matters for Units 6–7: use the **highest-throughput dtype your hardware supports** for `tl.dot` / `wmma`.

| Architecture | Compute capability | Tensor-core data types |
|---|---|---|
| Volta | 7.0 | FP16 |
| Turing | 7.5 | FP16, INT8, INT4, INT1 |
| Ampere | 8.0 / 8.6 / 8.7 | FP16, BF16, **TF32**, INT8, INT4 (+ FP64 on 8.0) |
| Ada Lovelace | 8.9 | FP16, BF16, TF32, **FP8**, INT8, INT4 |
| Hopper | 9.0 | FP16, BF16, TF32, FP8, INT8, FP64 |
| Blackwell | 10.x / 12.0 | FP16, BF16, TF32, FP8, **FP6, FP4**, INT8 |

Reading: **TF32 and BF16 are Ampere (8.0)+** — *not* available on Turing. **FP8 is Ada/Hopper+**. **FP4/FP6 are Blackwell**. A tutorial that tells a Turing user to use TF32 is wrong; one that tells a Hopper user to use only FP16 is leaving ~half the throughput on the table. Your `docs/my-gpu-spec.md` lists exactly what *your* card supports. (Newer SMs than this table get a "confirm against the CUDA C++ Programming Guide" flag rather than a guess.)

---

## R.4 Worked Example — RTX 2060 Max-Q (the "reference GPU")

The units use one concrete machine for their worked arithmetic so the math stays tangible. That machine is a power-limited mobile Turing card; **substitute your own numbers** from `docs/my-gpu-spec.md`.

| Quantity | Theoretical | Measured (this card) |
|---|---|---|
| Memory bandwidth | ~264 GB/s | ~233 GB/s (88% of theoretical) |
| FP32 GEMM | ~4.55 TFLOP/s | ~4.0 TFLOP/s |
| Tensor-core GEMM (FP16) | ~36 (fp16-acc) / ~18 (fp32-acc) | ~18.5 TFLOP/s (`torch.matmul` uses fp32 accumulate) |
| **FP32 ridge** | ~4550/264 ≈ **17 FLOP/byte** | ~4000/233 ≈ **17 FLOP/byte** |
| **Tensor ridge** | — | ~18500/233 ≈ **79 FLOP/byte** |
| Compute capability | 7.5 (Turing) | |
| Tensor-core dtypes | FP16, INT8, INT4, INT1 (no TF32/BF16/FP8) | |
| Shared memory / block | up to 64 KB | |
| VRAM | 6 GB | |

When a unit's roofline section computes "intensity vs the ridge (≈17)" or quotes "264 GB/s," that is this reference card. Your card's numbers differ; your *conclusions* (memory- vs compute-bound) will almost always match.

---

## R.5 Theoretical vs Measured

- **Theoretical peaks** (computable from clocks/bus width) tell you *which régime* a kernel is in — memory- or compute-bound. The régime is what drives every optimization decision.
- **Measured ceilings** (what the calibrator records) tell you *how close to done* you are. You can never beat your measured copy bandwidth on a memory-bound kernel, or your measured cuBLAS GEMM on a compute-bound one.

Note the reference card above: `torch.matmul` fp16 measures ~18.5 TFLOP/s, not the ~36 "fp16-accumulate" theoretical figure, because GeForce Turing caps fp32-accumulate tensor throughput at half rate and `torch.matmul` accumulates in fp32. This is exactly why the course grades against **measured** numbers — theory would have set an unreachable target. Trust your `my-gpu-spec.md`.

---

**Referenced by:** every unit. Run the calibrator, then keep `docs/my-gpu-spec.md` open while you work.
