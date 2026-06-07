<img src="assets/featured.png" alt="smol-kernels" width="100%">

# smol-kernels

GPU programming from scratch — learn every core idea **twice**, once in CUDA C++
and once in Triton, from a memcpy bandwidth benchmark all the way to
FlashAttention. You write every kernel; the roofline grades it against **your own
GPU**.

## What this is

A structured, self-paced kernel curriculum. Nine units take you from "what is a
warp" to a working FlashAttention, each concept implemented in both CUDA and
Triton and benchmarked against a PyTorch reference.

It is **not** a place to get AI-generated kernels. The student writes every
kernel body. The materials provide the specs, the docs, the skeletons, the
benchmark harnesses, and the reviews — never the solution.

## Hardware-agnostic

The curriculum recalibrates to whatever GPU you have. Generate your spec once:

```bash
uv run python3 scripts/calibrate_gpu.py
```

This measures your card's real bandwidth and compute ceilings and writes
`docs/my-gpu-spec.md` (gitignored). Every roofline target in the units grades
against that spec, so the lessons work on a laptop 2060 or a cloud H100 alike.
Re-run it after switching GPUs.

## The climb

| Unit | Topic | Regime |
|------|-------|--------|
| 00 | GPU internals & the Triton model | — |
| 01 | Vector add | memory-bound |
| 02 | Fused activation | memory-bound |
| 03 | Matrix transpose | memory-bound |
| 04 | Softmax | memory-bound |
| 05 | LayerNorm | memory-bound |
| 06 | Tiled GEMM | compute-bound |
| 07 | FlashAttention | compute-bound |
| 08 | Pick your battle | mixed |

Start at [`docs/unit-00-gpu-internals-and-triton-model.md`](docs/unit-00-gpu-internals-and-triton-model.md).

## The web reader

The curriculum is also a polished web reader (Vite + React + Tailwind):

```bash
cd web
npm install
npm run dev      # http://localhost:5173
```

See [`web/README.md`](web/README.md) for build and deploy details.

## Layout

```
docs/      the curriculum — one markdown file per unit
src/       TODO scaffolds you fill in (one folder per unit)
scripts/   calibrate_gpu.py and tooling
web/       the web reader
AGENTS.md  full project charter and conventions
```
