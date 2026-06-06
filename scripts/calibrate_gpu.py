"""
GPU calibration — generate a personal hardware spec sheet for THIS machine.

Run:   uv run python3 scripts/calibrate_gpu.py
Writes: docs/my-gpu-spec.md  (gitignored; yours, not committed)

The curriculum's roofline math, bandwidth ceilings, and "what good looks like"
targets are all relative to YOUR GPU. This script measures them so the lessons
recalibrate to any CUDA GPU instead of being hardwired to one card.

Design principle: MEASURE, don't guess.
- Achievable bandwidth and GEMM throughput are *measured* with library ops
  (tensor copy, torch.matmul) timed with CUDA events. No "cores x clock" tables
  that silently break on GPUs this script has never seen.
- The only non-measured fact is which data types your tensor cores support, which
  is derived from compute capability (with a documented fallback for new archs).
- It uses ONLY library ops, never your kernels, so you can run it on day one
  before implementing anything.
"""

from __future__ import annotations

import datetime as _dt
import statistics
from pathlib import Path

import torch


# --- Tensor-core data types by compute capability ---------------------------
# Verified against NVIDIA architecture whitepapers / CUDA C++ Programming Guide
# compute-capability table (as of 2026). All tensor-core GPUs (>= 7.0) do FP16.
# For an unknown / newer SM, we fall back to "FP16 (+ check NVIDIA docs)" rather
# than guess — a wrong dtype claim is worse than an honest "look it up".
_TENSOR_DTYPES: dict[tuple[int, int], list[str]] = {
    (7, 0): ["FP16"],                                              # Volta
    (7, 2): ["FP16"],                                              # Volta (Xavier)
    (7, 5): ["FP16", "INT8", "INT4", "INT1"],                      # Turing
    (8, 0): ["FP16", "BF16", "TF32", "INT8", "INT4", "INT1", "FP64"],   # Ampere A100
    (8, 6): ["FP16", "BF16", "TF32", "INT8", "INT4", "INT1"],      # Ampere consumer
    (8, 7): ["FP16", "BF16", "TF32", "INT8", "INT4", "INT1"],      # Ampere Orin
    (8, 9): ["FP16", "BF16", "TF32", "FP8", "INT8", "INT4"],       # Ada Lovelace
    (9, 0): ["FP16", "BF16", "TF32", "FP8", "INT8", "FP64"],       # Hopper
    (10, 0): ["FP16", "BF16", "TF32", "FP8", "FP6", "FP4", "INT8", "FP64"],  # Blackwell
    (10, 1): ["FP16", "BF16", "TF32", "FP8", "FP6", "FP4", "INT8", "FP64"],  # Blackwell
    (12, 0): ["FP16", "BF16", "TF32", "FP8", "FP6", "FP4", "INT8"],          # Blackwell consumer
}


def _tensor_core_dtypes(major: int, minor: int) -> tuple[list[str], bool]:
    """Return (dtype list, is_known). Unknown SMs get a safe FP16 baseline."""
    key = (major, minor)
    if key in _TENSOR_DTYPES:
        return _TENSOR_DTYPES[key], True
    if major < 7:
        return [], True  # pre-Volta: no tensor cores
    return ["FP16"], False  # known to have tensor cores, exact dtype list unverified


def _time_ms(fn, *, warmup: int = 10, iters: int = 50) -> float:
    """Median wall time (ms) of a GPU op, timed with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def measure_bandwidth(total_bytes: int) -> float:
    """Achievable HBM bandwidth in GB/s via a large device-to-device copy."""
    # size one tensor to ~ total_bytes/16 (read+write fits comfortably), min 64 MiB
    nbytes = max(min(total_bytes // 16, 512 * 1024**2), 64 * 1024**2)
    n = nbytes // 4
    src = torch.randn(n, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    ms = _time_ms(lambda: dst.copy_(src))
    moved = 2 * n * 4  # read + write
    return moved / (ms / 1e3) / 1e9


def measure_gemm_tflops(dtype: torch.dtype, total_bytes: int) -> tuple[float, int]:
    """Achievable matmul throughput (TFLOP/s) for a large square GEMM.

    For fp32 we force true FP32 (no TF32): on Ampere+ `torch.matmul` may otherwise
    silently dispatch fp32 to TF32 tensor cores, which would mislabel tensor-core
    throughput as FP32-core throughput and collapse the FP32-vs-tensor distinction
    Unit 6 depends on. fp16 inputs always engage tensor cores regardless.
    """
    if dtype == torch.float32:
        torch.set_float32_matmul_precision("highest")  # disable TF32 for a true FP32 reading
    bytes_per = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    # three NxN matrices must fit in ~ total_bytes/4; cap N at 8192
    max_n = int(((total_bytes / 4) / (3 * bytes_per)) ** 0.5)
    n = min(8192, max(1024, (max_n // 256) * 256))
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    ms = _time_ms(lambda: torch.matmul(a, b), warmup=10, iters=30)
    flops = 2 * n * n * n
    return flops / (ms / 1e3) / 1e12, n


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available — this curriculum needs an NVIDIA GPU.")

    dev = 0
    p = torch.cuda.get_device_properties(dev)
    cc = (p.major, p.minor)
    total_mem = p.total_memory

    # Computed-from-properties theoretical peak bandwidth (reliable: DDR factor 2).
    theo_bw = 2 * (p.memory_clock_rate * 1e3) * (p.memory_bus_width / 8) / 1e9

    dtypes, dtypes_known = _tensor_core_dtypes(*cc)
    has_tc = p.major >= 7

    print(f"Calibrating {p.name} (SM {p.major}.{p.minor})...")
    meas_bw = measure_bandwidth(total_mem)
    fp32_tflops, fp32_n = measure_gemm_tflops(torch.float32, total_mem)
    if has_tc and ("FP16" in dtypes):
        fp16_tflops, fp16_n = measure_gemm_tflops(torch.float16, total_mem)
    else:
        fp16_tflops, fp16_n = 0.0, 0

    ridge_fp32 = (fp32_tflops * 1e12) / (meas_bw * 1e9) if meas_bw else 0.0
    ridge_fp16 = (fp16_tflops * 1e12) / (meas_bw * 1e9) if (meas_bw and fp16_tflops) else 0.0

    now = _dt.date.today().isoformat()
    dtype_str = ", ".join(dtypes) if dtypes else "none (no tensor cores)"
    if not dtypes_known:
        dtype_str += "  ⚠️ unverified SM — confirm against the CUDA C++ Programming Guide"

    fp16_rows = ""
    if fp16_tflops:
        fp16_rows = (
            f"| **Measured FP16 tensor-core GEMM** | **{fp16_tflops:.1f} TFLOP/s** "
            f"| `torch.matmul` fp16, {fp16_n}³ |\n"
            f"| **FP16 ridge point** | **~{ridge_fp16:.0f} FLOP/byte** "
            f"| measured GEMM / measured BW |\n"
        )

    md = f"""# My GPU Spec Sheet — {p.name}

> Generated by `scripts/calibrate_gpu.py` on {now}. This file is gitignored: it is
> *yours*, measured on *your* machine. The curriculum's roofline math and targets
> are graded against these numbers. Re-run the script if you change GPUs.

## Measured ceilings (what the units grade against)

| Quantity | Value | How |
|---|---|---|
| **Measured memory bandwidth** | **{meas_bw:.0f} GB/s** | device-to-device copy, CUDA-event timed |
| Theoretical memory bandwidth | {theo_bw:.0f} GB/s | `2 × mem_clock × bus_width/8` |
| Bandwidth efficiency | {100 * meas_bw / theo_bw:.0f}% of theoretical | |
| **Measured FP32 GEMM** | **{fp32_tflops:.1f} TFLOP/s** | `torch.matmul` fp32 (true FP32, TF32 off), {fp32_n}³ |
| **FP32 ridge point** | **~{ridge_fp32:.0f} FLOP/byte** | measured GEMM / measured BW |
{fp16_rows}
> FP32 is measured with TF32 **disabled** (`set_float32_matmul_precision('highest')`),
> so it reflects true FP32 CUDA cores — not TF32 tensor cores. The tensor-core row is
> measured at **FP16**; on Ada/Hopper/Blackwell the absolute tensor peak is higher
> still (FP8/FP4), but this curriculum's kernels use FP16/FP32-accumulate, so FP16 is
> the relevant ceiling.
> **Régime rule:** a kernel whose arithmetic intensity (FLOP/byte) is **below** your
> ridge point is **memory-bound** — optimize traffic. Above it, **compute-bound** —
> optimize math. Units 1–5 sit far below your ridge; Unit 6 (GEMM) is the first to cross it.

## Hardware facts

| Item | Value |
|---|---|
| Name | {p.name} |
| Compute capability | {p.major}.{p.minor} |
| SM count | {p.multi_processor_count} |
| SM clock | {p.clock_rate / 1e6:.2f} GHz |
| VRAM | {total_mem / 1024**3:.1f} GB |
| L2 cache | {p.L2_cache_size / 1024**2:.1f} MB |
| Max shared memory / block (opt-in) | {p.shared_memory_per_block_optin / 1024:.0f} KB |
| Max shared memory / SM | {p.shared_memory_per_multiprocessor / 1024:.0f} KB |
| Registers / SM | {p.regs_per_multiprocessor} |
| Warp size | {p.warp_size} |
| Tensor cores | {"yes" if has_tc else "no"} |
| Tensor-core data types | {dtype_str} |

## What this means for the units

- **Units 1–5 (memory-bound):** target a healthy fraction of your **{meas_bw:.0f} GB/s**
  measured bandwidth. Your memcpy (Unit 0) sets the practical ceiling.
- **Unit 6 (GEMM):** your FP32 ceiling is ~{fp32_tflops:.1f} TFLOP/s; your tensor-core
  ceiling is {"~" + format(fp16_tflops, ".1f") + " TFLOP/s" if fp16_tflops else "n/a"}.
  Grade against the *relevant* one, and against cuBLAS (`torch.matmul`).
- **Units 6–7 (tensor cores):** your tensor cores accept **{dtype_str.split('  ⚠')[0]}**.
  Use the highest-throughput dtype your hardware supports for `tl.dot` / `wmma`
  (e.g. FP16 on Turing; BF16/TF32 on Ampere+; FP8 on Ada/Hopper).
- **Unit 7 (FlashAttention):** your shared-memory budget is
  ~{p.shared_memory_per_block_optin / 1024:.0f} KB/block — size your attention tiles to fit it.
"""

    out = Path(__file__).resolve().parent.parent / "docs" / "my-gpu-spec.md"
    out.write_text(md)
    print(f"\n  measured bandwidth : {meas_bw:.0f} GB/s ({100*meas_bw/theo_bw:.0f}% of {theo_bw:.0f} theoretical)")
    print(f"  FP32 GEMM          : {fp32_tflops:.1f} TFLOP/s  -> FP32 ridge ~{ridge_fp32:.0f} FLOP/byte")
    if fp16_tflops:
        print(f"  FP16 tensor GEMM   : {fp16_tflops:.1f} TFLOP/s  -> FP16 ridge ~{ridge_fp16:.0f} FLOP/byte")
    print(f"  tensor-core dtypes : {dtype_str}")
    print(f"\nWrote {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")


if __name__ == "__main__":
    main()
