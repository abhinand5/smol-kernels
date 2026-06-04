"""
Unit 00 - Exercise: Bandwidth Benchmark

Goal: implement the same memcpy-style kernel in CUDA and Triton, then compare
both against a PyTorch baseline.

Student-owned work:
- Write the CUDA kernel in src/unit00/bandwidth_cuda.cu or an equivalent file.
- Write the Triton kernel in this file or split it into bandwidth_triton.py.
- Understand every line of launch/grid math.

AI/helper-owned work allowed:
- benchmark harnesses
- correctness checks
- build glue
- review feedback

Do not treat torch.copy_ or cudaMemcpy as your own kernel. They are baselines.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# === TRITON KERNEL TODO ===
# Write a @triton.jit kernel called `memcpy_kernel` that:
# - Takes x_ptr, y_ptr, n_elements, BLOCK_SIZE
# - Computes pid using tl.program_id(0)
# - Computes offsets using tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
# - Creates a mask: offsets < n_elements
# - Loads from x_ptr with the mask
# - Stores to y_ptr with the mask


# === CUDA KERNEL TODO ===
# Write the CUDA version separately, suggested path:
#   src/unit00/bandwidth_cuda.cu
# It should:
# - Use a __global__ kernel
# - Compute global thread index from blockIdx/threadIdx
# - Guard idx < n
# - Copy x[idx] to y[idx]


# === BENCHMARK TODO ===
# Implement a benchmark that:
# - Allocates x and y on CUDA once, outside the timed region
# - Uses N = 32 * 1024 * 1024 float32 elements by default
# - Warms up before measuring
# - Measures median runtime
# - Computes bandwidth as N * 8 / seconds for float32 copy
# - Verifies y matches x
# - Reports CUDA kernel, Triton kernel, and PyTorch baseline
# - Prints device name, dtype, N, time, GB/s, and efficiency vs 264 GB/s


def main() -> None:
    # TODO: benchmark after student implements kernels.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print(f"device: {torch.cuda.get_device_name(0)}")
    print("Unit 00 starter loaded. Implement CUDA + Triton memcpy kernels next.")


if __name__ == "__main__":
    main()
