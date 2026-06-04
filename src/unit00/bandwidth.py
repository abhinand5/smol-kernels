"""
Unit 00 - Exercise: Bandwidth Benchmark

Task: Write a Triton kernel that copies one tensor into another (a memcpy),
then benchmark it for different BLOCK_SIZE values.

The goal is to see how close we can get to the theoretical peak bandwidth
of the RTX 2060 Max-Q (264 GB/s).

1. Write the triton kernel
2. Write the benchmark loop
3. Run it and observe the pattern

Hint: The kernel should be very similar to the one in the unit doc.
"""
import torch
import triton
import triton.language as tl


# === YOUR KERNEL HERE ===
# Write a @triton.jit kernel called `memcpy_kernel` that:
# - Takes x_ptr, y_ptr, n_elements, BLOCK_SIZE
# - Computes pid using tl.program_id(0)
# - Computes offsets using tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
# - Creates a mask: offsets < n_elements
# - Loads from x_ptr, stores to y_ptr


# === YOUR BENCHMARK FUNCTION HERE ===
# Write a function `benchmark_memcpy` that:
# - Creates a random tensor of size N (e.g., 32M elements = 128 MB)
# - For each BLOCK_SIZE in [128, 256, 512, 1024, 2048, 4096, 8192]:
#   - Runs the kernel 10 times
#   - Measures median time
#   - Computes bandwidth = (bytes_read + bytes_written) / time
#   - Prints: BLOCK_SIZE, time (us), bandwidth (GB/s), efficiency (%)


def main():
    # TODO: benchmark
    pass


if __name__ == "__main__":
    main()
