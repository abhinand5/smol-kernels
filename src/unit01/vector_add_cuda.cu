// Unit 01 CUDA starter: Vector Add
//
// Student task (see docs/unit-01-vector-add.md):
// - Write a __global__ kernel computing c[idx] = a[idx] + b[idx].
// - Compute the global index from blockIdx.x, blockDim.x, threadIdx.x.
// - Guard out-of-bounds access with `if (idx < n)`.
// - Then write a SECOND kernel using a grid-stride loop:
//     for (idx = gid; idx < n; idx += gridDim.x * blockDim.x) ...
// - Use size_t / int64_t for the index in code meant to generalize past 2^31.
//
// Benchmark block sizes 128, 256, 512, 1024. `a + b` in PyTorch is the baseline.
// Keep this file minimal until you write the kernels yourself.
// Do not paste generated kernel code here.
