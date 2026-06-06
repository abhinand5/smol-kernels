// Unit 06 CUDA starter: Tiled GEMM
//
// Student task (see docs/unit-06-tiled-gemm.md). Write and benchmark in order:
//   1. naive          -- one thread per C_ij, global reads (memory-bound, intensity ~0.25)
//   2. shared tiled   -- stage TILE x TILE tiles of A,B through __shared__; loop over K
//   3. register-blocked-- each thread accumulates an R x R micro-tile in registers (-> peak)
//   4. (stretch) wmma -- tensor cores: FP16 inputs, FP32 accumulate (NOT TF32 on Turing)
//
// Requirements:
// - Two __syncthreads() per K-iteration: after the shared load, and before overwriting tiles.
// - Handle non-tile-divisible M, N, K (mask all three dims).
// - Accumulate in FP32 even with FP16 inputs.
// - Report TFLOP/s = 2*M*N*K / s / 1e12; grade vs FP32 (~4.55) or tensor (~36/18) ceiling.
//
// torch.matmul (cuBLAS) is the baseline. Keep this file minimal until you write the kernels.
// Do not paste generated kernel code here.
