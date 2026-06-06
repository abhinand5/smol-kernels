// Unit 03 CUDA starter: Matrix Transpose
//
// Student task (see docs/unit-03-matrix-transpose.md). Write THREE kernels and
// benchmark all three in sequence -- the three numbers are the deliverable:
//   1. naive    -- direct global->global transpose (uncoalesced writes; slow baseline)
//   2. tiled    -- stage a TILE x TILE tile through __shared__ memory (no padding)
//   3. tiled+pad-- shared tile declared [TILE][TILE + 1] to kill the 32-way bank conflict
//
// Requirements:
// - 2D grid + 2D blocks; __syncthreads() between the shared load and the shared store.
// - Handle non-square, non-tile-divisible matrices with bounds checks on both dims.
// - Tile sizes 16x16 and 32x32.
//
// A.t() is the baseline. Keep this file minimal until you write the kernels yourself.
// Do not paste generated kernel code here.
