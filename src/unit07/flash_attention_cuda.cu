// Unit 07 CUDA starter: FlashAttention -- execution-model explanation (impl optional)
//
// A full hand-written CUDA FlashAttention is a stretch goal. The REQUIRED deliverable
// (see docs/unit-07-flash-attention.md) is to explain, precisely, the execution model:
//
//   - how one Triton program maps to a CUDA block (one query block per CTA)
//   - where Q_i, K_j, V_j live (shared memory / registers) and how 64 KB SRAM bounds tiles
//   - how Q@Kt and P@V would use wmma tensor-core fragments (FP16 in, FP32 accumulate)
//   - where __syncthreads() barriers sit in the streaming K/V loop
//   - why the online (m, l, O) recurrence avoids the N x N HBM round-trips the naive
//     CUDA version pays -- same FLOPs, far less HBM traffic and memory footprint
//
// Write that explanation here as comments (or implement the kernel if you're ready).
// Do not paste generated kernel code here.
