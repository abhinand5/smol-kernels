// Unit 05 CUDA starter: LayerNorm (forward)
//
// Student task (see docs/unit-05-layernorm.md): one block per row.
// - Compute mu and var with a reduction. Start with the stable TWO-PASS form
//     mu  = (sum x) / N ;  var = (sum (x - mu)^2) / N
//   then implement WELFORD (one pass) using the parallel merge of (count, mean, M2).
//   Avoid naive E[x^2] - E[x]^2 (catastrophic cancellation -> fails the stability test).
// - rstd = rsqrt(var + eps)  (eps INSIDE the sqrt).
// - Apply (x - mu) * rstd * gamma + beta as a FUSED epilogue before storing.
//   gamma/beta are indexed by feature (column), shared across rows.
//
// M=4096, N in {1024,4096,16384}, eps=1e-5. F.layer_norm is the baseline.
// Reuse the warp+block reduction from Unit 04. Keep this file minimal.
// Do not paste generated kernel code here.
