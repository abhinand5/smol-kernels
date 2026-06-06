// Unit 02 CUDA starter: Fused ReLU / GeLU
//
// Student task (see docs/unit-02-fused-activation.md):
// - Write a scalar __global__ kernel: y = activation(x + b), activation in {relu, gelu}.
//   Keep (x + b) in a register; do NOT write it to global memory between add and act.
// - Write a SECOND, vectorized variant using float4 loads/stores (4 elems/thread).
// - Implement both ReLU and a GeLU variant to observe the SFU gap (erf/tanh on the SFU).
//
// Block sizes 128, 256, 512, 1024. torch.relu / F.gelu are baselines.
// Keep this file minimal until you write the kernels yourself.
// Do not paste generated kernel code here.
