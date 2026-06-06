// Unit 04 CUDA starter: Softmax
//
// Student task (see docs/unit-04-softmax.md):
// - Write a row-softmax kernel: one block per row.
//   1. reduce to the row max  (warp __shfl_down_sync, then block reduction via __shared__)
//   2. reduce sum of exp(x - max)
//   3. write exp(x - max) / sum
//   The `- max` subtraction is mandatory (overflow -> NaN otherwise).
// - Then write the ONLINE (single-pass) variant: maintain running (m, l), rescaling
//     l = l * exp(m_old - m_new) + exp(x - m_new)
//   and verify it matches the two-pass version. This is the FlashAttention building block.
//
// Handle n_cols not a multiple of block size; masked lanes must not pollute reductions.
// torch.softmax is the baseline. Keep this file minimal until you write the kernels.
// Do not paste generated kernel code here.
