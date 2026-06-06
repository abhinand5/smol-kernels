# Unit 08 - Kernel Annotation

> Student deliverable. Pick ONE production kernel, answer the eight dissection
> questions, and walk it line by line. See docs/unit-08-pick-your-battle.md.

## Kernel under study

- **Project / file:** _(e.g. vllm/csrc/layernorm_kernels.cu)_
- **Commit / line range:** _(pin it)_
- **What it computes (math + PyTorch equivalent):** _(...)_

## The eight dissection questions

1. **Signature & contract** — inputs, outputs, dtypes, shapes, strides:
2. **Launch configuration** — grid; what one block/program owns:
3. **Memory hierarchy** — what's in shared/SRAM vs registers; global access pattern; coalesced?
4. **Reductions & sync** — where the reductions are; what race each barrier prevents:
5. **Numerics** — stability tricks (max-subtraction, FP32 accumulate, eps, rescaling) and why:
6. **Roofline** — memory- or compute-bound; arithmetic intensity; which ceiling:
7. **Tuning surface** — what's autotuned/templated; baked-in hardware assumptions:
8. **The clever bit** — the one non-obvious trick:

## Line-by-line walk (execution order)

_(annotate every non-trivial line)_

## Hardware mismatch

_(kernel's intended target, e.g. A100/H100, vs this 2060 Max-Q: 64 KB SRAM,
FP16-only tensor cores, no TF32/BF16/FP8 — what would you change?)_
