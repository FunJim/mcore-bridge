# Copyright (c) ModelScope Contributors. All rights reserved.
"""Experimental FP32 elementwise fusion; GEMM and head reduction stay in PyTorch."""
import torch


def validate_inputs(scores, weights):
    if scores.dtype != torch.float32 or weights.dtype != torch.float32:
        raise ValueError('KPool ReLU/mul fusion requires FP32 scores and weights')
    if scores.ndim != 4 or tuple(weights.shape) != tuple(scores.shape[:3]):
        raise ValueError('Expected scores[query,batch,heads,keys] and matching weights')
    if scores.device != weights.device:
        raise ValueError('Scores and weights must be on the same device')
    if scores.requires_grad or weights.requires_grad:
        raise ValueError('KPool ReLU/mul fusion is restricted to detached selection')
    expected = 1
    for stride, size in sorted(zip(scores.stride(), scores.shape)):
        if size > 1:
            if stride != expected:
                raise ValueError('KPool ReLU/mul fusion requires dense non-overlapping scores')
            expected *= size
    if any(stride < 0 for stride in weights.stride()):
        raise ValueError('Negative weight strides are unsupported')


_KERNEL = None


def _kernel():
    global _KERNEL, triton, tl
    if _KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(S, W, O, N: tl.constexpr,
                   Q: tl.constexpr, B: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                   SQ: tl.constexpr, SB: tl.constexpr, SH: tl.constexpr, SK: tl.constexpr,
                   WQ: tl.constexpr, WB: tl.constexpr, WH: tl.constexpr,
                   BLOCK: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            q = (offsets // SQ) % Q
            b = (offsets // SB) % B
            h = (offsets // SH) % H
            values = tl.load(S + offsets, offsets < N, other=0)
            weights = tl.load(W + q * WQ + b * WB + h * WH, offsets < N, other=0)
            positive = tl.where(values < 0.0, 0.0, values)
            products = tl.inline_asm_elementwise(
                'mul.rn.f32 $0, $1, $2;', constraints='=f,f,f',
                args=[positive, weights], dtype=tl.float32, is_pure=True, pack=1)
            tl.store(O + offsets, products, offsets < N)

        _KERNEL = kernel
    return _KERNEL


def relu_mul(scores, weights):
    if not torch.is_grad_enabled():
        scores, weights = scores.detach(), weights.detach()
    validate_inputs(scores, weights)
    if not scores.is_cuda:
        raise ValueError('KPool ReLU/mul fusion requires CUDA; there is no silent fallback')
    if any(stride == 0 for stride in scores.stride()):
        raise ValueError('Zero score strides are unsupported')
    output = torch.empty_strided(scores.shape, scores.stride(), dtype=scores.dtype,
                                 device=scores.device)
    if scores.numel():
        import triton
        with torch.cuda.device(scores.device):
            _kernel()[(triton.cdiv(scores.numel(), 256),)](
                scores, weights, output, scores.numel(), *scores.shape, *scores.stride(),
                *weights.stride(), BLOCK=256, enable_fp_fusion=False, num_warps=4)
    return output
