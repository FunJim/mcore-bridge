# Copyright (c) ModelScope Contributors. All rights reserved.
"""FP32 ReLU/weight/32-head reduction fusion for detached KPool selection; GEMM is unchanged."""
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


_SUM_KERNEL = None


def _sum_kernel():
    global _SUM_KERNEL, triton, tl
    if _SUM_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def mcore_kpool_relu_mul_head_sum(S, W, O, Q: tl.constexpr, B: tl.constexpr,
                                         H: tl.constexpr, K: tl.constexpr,
                                         SQ: tl.constexpr, SB: tl.constexpr, SH: tl.constexpr,
                                         WQ: tl.constexpr, WB: tl.constexpr, WH: tl.constexpr,
                                         OQ: tl.constexpr, OB: tl.constexpr, OK: tl.constexpr,
                                         BLOCK: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            k = offsets % K
            b = (offsets // K) % B
            q = offsets // (K * B)
            valid = q < Q
            a0 = tl.full((BLOCK,), 0.0, tl.float32)
            a1 = tl.full((BLOCK,), 0.0, tl.float32)
            a2 = tl.full((BLOCK,), 0.0, tl.float32)
            a3 = tl.full((BLOCK,), 0.0, tl.float32)
            for h in tl.static_range(H):
                value = tl.load(S + q * SQ + b * SB + h * SH + k, valid, other=0)
                weight = tl.load(W + q * WQ + b * WB + h * WH, valid, other=0)
                positive = tl.where(value < 0.0, 0.0, value)
                product = tl.inline_asm_elementwise(
                    'mul.rn.f32 $0, $1, $2;', '=f,f,f', [positive, weight],
                    dtype=tl.float32, is_pure=True, pack=1)
                if h % 4 == 0:
                    a0 = a0 + product
                elif h % 4 == 1:
                    a1 = a1 + product
                elif h % 4 == 2:
                    a2 = a2 + product
                else:
                    a3 = a3 + product
            result = ((a0 + a1) + a2) + a3
            tl.store(O + q * OQ + b * OB + k * OK, result, valid)

        _SUM_KERNEL = mcore_kpool_relu_mul_head_sum
    return _SUM_KERNEL


def relu_mul_sum(scores: torch.Tensor, weights: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """Detached FP32 selection with ATen's four-accumulator, non-warp head reduction."""
    if not torch.is_grad_enabled():
        scores, weights = scores.detach(), weights.detach()
    validate_inputs(scores, weights)
    if not scores.is_cuda:
        raise ValueError('KPool ReLU/mul/head-sum fusion requires CUDA; there is no silent fallback')
    from megatron.core.utils import nvtx_range

    with nvtx_range('kpool/head_sum'):
        q, batch, heads, keys = scores.shape
        if heads != 32 or scores.stride(-1) != 1 or scores.stride(2) != keys:
            raise ValueError('KPool head-sum fusion requires 32 heads and contiguous head/key axes')
        shape = (q, batch, keys)
        if out is None:
            out = torch.empty(shape, dtype=torch.float32, device=scores.device)
        if (tuple(out.shape) != shape or out.dtype != torch.float32 or out.device != scores.device
                or out.requires_grad or not out.is_contiguous()):
            raise ValueError('Expected a detached contiguous FP32 output[query,batch,keys]')
        if (out.untyped_storage().data_ptr() in
                (scores.untyped_storage().data_ptr(), weights.untyped_storage().data_ptr()) and out.numel()):
            raise ValueError('KPool head-sum output must not alias its inputs')
        if out.numel():
            import triton
            with torch.cuda.device(scores.device):
                _sum_kernel()[(triton.cdiv(out.numel(), 256),)](
                    scores, weights, out, *scores.shape, *scores.stride()[:3], *weights.stride(),
                    *out.stride(), BLOCK=256, enable_fp_fusion=False, num_warps=4)
        return out
