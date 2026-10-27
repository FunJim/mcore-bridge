# Copyright (c) ModelScope Contributors. All rights reserved.
"""Default-off four-stream FP32 post-BDA; no Sinkhorn or mapping changes."""
from typing import Optional

import torch


_KERNELS = None


def validate_inputs(h_res: torch.Tensor, residual: torch.Tensor, h_post: torch.Tensor,
                    x: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
    if residual.ndim != 4 or residual.shape[2:] != (4, 4096):
        raise ValueError('mHC post-BDA requires residual[sequence,batch,4,4096]')
    shape = residual.shape[:2]
    if h_res.shape != shape + (4, 4) or h_post.shape != shape + (4,) or x.shape != shape + (4096,):
        raise ValueError('Invalid mHC mapping or activation shape')
    if h_res.dtype != torch.float32 or h_post.dtype != torch.float32:
        raise ValueError('mHC mappings must stay FP32')
    if residual.dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
        raise ValueError('mHC post-BDA requires BF16 residual and activation')
    values = (h_res, residual, h_post, x) + (() if bias is None else (bias,))
    if bias is not None and (bias.shape != (4096,) or bias.dtype not in (torch.float32, torch.bfloat16)):
        raise ValueError('mHC bias must be FP32/BF16 bias[4096]')
    if any(value.device != residual.device or not value.is_contiguous() for value in values):
        raise ValueError('mHC inputs must be contiguous and on the same device')


def _kernels():
    global _KERNELS, triton, tl
    if _KERNELS is None:
        import triton
        import triton.language as tl

        @triton.jit
        def mcore_mhc_post_bda_fwd(HR, R, HP, X, BIAS, O,
                                   HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
            row = tl.program_id(0)
            c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            xv = tl.load(X + row * 4096 + c).to(tl.float32)
            if HAS_BIAS:
                bv = tl.load(BIAS + c).to(tl.float32)
            r0 = tl.load(R + row * 16384 + c).to(tl.float32)
            r1 = tl.load(R + row * 16384 + 4096 + c).to(tl.float32)
            r2 = tl.load(R + row * 16384 + 8192 + c).to(tl.float32)
            r3 = tl.load(R + row * 16384 + 12288 + c).to(tl.float32)
            for i in tl.static_range(4):
                mixed = tl.load(HR + row * 16 + i) * r0
                mixed = mixed + tl.load(HR + row * 16 + 4 + i) * r1
                mixed = mixed + tl.load(HR + row * 16 + 8 + i) * r2
                mixed = mixed + tl.load(HR + row * 16 + 12 + i) * r3
                hp = tl.load(HP + row * 4 + i)
                post = hp * xv
                if HAS_BIAS:
                    post = post + hp * bv
                tl.store(O + row * 16384 + i * 4096 + c, post + mixed)

        @triton.jit
        def mcore_mhc_post_bda_bwd_inputs(HR, HP, G, DR, DX, BLOCK: tl.constexpr):
            row = tl.program_id(0)
            c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            g0 = tl.load(G + row * 16384 + c).to(tl.float32)
            g1 = tl.load(G + row * 16384 + 4096 + c).to(tl.float32)
            g2 = tl.load(G + row * 16384 + 8192 + c).to(tl.float32)
            g3 = tl.load(G + row * 16384 + 12288 + c).to(tl.float32)
            dx = tl.load(HP + row * 4) * g0
            dx = dx + tl.load(HP + row * 4 + 1) * g1
            dx = dx + tl.load(HP + row * 4 + 2) * g2
            dx = dx + tl.load(HP + row * 4 + 3) * g3
            tl.store(DX + row * 4096 + c, dx)
            for j in tl.static_range(4):
                dr = tl.load(HR + row * 16 + j * 4) * g0
                dr = dr + tl.load(HR + row * 16 + j * 4 + 1) * g1
                dr = dr + tl.load(HR + row * 16 + j * 4 + 2) * g2
                dr = dr + tl.load(HR + row * 16 + j * 4 + 3) * g3
                tl.store(DR + row * 16384 + j * 4096 + c, dr)

        @triton.jit
        def mcore_mhc_post_bda_bwd_mappings(R, X, BIAS, G, DHR, DHP,
                                            HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
            row = tl.program_id(0)
            i = tl.program_id(1)
            c = tl.arange(0, BLOCK)
            g = tl.load(G + row * 16384 + i * 4096 + c).to(tl.float32)
            xv = tl.load(X + row * 4096 + c).to(tl.float32)
            dhp = tl.sum(g * xv, 0)
            if HAS_BIAS:
                bv = tl.load(BIAS + c).to(tl.float32)
                dhp = dhp + tl.sum(g * bv, 0)
            tl.store(DHP + row * 4 + i, dhp)
            for j in tl.static_range(4):
                rv = tl.load(R + row * 16384 + j * 4096 + c).to(tl.float32)
                tl.store(DHR + row * 16 + j * 4 + i, tl.sum(g * rv, 0))

        _KERNELS = (mcore_mhc_post_bda_fwd, mcore_mhc_post_bda_bwd_inputs,
                    mcore_mhc_post_bda_bwd_mappings)
    return _KERNELS


class _PostBDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_res, residual, h_post, x, bias):
        from megatron.core.utils import nvtx_range

        with nvtx_range('mhc/post_bda_fwd'):
            rows = residual.shape[0] * residual.shape[1]
            out = torch.empty_like(residual)
            ctx.has_bias = bias is not None
            ctx.save_for_backward(h_res, residual, h_post, x, *(() if bias is None else (bias,)))
            if rows:
                with torch.cuda.device(residual.device):
                    _kernels()[0][(rows, 16)](
                        h_res, residual, h_post, x, bias if bias is not None else x, out,
                        HAS_BIAS=bias is not None, BLOCK=256, num_warps=4, enable_fp_fusion=False)
            return out

    @staticmethod
    def backward(ctx, grad_output):
        if torch.is_grad_enabled():
            raise RuntimeError('Fused mHC post-BDA does not support higher-order gradients')
        from megatron.core.utils import nvtx_range

        with nvtx_range('mhc/post_bda_bwd'):
            h_res, residual, h_post, x, *saved_bias = ctx.saved_tensors
            bias = saved_bias[0] if ctx.has_bias else None
            rows = residual.shape[0] * residual.shape[1]
            with nvtx_range('mhc/post_bda_bwd_prepare'):
                g = grad_output.contiguous()
                dhr, dr, dhp = torch.empty_like(h_res), torch.empty_like(residual), torch.empty_like(h_post)
                dx_fp32 = torch.empty(x.shape, dtype=torch.float32, device=x.device)
            if rows:
                with torch.cuda.device(residual.device):
                    _, inputs, mappings = _kernels()
                    with nvtx_range('mhc/post_bda_bwd_inputs'):
                        inputs[(rows, 16)](h_res, h_post, g, dr, dx_fp32, BLOCK=256,
                                           num_warps=4, enable_fp_fusion=False)
                    with nvtx_range('mhc/post_bda_bwd_mappings'):
                        mappings[(rows, 4)](residual, x, bias if bias is not None else x, g, dhr, dhp,
                                            HAS_BIAS=ctx.has_bias, BLOCK=4096,
                                            num_warps=8, enable_fp_fusion=False)
            with nvtx_range('mhc/post_bda_bwd_postprocess'):
                dbias = dx_fp32.reshape(rows, 4096).sum(0).to(bias.dtype) if ctx.has_bias else None
                return dhr, dr, dhp, dx_fp32.to(x.dtype), dbias


def post_bda(h_res: torch.Tensor, residual: torch.Tensor, h_post: torch.Tensor,
             x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mix in FP32 and round once to BF16, saving original inputs rather than FP32 copies."""
    validate_inputs(h_res, residual, h_post, x, bias)
    if not residual.is_cuda:
        raise ValueError('Fused mHC post-BDA requires CUDA; there is no silent fallback')
    with torch.autocast(device_type='cuda', enabled=False):
        return _PostBDA.apply(h_res, residual, h_post, x, bias)
