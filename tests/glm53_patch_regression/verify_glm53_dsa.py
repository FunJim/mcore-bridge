"""Check GLM-5.3 sparse attention output/gradients and a full-length GPU case."""
import argparse
import gc
import math
from types import SimpleNamespace

import torch
from megatron.core.transformer.experimental_attention_variant.dsa import (
    _unfused_absorbed_dsa_fn, fused_qk_topk_kpool,
)
from megatron.core.transformer.experimental_attention_variant import dsa_kernels


def run_case(length: int, topk: int, compare: bool) -> None:
    torch.manual_seed(17)
    query = torch.randn(length, 1, 8, 512, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(length, 1, 1, 512, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    with torch.no_grad():
        index_q = torch.randn(length, 1, 32, 128, device='cuda', dtype=torch.bfloat16)
        index_k = torch.randn(length, 1, 128, device='cuda', dtype=torch.bfloat16)
        weights = torch.randn(length, 1, 32, device='cuda')
        gate = torch.randn_like(index_k)
        ape = torch.randn(4, 128, device='cuda', dtype=torch.bfloat16)
        scores, indices = fused_qk_topk_kpool(
            index_q, index_k, weights, topk, 4, gate, ape, use_relu=False, always_select_tail=True,
        )
        del scores, index_q, index_k, weights, gate, ape
    assert indices.size(-1) % 64 != 0, 'Test must include the GLM KPool tail'
    scale = 1 / math.sqrt(512)
    output = dsa_kernels.run_fused_absorbed_sparse_attention(
        SimpleNamespace(dsa_kernel_backend='tilelang'), query, key, indices, scale, 512,
    )
    assert output is not None, 'Fused kernel declined the GLM shape'
    grad = torch.randn_like(output) * 0.01
    output.backward(grad)
    assert torch.isfinite(output).all() and torch.isfinite(query.grad).all() and torch.isfinite(key.grad).all()
    if compare:
        q_ref = query.detach().clone().requires_grad_()
        k_ref = key.detach().clone().requires_grad_()
        reference = _unfused_absorbed_dsa_fn(q_ref, k_ref, indices, scale, 512)
        reference.backward(grad)
        for label, actual, expected in (
            ('output', output, reference), ('dq', query.grad, q_ref.grad), ('dk', key.grad, k_ref.grad),
        ):
            error = (actual.float() - expected.float()).norm() / expected.float().norm()
            print(f'{label} relative L2 error={error.item():.6f}', flush=True)
            assert error < 0.02, (label, error.item())
            torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.01)
    torch.cuda.synchronize()
    print(f'PASS length={length}, topk={topk}, width={indices.size(-1)}, peak_GiB={torch.cuda.max_memory_allocated() / 2**30:.3f}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--long-length', type=int, default=0)
    args = parser.parse_args()
    run_case(257, 64, compare=True)
    if args.long_length:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        run_case(args.long_length, 2048, compare=False)
    print('GLM53_DSA_REGRESSION_PASS', flush=True)


if __name__ == '__main__':
    main()
