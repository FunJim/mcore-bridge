"""GPU regression for native FP32 shards with Megatron CPU optimizer offload."""
import argparse
import copy
from types import SimpleNamespace

import torch
from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from transformer_engine.pytorch.optimizers import FusedAdam


def make_groups() -> tuple[list[torch.nn.Parameter], list[dict], list[torch.Tensor]]:
    """Use the real Megatron shard builder with mixed dtypes and partial ranges."""
    params = [
        torch.nn.Parameter(torch.linspace(0.1, 1.6, 16, device='cuda', dtype=dtype))
        for dtype in (torch.float32, torch.bfloat16, torch.float32, torch.bfloat16)
    ]
    ranges = []
    mapping = {}
    for i, param in enumerate(params):
        param.shared = False
        param.tensor_model_parallel = True
        param.partition_dim = 0
        param.partition_stride = 1
        dtype = (param.dtype, torch.float32)
        ranges.append({dtype: [{'param_map': {param: {'param': Range(2, 14)}}}]})
        mapping[param] = (i, dtype, 0)
    groups = [
        {'params': params[i:i + 2], 'orig_group': {'params': params[i:i + 2]}}
        for i in (0, 2)
    ]
    result = DistributedOptimizer._build_model_and_main_param_groups(
        ranges, mapping, groups,
        SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False, fp8_recipe='delayed'),
    )
    shards = [p for group in groups for p in group['orig_group']['params']]
    for param, shard in zip(params, shards):
        assert shard.shape == (12,)
        assert shard.shared is False and shard.tensor_model_parallel is True
        if param.dtype == torch.float32:
            assert shard.data_ptr() == param.data_ptr() + 2 * param.element_size()
    assert len(result[3]) == 2 and len(result[4]) == 2
    return params, [g['orig_group'] for g in groups], shards


def make_optimizer(groups: list[dict], fraction: float) -> HybridDeviceOptimizer:
    return HybridDeviceOptimizer(
        groups, offload_fraction=fraction, cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=FusedAdam, param_update_in_fp32=True,
        overlap_cpu_optimizer_d2h_h2d=True, lr=0.01, weight_decay=0.01,
        betas=(0.9, 0.999), eps=1e-8,
    )


def verify(fraction: float) -> None:
    params, groups, shards = make_groups()
    assert all(p.is_leaf and p.grad_fn is None for p in shards)
    optimizer = make_optimizer(groups, fraction)
    reference = [torch.nn.Parameter(p.detach().float().cpu().clone()) for p in shards]
    reference_optimizer = torch.optim.AdamW(reference, lr=0.01, weight_decay=0.01)
    untouched = [p.detach().clone() for p in params]
    restored_optimizer = None
    restored_shards = None
    for step in range(3):
        for param in params:
            param.grad = None
        loss = sum(p.float().square().sum() for p in params)
        loss.backward()
        grads = [p.grad[2:14].float().clone() for p in params]
        for shard, ref, grad in zip(shards, reference, grads):
            shard.grad = grad
            ref.grad = grad.cpu()
        if restored_optimizer is not None:
            for shard, grad in zip(restored_shards, grads):
                shard.grad = grad.clone()
            restored_optimizer.step()
        optimizer.step()
        reference_optimizer.step()
        torch.cuda.synchronize()
        for i, (param, shard, ref) in enumerate(zip(params, shards, reference)):
            torch.testing.assert_close(shard.cpu(), ref.detach(), atol=2e-6, rtol=2e-6)
            if restored_shards is not None:
                torch.testing.assert_close(shard, restored_shards[i], atol=2e-6, rtol=2e-6)
            with torch.no_grad():
                param[2:14].copy_(shard)
            torch.testing.assert_close(param[:2], untouched[i][:2])
            torch.testing.assert_close(param[14:], untouched[i][14:])
        if step == 1:
            saved_state = copy.deepcopy(optimizer.state_dict())
            # Megatron saves the step separately from HybridDeviceOptimizer state.
            if fraction:
                metadata = DistributedOptimizer.state_dict(
                    SimpleNamespace(optimizer=optimizer, grad_scaler=None)
                )
                for group, saved in zip(saved_state['param_groups'], metadata['optimizer']['param_groups']):
                    group['step'] = saved['step']
            else:
                for group, saved in zip(saved_state['param_groups'], optimizer.gpu_optimizer.param_groups):
                    group['step'] = saved['step']
            _, restored_groups, restored_shards = make_groups()
            for target, source in zip(restored_shards, shards):
                target.copy_(source)
            restored_optimizer = make_optimizer(restored_groups, fraction)
            restored_optimizer.load_state_dict(saved_state)
    print(f'PASS offload={fraction}: backward, AdamW parity, shared storage, state restore', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--expect-nonleaf', action='store_true')
    args = parser.parse_args()
    if args.expect_nonleaf:
        _, groups, shards = make_groups()
        assert any(not p.is_leaf for p in shards)
        try:
            make_optimizer(groups, 0.5)
        except ValueError as exc:
            assert 'non-leaf' in str(exc)
            print('BASELINE_REPRODUCED: native FP32 shard rejected by HybridDeviceOptimizer', flush=True)
        else:
            raise AssertionError('Expected the original non-leaf failure')
        return
    for fraction in (0.0, 0.5, 1.0):
        verify(fraction)
    print('GLM53_FP32_SHARD_REGRESSION_PASS', flush=True)


if __name__ == '__main__':
    main()
