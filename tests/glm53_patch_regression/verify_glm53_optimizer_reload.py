"""GPU regression for weights-only reload with distributed CPU-offloaded masters."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from transformer_engine.pytorch.optimizers import FusedAdam


def make_optimizer(params: list, fraction: float, overlap: bool) -> HybridDeviceOptimizer:
    return HybridDeviceOptimizer(
        params, offload_fraction=fraction, cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=FusedAdam, param_update_in_fp32=True,
        overlap_cpu_optimizer_d2h_h2d=overlap, lr=0.01, weight_decay=0.0,
        betas=(0.9, 0.999), eps=1e-8,
    )


def verify_distributed(fraction: float, overlap: bool, initial: float,
                       from_state: bool) -> None:
    """Use real shard construction and reload, including partial checkpoint ranges."""
    params = [torch.nn.Parameter(torch.full((32,), initial, device='cuda', dtype=dtype))
              for dtype in (torch.float32, torch.bfloat16)]
    mapping, ranges = {}, []
    shard_range = Range(3, 29)
    for index, param in enumerate(params):
        dtype = (param.dtype, torch.float32)
        ranges.append({dtype: [{'param_map': {param: {'param': shard_range}}}]})
        mapping[param] = (index, dtype, 0)
    group_ranges = [{'params': params, 'orig_group': {'params': params}}]
    config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
                             fp8_recipe='delayed')
    groups = DistributedOptimizer._build_model_and_main_param_groups(
        ranges, mapping, group_ranges, config)
    masters = group_ranges[0]['orig_group']['params']
    optimizer = make_optimizer(masters, fraction, overlap)
    distributed = SimpleNamespace(
        optimizer=optimizer, config=config,
        ddp_config=SimpleNamespace(use_megatron_fsdp=False),
        model_float16_groups=groups[0], model_fp32_groups=groups[1],
        shard_fp32_groups=groups[3], shard_fp32_from_float16_groups=groups[4],
        ensure_master_weights_for_param_sync=lambda: None,
        assert_master_weights_resident=lambda _: None,
        _get_model_param_range_map=lambda _: {'param': shard_range},
        _is_distopt_quantized_param=DistributedOptimizer._is_distopt_quantized_param,
        _build_model_param_to_state_dict_param_map=lambda state: state,
    )
    values = [torch.linspace(2.0, 4.0, 32, device='cuda', dtype=param.dtype)
              for param in params]
    with torch.no_grad():
        for param, value in zip(params, values):
            param.copy_(value)
    state = {param: value.float() + 0.25 for param, value in zip(params, values)} if from_state else None
    expected = [(state[param] if state else param).detach()[3:29].float().cpu().clone()
                for param in params]
    # Public reload also guards empty optimizer groups; bind the actual distributed method.
    distributed.param_groups = optimizer.param_groups
    distributed._copy_model_params_to_main_params = (
        lambda state_dict=None: DistributedOptimizer._copy_model_params_to_main_params(
            distributed, state_dict=state_dict))
    DistributedOptimizer.reload_model_params(distributed, state_dict=state)
    assert not optimizer.state, 'weights-only reload must not create optimizer moments'
    for master, value in zip(masters, expected):
        torch.testing.assert_close(master.cpu(), value, atol=0, rtol=0)
        torch.testing.assert_close(optimizer.param_to_inner_param[master].cpu(), value, atol=0, rtol=0)
    reference = [torch.nn.Parameter(value.clone()) for value in expected]
    reference_optimizer = torch.optim.AdamW(reference, lr=0.01, weight_decay=0.0)
    for step in range(3):
        for master, ref in zip(masters, reference):
            grad = torch.zeros_like(master) if step == 0 else torch.linspace(-0.5, 0.5, 26, device='cuda')
            master.grad = grad
            ref.grad = grad.cpu()
        optimizer.step()
        reference_optimizer.step()
        torch.cuda.synchronize()
        for master, ref in zip(masters, reference):
            torch.testing.assert_close(master.cpu(), ref.detach(), atol=2e-6, rtol=2e-6)
    for param, value in zip(params, values):
        torch.testing.assert_close(param[:3], value[:3], atol=0, rtol=0)
        torch.testing.assert_close(param[29:], value[29:], atol=0, rtol=0)


def verify_hybrid(fraction: float, overlap: bool, dtype: torch.dtype) -> None:
    """Cover direct BF16 masters and native FP32 copies used by precision-aware paths."""
    params = [torch.nn.Parameter(torch.full((16,), float('nan'), device='cuda', dtype=dtype))
              for _ in range(2)]
    optimizer = make_optimizer(params, fraction, overlap)
    with torch.no_grad():
        for param in params:
            param.fill_(3.0)
    optimizer.update_fp32_param_by_new_param()
    for param in params:
        torch.testing.assert_close(optimizer.param_to_inner_param[param].float(),
                                   torch.full_like(optimizer.param_to_inner_param[param].float(), 3.0))
        param.grad = torch.zeros_like(param)
    optimizer.step()
    torch.cuda.synchronize()
    for param in params:
        torch.testing.assert_close(param, torch.full_like(param, 3.0), atol=0, rtol=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--expect-stale', action='store_true', help='Negative control on the previous patch')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    cases = []
    for fraction in (0.0, 0.5, 1.0):
        for overlap in (False, True):
            for initial in (1.0, float('nan')):
                for from_state in (False, True):
                    try:
                        verify_distributed(fraction, overlap, initial, from_state)
                    except AssertionError:
                        if not args.expect_stale:
                            raise
                    else:
                        assert not args.expect_stale, 'negative control failed to reproduce stale masters'
                    cases.append({'path': 'distributed', 'offload': fraction, 'overlap': overlap,
                                  'initial': str(initial), 'from_state': from_state})
            if not args.expect_stale:
                for dtype in (torch.float32, torch.bfloat16):
                    verify_hybrid(fraction, overlap, dtype)
                    cases.append({'path': 'hybrid', 'offload': fraction, 'overlap': overlap,
                                  'dtype': str(dtype)})
    report = {'passed': True, 'negative_control': args.expect_stale,
              'cases': cases, 'gpu': torch.cuda.get_device_name()}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    print('GLM53_OPTIMIZER_RELOAD_REGRESSION_PASS', flush=True)


if __name__ == '__main__':
    main()
