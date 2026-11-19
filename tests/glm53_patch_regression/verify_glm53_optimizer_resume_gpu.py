"""Real HDO/DCP save and cold load on bounded synthetic FP32/BF16 master shards."""

import argparse
import ast
import importlib.util
import json
import logging
import math
import os
import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedObject, ShardedTensor
from pathlib import Path
from transformer_engine.pytorch.optimizers import FusedAdam
from types import MethodType, SimpleNamespace


def load(root: Path) -> tuple:
    spec = importlib.util.spec_from_file_location(
        'resume_hybrid',
        root / 'megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = (root / 'megatron/core/optimizer/distrib_optimizer.py').read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'DistributedOptimizer')
    names = {
        '_restore_checkpoint_steps',
        'state_dict',
        'load_state_dict',
        '_get_main_param_and_optimizer_states',
        '_set_main_param_and_optimizer_states',
        'get_parameter_state_dp_reshardable',
        'sharded_param_state_dp_reshardable',
        'load_parameter_state_from_dp_reshardable',
        'sharded_state_dict',
    }
    funcs = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {
        'torch': torch,
        'math': math,
        'HybridDeviceOptimizer': module.HybridDeviceOptimizer,
        'HAVE_APEX_OR_TE': True,
        'USING_TE_OPTIMIZER': True,
        'USING_APEX_OPTIMIZER': False,
        'logger': logging.getLogger('cold-resume'),
        'logging': logging,
        'log_single_rank': lambda *a, **k: None,
        'ShardedObject': ShardedObject,
        'ShardedTensor': ShardedTensor,
        'LocalNonpersistentObject': LocalNonpersistentObject,
        'param_group_identifier_keys': (
            'lr_mult',
            'is_expert_parallel',
            'is_decoupled_lr',
        ),
    }
    exec(
        compile(
            ast.Module(
                body=ast.parse('from __future__ import annotations').body + funcs,
                type_ignores=[],
            ),
            '<runtime-resume>',
            'exec',
        ),
        ns,
    )
    return module.HybridDeviceOptimizer, ns


def make(root: Path, fraction: float, overlap: bool) -> tuple:
    hybrid, ns = load(root)
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range

    params = [
        torch.nn.Parameter(torch.arange(32, device='cuda', dtype=dtype) / 32 + i + 1)
        for i, dtype in enumerate((torch.float32, torch.bfloat16))
    ]
    ranges = []
    mapping = {}
    buffers = []
    shard_size = 32 // dist.get_world_size()
    shard_start = dist.get_rank() * shard_size
    for i, p in enumerate(params):
        dtype = (p.dtype, torch.float32)
        range_map = {
            'param': Range(shard_start, shard_start + shard_size),
            'gbuf_local': Range(0, shard_size),
            'gbuf_world': Range(shard_start, shard_start + shard_size),
        }
        ranges.append({dtype: [{'param_map': {p: range_map}}]})
        mapping[p] = (i, dtype, 0)
        buffers.append(
            SimpleNamespace(buckets=[SimpleNamespace(numel_unpadded=32, grad_data=torch.empty(32, device='cuda'))]))
    group_ranges = [{'params': params, 'orig_group': {'params': params}}]
    config = SimpleNamespace(
        optimizer='adam',
        fp16=False,
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
        fp8_recipe='delayed',
    )
    DistributedOptimizer._build_model_and_main_param_groups(ranges, mapping, group_ranges, config)
    masters = group_ranges[0]['orig_group']['params']
    opt = hybrid(
        [{
            'params': masters,
            'lr_mult': 1,
            'is_expert_parallel': False,
            'is_decoupled_lr': False,
        }],
        offload_fraction=fraction,
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=FusedAdam,
        param_update_in_fp32=True,
        overlap_cpu_optimizer_d2h_h2d=overlap,
        lr=0.01,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    owner = SimpleNamespace(
        optimizer=opt,
        config=config,
        ddp_config=SimpleNamespace(use_megatron_fsdp=False),
        _optimizer_state_offloader=None,
        grad_scaler=None,
        optimizer_state_keys=('exp_avg', 'exp_avg_sq'),
        model_param_group_index_map={
            p: (0, i)
            for i, p in enumerate(params)
        },
        gbuf_ranges=ranges,
        buffers=buffers,
        per_bucket_numel=[shard_size, shard_size],
        per_bucket_numel_unpadded=[32, 32],
        data_parallel_group=dist.group.WORLD,
        data_parallel_group_idx=0,
        distributed_optimizer_instance_id=0,
        checkpoint_fully_reshardable_formats={
            'fully_reshardable',
            'fully_sharded_model_space',
            'fsdp_dtensor',
        },
    )
    for name, value in ns.items():
        if name in {
                'state_dict',
                'load_state_dict',
                '_restore_checkpoint_steps',
                '_get_main_param_and_optimizer_states',
                '_set_main_param_and_optimizer_states',
                'get_parameter_state_dp_reshardable',
                'sharded_param_state_dp_reshardable',
                'load_parameter_state_from_dp_reshardable',
                'sharded_state_dict',
        }:
            setattr(owner, name, MethodType(value, owner))
    owner._resume_test_uses_new_step_rule = '_restore_checkpoint_steps' in ns
    return owner, params, masters


def step(owner: SimpleNamespace, masters: list[torch.Tensor], index: int) -> None:
    for i, p in enumerate(masters):
        p.grad = (torch.linspace(-0.5, 0.5, p.numel(), device='cuda') + index * 0.01 + i * 0.1)
    owner.optimizer.step()
    owner.optimizer.zero_grad()
    torch.cuda.synchronize()


def capture(owner: SimpleNamespace, masters: list[torch.Tensor]) -> dict:
    opt = owner.optimizer
    result = {
        'masters': [p.detach().cpu().clone() for p in masters],
        'inner_parameters': [opt.param_to_inner_param[p].detach().cpu().clone() for p in masters],
        'moments': [{
            k: v.detach().cpu().clone()
            for k, v in opt.state[p].items() if k in ('exp_avg', 'exp_avg_sq')
        } for p in masters],
        'group_steps': [g['step'] for g in owner.state_dict()['optimizer']['param_groups']],
        'native_steps': [
            opt.state[p]['step'].item() if any(
                isinstance(child, (torch.optim.Adam, torch.optim.AdamW)) and any(inner is opt.param_to_inner_param[p]
                                                                                 for group in child.param_groups
                                                                                 for inner in group['params'])
                for child in opt.sub_optimizers) else None for p in masters
        ],
        'child_steps': [
            g.get('step') for child in opt.sub_optimizers
            if not isinstance(child, (torch.optim.Adam, torch.optim.AdamW)) for g in child.param_groups
        ],
    }
    return result


def compare(actual: dict, expected: dict, stage: str) -> None:
    assert actual['group_steps'] == expected['group_steps'], (
        stage,
        actual['group_steps'],
        expected['group_steps'],
    )
    assert actual['native_steps'] == expected['native_steps'], (
        stage,
        actual['native_steps'],
        expected['native_steps'],
    )
    assert actual['child_steps'] == expected['child_steps'], (
        stage,
        actual['child_steps'],
        expected['child_steps'],
    )
    for key in ('masters', 'inner_parameters'):
        for a, b in zip(actual[key], expected[key]):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6, msg=stage + ' ' + key)
    for a, b in zip(actual['moments'], expected['moments']):
        for key in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(a[key], b[key], rtol=2e-6, atol=2e-6, msg=stage + ' ' + key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['save', 'load'], required=True)
    parser.add_argument('--fraction', type=float, required=True)
    parser.add_argument('--overlap', type=int, required=True)
    parser.add_argument('--expect-load-failure', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    torch.cuda.set_per_process_memory_fraction(0.01)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ.setdefault('MASTER_PORT', '29763')
    if args.distributed:
        dist.init_process_group('gloo')
    else:
        dist.init_process_group('gloo', rank=0, world_size=1)
    try:
        owner, params, masters = make(args.root, args.fraction, bool(args.overlap))
        directory = args.output / f"fraction{args.fraction}-overlap{args.overlap}"
        expected_file = directory / f"expected-rank{dist.get_rank()}.pt"
        if args.phase == 'save':
            directory.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                step(owner, masters, i)
            expected = [capture(owner, masters)]
            sd = {'optimizer': owner.sharded_state_dict({}, metadata={'distrib_optim_sharding_type': 'dp_reshardable'})}
            if dist.get_rank() == 0:
                (directory / 'checkpoint').mkdir()
            dist.barrier()
            dist_checkpointing.save(sd, str(directory / 'checkpoint'))
            for i in (3, 4):
                step(owner, masters, i)
                expected.append(capture(owner, masters))
            torch.save(expected, expected_file)
        else:
            expected = torch.load(expected_file, weights_only=False, map_location='cpu')
            sd = {
                'optimizer':
                owner.sharded_state_dict(
                    {},
                    is_loading=True,
                    metadata={'distrib_optim_sharding_type': 'dp_reshardable'},
                )
            }
            loaded = dist_checkpointing.load(sd, str(directory / 'checkpoint'))
            try:
                owner.load_state_dict(loaded['optimizer'])
            except KeyError as error:
                if not args.expect_load_failure or error.args != ('step', ):
                    raise
                print(
                    json.dumps({
                        'negative_control_passed': True,
                        'exception': "KeyError: 'step'",
                    }),
                    flush=True,
                )
                return
            if args.expect_load_failure:
                raise AssertionError('Previous loader did not reproduce the missing-step failure')
            compare(capture(owner, masters), expected[0], 'cold restore')
            for j, i in enumerate((3, 4), 1):
                step(owner, masters, i)
                compare(capture(owner, masters), expected[j], f"update {i + 1}")
        print(
            json.dumps({
                'passed': True,
                'phase': args.phase,
                'offload': args.fraction,
                'overlap': args.overlap,
                'peak_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
                'group_steps': capture(owner, masters)['group_steps'],
                'rank': dist.get_rank(),
                'dp': dist.get_world_size(),
            }),
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
