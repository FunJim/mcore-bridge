# Copyright (c) ModelScope Contributors. All rights reserved.
"""Check the real Swift metadata entry and cached THD routing, on CPU or allocated GPUs."""

import argparse
import json
import os
import torch
import torch.distributed as dist
import warnings
from datetime import timedelta
from megatron.core import parallel_state
from megatron.core.context_parallel_layout import conversion, routes
from megatron.core.packed_seq_params import PackedSeqParams
from pathlib import Path
from swift.megatron.utils import utils as swift_utils
from types import SimpleNamespace
from typing import Union
from unittest.mock import patch


class Group:

    def __init__(self, size: int, rank: int) -> None:
        self._size, self._rank = size, rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


def positions(boundaries: list[int], device: str) -> torch.Tensor:
    return torch.cat([torch.arange(b - a, device=device) for a, b in zip(boundaries, boundaries[1:])]).unsqueeze(0)


def token_indices(boundaries: list[int], size: int, rank: int, mode: str) -> list[int]:
    if mode == 'contiguous':
        n = boundaries[-1] // size
        return list(range(rank * n, (rank + 1) * n))
    result = []
    for a, b in zip(boundaries, boundaries[1:]):
        n = (b - a) // (2 * size)
        for block in (rank, 2 * size - rank - 1):
            result.extend(range(a + block * n, a + (block + 1) * n))
    return result


def packed(boundaries: list[int], group: Union[Group, dist.ProcessGroup], enabled: bool,
           device: str) -> PackedSeqParams:
    args = SimpleNamespace(cp_partition_mode='zigzag', context_parallel_size=group.size())
    with patch.dict(os.environ, {'MCORE_CP_ROUTE_PREBUILD': str(int(enabled))}), \
            patch.object(parallel_state, 'get_context_parallel_group', return_value=group):
        return swift_utils.get_packed_seq_params(args, positions(boundaries, device))


def simulate(boundaries: list[int], size: int, source: str, target: str,
             cached: bool) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    metadata = [packed(boundaries, Group(size, r), cached, 'cpu') for r in range(size)]
    inputs, sends, plans = [], [], []
    for rank, params in enumerate(metadata):
        route = params.cp_partition_route if cached else routes.build_thd_cp_partition_route(
            params.cu_seqlens_q, size, rank)
        ids = token_indices(boundaries, size, rank, source)
        value = torch.tensor(ids, dtype=torch.float64).requires_grad_()
        inputs.append(value)
        if source == 'zigzag':
            send_idx, recv_idx = route.zigzag_index, route.contiguous_index
            send_split, recv_split = route.zigzag_split_sizes, route.contiguous_split_sizes
        else:
            send_idx, recv_idx = route.contiguous_index, route.zigzag_index
            send_split, recv_split = route.contiguous_split_sizes, route.zigzag_split_sizes
        sends.append(value if send_idx is None else value.index_select(0, send_idx))
        plans.append((recv_idx, send_split, recv_split))
    outputs = []
    for rank in range(size):
        recv = torch.cat(
            [value.narrow(0, sum(plans[src][1][:rank]), plans[src][1][rank]) for src, value in enumerate(sends)])
        index = plans[rank][0]
        output = recv if index is None else torch.empty_like(recv).index_copy(0, index, recv)
        expected = torch.tensor(token_indices(boundaries, size, rank, target), dtype=torch.float64)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        outputs.append(output)
    sum(value.square().sum() for value in outputs).backward()
    for value in inputs:
        torch.testing.assert_close(value.grad, 2 * value.detach(), rtol=0, atol=0)
    return [x.detach() for x in outputs], [x.grad for x in inputs]


def cpu_check() -> dict:
    cases = []
    for size in (2, 4, 8):
        for boundaries in ([0, 16 * size], [0, 4 * size, 12 * size, 32 * size]):
            group = Group(size, 0)
            with patch.object(
                    routes, 'build_thd_cp_partition_route', wraps=routes.build_thd_cp_partition_route) as builder:
                reference = packed(boundaries, group, False, 'cpu')
                assert reference.cp_partition_route is None
                ready = packed(boundaries, group, True, 'cpu')
                assert builder.call_count == 1
                route = ready.cp_partition_route
                assert route is not None and ready.cp_group is group
                for _ in range(20):
                    for a, b in (('zigzag', 'contiguous'), ('contiguous', 'zigzag')):
                        assert routes.get_thd_cp_partition_route(ready, a, b) is route
                assert builder.call_count == 1
                next_batch = packed([0, 20 * size], group, True, 'cpu')
                assert builder.call_count == 2 and next_batch.cp_partition_route is not route
            for a, b in (('zigzag', 'contiguous'), ('contiguous', 'zigzag')):
                before = simulate(boundaries, size, a, b, False)
                after = simulate(boundaries, size, a, b, True)
                for old, new in zip(before[0] + before[1], after[0] + after[1]):
                    torch.testing.assert_close(old, new, rtol=0, atol=0)
                cases.append({'cp': size, 'boundaries': boundaries, 'direction': [a, b], 'output_gradient_exact': True})
    no_cp = packed([0, 32], Group(1, 0), True, 'cpu')
    assert no_cp.cp_partition_route is None
    return {
        'passed': True,
        'cases': cases,
        'route_identity_reused': True,
        'next_microbatch_new_route': True,
        'cp1_no_route': True,
        'gpu': False
    }


def gpu_check(output: Path) -> dict:
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=5))
    assert dist.get_world_size() == 8
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=4)
    cp = parallel_state.get_context_parallel_group()
    tp = parallel_state.get_tensor_model_parallel_group()
    cases = []
    for boundaries in ([0, 1024], [0, 256, 640, 1024], [0, 512, 2048]):
        params = packed(boundaries, cp, True, 'cuda')
        route = params.cp_partition_route
        assert route is not None
        for source, target in (('zigzag', 'contiguous'), ('contiguous', 'zigzag')):
            ids = token_indices(boundaries, cp.size(), cp.rank(), source)
            dst = token_indices(boundaries, cp.size(), cp.rank(), target)
            n = len(ids) // tp.size()
            x0 = torch.tensor(
                ids[tp.rank() * n:(tp.rank() + 1) * n], device='cuda', dtype=torch.float32).reshape(-1, 1, 1)
            wanted = torch.tensor(
                dst[tp.rank() * n:(tp.rank() + 1) * n], device='cuda', dtype=torch.float32).reshape(-1, 1, 1)
            outputs, grads = [], []
            for reuse in (False, True, True):
                x = x0.clone().requires_grad_()
                converter = conversion.CpPartitionModeConverter(
                    packed_seq_params=params if reuse else packed(boundaries, cp, False, 'cuda'),
                    source_partition_mode=source,
                    target_partition_mode=target,
                    config=SimpleNamespace(cuda_graph_impl='none'),
                    cp_group=cp,
                    tp_group=tp)
                with warnings.catch_warnings(), patch.object(
                        conversion, 'build_thd_cp_partition_route',
                        wraps=conversion.build_thd_cp_partition_route) as builder:
                    warnings.simplefilter('ignore')
                    y = converter.convert(x, sequence_parallel=True)
                    y.square().sum().backward()
                    assert builder.call_count == (0 if reuse else 1)
                torch.testing.assert_close(y, wanted, rtol=0, atol=0)
                torch.testing.assert_close(x.grad, 2 * x0, rtol=0, atol=0)
                outputs.append(y.detach())
                grads.append(x.grad)
                if reuse:
                    assert params.cp_partition_route is route
            for actual in outputs[1:]:
                torch.testing.assert_close(actual, outputs[0], rtol=0, atol=0)
            for actual in grads[1:]:
                torch.testing.assert_close(actual, grads[0], rtol=0, atol=0)
            cases.append({'boundaries': boundaries, 'direction': [source, target], 'passed': True})
    result = {
        'passed': True,
        'rank': rank,
        'tp': 2,
        'cp': 4,
        'cases': cases,
        'gpu': True,
        'prefix': os.sys.prefix,
        'memory_bytes': torch.cuda.max_memory_allocated()
    }
    (output / f'route_gpu_rank{rank}.json').write_text(json.dumps(result, indent=2) + '\n')
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.gpu:
        result = gpu_check(args.output)
    else:
        result = cpu_check()
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
