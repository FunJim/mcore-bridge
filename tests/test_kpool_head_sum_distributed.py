# Copyright (c) ModelScope Contributors. All rights reserved.
"""Two-GPU NCCL gate for the bundled TP dedup and checkpoint replay selection path."""
import os
import sys
import tempfile
import types
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import test_kpool_tp_dedup as bundle
from test_kpool_checkpoint_replay import load_replay
from test_kpool_head_sum import MODULE as fusion


def worker(rank: int, rendezvous: str, sources: tuple) -> None:
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    replay = load_replay()
    package = types.ModuleType('mcore_bridge')
    package.__path__ = []
    package.kpool_checkpoint_replay = replay
    modules = {'mcore_bridge': package, 'mcore_bridge.kpool_checkpoint_replay': replay,
               'mcore_bridge.kpool_relu_mul': fusion}
    with patch.dict(sys.modules, modules):
        adapter, dsa = bundle.load_sources(sources)
        dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=2,
                                timeout=timedelta(seconds=120))
        try:
            flags = {'MCORE_KPOOL_SELECTION_MODE': 'tp_dedup', 'MCORE_KPOOL_TP_VALIDATION': 'first',
                     'MCORE_KPOOL_HEAD_SUM_FUSION': '1',
                     'MCORE_KPOOL_CHECKPOINT_REPLAY': 'replay'}
            with patch.dict(os.environ, flags), torch.no_grad():
                adapter.initialize_mode(dist.group.WORLD, loss_coeff=0, pool_size=4, index_share=False)
                for large in (False, True):
                    torch.manual_seed(319)
                    kwargs = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                              for k, v in bundle.sample('packed').items()}
                    kwargs['q'] = kwargs['q'].repeat(1, 1, 8, 16)
                    kwargs['weights'] = kwargs['weights'].repeat(1, 1, 8)
                    for name in ('k', 'gate_score'):
                        kwargs[name] = kwargs[name].repeat(1, 1, 16)
                    kwargs['ape'] = kwargs['ape'].repeat(1, 16)
                    kwargs['use_relu'] = True
                    if large:
                        kwargs.update(q=torch.randn(1025, 1, 32, 128, device='cuda', dtype=torch.bfloat16),
                                      k=torch.randn(32768, 1, 128, device='cuda', dtype=torch.bfloat16),
                                      weights=torch.randn(1025, 1, 32, device='cuda'),
                                      gate_score=torch.randn(32768, 1, 128, device='cuda', dtype=torch.bfloat16),
                                      cu_seqlens_kv=None,
                                      varlen_starts=torch.zeros(1025, device='cuda', dtype=torch.int64),
                                      varlen_ends=torch.arange(31744, 32769, device='cuda'),
                                      key_positions=torch.arange(32768, device='cuda'))
                    with patch.dict(os.environ, {'MCORE_KPOOL_HEAD_SUM_FUSION': '0'}):
                        expected = dsa.fused_qk_topk_kpool(**kwargs)[1]
                    for mode in ('observe', 'replay'):
                        with patch.dict(os.environ, {'MCORE_KPOOL_CHECKPOINT_REPLAY': mode}):
                            ctx = types.SimpleNamespace()
                            with patch.object(dsa, '_compute_index_scores', wraps=dsa._compute_index_scores) as score:
                                with replay.checkpoint_region(ctx, 'original'):
                                    first = adapter.run_selection(**kwargs, mode='tp_dedup',
                                                                  tp_group=dist.group.WORLD, check_replicas=True)
                                calls = score.call_count
                                with replay.checkpoint_region(ctx, 'recompute'):
                                    second = adapter.run_selection(**kwargs, mode='tp_dedup',
                                                                   tp_group=dist.group.WORLD, check_replicas=True)
                                assert torch.equal(first.indices, expected)
                                assert torch.equal(second.indices, expected)
                                assert score.call_count == calls * (2 if mode == 'observe' else 1)
                                assert first.local_rows == ((1024 if rank == 0 else 1) if large
                                                            else (41 if rank == 0 else 0))
                            assert replay.replay_stats()['live_bytes'] == 0
                    ctx = types.SimpleNamespace()
                    with replay.checkpoint_region(ctx, 'original'):
                        adapter.run_selection(**kwargs, mode='tp_dedup', tp_group=dist.group.WORLD,
                                              check_replicas=False)
                    changed = {'MCORE_KPOOL_HEAD_SUM_FUSION': '0' if rank == 0 else '1'}
                    try:
                        with patch.dict(os.environ, changed), replay.checkpoint_region(ctx, 'recompute'):
                            adapter.run_selection(**kwargs, mode='tp_dedup', tp_group=dist.group.WORLD,
                                                  check_replicas=False)
                    except replay.ReplayError:
                        pass
                    else:
                        raise AssertionError('Changed fusion flag failed to terminate both ranks')
                    assert replay.replay_stats()['live_bytes'] == 0
                with patch.dict(os.environ, {'MCORE_KPOOL_HEAD_SUM_FUSION': str(rank)}):
                    try:
                        adapter.initialize_mode(dist.group.WORLD, loss_coeff=0, pool_size=4, index_share=False)
                    except RuntimeError as error:
                        assert 'contract mismatch' in str(error)
                    else:
                        raise AssertionError('Mixed fusion modes accepted at initialization')
            dist.barrier()
        finally:
            dist.destroy_process_group()


@unittest.skipUnless(bundle.BASE_WHEEL and torch.cuda.device_count() >= 2, 'Pinned wheel and two GPUs required')
class HeadSumDistributedTest(unittest.TestCase):
    def test_nccl_tp_replay_indices_and_symmetric_failures(self):
        with tempfile.TemporaryDirectory(prefix='headsum-nccl-') as directory:
            rendezvous = (Path(directory) / 'rendezvous').as_uri()
            mp.spawn(worker, args=(rendezvous, bundle.bundled_sources()), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
