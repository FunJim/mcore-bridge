"""Allocated GPU regression: real bundled scoring, TP4 selection and replay."""
import argparse
import importlib.util
import json
import os
import sys
import torch
import torch.distributed as dist
import types
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'tests'))
import test_kpool_tp_dedup as bundle  # noqa: E402


def sources(root):
    area = root / 'megatron/core/transformer/experimental_attention_variant'
    return tuple((area / name).read_text() for name in ('dsa.py', 'dsa_masking.py', 'dsa_kpool_kernels.py'))


def bridge_modules():
    package = types.ModuleType('mcore_bridge')
    package.__path__ = []
    sys.modules['mcore_bridge'] = package
    for name in ('kpool_checkpoint_replay', 'kpool_relu_mul'):
        path = REPO / 'src/mcore_bridge' / f'{name}.py'
        spec = importlib.util.spec_from_file_location('mcore_bridge.' + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        setattr(package, name, module)
    return package.kpool_checkpoint_replay


def verify_scores(a, b, root, label):
    mask_a = torch.isneginf(a)
    mask_b = torch.isneginf(b)
    invalid = (not torch.equal(mask_a, mask_b) or bool(torch.isnan(a).any()) or bool(torch.isnan(b).any())
               or bool(torch.isposinf(a).any()) or bool(torch.isposinf(b).any()))
    valid = ~mask_a
    differs = invalid or not torch.equal(a[valid], b[valid])
    maximum = (a[valid] - b[valid]).abs().max().item() if not invalid and valid.any() else 0.0
    failure = torch.tensor(int(differs), device=a.device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(failure, op=dist.ReduceOp.MAX)
    if failure.item():
        if differs:
            changed = (a != b) | torch.isnan(a) | torch.isnan(b) | torch.isposinf(a) | torch.isposinf(b)
            row = int(torch.nonzero(changed.any(-1), as_tuple=False)[0, -1])
            try:
                torch.save(
                    {
                        'reference': a[:, row].cpu(),
                        'candidate': b[:, row].cpu(),
                        'max_delta': maximum,
                        'label': label
                    }, root / f'score-mismatch-rank{dist.get_rank() if dist.is_initialized() else 0}.pt')
            except OSError as error:
                print(f'Failed to save score mismatch: {error}', flush=True)
        raise AssertionError('Score drift ' + label + ': ' + str(maximum))
    return maximum


def verify_selector(adapter, dsa):
    """Exercise cutoff ties and non-tied neighbors on whole and sliced CUDA inputs."""
    rows, keys = 257, 8192
    bases = torch.arange(keys, device='cuda') * 4
    starts = torch.zeros(rows, device='cuda', dtype=torch.int64)
    ends = torch.full_like(starts, keys * 4)
    for case in ('ties', 'random', 'quantized', 'masked', 'cutoff', 'signed_zero'):
        torch.manual_seed(713)
        scores = torch.randn(1, rows, keys, device='cuda')
        if case == 'ties':
            scores.zero_()
        elif case == 'quantized':
            scores = scores.bfloat16().float()
        elif case == 'masked':
            scores.masked_fill_(
                torch.arange(keys, device='cuda')[None, None] >= torch.arange(rows, device='cuda')[None, :, None],
                float('-inf'))
        elif case == 'cutoff':
            scores.fill_(-2)
            scores[..., :511] = 2
            scores[..., 511:521] = 1
            scores[..., 520] = torch.nextafter(scores[..., 520], torch.full_like(scores[..., 520], 2))
        elif case == 'signed_zero':
            scores.zero_()
            scores[..., ::2] = -0.0
        oracle = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :512].clone()
        values, ids = dsa._kpool_ordered_topk(scores, 512)
        assert torch.equal(ids, oracle), case
        assert torch.equal(values, torch.gather(scores, -1, oracle)), case
        assert ids.untyped_storage().nbytes() == ids.numel() * ids.element_size(), case
        whole = adapter._select(scores, bases, starts, ends, 2048, 4, True)
        for width in (1, 64, 256):
            pieces = [
                adapter._select(scores[:, lo:lo + width], bases, starts[lo:lo + width], ends[lo:lo + width], 2048, 4,
                                True) for lo in range(0, rows, width)
            ]
            assert torch.equal(torch.cat(pieces, dim=1), whole), (case, width)
        print(json.dumps({'selector_case': case, 'passed': True}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    torch.backends.cuda.matmul.allow_tf32 = False
    text = sources(args.root)
    bundle.HELPERS.add('_kpool_ordered_topk')
    adapter, dsa = bundle.load_sources(text)
    replay = bridge_modules()
    verify_selector(adapter, dsa)
    if args.distributed:
        dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    rank = dist.get_rank() if args.distributed else 0
    try:
        # Five reference score chunks spread across all four ranks; production heads/dim.
        rows, keys = 2051, 65536
        args.output.mkdir(parents=True, exist_ok=True)
        for case in ('ties', 'random', 'packed'):
            torch.manual_seed(713)
            q = torch.randn(rows, 1, 32, 128, device='cuda', dtype=torch.bfloat16)
            if case == 'ties':
                q.zero_()
            kwargs = dict(
                q=q,
                k=torch.randn(keys, 1, 128, device='cuda', dtype=torch.bfloat16),
                weights=torch.randn(rows, 1, 32, device='cuda'),
                gate_score=torch.randn(keys, 1, 128, device='cuda', dtype=torch.bfloat16),
                ape=torch.randn(4, 128, device='cuda'),
                pool_size=4,
                index_topk=2048,
                varlen_starts=torch.zeros(rows, device='cuda', dtype=torch.int64),
                varlen_ends=torch.full((rows, ), keys, device='cuda', dtype=torch.int64),
                use_relu=True,
                always_select_tail=True)
            if case == 'packed':
                kwargs.update(
                    cu_seqlens_kv=torch.tensor([0, 32767, keys], device='cuda', dtype=torch.int32),
                    varlen_ends=torch.full((rows, ), 32767, device='cuda', dtype=torch.int64))
            for fusion in ('0', '1'):
                os.environ['MCORE_KPOOL_HEAD_SUM_FUSION'] = fusion
                os.environ['MCORE_KPOOL_CHECKPOINT_REPLAY'] = 'off'
                scores, expected = dsa.fused_qk_topk_kpool(**kwargs)
                pooled, bases = adapter._prepare(kwargs['k'], kwargs['gate_score'], kwargs['ape'], 4,
                                                 kwargs.get('cu_seqlens_kv'))
                starts, ends, positions = adapter._bounds(q, bases, 4, kwargs['varlen_starts'], kwargs['varlen_ends'],
                                                          None, kwargs.get('cu_seqlens_kv'))
                chunk_rows = min(rows, max(1, 1024**3 // (32 * bases.numel() * 4)))
                pieces = []
                for lo in range(0, rows, chunk_rows):
                    hi = min(rows, lo + chunk_rows)
                    part = dsa._compute_index_scores(q[lo:hi], kwargs['weights'][lo:hi], pooled)
                    part = dsa.dsa_masking.apply_starts_ends_mask_to_scores(part, starts[lo:hi], ends[lo:hi], positions)
                    verify_scores(scores[:, lo:hi], part, args.output, f'{case}/fusion{fusion}/{lo}:{hi}')
                    pieces.append(adapter._select(part, bases, starts[lo:hi], ends[lo:hi], 2048, 4, True))
                assert torch.equal(torch.cat(pieces, 1), expected), 'Sharded selector parity'
                if args.distributed:
                    for mode in ('off', 'observe', 'replay'):
                        os.environ['MCORE_KPOOL_CHECKPOINT_REPLAY'] = mode
                        ctx = types.SimpleNamespace()
                        original = dsa._compute_index_scores
                        score_calls = []

                        def score(*a, **k):
                            score_calls.append(a[0].shape[0])
                            return original(*a, **k)

                        with patch.object(dsa, '_compute_index_scores', score):
                            if mode == 'off':
                                first = adapter.select_and_validate(
                                    **kwargs, mode='tp_dedup', tp_group=dist.group.WORLD)
                                second = first
                                first_calls = len(score_calls)
                            else:
                                with replay.checkpoint_region(ctx, 'original'):
                                    first = adapter.select_and_validate(
                                        **kwargs, mode='tp_dedup', tp_group=dist.group.WORLD)
                                first_calls = len(score_calls)
                                with replay.checkpoint_region(ctx, 'recompute'):
                                    second = adapter.select_and_validate(
                                        **kwargs, mode='tp_dedup', tp_group=dist.group.WORLD)
                            assert first.local_rows > 0, 'Every TP rank must compute rows'
                            assert torch.equal(first.indices, expected) and torch.equal(second.indices, expected)
                            stats = replay.replay_stats()
                            assert stats['live_bytes'] == stats['live_entries'] == 0
                            expected_calls = first_calls if mode == 'off' else (
                                first_calls * 2 if mode == 'observe' else first_calls + 1)
                            assert len(score_calls) == expected_calls, (mode, score_calls)
                        print(
                            json.dumps({
                                'case': case,
                                'fusion': fusion,
                                'mode': mode,
                                'tp': dist.get_world_size(),
                                'rank': rank,
                                'local_rows': first.local_rows,
                                'passed': True,
                                'score_calls': len(score_calls),
                                'original_score_calls': first_calls
                            }),
                            flush=True)
                else:
                    print(
                        json.dumps({
                            'case': case,
                            'fusion': fusion,
                            'exact_score_parity': True,
                            'index_parity': True,
                            'passed': True
                        }),
                        flush=True)
                del scores, expected, pooled, bases, pieces, part
            del kwargs, q
        print(
            json.dumps({
                'GPU_GATE_PASS': True,
                'rank': rank,
                'peak_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20
            }),
            flush=True)
    finally:
        if args.distributed:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
