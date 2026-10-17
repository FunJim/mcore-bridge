# Copyright (c) ModelScope Contributors. All rights reserved.
"""Independent strict-parity gates and frozen-input ABBA timing for ReLU/mul."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
import test_kpool_tp_dedup as bundle

spec = importlib.util.spec_from_file_location(
    'mcore_bridge.kpool_relu_mul', ROOT / 'src/mcore_bridge/kpool_relu_mul.py')
fusion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion)


def reference(scores, weights):
    return scores.clone(memory_format=torch.preserve_format).relu_() * weights.unsqueeze(-1)


def assert_bits(left, right):
    if left.shape != right.shape or left.stride() != right.stride():
        raise AssertionError('Shape or stride changed')
    if not torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32)):
        raise AssertionError('FP32 products differ bitwise, including NaN and signed zero')


def tensors(device='cpu', dtype=torch.float32, batch=2):
    generator = torch.Generator().manual_seed(19)
    q = torch.randn(19, batch, 4, 8, generator=generator).to(device=device, dtype=dtype)
    k = torch.randn(37, batch, 8, generator=generator).to(device=device, dtype=dtype)
    weights = torch.randn(19, batch, 4, generator=generator).to(device)
    scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())
    return scores, weights


def load_dsa():
    return bundle.load_sources(bundle.bundled_sources())[1]


def selection_pair(dsa, kwargs):
    package = types.ModuleType('mcore_bridge')
    package.__path__ = []
    modules = {'mcore_bridge': package, 'mcore_bridge.kpool_relu_mul': fusion}
    with patch.dict(sys.modules, modules), torch.no_grad():
        with patch.dict(os.environ, {'MCORE_KPOOL_RELU_MUL_FUSION': '0'}):
            expected = dsa.fused_qk_topk_kpool(**kwargs)
        with patch.dict(os.environ, {'MCORE_KPOOL_RELU_MUL_FUSION': '1'}):
            actual = dsa.fused_qk_topk_kpool(**kwargs)
    assert_bits(expected[0], actual[0])
    if not torch.equal(expected[1], actual[1]):
        raise AssertionError('Ordered KPool token indices changed')
    return expected


class Contracts(unittest.TestCase):
    def test_dense_layout_and_broadcast_weights(self):
        scores, weights = tensors()
        fusion.validate_inputs(scores, weights)
        self.assertEqual(reference(scores, weights).stride(), scores.stride())
        fusion.validate_inputs(scores, weights[:, :1].expand_as(weights))

    def test_invalid_inputs_fail(self):
        scores, weights = tensors()
        for left, right in [(scores.bfloat16(), weights),
                            (scores, weights.bfloat16()),
                            (scores[..., ::2], weights),
                            (scores.clone().requires_grad_(), weights)]:
            with self.assertRaises(ValueError):
                fusion.validate_inputs(left, right)

    def test_no_silent_cpu_fallback(self):
        scores, weights = tensors()
        with self.assertRaisesRegex(ValueError, 'requires CUDA'):
            fusion.relu_mul(scores, weights)
        weights.requires_grad_()
        with torch.no_grad(), self.assertRaisesRegex(ValueError, 'requires CUDA'):
            fusion.relu_mul(scores, weights)
        self.assertTrue(weights.requires_grad)
        with self.assertRaisesRegex(ValueError, 'detached selection'):
            fusion.relu_mul(scores, weights)

    @unittest.skipUnless(bundle.BASE_WHEEL, 'MEGATRON_PATCH_BASE_WHEEL required')
    def test_default_path_and_flag(self):
        dsa = load_dsa()
        kwargs = bundle.sample()
        with patch.dict(os.environ, {'MCORE_KPOOL_RELU_MUL_FUSION': '0'}):
            actual = dsa._compute_index_scores(kwargs['q'], kwargs['weights'], kwargs['k'])
        scores = torch.einsum('sbhd,tbd->sbht', kwargs['q'].float(), kwargs['k'].float())
        expected = reference(scores, kwargs['weights']).sum(2).transpose(0, 1)
        assert_bits(expected, actual)
        with patch.dict(os.environ, {'MCORE_KPOOL_RELU_MUL_FUSION': 'invalid'}):
            with self.assertRaises(ValueError):
                dsa._compute_index_scores(kwargs['q'], kwargs['weights'], kwargs['k'])


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required; parity not yet validated')
class CudaParity(unittest.TestCase):
    def test_product_bits_and_layout(self):
        for batch in (1, 2):
            for dtype in (torch.float32, torch.bfloat16):
                scores, weights = tensors('cuda', dtype, batch)
                for s, w in [(scores, weights), (scores.contiguous(), weights),
                             (scores, weights[:, :1].expand_as(weights)),
                             (scores, torch.ones(38, batch, 4, device='cuda')[::2])]:
                    with self.subTest(batch=batch, dtype=dtype, stride=s.stride()):
                        original = s.clone()
                        assert_bits(reference(s, w), fusion.relu_mul(s, w))
                        assert_bits(original, s)

    def test_ieee_edges(self):
        bits = torch.tensor([0, -2147483648, 1, -2147483647, 8388607, 8388608,
                             2139095039, 2139095040, -8388608, 2143289345, -4194303],
                            dtype=torch.int32, device='cuda')
        values = bits.view(torch.float32)
        scores = values.repeat(4).reshape(1, 1, 4, -1)
        for w in [0., -0., 1., -1., float('inf'), float('nan'), 1.e-20]:
            weights = torch.full((1, 1, 4), w, device='cuda')
            with self.subTest(weight=w):
                assert_bits(reference(scores, weights), fusion.relu_mul(scores, weights))

    @unittest.skipUnless(bundle.BASE_WHEEL, 'MEGATRON_PATCH_BASE_WHEEL required')
    def test_scores_and_ordered_selection(self):
        dsa = load_dsa()
        for case in ('packed', 'unpacked', 'zigzag', 'ties', 'near_ties',
                     'explicit_positions', 'empty_rows'):
            for dtype in (torch.float32, torch.bfloat16):
                kwargs = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                          for k, v in bundle.sample(case, dtype=dtype).items()}
                with self.subTest(case=case, dtype=dtype):
                    selection_pair(dsa, kwargs)


def abba(args):
    if not torch.cuda.is_available():
        raise RuntimeError('ABBA timing needs CUDA; no CPU substitute is accepted')
    payload = torch.load(args.input, map_location='cuda', weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError('Frozen input must be a tensor dictionary')
    if 'scores' in payload:
        scores, weights = payload['scores'], payload['weights']
    else:
        if not bundle.BASE_WHEEL:
            raise ValueError('MEGATRON_PATCH_BASE_WHEEL is required for frozen KPool inputs')
        selection_pair(load_dsa(), payload)
        q, k, weights = payload['q'], payload['k'], payload['weights']
        # Time one production chunk of the 4D pre-reduction scores, exactly the
        # unit _compute_index_scores processes per iteration (~1GiB by design);
        # the whole 4D tensor cannot fit on a single GPU.
        import os
        sq, batch, n_heads, head_dim = q.shape
        sk = k.size(0)
        bytes_per_token = batch * n_heads * sk * 4
        chunk_size = min(sq, max(1, 1024 * 1024 * 1024 // max(1, bytes_per_token)))
        scores = torch.einsum('sbhd,tbd->sbht', q[:chunk_size].float(), k.float())
        if payload.get('use_relu', True):
            pass  # keep raw GEMM output; ReLU belongs to the timed operation
        else:
            raise ValueError('ABBA requires use_relu=True frozen inputs')
        weights = weights[:chunk_size]
    fusion.validate_inputs(scores, weights)
    assert_bits(reference(scores, weights), fusion.relu_mul(scores, weights))
    operations = {'reference': lambda: reference(scores, weights),
                  'fused': lambda: fusion.relu_mul(scores, weights)}
    for operation in operations.values():
        for _ in range(args.warmup):
            operation()
    torch.cuda.synchronize()
    segments = []
    for round_index in range(args.rounds):
        for slot, mode in enumerate(('reference', 'fused', 'fused', 'reference')):
            measurements = []
            for _ in range(args.steps):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                result = operations[mode]()
                end.record()
                end.synchronize()
                measurements.append(start.elapsed_time(end))
                del result
            segments.append(dict(round=round_index, slot=slot, mode=mode,
                                 median_ms=statistics.median(measurements), samples_ms=measurements))
    pairs = []
    for r in range(args.rounds):
        a = statistics.mean(s['median_ms'] for s in segments if s['round'] == r and s['mode'] == 'reference')
        b = statistics.mean(s['median_ms'] for s in segments if s['round'] == r and s['mode'] == 'fused')
        pairs.append(dict(round=r, reference_ms=a, fused_ms=b, speedup=a/b))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(input=str(args.input.resolve()),
        parity='products bitwise; shape/stride unchanged', scope='postprocessing only, not training speedup',
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), shape=list(scores.shape),
        stride=list(scores.stride()), segments=segments, rounds=pairs), indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    if '--abba' in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--abba', action='store_true')
        parser.add_argument('--input', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        parser.add_argument('--warmup', type=int, default=5)
        parser.add_argument('--steps', type=int, default=20)
        parser.add_argument('--rounds', type=int, default=3)
        args = parser.parse_args()
        if min(args.warmup, args.steps, args.rounds) < 1:
            parser.error('warmup, steps and rounds must be positive')
        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        abba(args)
    else:
        unittest.main()
