# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def load_module():
    path = Path(__file__).parents[1] / 'src/mcore_bridge/kpool_relu_mul.py'
    spec = importlib.util.spec_from_file_location('head_sum_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def reference(scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (scores.relu() * weights.unsqueeze(-1)).sum(2)


class HeadSumValidationTest(unittest.TestCase):
    def test_cpu_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'requires CUDA'):
            MODULE.relu_mul_sum(torch.zeros(2, 1, 32, 129), torch.ones(2, 1, 32))

    def test_rejects_autograd(self):
        with self.assertRaisesRegex(ValueError, 'detached'):
            MODULE.relu_mul_sum(torch.zeros(2, 1, 32, 129, requires_grad=True), torch.ones(2, 1, 32))

    def test_patch_has_single_flag_and_replay_contract(self):
        text = (Path(__file__).parents[1] / 'src/mcore_bridge/patches/megatron_glm53_dev.patch').read_text()
        self.assertIn("head_sum = os.environ.get('MCORE_KPOOL_HEAD_SUM_FUSION', '0')", text)
        self.assertIn('relu_mul_sum(scores, weights[start:end], out=index_scores[start:end])', text)
        self.assertGreaterEqual(text.count("os.environ.get('MCORE_KPOOL_HEAD_SUM_FUSION', '0')"), 3)

    @unittest.skipUnless(os.environ.get('MEGATRON_PATCH_BASE_WHEEL'), 'Base wheel required')
    def test_default_invalid_flag_and_non_relu_path(self):
        import test_kpool_tp_dedup as bundle
        _, dsa = bundle.load_sources(bundle.bundled_sources())
        kwargs = bundle.sample()
        q, k, weights = kwargs['q'], kwargs['k'], kwargs['weights']
        raw = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())
        with patch.dict(os.environ, {}, clear=True):
            expected = (raw.relu() * weights.unsqueeze(-1)).sum(2).transpose(0, 1)
            actual = dsa._compute_index_scores(q, weights, k)
            self.assertTrue(torch.equal(actual, expected))
        with patch.dict(os.environ, {'MCORE_KPOOL_HEAD_SUM_FUSION': 'invalid'}):
            with self.assertRaisesRegex(ValueError, 'HEAD_SUM_FUSION must be 0 or 1'):
                dsa._compute_index_scores(q, weights, k)
        with patch.dict(os.environ, {'MCORE_KPOOL_HEAD_SUM_FUSION': '1'}):
            actual = dsa._compute_index_scores(q, weights, k, use_relu=False)
            self.assertTrue(torch.equal(actual, (raw * weights.unsqueeze(-1)).sum(2).transpose(0, 1)))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class HeadSumCudaTest(unittest.TestCase):
    def test_bitwise_scores_and_ordered_topk(self):
        torch.manual_seed(731)
        for batch in (1, 2):
            for keys in (129, 257, 1024, 8192):
                with self.subTest(batch=batch, keys=keys):
                    scores = torch.randn(batch, 9, 32, keys, device='cuda').permute(1, 0, 2, 3)
                    weights = torch.randn(18, batch, 32, device='cuda')[::2]
                    expected = reference(scores, weights)
                    actual = MODULE.relu_mul_sum(scores, weights)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertTrue(torch.equal(actual.topk(8, -1).indices, expected.topk(8, -1).indices))

    def test_einsum_layout_and_output_slice(self):
        torch.manual_seed(191)
        q = torch.randn(7, 1, 32, 128, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(263, 1, 128, device='cuda', dtype=torch.bfloat16)
        scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())
        weights = torch.randn(7, 1, 32, device='cuda')
        out = torch.full((11, 1, 263), 123.0, device='cuda')
        actual = MODULE.relu_mul_sum(scores, weights, out[2:9])
        self.assertEqual(actual.data_ptr(), out[2:9].data_ptr())
        torch.testing.assert_close(actual, reference(scores, weights), rtol=0, atol=0)
        self.assertTrue(torch.all(out[:2] == 123))
        self.assertTrue(torch.all(out[9:] == 123))

    def test_ties_close_scores_mask_and_order(self):
        scores = torch.ones(5, 1, 32, 257, device='cuda')
        scores[..., 4] += torch.finfo(torch.float32).eps
        scores[..., 6] -= torch.finfo(torch.float32).eps
        scores[..., 10] = 0
        weights = torch.linspace(-1, 2, 32, device='cuda').expand(5, 1, 32).contiguous()
        expected = reference(scores, weights)
        actual = MODULE.relu_mul_sum(scores, weights)
        expected[..., 128:] = -torch.inf
        actual[..., 128:] = -torch.inf
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(actual.topk(16, -1).indices, expected.topk(16, -1).indices))

    def test_invalid_layout_out_and_empty(self):
        scores = torch.zeros(2, 1, 32, 257, device='cuda')
        weights = torch.ones(2, 1, 32, device='cuda')
        with self.assertRaisesRegex(ValueError, 'dense non-overlapping'):
            MODULE.relu_mul_sum(scores[..., ::2], weights)
        with self.assertRaisesRegex(ValueError, 'must not alias'):
            MODULE.relu_mul_sum(scores[:1], weights[:1], scores[:1, :, 0])
        empty = MODULE.relu_mul_sum(scores[:0], weights[:0])
        self.assertEqual(empty.shape, (0, 1, 257))
        with self.assertRaisesRegex(ValueError, '32 heads'):
            MODULE.relu_mul_sum(scores[:, :, :4].contiguous(), weights[:, :, :4])

    def test_broadcast_weights_no_grad_and_inputs_unchanged(self):
        scores = torch.randn(2, 3, 32, 257, device='cuda')
        weights = torch.randn(2, 1, 32, device='cuda').expand(2, 3, 32).requires_grad_()
        original = scores.clone()
        with torch.no_grad():
            actual = MODULE.relu_mul_sum(scores, weights)
            expected = reference(scores, weights)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(scores, original))
        self.assertTrue(weights.requires_grad)
        self.assertFalse(actual.requires_grad)

    def test_ieee_edges_and_small_pools(self):
        for keys in (1, 3, 64, 257):
            scores = torch.randn(2, 1, 32, keys, device='cuda')
            scores[0, 0, 0, 0] = float('nan')
            scores[1, 0, 0, 0] = float('inf')
            weights = torch.randn(2, 1, 32, device='cuda')
            expected = reference(scores, weights)
            actual = MODULE.relu_mul_sum(scores, weights)
            self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))

    def test_ieee_bit_patterns(self):
        bits = torch.tensor([0, -2147483648, 1, -2147483647, 8388607, 8388608,
                             2139095039, 2139095040, -8388608, 2143289345, -4194303],
                            dtype=torch.int32, device='cuda')
        scores = bits.view(torch.float32).repeat(32).reshape(1, 1, 32, -1)
        for weight in (0., -0., 1., -1., float('inf'), float('nan'), 1.e-20):
            weights = torch.full((1, 1, 32), weight, device='cuda')
            expected, actual = reference(scores, weights), MODULE.relu_mul_sum(scores, weights)
            self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))

    @unittest.skipUnless(os.environ.get('MEGATRON_PATCH_BASE_WHEEL'), 'Base wheel required')
    def test_bundled_packed_mask_tail_selection(self):
        import test_kpool_tp_dedup as bundle
        _, dsa = bundle.load_sources(bundle.bundled_sources())
        package = types.ModuleType('mcore_bridge')
        package.__path__ = []
        modules = {'mcore_bridge': package, 'mcore_bridge.kpool_relu_mul': MODULE}
        with patch.dict(sys.modules, modules), torch.no_grad():
            for case in ('packed', 'unpacked', 'zigzag', 'ties', 'near_ties',
                         'explicit_positions', 'empty_rows'):
                kwargs = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                          for k, v in bundle.sample(case).items()}
                kwargs['q'] = kwargs['q'].repeat(1, 1, 8, 16)
                kwargs['weights'] = kwargs['weights'].repeat(1, 1, 8)
                kwargs['k'] = kwargs['k'].repeat(1, 1, 16)
                kwargs['gate_score'] = kwargs['gate_score'].repeat(1, 1, 16)
                kwargs['ape'] = kwargs['ape'].repeat(1, 16)
                kwargs['use_relu'] = True
                with patch.dict(os.environ, {'MCORE_KPOOL_HEAD_SUM_FUSION': '0'}):
                    expected = dsa.fused_qk_topk_kpool(**kwargs)
                with patch.dict(os.environ, {'MCORE_KPOOL_HEAD_SUM_FUSION': '1'}):
                    actual = dsa.fused_qk_topk_kpool(**kwargs)
                self.assertTrue(torch.equal(actual[0].contiguous().view(torch.int32),
                                            expected[0].contiguous().view(torch.int32)), case)
                self.assertTrue(torch.equal(actual[1], expected[1]), case)


if __name__ == '__main__':
    unittest.main()
