# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exact tie and workspace regressions using the bundled production selector."""
import tempfile
import test_kpool_tp_dedup as bundle
import unittest
from pathlib import Path
from unittest.mock import patch

torch = bundle.torch


@unittest.skipUnless(torch is not None and bundle.BASE_WHEEL, 'requires Torch and MEGATRON_PATCH_BASE_WHEEL')
class OrderedTopKTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.adapter, cls.dsa = bundle.load_sources(bundle.bundled_sources())

    def test_ordered_topk_matches_exact_tie_oracle(self):
        generator = torch.Generator().manual_seed(713)
        cases = [
            torch.zeros(2, 7, 37),
            torch.randn(2, 7, 37, generator=generator),
            torch.randn(2, 7, 37, generator=generator).bfloat16().float(),
            torch.tensor([2.] * 4 + [1.] * 20 + [-2.] * 13).repeat(2, 7, 1),
            torch.tensor([1., 1.0000001192092896, 0.9999999403953552]).repeat(2, 7, 13)[..., :37]
        ]
        masked = torch.full((2, 7, 37), float('-inf'))
        for row in range(7):
            masked[:, row, :row] = torch.arange(row).float()
        cases.append(masked)
        for scores in cases:
            for count in (0, 1, 5, 16, 37):
                expected = torch.tensor(
                    [[sorted(range(37), key=lambda col: (-float(row[col]), col))[:count] for row in batch]
                     for batch in scores],
                    dtype=torch.int64)
                values, indices = self.dsa._kpool_ordered_topk(scores, count)
                self.assertTrue(torch.equal(indices, expected))
                self.assertTrue(torch.equal(values, torch.gather(scores, -1, expected)))
                self.assertEqual(indices.untyped_storage().nbytes(), indices.numel() * indices.element_size())

    def test_large_tie_selection_matches_query_chunks(self):
        scores = torch.zeros(1, 257, 8192)
        bases = torch.arange(8192) * 4
        starts = torch.zeros(257, dtype=torch.int64)
        ends = torch.full_like(starts, 32768)
        whole = self.adapter._select(scores, bases, starts, ends, 2048, 4, True)
        for width in (1, 64, 256):
            pieces = [
                self.adapter._select(scores[:, lo:lo + width], bases, starts[lo:lo + width], ends[lo:lo + width], 2048,
                                     4, True) for lo in range(0, 257, width)
            ]
            self.assertTrue(torch.equal(torch.cat(pieces, dim=1), whole))
        self.assertTrue(torch.equal(whole[0, 0, :2048], torch.arange(2048, dtype=torch.int32)))

    def test_ieee_edges_empty_inputs_and_noncontiguous_scores(self):
        bits = torch.tensor(
            [0, -2147483648, 1, -2147483647, 8388607, 8388608, 2139095039, 2139095040, -8388608, 2143289345, -4194303],
            dtype=torch.int32)
        scores = bits.view(torch.float32).repeat(2, 7, 4).transpose(0, 1)
        for count in (0, 1, 5, scores.shape[-1]):
            oracle = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :count]
            values, ids = self.dsa._kpool_ordered_topk(scores, count)
            self.assertTrue(torch.equal(ids, oracle))
            self.assertTrue(
                torch.equal(values.contiguous().view(torch.int32),
                            torch.gather(scores, -1, oracle).contiguous().view(torch.int32)))
        for scores in (torch.empty(2, 0, 37), torch.empty(0, 3, 37), torch.empty(2, 3, 0)):
            values, ids = self.dsa._kpool_ordered_topk(scores, min(5, scores.shape[-1]))
            self.assertEqual(values.shape, scores.shape[:-1] + (min(5, scores.shape[-1]), ))
            self.assertEqual(ids.shape, values.shape)

    def test_sort_chunks_bound_explicit_index_storage(self):
        scores = torch.zeros(1, 257, 8192)
        sort = torch.argsort
        calls = []

        def tracked(chunk, **kwargs):
            calls.append(chunk.shape)
            return sort(chunk, **kwargs)

        with patch.object(torch, 'argsort', side_effect=tracked):
            values, ids = self.dsa._kpool_ordered_topk(scores, 512)
        self.assertEqual(sum(shape[1] for shape in calls), 257)
        self.assertTrue(all(shape[0] * shape[1] * shape[2] * 16 <= 64 * 1024 * 1024 for shape in calls))
        self.assertEqual(ids.untyped_storage().nbytes(), ids.numel() * ids.element_size())
        self.assertEqual(values.untyped_storage().nbytes(), values.numel() * values.element_size())

    def test_score_gate_rejects_drift_and_invalid_values(self):
        from glm53_patch_regression.verify_glm53_kpool_ordered_selection import verify_scores

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = torch.tensor([[[1., 2., -float('inf')]]])
            self.assertEqual(verify_scores(baseline, baseline.clone(), root, 'equal'), 0)
            for values in ([1., 3., -float('inf')], [1., 2., 0.], [1., float('nan'),
                                                                   -float('inf')], [1., float('inf'), -float('inf')]):
                with self.subTest(values=values), self.assertRaisesRegex(AssertionError, 'Score drift'):
                    verify_scores(baseline, torch.tensor([[values]]), root, 'invalid')
            self.assertTrue((root / 'score-mismatch-rank0.pt').is_file())
            with patch.object(torch, 'save', side_effect=OSError('unwritable')):
                with self.assertRaisesRegex(AssertionError, 'Score drift'):
                    verify_scores(baseline, baseline + 1, root, 'write-failure')

    def test_score_gate_propagates_peer_failure(self):
        from glm53_patch_regression.verify_glm53_kpool_ordered_selection import verify_scores

        value = torch.ones(1, 2, 3)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(torch.distributed, 'is_initialized', return_value=True), \
                    patch.object(torch.distributed, 'all_reduce', side_effect=lambda flag, **kwargs: flag.fill_(1)):
                with self.assertRaisesRegex(AssertionError, 'Score drift'):
                    verify_scores(value, value, Path(directory), 'peer-failure')


if __name__ == '__main__':
    unittest.main()
