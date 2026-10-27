# Copyright (c) ModelScope Contributors. All rights reserved.
"""Frozen-input ABBA timing for complete KPool postprocessing against native PyTorch."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import unittest

import torch

from test_kpool_head_sum import MODULE as fusion, reference


def score_chunk(payload: dict) -> tuple:
    if not isinstance(payload, dict):
        raise ValueError('Frozen input must be a tensor dictionary')
    if 'scores' in payload:
        scores, weights = payload['scores'], payload['weights']
    else:
        if not payload.get('use_relu', True):
            raise ValueError('ABBA requires use_relu=True frozen inputs')
        q, k, weights = payload['q'], payload['k'], payload['weights']
        rows, batch, heads, _ = q.shape
        chunk_rows = min(rows, max(1, 1024 ** 3 // max(1, batch * heads * k.size(0) * 4)))
        scores = torch.einsum('sbhd,tbd->sbht', q[:chunk_rows].float(), k.float())
        weights = weights[:chunk_rows]
    fusion.validate_inputs(scores, weights)
    return scores, weights


class FrozenInputContracts(unittest.TestCase):
    def test_rejects_non_dictionary(self):
        with self.assertRaisesRegex(ValueError, 'tensor dictionary'):
            score_chunk([])

    def test_rejects_non_relu(self):
        with self.assertRaisesRegex(ValueError, 'use_relu=True'):
            score_chunk({'use_relu': False})

    def test_fp32_scores_pass_without_cuda_initialization(self):
        scores = torch.zeros(2, 1, 32, 129)
        weights = torch.ones(2, 1, 32)
        actual_scores, actual_weights = score_chunk({'scores': scores, 'weights': weights})
        self.assertIs(actual_scores, scores)
        self.assertIs(actual_weights, weights)


def abba(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('ABBA timing needs CUDA; no CPU substitute is accepted')
    payload = torch.load(args.input, map_location='cuda', weights_only=True)
    with torch.no_grad():
        scores, weights = score_chunk(payload)
        expected, actual = reference(scores, weights), fusion.relu_mul_sum(scores, weights)
        if not torch.equal(expected.contiguous().view(torch.int32), actual.contiguous().view(torch.int32)):
            raise AssertionError('FP32 reduced scores differ bitwise')
        topk = min(32, scores.shape[-1])
        if not torch.equal(expected.topk(topk, -1).indices, actual.topk(topk, -1).indices):
            raise AssertionError('Ordered top-k indices changed')
        del expected, actual
        operations = {'reference': lambda: reference(scores, weights),
                      'fused': lambda: fusion.relu_mul_sum(scores, weights)}
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
        for round_index in range(args.rounds):
            a = statistics.mean(s['median_ms'] for s in segments
                                if s['round'] == round_index and s['mode'] == 'reference')
            b = statistics.mean(s['median_ms'] for s in segments
                                if s['round'] == round_index and s['mode'] == 'fused')
            pairs.append(dict(round=round_index, reference_ms=a, fused_ms=b, speedup=a / b))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(input=str(args.input.resolve()),
        parity='reduced scores bitwise; ordered top-k unchanged',
        baseline='native PyTorch ReLU/mul/head sum', scope='postprocessing only, not training speedup',
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
        abba(args)
    else:
        unittest.main()
