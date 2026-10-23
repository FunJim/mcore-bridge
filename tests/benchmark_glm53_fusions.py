# Copyright (c) ModelScope Contributors. All rights reserved.
"""Opt-in single-GPU benchmarks; no training, environment installation or distributed launch."""
import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Callable, Optional

import torch
import triton

from test_kpool_head_sum import MODULE as kpool, reference as kpool_reference
from test_mhc_post_bda import MODULE as mhc, inputs, reference


def measure(fn: Callable, repetitions: int) -> dict:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        fn()
    end.record()
    end.synchronize()
    return {'ms': start.elapsed_time(end) / repetitions,
            'extra_peak_mib': (torch.cuda.max_memory_allocated() - allocated) / 1024 ** 2}


def measure_pair(base: Callable, fused: Callable, repetitions: int, rounds: int) -> dict:
    samples = []
    for round_index in range(rounds):
        for slot, (name, fn) in enumerate((('baseline', base), ('fused', fused),
                                         ('fused', fused), ('baseline', base))):
            samples.append({'round': round_index, 'slot': slot, 'mode': name, **measure(fn, repetitions)})
    summary = {}
    for name in ('baseline', 'fused'):
        selected = [sample for sample in samples if sample['mode'] == name]
        summary[name] = {'ms': statistics.median(sample['ms'] for sample in selected),
                         'extra_peak_mib': max(sample['extra_peak_mib'] for sample in selected)}
    return {**summary, 'speedup': summary['baseline']['ms'] / summary['fused']['ms'], 'segments': samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--rows', type=int, default=16384)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or min(args.repetitions, args.rounds, args.rows) < 1:
        parser.error('CUDA and positive rows/repetitions/rounds are required')
    torch.manual_seed(281)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'triton': triton.__version__,
              'scope': 'synthetic single-GPU local operators, not training throughput',
              'rounds': args.rounds, 'repetitions': args.repetitions, 'kpool': [], 'mhc': {}}
    with torch.no_grad():
        for q, keys in ((32, 1024), (512, 16384)):
            scores = torch.randn(q, 1, 32, keys, device='cuda')
            weights = torch.randn(q, 1, 32, device='cuda')
            base = lambda scores=scores, weights=weights: kpool_reference(scores, weights)
            fused = lambda scores=scores, weights=weights: kpool.relu_mul_sum(scores, weights)
            expected, actual = base(), fused()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if not torch.equal(actual.topk(32, -1).indices, expected.topk(32, -1).indices):
                raise AssertionError('KPool ordered top-k changed')
            del expected, actual
            result['kpool'].append({'shape': list(scores.shape), 'baseline_implementation': 'native PyTorch ReLU/mul/head sum',
                                    **measure_pair(base, fused, args.repetitions, args.rounds)})
            del base, fused, scores, weights
    values = inputs('cuda', seq=args.rows)
    compiled = torch.compile(reference)
    native: Optional[Callable] = None
    if os.environ.get('MEGATRON_PATCH_BASE_WHEEL'):
        from test_glm53_fusion_bundle import BundleInstallationTest
        BundleInstallationTest.setUpClass()
        native = BundleInstallationTest.native
        BundleInstallationTest.tearDownClass()
        compiled = native
    result['mhc']['baseline'] = 'bundled native_h_post_bda_fp32' if native else 'compiled equivalent reference'
    gradient = torch.randn_like(values[1])
    with torch.no_grad():
        expected, actual = compiled(*values), mhc.post_bda(*values)
        result['mhc']['output_max_abs'] = (actual.float() - expected.float()).abs().max().item()
        result['mhc']['output_mismatch_fraction'] = (actual != expected).float().mean().item()
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        if error.item() >= 1e-4 or result['mhc']['output_mismatch_fraction'] >= 1e-3:
            raise AssertionError('mHC output parity gate failed')
        result['mhc']['output_relative_l2'] = error.item()
        del actual, expected
        result['mhc']['forward'] = measure_pair(lambda: compiled(*values), lambda: mhc.post_bda(*values),
                                               args.repetitions, args.rounds)

    def training(fn):
        for value in values:
            if value is not None:
                value.grad = None
        fn(*values).backward(gradient)

    result['mhc']['rows'] = args.rows
    result['mhc']['forward_backward'] = measure_pair(lambda: training(compiled), lambda: training(mhc.post_bda),
                                                    args.repetitions, args.rounds)
    for value in values:
        if value is not None:
            value.grad = None
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
