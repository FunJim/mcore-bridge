# Copyright (c) ModelScope Contributors. All rights reserved.
"""NVTX window regressions and an opt-in, prewarmed Nsight smoke workload."""
import argparse
import sys
import unittest
from unittest.mock import patch

import torch
from megatron.core import utils as mcore_utils

from test_kpool_head_sum import MODULE as kpool
from test_mhc_post_bda import MODULE as mhc, inputs


KERNEL_NAMES = ('mcore_kpool_relu_mul_head_sum', 'mcore_mhc_post_bda_fwd',
                'mcore_mhc_post_bda_bwd_inputs', 'mcore_mhc_post_bda_bwd_mappings')
RANGES = ('kpool/head_sum', 'mhc/post_bda_fwd', 'mhc/post_bda_bwd',
          'mhc/post_bda_bwd_prepare', 'mhc/post_bda_bwd_inputs',
          'mhc/post_bda_bwd_mappings', 'mhc/post_bda_bwd_postprocess')


class KernelNamingTest(unittest.TestCase):
    def test_jit_entry_names(self):
        kernels = (kpool._sum_kernel(), *mhc._kernels())
        self.assertEqual(tuple(kernel.__name__ for kernel in kernels), KERNEL_NAMES)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class FusionRangesCudaTest(unittest.TestCase):
    def setUp(self):
        self.enabled = mcore_utils._nvtx_enabled
        self.depth = len(mcore_utils._nvtx_range_messages)
        mcore_utils.configure_nvtx_profiling(False)
        self.scores = torch.randn(3, 1, 32, 257, device='cuda')
        self.weights = torch.randn(3, 1, 32, device='cuda')
        self.values = inputs('cuda', seq=3, bias_dtype=torch.float32)
        kpool.relu_mul_sum(self.scores, self.weights)
        self.gradient = torch.randn_like(self.values[1])
        mhc.post_bda(*self.values).backward(self.gradient)
        for value in self.values:
            value.grad = None

    def tearDown(self):
        mcore_utils.configure_nvtx_profiling(self.enabled)
        self.assertEqual(len(mcore_utils._nvtx_range_messages), self.depth)

    def test_callback_window_toggling_and_parity(self):
        events = []
        stack = []

        def push(name):
            events.append(('push', name))
            stack.append(name)

        def pop():
            events.append(('pop', stack.pop()))

        with patch.object(torch.cuda.nvtx, 'range_push', side_effect=push), \
                patch.object(torch.cuda.nvtx, 'range_pop', side_effect=pop):
            expected_scores = kpool.relu_mul_sum(self.scores, self.weights)
            expected_output = mhc.post_bda(*self.values)
            expected_output.backward(self.gradient)
            expected_grads = [value.grad.clone() for value in self.values]
            self.assertEqual(events, [])
            for value in self.values:
                value.grad = None
            mcore_utils.configure_nvtx_profiling(True)
            actual_scores = kpool.relu_mul_sum(self.scores, self.weights)
            actual_output = mhc.post_bda(*self.values)
            actual_output.backward(self.gradient)
            torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)
            torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
            for value, expected in zip(self.values, expected_grads):
                torch.testing.assert_close(value.grad, expected, rtol=0, atol=0)
            self.assertEqual([name for kind, name in events if kind == 'push'], list(RANGES))
            self.assertEqual(stack, [])
            count = len(events)
            mcore_utils.configure_nvtx_profiling(False)
            kpool.relu_mul_sum(self.scores, self.weights)
            mhc.post_bda(*self.values).backward(self.gradient)
            self.assertEqual(len(events), count)

    def test_exception_unwinds_nested_ranges(self):
        events = []
        stack = []

        def push(name):
            events.append(('push', name))
            stack.append(name)

        def pop():
            events.append(('pop', stack.pop()))

        class BrokenKernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    raise RuntimeError('synthetic launch failure')
                return launch

        with patch.object(torch.cuda.nvtx, 'range_push', side_effect=push), \
                patch.object(torch.cuda.nvtx, 'range_pop', side_effect=pop):
            mcore_utils.configure_nvtx_profiling(True)
            with patch.object(kpool, '_sum_kernel', return_value=BrokenKernel()):
                with self.assertRaisesRegex(RuntimeError, 'synthetic launch'):
                    kpool.relu_mul_sum(self.scores, self.weights)
            with patch.object(mhc, '_kernels', return_value=(BrokenKernel(),) * 3):
                with self.assertRaisesRegex(RuntimeError, 'synthetic launch'):
                    mhc.post_bda(*self.values)
            output = mhc.post_bda(*self.values)
            with patch.object(mhc, '_kernels', return_value=(BrokenKernel(),) * 3):
                with self.assertRaisesRegex(RuntimeError, 'synthetic launch'):
                    output.backward(self.gradient)
            self.assertEqual(stack, [])
            self.assertEqual(len(mcore_utils._nvtx_range_messages), self.depth)
            for name in set(name for _, name in events):
                self.assertEqual(events.count(('push', name)), events.count(('pop', name)))


def nsys_smoke() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('Nsight smoke workload requires CUDA')
    torch.manual_seed(739)
    scores = torch.randn(7, 1, 32, 1024, device='cuda')
    weights = torch.randn(7, 1, 32, device='cuda')
    values = inputs('cuda', seq=129, bias_dtype=torch.float32)
    gradient = torch.randn_like(values[1])
    previous = mcore_utils._nvtx_enabled
    mcore_utils.configure_nvtx_profiling(False)
    for _ in range(3):
        kpool.relu_mul_sum(scores, weights)
        mhc.post_bda(*values).backward(gradient)
        for value in values:
            value.grad = None
    torch.cuda.synchronize()
    torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
    try:
        for enabled, label in ((False, 'before_window'), (True, 'capture_window'), (False, 'after_window')):
            mcore_utils.configure_nvtx_profiling(enabled)
            with torch.cuda.nvtx.range(label):
                kpool.relu_mul_sum(scores, weights)
                mhc.post_bda(*values).backward(gradient)
            for value in values:
                value.grad = None
        torch.cuda.synchronize()
    finally:
        mcore_utils.configure_nvtx_profiling(previous)
        torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
    print('prewarmed fusion Nsight smoke passed')


if __name__ == '__main__':
    if '--nsys-smoke' in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--nsys-smoke', action='store_true')
        parser.parse_args()
        nsys_smoke()
    else:
        unittest.main()
