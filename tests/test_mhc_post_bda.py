# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import unittest
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint


def load_module():
    path = Path(__file__).parents[1] / 'src/mcore_bridge/mhc_post_bda.py'
    spec = importlib.util.spec_from_file_location('mhc_post_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def reference(h_res, residual, h_post, x, bias=None):
    seq, batch, streams, hidden = residual.shape
    mixed = torch.bmm(h_res.reshape(-1, streams, streams).transpose(1, 2),
                      residual.float().reshape(-1, streams, hidden)).reshape(seq, batch, streams, hidden)
    post = h_post.unsqueeze(-1) * x.float().unsqueeze(-2)
    if bias is not None:
        post = post + h_post.unsqueeze(-1) * bias.float().unsqueeze(-2)
    return (post + mixed).to(residual.dtype)


def inputs(device, seq=3, batch=1, bias_dtype=None):
    hr = torch.rand(seq, batch, 4, 4, device=device)
    hr = (hr / hr.sum(-1, keepdim=True)).detach().requires_grad_()
    r = torch.randn(seq, batch, 4, 4096, device=device, dtype=torch.bfloat16).requires_grad_()
    hp = (2 * torch.rand(seq, batch, 4, device=device)).requires_grad_()
    x = torch.randn(seq, batch, 4096, device=device, dtype=torch.bfloat16).requires_grad_()
    bias = None if bias_dtype is None else torch.randn(4096, device=device, dtype=bias_dtype).requires_grad_()
    return hr, r, hp, x, bias


class PostBDAValidationTest(unittest.TestCase):
    def test_shape_dtype_device_validation(self):
        values = inputs('cpu')
        MODULE.validate_inputs(*values)
        with self.assertRaisesRegex(ValueError, 'requires CUDA'):
            MODULE.post_bda(*values)
        with self.assertRaisesRegex(ValueError, 'stay FP32'):
            MODULE.validate_inputs(values[0].bfloat16(), *values[1:])
        with self.assertRaisesRegex(ValueError, 'BF16'):
            MODULE.validate_inputs(values[0], values[1].float(), *values[2:])
        with self.assertRaisesRegex(ValueError, '4,4096'):
            MODULE.validate_inputs(values[0], values[1][:, :, :3], *values[2:])

    def test_patch_only_switches_post_bda(self):
        text = (Path(__file__).parents[1] / 'src/mcore_bridge/patches/megatron_glm53_dev.patch').read_text()
        self.assertIn("os.environ.get('MCORE_MHC_POST_BDA_FP32_FUSION', '0')", text)
        self.assertIn('self._h_post_bda_op = post_bda', text)
        self.assertIn('config.num_residual_streams != 4 or config.hidden_size != 4096', text)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PostBDACudaTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(931)
        self.tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.tf32

    def test_outputs_gradients_and_parameter_update(self):
        for batch in (1, 2):
            for bias_dtype in (None, torch.bfloat16, torch.float32):
                with self.subTest(batch=batch, bias_dtype=bias_dtype):
                    fused_values = inputs('cuda', batch=batch, bias_dtype=bias_dtype)
                    base_values = tuple(None if v is None else v.detach().clone().requires_grad_()
                                        for v in fused_values)
                    expected = reference(*base_values)
                    actual = MODULE.post_bda(*fused_values)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0.015625)
                    grad = torch.randn_like(actual)
                    actual.backward(grad)
                    expected.backward(grad)
                    for fused, base in zip(fused_values, base_values):
                        if fused is None:
                            continue
                        if fused.dtype == torch.float32:
                            torch.testing.assert_close(fused.grad, base.grad, rtol=2e-4, atol=2e-3)
                        else:
                            torch.testing.assert_close(fused.grad.float(), base.grad.float(), rtol=0.016, atol=0.03125)
                        updated_fused = fused.detach().float() - 8e-6 * fused.grad.float()
                        updated_base = base.detach().float() - 8e-6 * base.grad.float()
                        torch.testing.assert_close(updated_fused, updated_base, rtol=1e-5, atol=1e-6)

    def test_checkpoint_and_autocast(self):
        first = inputs('cuda')
        second = tuple(None if v is None else v.detach().clone().requires_grad_() for v in first)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            direct = MODULE.post_bda(*first)
            recomputed = checkpoint(MODULE.post_bda, *second, use_reentrant=False)
        torch.testing.assert_close(direct, recomputed, rtol=0, atol=0)
        grad = torch.randn_like(direct)
        direct.backward(grad)
        recomputed.backward(grad)
        for a, b in zip(first[:-1], second[:-1]):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_saves_original_inputs_not_fp32_activation_copies(self):
        values = inputs('cuda')
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda x: saved.append(x) or x, lambda x: x):
            output = MODULE.post_bda(*values)
        self.assertEqual(len(saved), 4)
        self.assertEqual([v.data_ptr() for v in saved], [v.data_ptr() for v in values[:-1]])
        output.sum().backward()

    def test_empty_and_noncontiguous_upstream_gradient(self):
        empty = inputs('cuda', seq=0)
        out = MODULE.post_bda(*empty)
        out.sum().backward()
        self.assertEqual(out.numel(), 0)
        values = inputs('cuda')
        out = MODULE.post_bda(*values)
        grad = torch.randn(3, 1, 4, 8192, device='cuda', dtype=torch.bfloat16)[..., ::2]
        out.backward(grad)
        self.assertTrue(all(v.grad is not None for v in values[:-1]))

    def test_reentrant_checkpoint_and_higher_order_rejection(self):
        values = inputs('cuda')
        output = checkpoint(MODULE.post_bda, *values, use_reentrant=True)
        output.backward(torch.randn_like(output))
        self.assertTrue(all(v.grad is not None for v in values[:-1]))
        values = inputs('cuda')
        with self.assertRaisesRegex(RuntimeError, 'higher-order'):
            torch.autograd.grad(MODULE.post_bda(*values).float().sum(), values[:-1], create_graph=True)


if __name__ == '__main__':
    unittest.main()
