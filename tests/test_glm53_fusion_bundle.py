# Copyright (c) ModelScope Contributors. All rights reserved.
"""Install the actual bundled patch into a disposable wheel tree, never a training environment."""
import ast
import importlib.util
import os
import tempfile
import types
import unittest
import zipfile
from pathlib import Path

import torch

from test_mhc_post_bda import MODULE as mhc, inputs


ROOT = Path(__file__).parents[1]
BASE_WHEEL = os.environ.get('MEGATRON_PATCH_BASE_WHEEL')


@unittest.skipUnless(BASE_WHEEL, 'MEGATRON_PATCH_BASE_WHEEL required')
class BundleInstallationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            'fusion_bundle_installer', ROOT / 'src/mcore_bridge/tools/apply_megatron_patch.py')
        cls.installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.installer)
        cls.temporary = tempfile.TemporaryDirectory(prefix='glm53-fusion-bundle-')
        cls.root = Path(cls.temporary.name)
        try:
            with zipfile.ZipFile(BASE_WHEEL) as wheel:
                for path in cls.installer.patch_files(False):
                    if path in wheel.namelist():
                        target = cls.root / path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(wheel.read(path))
            cls.installer.apply_patch(cls.root)
        except Exception:
            cls.temporary.cleanup()
            raise
        text = (cls.root / 'megatron/core/transformer/hyper_connection.py').read_text()
        tree = ast.parse(text)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('native_h_post_bda', 'native_h_post_bda_fp32')]
        namespace = {'torch': torch}
        future = ast.parse('from __future__ import annotations').body
        exec(compile(ast.fix_missing_locations(ast.Module(body=future + functions, type_ignores=[])),
                     '<installed-post-bda>', 'exec'), namespace)
        cls.native = staticmethod(namespace['native_h_post_bda_fp32'])
        cls.hyper_tree = tree

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_installation_is_complete_and_idempotent(self):
        self.assertTrue(self.installer.is_applied(self.root))
        before = {p: (self.root / p).read_bytes() for p in self.installer.patch_files(False)}
        self.installer.apply_patch(self.root, check_only=True)
        self.installer.apply_patch(self.root)
        self.assertEqual(before, {p: (self.root / p).read_bytes() for p in before})

    def test_default_and_enabled_post_bda_dispatch(self):
        from unittest.mock import patch
        import sys

        cls = next(node for node in self.hyper_tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'HyperConnectionModule')
        constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                           and node.name == '__init__')
        start = next(i for i, node in enumerate(constructor.body)
                     if isinstance(node, ast.Import) and any(alias.name == 'os' for alias in node.names))
        end = next(i for i, node in enumerate(constructor.body[start:], start)
                   if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                   and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == '_init_weights')
        flag_nodes = constructor.body[start:end]
        function = ast.parse('def dispatch(self, config):\n    pass\n').body[0]
        function.body = flag_nodes
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     '<installed-post-bda-dispatch>', 'exec'), namespace)
        config = types.SimpleNamespace(use_fused_mhc=False, mhc_keep_mappings_in_fp32=True,
                                       num_residual_streams=4, hidden_size=4096)
        owner = types.SimpleNamespace(_h_post_bda_op=self.native)
        with patch.dict(os.environ, {}, clear=True):
            namespace['dispatch'](owner, config)
            self.assertIs(owner._h_post_bda_op, self.native)
        package = types.ModuleType('mcore_bridge')
        package.__path__ = []
        with patch.dict(sys.modules, {'mcore_bridge': package, 'mcore_bridge.mhc_post_bda': mhc}):
            with patch.dict(os.environ, {'MCORE_MHC_POST_BDA_FP32_FUSION': '1'}):
                namespace['dispatch'](owner, config)
                self.assertIs(owner._h_post_bda_op, mhc.post_bda)
                for field, value in (('hidden_size', 2048), ('num_residual_streams', 8),
                                     ('mhc_keep_mappings_in_fp32', False), ('use_fused_mhc', True)):
                    invalid = types.SimpleNamespace(**vars(config))
                    setattr(invalid, field, value)
                    with self.assertRaisesRegex(ValueError, 'FP32 post-BDA fusion requires'):
                        namespace['dispatch'](owner, invalid)
            with patch.dict(os.environ, {'MCORE_MHC_POST_BDA_FP32_FUSION': 'invalid'}):
                with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                    namespace['dispatch'](owner, config)

    def test_real_model_config_full_hyper_connection_construction(self):
        import subprocess
        import sys

        script = '''
import importlib.util
import os
import sys
from unittest.mock import patch
import torch
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
from mcore_bridge.config import ModelConfig
from mcore_bridge.mhc_post_bda import post_bda

spec = importlib.util.spec_from_file_location('hyper_connection_construction_gate', sys.argv[1])
hyper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hyper)
with patch('mcore_bridge.bridge.gpt_bridge.GPTBridge.__init__', return_value=None):
    config = ModelConfig(hf_model_type='glm5_next', mcore_model_type='glm5_next',
                         hf_config=Glm5NextConfig(), num_layers=2, hidden_size=4096,
                         num_attention_heads=32, num_residual_streams=4,
                         enable_hyper_connections=True, mhc_norm_eps_inside_sqrt=True,
                         mhc_keep_mappings_in_fp32=True, use_fused_mhc=False)
assert not hasattr(config, 'mhc_num_streams')
for flag in ('0', '1'):
    os.environ['MCORE_MHC_POST_BDA_FP32_FUSION'] = flag
    with torch.device('cpu'):
        module = hyper.HyperConnectionModule(config, layer_number=1)
    assert module.n == 4
    assert module.mapping_proj.weight.shape == (24, 16384)
    assert module.mapping_proj.weight.dtype == torch.float32
    assert module._h_post_bda_op is (post_bda if flag == '1' else hyper.native_h_post_bda_fp32)
    assert module._sinkhorn_op is hyper.native_sinkhorn
    assert module._proj_rms_compute_h_op is None
    del module
print('real ModelConfig and complete HyperConnectionModule construction passed, fusion=0/1')
'''
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(ROOT / 'src')
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        result = subprocess.run(
            [sys.executable, '-c', script, str(self.root / 'megatron/core/transformer/hyper_connection.py')],
            env=environment, text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('construction passed', result.stdout)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_actual_native_compiled_vjp_and_optimizer_update(self):
        torch.manual_seed(332)
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for sequence, bias_dtype in ((129, None), (129, torch.float32), (16384, None)):
                actual_values = inputs('cuda', seq=sequence, bias_dtype=bias_dtype)
                base_values = tuple(None if v is None else v.detach().clone().requires_grad_()
                                    for v in actual_values)
                actual, expected = mhc.post_bda(*actual_values), self.native(*base_values)
                error = (actual.float() - expected.float()).norm() / expected.float().norm()
                self.assertLess(error.item(), 1e-4)
                self.assertLess((actual != expected).float().mean().item(), 1e-3)
                torch.testing.assert_close(actual.float(), expected.float(), rtol=0.008, atol=2e-6)
                grad = torch.randn_like(expected)
                actual.backward(grad)
                expected.backward(grad)
                for a, b in zip(actual_values, base_values):
                    if a is None:
                        continue
                    delta = (a.grad.float() - b.grad.float()).norm() / b.grad.float().norm().clamp_min(1e-12)
                    self.assertLess(delta.item(), 1e-4)
                    tolerance = 0.008 if a.dtype == torch.bfloat16 else 1e-4
                    torch.testing.assert_close(a.grad.float(), b.grad.float(), rtol=tolerance, atol=0.002)
                a_params = [v for v in actual_values if v is not None and v.dtype == torch.float32]
                b_params = [v for v in base_values if v is not None and v.dtype == torch.float32]
                opt_a = torch.optim.AdamW(a_params, lr=8e-6)
                opt_b = torch.optim.AdamW(b_params, lr=8e-6)
                opt_a.step()
                opt_b.step()
                for a, b in zip(a_params, b_params):
                    torch.testing.assert_close(a, b, rtol=0, atol=2e-6)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32


if __name__ == '__main__':
    unittest.main()
