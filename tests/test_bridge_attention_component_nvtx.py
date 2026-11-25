# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU checks of the actual Bridge output-projection callsites."""

import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load_helper():
    path = os.environ.get("MEGATRON_COMPONENT_NVTX_SOURCE")
    if not path:
        raise RuntimeError("set MEGATRON_COMPONENT_NVTX_SOURCE to the development helper")
    spec = importlib.util.spec_from_file_location("bridge_attention_nvtx", path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    with patch.dict(os.environ, {"MCORE_COMPONENT_NVTX": "1", "MCORE_COMPONENT_NVTX_CAPTURE_MANAGED": "0"}):
        spec.loader.exec_module(helper)
    return helper


class AttentionComponentTests(unittest.TestCase):
    def setUp(self):
        self.names = []
        self.opened = set()

        def start(name):
            self.names.append(name)
            handle = len(self.names)
            self.opened.add(handle)
            return handle

        nvtx = SimpleNamespace(range_push=self.names.append, range_pop=lambda: None,
                               range_start=start, range_end=self.opened.remove)
        self.mock = patch.object(torch.cuda, "nvtx", nvtx)
        self.mock.start()
        self.helper = load_helper()
        self.path = ROOT / "src/mcore_bridge/model/gpts/deepseek_v4.py"
        tree = ast.parse(self.path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "DSv4HybridSelfAttention")
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("_grouped_output_projection", "_final_output_projection")]
        self.namespace = {"torch": torch, "component": self.helper.component}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(self.path), "exec"), self.namespace)

    def tearDown(self):
        self.assertEqual(self.opened, set())
        self.mock.stop()
        self.assertFalse(torch.cuda.is_initialized())

    def test_grouped_and_dense_outputs_gradients_match_reference(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(27)
                x = torch.randn(3, 2, 8, dtype=dtype, requires_grad=True)
                weight = nn.Parameter(torch.randn(6, 4, dtype=dtype))
                dense = nn.Linear(6, 5).to(dtype)
                holder = SimpleNamespace(_o_group_proj_is_grouped_linear=False, o_local_groups=2,
                                         config=SimpleNamespace(o_lora_rank=3), linear_o_group_proj=weight,
                                         linear_proj=lambda value: (dense(value), None))
                expected = dense(torch.einsum("sbgd,grd->sbgr", x.reshape(3, 2, 2, 4),
                                               weight.reshape(2, 3, 4)).reshape(3, 2, 6))
                projected = self.namespace["_grouped_output_projection"](holder, x)
                actual, bias = self.namespace["_final_output_projection"](holder, projected)
                parameters = (x, weight, *dense.parameters())
                reference_grad = torch.autograd.grad(expected.sum(), parameters)
                with self.helper.backward_scope(actual):
                    actual_grad = torch.autograd.grad(actual.sum(), parameters)
                self.assertIsNone(bias)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for left,right in zip(reference_grad, actual_grad):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
        for name in ("attention.grouped_output_projection", "attention.dense_output_projection"):
            self.assertTrue(any(value.startswith(f"comp:{name}:") for value in self.names))
            self.assertTrue(any(value.startswith(f"comp:{name}.bwd:") for value in self.names))

    def test_all_actual_component_callsite_names_are_present(self):
        names = set()
        for relative in ("src/mcore_bridge/model/gpts/deepseek_v4.py",
                         "src/mcore_bridge/model/modules/engram.py"):
            tree = ast.parse((ROOT / relative).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "component":
                    names.update(arg.value for arg in node.args if isinstance(arg, ast.Constant)
                                 and isinstance(arg.value, str))
        self.assertTrue({"attention.qkv_projection", "attention.grouped_output_projection",
                         "attention.dense_output_projection", "engram.hash", "engram.lookup",
                         "engram.projection", "engram.gate"} <= names)

    def test_projection_exception_does_not_leave_backward_or_forward_ranges(self):
        holder = SimpleNamespace(linear_proj=lambda value: (_ for _ in ()).throw(ValueError("fixture")))
        with self.assertRaisesRegex(ValueError, "fixture"):
            self.namespace["_final_output_projection"](holder, torch.ones(2))
        self.assertEqual(self.helper._claimed.get(), ())


if __name__ == "__main__":
    unittest.main()
