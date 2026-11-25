# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real CPU PyTorch checks of isolated Engram profiling methods."""

import ast
import importlib.util
import os
import sys
import torch
import unittest
from pathlib import Path
from torch import nn
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_methods(component):
    path = ROOT / "src/mcore_bridge/model/modules/engram.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Engram")
    names = ("_lookup", "_project_memory", "_context_gate")
    namespace = {"torch": torch, "Tensor": torch.Tensor, "component": component}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class EngramComponentTests(unittest.TestCase):

    def test_split_projection_gate_outputs_gradients_and_adam_are_unchanged(self):
        methods = load_methods(lambda name: lambda fn: fn)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(27)
                holder = SimpleNamespace(
                    num_streams=2,
                    hidden_size=8,
                    value_projection=nn.Linear(12, 8, bias=False).to(dtype),
                    key_projection=nn.Linear(12, 16, bias=False).to(dtype),
                    key_norm=nn.RMSNorm(16).to(dtype),
                    query_norm=nn.RMSNorm(16).to(dtype),
                )
                hidden = torch.randn(9, 1, 16, dtype=dtype, requires_grad=True)
                memory = torch.randn(9, 1, 12, dtype=dtype, requires_grad=True)
                live = torch.ones(1, 9, dtype=torch.bool)
                live[:, 3] = False
                params = [
                    p for name in ("value_projection", "key_projection", "key_norm", "query_norm")
                    for p in getattr(holder, name).parameters()
                ]
                value = holder.value_projection(memory)
                key = holder.key_norm(holder.key_projection(memory)).reshape(9, 1, 2, 8)
                query = holder.query_norm(hidden).reshape(9, 1, 2, 8)
                dot = torch.sum(key * query, dim=-1) / 8**0.5
                gate = (dot.abs().clamp_min(1e-6).sqrt() * dot.sign()).sigmoid()
                expected = (gate[..., None] * value[..., None, :] * live.T[..., None, None]).reshape_as(hidden)
                grad = torch.randn_like(expected)
                reference_grads = torch.autograd.grad(expected, (hidden, memory, *params), grad)
                streams, value, key, query = methods["_project_memory"](holder, hidden, memory)
                actual = methods["_context_gate"](holder, hidden, streams, value, key, query, live)
                actual_grads = torch.autograd.grad(actual, (hidden, memory, *params), grad)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                for left, right in zip(reference_grads, actual_grads):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
                # Real FP32 Adam parameters/moments receive the same gradients.
                masters = [[nn.Parameter(p.detach().float().clone()) for p in params] for _ in range(2)]
                optimizers = [torch.optim.AdamW(values, lr=1e-3) for values in masters]
                for values, optimizer, grads in zip(masters, optimizers, (reference_grads, actual_grads)):
                    for value, g in zip(values, grads[2:]):
                        value.grad = g.float()
                    optimizer.step()
                for left, right in zip(*masters):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
                    for key in ("step", "exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(optimizers[0].state[left][key],
                                                   optimizers[1].state[right][key], atol=0, rtol=0)

    def test_real_component_helper_marks_projection_and_backward(self):
        path = os.environ.get("MEGATRON_COMPONENT_NVTX_SOURCE")
        if path is None:
            self.skipTest("Set MEGATRON_COMPONENT_NVTX_SOURCE to the adapted helper")
        spec = importlib.util.spec_from_file_location("engram_nvtx_helper", path)
        helper = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = helper
        with patch.dict(os.environ, {"MCORE_COMPONENT_NVTX": "1"}):
            spec.loader.exec_module(helper)
        names = []
        opened = set()

        def start(name):
            names.append(name)
            handle = len(names)
            opened.add(handle)
            return handle

        nvtx = SimpleNamespace(
            range_push=lambda name: names.append(name),
            range_pop=lambda: None,
            range_start=start,
            range_end=opened.remove,
        )
        methods = load_methods(helper.component)
        holder = SimpleNamespace(embedding=nn.Embedding(16, 8))
        hashes = torch.tensor([[[0, 1], [1, 2]]])
        with patch.object(torch.cuda, "nvtx", nvtx):
            output = methods["_lookup"](holder, hashes)
            output.sum().backward()
        self.assertEqual(opened, set())
        self.assertTrue(any(name.startswith("comp:engram.lookup:") for name in names))
        self.assertTrue(any(name.startswith("comp:engram.lookup.bwd:") for name in names))


if __name__ == "__main__":
    unittest.main()
