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
