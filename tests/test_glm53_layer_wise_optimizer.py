# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU regression for PR #6099 using functions from the actual installed patch."""
import ast
import importlib.util
import math
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = os.environ.get('MEGATRON_PATCH_BASE_SOURCE')
BASE_WHEEL = os.environ.get('MEGATRON_PATCH_BASE_WHEEL')


def load_functions(root, name, names, namespace, class_name=None):
    """Compile unchanged production functions without importing CUDA-only dependencies."""
    tree = ast.parse((root / name).read_text())
    if class_name is not None:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise AssertionError(f'Missing production functions in {name}: {names}')
    future = ast.parse('from __future__ import annotations').body
    exec(compile(ast.Module(body=future + functions, type_ignores=[]), name, 'exec'), namespace)


@unittest.skipUnless(torch is not None and (BASE_SOURCE or BASE_WHEEL), 'PyTorch and baseline source or wheel required')
class LayerWiseGradientStatisticsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(prefix='glm53-layerwise-')
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        spec = importlib.util.spec_from_file_location('layerwise_patch_installer',
                                                      ROOT / 'src/mcore_bridge/tools/apply_megatron_patch.py')
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        if BASE_WHEEL:
            with zipfile.ZipFile(BASE_WHEEL) as wheel:
                wheel.extractall(cls.root)
        else:
            for name, section in installer.patch_files(False).items():
                if installer.new_file_content(section) is None:
                    target = cls.root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes((Path(BASE_SOURCE) / name).read_bytes())
        installer.apply_patch(cls.root)
        installer.apply_patch(cls.root, check_only=True)

    def setUp(self):

        def cpu_zeros(*args, **kwargs):
            kwargs['device'] = 'cpu'
            return torch.zeros(*args, **kwargs)

        def multi_tensor_applier(implementation, overflow, gradients, per_parameter):
            return torch.cat([grad.flatten() for grad in gradients[0]]).norm(), None

        # Keep tensor math intact; replace CUDA allocation and collective transport only.
        self.namespace = {
            'torch': SimpleNamespace(**{
                **vars(torch), 'zeros': cpu_zeros
            }),
            'param_is_not_shared': lambda param: not getattr(param, 'shared', False),
            'get_tensor_model_parallel_rank': lambda: 1,
            'get_data_parallel_group_if_dtensor': lambda grad, group: group,
            'to_local_if_dtensor': lambda grad: grad,
            'multi_tensor_applier': multi_tensor_applier,
            'l2_norm_impl': None,
            'multi_tensor_scale_tensor_impl': None,
            'inf': math.inf,
            '_get_param_grad_norm_group': lambda param: None,
            '_is_separate_grad_norm_group': lambda group: False,
        }
        load_functions(self.root, 'megatron/core/tensor_parallel/layers.py', ['param_is_not_tensor_parallel_duplicate'],
                       self.namespace)
        self.namespace['tensor_parallel'] = SimpleNamespace(
            param_is_not_tensor_parallel_duplicate=self.namespace['param_is_not_tensor_parallel_duplicate'])
        load_functions(self.root, 'megatron/core/optimizer/clip_grads.py', ['get_grad_norm_fp32', 'count_zeros_fp32'],
                       self.namespace)
        load_functions(self.root, 'megatron/core/optimizer/optimizer.py',
                       ['_filter_grads_for_norm', 'get_grads_for_grad_norm'], self.namespace, 'MegatronOptimizer')

        namespace = self.namespace

        class Child:

            def __init__(self, params, config):
                self.params, self.config = params, config

            def get_parameters(self):
                return self.params

            _filter_grads_for_norm = namespace['_filter_grads_for_norm']
            get_grads_for_grad_norm = namespace['get_grads_for_grad_norm']

        class Float16Child(Child):

            def __init__(self, optimizer, config, scaler, init_state):
                super().__init__(optimizer.params, config)

        class FP32Child(Child):
            pass

        class Chained:

            def __init__(self, children):
                self.chained_optimizers = children
                self.config = children[0].config
                # Check what the parent receives, including newly created BF16 wrappers.
                self.groups_at_chain_init = [(getattr(child, 'tp_group', None), getattr(child, 'expert_tp_group', None))
                                             for child in children]

        self.namespace.update(
            Float16OptimizerWithFloat16Params=Float16Child,
            FP32Optimizer=FP32Child,
            is_float8tensor=lambda param: False,
            ChainedOptimizer=Chained)
        tree = ast.parse((self.root / 'megatron/core/optimizer/layer_wise_optimizer.py').read_text())
        layer_wise = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'LayerWiseDistributedOptimizer')
        layer_wise.body = [
            node for node in layer_wise.body
            if isinstance(node, ast.FunctionDef) and node.name in ('__init__', 'get_grad_norm', 'count_zeros')
        ]
        future = ast.parse('from __future__ import annotations').body
        # Preserve the class scope so the production constructor's super() works unchanged.
        exec(compile(ast.Module(body=future + [layer_wise], type_ignores=[]), '<layerwise>', 'exec'), self.namespace)
        self.child, self.layer_wise = Child, self.namespace['LayerWiseDistributedOptimizer']

        def shard_params(owner, optimizers, layouts, model_chunks):
            owner.dp_cp_params_list = owner.expt_dp_params_list = None

        self.layer_wise.shard_params = shard_params

    def make_optimizer(self, tp_rank, etp_rank, bf16=False, decoupled=False, missing_expert_group=False):
        config = SimpleNamespace(
            bf16=bf16,
            use_layer_wise_param_layout=False,
            overlap_param_gather=False,
            chunked_optimizer_state_offload=False,
            optimizer_state_offload_fraction=0.0,
            use_precision_aware_optimizer_no_fp8_or_ds_fp8=decoupled,
            use_precision_aware_optimizer=False)
        groups = SimpleNamespace(tp=SimpleNamespace(rank=lambda: tp_rank))
        if not missing_expert_group:
            groups.expt_tp = SimpleNamespace(rank=lambda: etp_rank)
        children = []
        for expert in (False, True):
            param = torch.nn.Parameter(torch.zeros(4))
            param.tensor_model_parallel = False
            param.allreduce = not expert
            grad = torch.tensor([0.0, 1.0, 0.0, 1.0])
            if decoupled:
                param.decoupled_grad = grad
            else:
                param.grad = grad
            children.append(self.child([param], config))
        return self.layer_wise(children, config, groups), groups

    def test_groups_reach_children_before_chain_initialization(self):
        for bf16 in (False, True):
            for fallback in (False, True):
                with self.subTest(bf16=bf16, fallback=fallback):
                    optimizer, groups = self.make_optimizer(1, 0, bf16=bf16, missing_expert_group=fallback)
                    expert_group = groups.tp if fallback else groups.expt_tp
                    self.assertEqual(optimizer.groups_at_chain_init, [(groups.tp, expert_group)] * 2)
                    self.assertIs(optimizer.tp_group, groups.tp)
                    self.assertIs(optimizer.expert_tp_group, expert_group)

    def test_norm_and_zeros_count_each_dense_and_expert_bias_once(self):
        # Each tuple represents (TP rank, ETP rank) at one simulated global rank.
        for topology in (((0, 0), (1, 0)), ((0, 0), (0, 1)), ((0, 0), (1, 1))):
            for bf16 in (False, True):
                for decoupled in (False, True):
                    with self.subTest(topology=topology, bf16=bf16, decoupled=decoupled):
                        squared_norm, zeros = 0.0, 0
                        with patch.object(torch.distributed, 'all_reduce'):
                            for tp_rank, etp_rank in topology:
                                optimizer, _ = self.make_optimizer(tp_rank, etp_rank, bf16, decoupled)
                                squared_norm += optimizer.get_grad_norm()**2
                                zeros += optimizer.count_zeros()
                        expected = 2 * (sum(tp == 0 for tp, _ in topology) + sum(etp == 0 for _, etp in topology))
                        self.assertAlmostEqual(squared_norm, expected, places=5)
                        self.assertEqual(zeros, expected)

    def test_missing_groups_reproduce_expert_bias_undercount(self):
        optimizer, _ = self.make_optimizer(1, 0)
        with patch.object(torch.distributed, 'all_reduce'):
            self.assertAlmostEqual(optimizer.get_grad_norm(), math.sqrt(2), places=5)
            self.assertEqual(optimizer.count_zeros(), 2)
            for child in optimizer.chained_optimizers:
                del child.tp_group, child.expert_tp_group
            self.assertEqual(optimizer.get_grad_norm(), 0)
            # Execute the pre-#6099 call with both groups absent.
            params = [param for child in optimizer.chained_optimizers for param in child.get_parameters()]
            self.assertEqual(self.namespace['count_zeros_fp32'](params, None), 0)


if __name__ == '__main__':
    unittest.main()
