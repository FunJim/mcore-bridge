"""Exercise actual patched optimizer methods with native CPU optimizer state."""

import argparse
import ast
import copy
import json
import logging
import math
import torch
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

METHODS = {
    '_restore_checkpoint_steps',
    'state_dict',
    'load_state_dict',
    '_get_main_param_and_optimizer_states',
    '_set_main_param_and_optimizer_states',
    'get_parameter_state_dp_reshardable',
    'load_parameter_state_from_dp_reshardable',
}


def namespace(root: Path) -> dict[str, object]:
    source = (root / 'megatron/core/optimizer/distrib_optimizer.py').read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'DistributedOptimizer')
    functions = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    ns = {
        'torch': torch,
        'math': math,
        'HAVE_APEX_OR_TE': True,
        'USING_TE_OPTIMIZER': True,
        'USING_APEX_OPTIMIZER': False,
        'HybridDeviceOptimizer': type('HybridMarker', (), {}),
        'logger': logging.getLogger('resume-test'),
        'logging': logging,
        'log_single_rank': lambda *a, **k: None,
        'param_group_identifier_keys': (
            'lr_mult',
            'is_expert_parallel',
            'is_decoupled_lr',
        ),
    }
    body = ast.parse('from __future__ import annotations').body + functions
    exec(
        compile(ast.Module(body=body, type_ignores=[]), '<actual-optimizer>', 'exec'),
        ns,
    )
    return ns


def fixture(ns: dict[str, object], groups: int = 2) -> tuple:
    params = [torch.nn.Parameter(torch.ones(4) * (i + 1)) for i in range(groups)]
    group_list = [{
        'params': [p],
        'lr_mult': i + 1,
        'is_expert_parallel': False,
        'is_decoupled_lr': False,
    } for i, p in enumerate(params)]
    optimizer = torch.optim.AdamW(group_list, lr=0.01, betas=(0.9, 0.95))
    for p in params:
        optimizer.state[p] = {
            'step': torch.tensor(1.0),
            'exp_avg': torch.zeros(4),
            'exp_avg_sq': torch.zeros(4),
        }
    dtype = (torch.float32, torch.float32)
    instance = SimpleNamespace(
        optimizer=optimizer,
        config=SimpleNamespace(
            optimizer='adam',
            fp16=False,
            use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
        ),
        ddp_config=SimpleNamespace(use_megatron_fsdp=False),
        _optimizer_state_offloader=None,
        grad_scaler=None,
        optimizer_state_keys=('exp_avg', 'exp_avg_sq'),
        model_param_group_index_map={
            p: (i, 0)
            for i, p in enumerate(params)
        },
        gbuf_ranges=[{
            dtype: [{
                'param_map': {
                    p: {
                        'gbuf_local': SimpleNamespace(start=0, end=4)
                    }
                }
            }]
        } for p in params],
        per_bucket_numel=[4] * groups,
        per_bucket_numel_unpadded=[4] * groups,
    )
    for name in METHODS:
        if name in ns:
            setattr(instance, name, MethodType(ns[name], instance))
    return instance, params, dtype


class ResumeTests(unittest.TestCase):

    def setUp(self) -> None:
        self.ns = namespace(ROOT)

    def test_group_steps_survive_shard_template_and_next_updates(self) -> None:
        instance, params, dtype = fixture(self.ns)
        groups = copy.deepcopy(instance.optimizer.state_dict()['param_groups'])
        for i, group in enumerate(groups):
            group.pop('params')
            group['step'] = [224, 17][i]
        payload = {
            i: {
                dtype: [[{
                    'param': torch.full((4, ), 3.0 + i),
                    'exp_avg': torch.full((4, ), 0.1),
                    'exp_avg_sq': torch.full((4, ), 0.2),
                    'step': torch.tensor(1.0),
                    'padding': False,
                }]]
            }
            for i in range(2)
        }
        instance.load_state_dict({
            'optimizer': {
                'param_groups': list(reversed(groups))
            },
            'param_state': payload,
            'param_state_sharding_type': 'dp_reshardable',
        })
        for i, p in enumerate(params):
            self.assertEqual(instance.optimizer.state[p]['step'].item(), [224, 17][i])
            self.assertTrue(torch.equal(p, torch.full((4, ), 3.0 + i)))
        for _ in range(2):
            for p in params:
                p.grad = torch.ones_like(p)
            instance.optimizer.step()
        self.assertEqual([instance.optimizer.state[p]['step'].item() for p in params], [226, 19])
        saved = instance.state_dict()
        self.assertEqual([g['step'] for g in saved['optimizer']['param_groups']], [226, 19])
        shard = instance.get_parameter_state_dp_reshardable()
        self.assertTrue(all('step' not in shard[i][dtype][0][0] for i in range(2)))

    def test_invalid_group_steps_fail(self) -> None:
        for bad in (None, -1, 1.5, float('nan'), float('inf'), True, torch.ones(2)):
            instance, _, _ = fixture(self.ns)
            groups = copy.deepcopy(instance.optimizer.state_dict()['param_groups'])
            for g in groups:
                g.pop('params')
                g['step'] = 224
            groups[-1]['step'] = bad
            with self.subTest(step=str(bad)), self.assertRaises(ValueError):
                instance.load_state_dict({'optimizer': {'param_groups': groups}})
            self.assertEqual(next(iter(instance.optimizer.state.values()))['step'].item(), 1)

    def test_missing_or_wrong_moments_fail_before_parameter_copy(self) -> None:
        for key in ('param', 'exp_avg', 'exp_avg_sq'):
            for invalid in ('missing', 'shape', 'dtype', 'non_tensor'):
                instance, params, _ = fixture(self.ns, 1)
                payload = {
                    'param': torch.ones(4) * 9,
                    'exp_avg': torch.ones(4),
                    'exp_avg_sq': torch.ones(4),
                }
                if invalid == 'missing':
                    payload.pop(key)
                elif invalid == 'shape':
                    payload[key] = torch.ones(3)
                elif invalid == 'dtype':
                    payload[key] = torch.ones(4, dtype=torch.float64)
                else:
                    payload[key] = None
                before = params[0].detach().clone()
                with (
                        self.subTest(key=key, invalid=invalid),
                        self.assertRaises((ValueError, KeyError)),
                ):
                    instance._set_main_param_and_optimizer_states(params[0], payload)
                self.assertTrue(torch.equal(before, params[0]))

    def test_precision_aware_skips_step_and_requires_moments(self) -> None:
        instance, params, _ = fixture(self.ns, 1)
        instance.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = True

        class HybridCPU(torch.optim.AdamW, self.ns['HybridDeviceOptimizer']):
            pass

        instance.optimizer = HybridCPU([params[0]])
        instance.optimizer.state[params[0]] = {
            'step': torch.tensor(224.0),
            'master_param': torch.zeros(4),
        }
        payload = {
            'param': torch.ones(4),
            'exp_avg': torch.ones(4),
            'exp_avg_sq': torch.ones(4),
            'step': torch.tensor(1.0),
        }
        instance._set_main_param_and_optimizer_states(params[0], payload)
        self.assertEqual(instance.optimizer.state[params[0]]['step'].item(), 224)
        self.assertTrue(torch.equal(instance.optimizer.state[params[0]]['master_param'], torch.ones(4)))
        with self.assertRaises(KeyError):
            instance._set_main_param_and_optimizer_states(params[0], {'param': torch.ones(4)})

    def test_non_adam_scalar_rule_is_unchanged(self) -> None:
        instance, _, _ = fixture(self.ns, 1)
        instance.config.optimizer = 'lion'
        self.ns['_restore_checkpoint_steps'](instance, {}, [{'params': [0]}])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    ROOT = args.root
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ResumeTests))
    print(
        json.dumps({
            'passed': result.wasSuccessful(),
            'tests': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
        }))
    raise SystemExit(not result.wasSuccessful())
