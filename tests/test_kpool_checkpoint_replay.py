# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU contracts, real bundled CheckpointFunction and TP-local replay regressions.

MEGATRON_PATCH_BASE_WHEEL enables pinned bundled-source integration. CPU RNG
adapters do not test CUDA RNG, SparseMLA, production optimizer or PP scheduling.
"""
import ast
import contextlib
import functools
import gc
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
import weakref
import zipfile
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import detach_variable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
import test_kpool_tp_dedup as bundle  # noqa: E402


def load_replay():
    name = 'mcore_bridge.kpool_checkpoint_replay'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'src/mcore_bridge/kpool_checkpoint_replay.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bridge_stub(module):
    package = types.ModuleType('mcore_bridge')
    package.__path__ = []
    package.kpool_checkpoint_replay = module
    return patch.dict(sys.modules, {'mcore_bridge': package, 'mcore_bridge.kpool_checkpoint_replay': module})


def patched_random():
    spec = importlib.util.spec_from_file_location('replay_installer',
                                                  ROOT / 'src/mcore_bridge/tools/apply_megatron_patch.py')
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    name = 'megatron/core/tensor_parallel/random.py'
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        target = root / name
        target.parent.mkdir(parents=True)
        with zipfile.ZipFile(bundle.BASE_WHEEL) as wheel:
            target.write_bytes(wheel.read(name))
        result = installer.run_patch(root, installer.patch_files(False)[name], dry_run=False)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        text = target.read_text()
    symbols = {
        '_kpool_checkpoint_region', '_set_checkpointing', '_unset_checkpointing', 'is_checkpointing',
        'CheckpointFunction'
    }
    body = [
        node for node in ast.parse(text).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in symbols
    ]
    if {node.name for node in body} != symbols:
        raise AssertionError('Missing bundled checkpoint replay hooks')
    module = types.ModuleType('replay_checkpoint_fixture')
    module.__dict__.update(
        torch=torch, os=os, functools=functools, detach_variable=detach_variable, IS_CHECKPOINTING=False)
    module._get_all_rng_states = lambda: (torch.get_rng_state(), )
    module._set_all_rng_states = lambda state: torch.set_rng_state(state)

    @contextlib.contextmanager
    def fork_rng():
        state = torch.get_rng_state()
        try:
            yield
        finally:
            torch.set_rng_state(state)

    module._fork_rng = fork_rng
    future = ast.parse('from __future__ import annotations').body
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=future + body, type_ignores=[])), '<bundled-checkpoint>', 'exec'),
        module.__dict__)
    return module


def distributed_worker(rank, rendezvous, sources):
    torch.set_num_threads(1)
    replay = load_replay()
    with bridge_stub(replay):
        adapter, dsa = bundle.load_sources(sources)
        dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
        try:
            kwargs = bundle.sample()
            group = dist.group.WORLD
            for mode in ('observe', 'replay'):
                with patch.dict(os.environ, {'MCORE_KPOOL_CHECKPOINT_REPLAY': mode}):
                    ctx = types.SimpleNamespace()
                    signatures = []
                    original_gather = dist.all_gather

                    def gather(outputs, tensor, **options):
                        signatures.append((tuple(tensor.shape), str(tensor.dtype)))
                        return original_gather(outputs, tensor, **options)

                    with patch.object(dist, 'all_gather', gather), patch.object(
                            dsa, '_compute_index_scores', wraps=dsa._compute_index_scores) as score:
                        with replay.checkpoint_region(ctx, 'original'):
                            first = adapter.run_selection(
                                **kwargs, mode='tp_dedup', tp_group=group, check_replicas=False)
                        original_signatures = list(signatures)
                        signatures.clear()
                        original_calls = score.call_count
                        with replay.checkpoint_region(ctx, 'recompute'):
                            second = adapter.run_selection(
                                **kwargs, mode='tp_dedup', tp_group=group, check_replicas=False)
                        assert torch.equal(first.indices, second.indices)
                        assert signatures == original_signatures
                        assert score.call_count == original_calls * (2 if mode == 'observe' else 1)
                    assert not ctx._kpool_replay_owner.entries
                    assert replay.replay_stats()['live_bytes'] == 0
                    ctx = types.SimpleNamespace()
                    with replay.checkpoint_region(ctx, 'original'):
                        adapter.run_selection(**kwargs, mode='tp_dedup', tp_group=group, check_replicas=False)
                    changed = dict(kwargs)
                    if rank == 0:
                        changed['varlen_ends'] = kwargs['varlen_ends'] + 1
                    try:
                        with replay.checkpoint_region(ctx, 'recompute'):
                            adapter.run_selection(**changed, mode='tp_dedup', tp_group=group, check_replicas=False)
                    except replay.ReplayError:
                        pass
                    else:
                        raise AssertionError('Replay mismatch did not fail all TP ranks')
                    assert replay.replay_stats()['live_bytes'] == 0
            with patch.dict(os.environ, {'MCORE_KPOOL_CHECKPOINT_REPLAY': 'replay'}):
                for failure in ('layout', 'eligibility', 'clone'):
                    ctx = types.SimpleNamespace()
                    with replay.checkpoint_region(ctx, 'original'):
                        adapter.run_selection(**kwargs, mode='tp_dedup', tp_group=group, check_replicas=False)
                    changed = dict(kwargs)
                    if failure == 'layout' and rank == 0:
                        changed['cu_seqlens_kv'] = torch.tensor([0, 1, 20, 41], dtype=torch.int32)
                    try:
                        with replay.checkpoint_region(ctx, 'recompute'):
                            if failure == 'eligibility' and rank == 0:
                                replay.mark_failure('unfrozen indexer')
                            if failure == 'clone' and rank == 0:
                                replay.mark_failure('cache allocation failed')
                            adapter.run_selection(**changed, mode='tp_dedup', tp_group=group, check_replicas=False)
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError('Failure did not terminate all TP ranks')
                    assert replay.replay_stats()['live_bytes'] == 0
                ctx = types.SimpleNamespace()
                with replay.checkpoint_region(ctx, 'original'):
                    first = adapter.run_selection(
                        **kwargs, mode='tp_dedup', tp_group=group, query_is_tp_local=True, check_replicas=False)
                with replay.checkpoint_region(ctx, 'recompute'):
                    second = adapter.run_selection(
                        **kwargs, mode='tp_dedup', tp_group=group, query_is_tp_local=True, check_replicas=False)
                assert first.executed_mode == second.executed_mode == 'reference'
                assert torch.equal(first.indices, second.indices)
            with patch.dict(os.environ, {'MCORE_KPOOL_CHECKPOINT_REPLAY': 'observe'}):
                ctx = types.SimpleNamespace()
                try:
                    with replay.checkpoint_region(ctx, 'original'):
                        manager = patch.object(
                            replay, '_fingerprint',
                            side_effect=ValueError('fingerprint')) if rank == 0 else contextlib.nullcontext()
                        with manager:
                            adapter.run_selection(**kwargs, mode='tp_dedup', tp_group=group, check_replicas=False)
                except replay.ReplayError:
                    pass
                else:
                    raise AssertionError('Fingerprint failure did not propagate to all ranks')
                assert replay.replay_stats()['live_bytes'] == 0
            with patch.dict(os.environ, {'MCORE_KPOOL_CHECKPOINT_REPLAY': 'replay'}):
                ctx = types.SimpleNamespace()
                with replay.checkpoint_region(ctx, 'original'):
                    adapter.select_and_validate(**kwargs, mode='tp_dedup', tp_group=group, check_replicas=True)
                with patch.object(dsa, '_compute_index_scores', wraps=dsa._compute_index_scores) as score:
                    with replay.checkpoint_region(ctx, 'recompute'):
                        adapter.select_and_validate(**kwargs, mode='tp_dedup', tp_group=group, check_replicas=True)
                    assert score.call_count == 1
                assert replay.replay_stats()['live_bytes'] == 0
            with patch.dict(os.environ, {'MCORE_KPOOL_CHECKPOINT_REPLAY': 'observe' if rank == 0 else 'replay'}):
                try:
                    adapter.initialize_mode(group, loss_coeff=0, pool_size=4, index_share=False)
                except RuntimeError as error:
                    assert 'contract mismatch' in str(error)
                else:
                    raise AssertionError('Mixed replay settings accepted')
            assert not torch.cuda.is_initialized()
        finally:
            dist.destroy_process_group()


class ReplayContracts(unittest.TestCase):

    def setUp(self):
        self.replay = load_replay()
        self.env = patch.dict(
            os.environ, {
                'MCORE_KPOOL_CHECKPOINT_REPLAY': 'replay',
                'MCORE_KPOOL_REPLAY_MAX_BYTES': str(1024**3),
                'MCORE_KPOOL_REPLAY_MAX_ENTRIES': '64'
            })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.stub = bridge_stub(self.replay)
        self.stub.start()
        self.addCleanup(self.stub.stop)
        self.value = torch.arange(12, dtype=torch.int32).reshape(1, 3, 4)
        self.inputs = {'q': torch.ones(3, 1, 2, 4), 'mask': None}
        self.bounds = (torch.zeros(3, dtype=torch.int64), torch.arange(1, 4))

    def select(self, compute=None, contract=('layer', ), inputs=None, bounds=None):
        return self.replay.local_selection(
            compute or (lambda: self.value.clone()),
            contract=contract,
            inputs=inputs if inputs is not None else self.inputs,
            bounds=bounds if bounds is not None else self.bounds)

    def test_off_and_outside_checkpoint(self):
        called = []
        self.select(lambda: called.append(1) or self.value)
        with patch.dict(os.environ, {'MCORE_KPOOL_CHECKPOINT_REPLAY': 'off'}):
            ctx = types.SimpleNamespace()
            with self.replay.checkpoint_region(ctx, 'original'):
                self.select(lambda: called.append(1) or self.value)
            self.assertFalse(hasattr(ctx, '_kpool_replay_owner'))
        self.assertEqual(len(called), 2)

    def test_multiple_owners_entries_and_mutation_isolation(self):
        contexts = []
        for i in range(8):
            ctx = types.SimpleNamespace()
            contexts.append(ctx)
            with self.replay.checkpoint_region(ctx, 'original'):
                for j in range(2):
                    value = self.select(lambda: self.value.clone() + i + j, contract=(i, j))
                    value.zero_()
        self.assertEqual(self.replay.replay_stats()['live_entries'], 16)
        for i in reversed(range(8)):
            with self.replay.checkpoint_region(contexts[i], 'recompute'):
                for j in range(2):
                    actual = self.select(lambda: (_ for _ in ()).throw(AssertionError('recomputed')), contract=(i, j))
                    self.assertTrue(torch.equal(actual, self.value + i + j))
        stats = self.replay.replay_stats()
        self.assertEqual(stats['live_bytes'], 0)
        self.assertEqual(stats['hits'], 16)

    def test_observe_compares_contents_and_order(self):
        with patch.dict(os.environ, {'MCORE_KPOOL_CHECKPOINT_REPLAY': 'observe'}):
            for failure in ('q', 'ids', 'none'):
                ctx = types.SimpleNamespace()
                with self.replay.checkpoint_region(ctx, 'original'):
                    self.select()
                changed = {'q': self.inputs['q'] + 1, 'mask': None} if failure == 'q' else self.inputs
                compute = (lambda: self.value.flip(-1)) if failure == 'ids' else None
                manager = self.assertRaises(self.replay.ReplayError) if failure != 'none' else contextlib.nullcontext()
                with manager, self.replay.checkpoint_region(ctx, 'recompute'):
                    self.select(compute, inputs=changed)
            self.assertEqual(self.replay.replay_stats()['live_bytes'], 0)

    def test_bounds_layout_and_call_count_reject(self):
        for kind in ('bounds', 'shape', 'count', 'contract'):
            ctx = types.SimpleNamespace()
            with self.replay.checkpoint_region(ctx, 'original'):
                self.select()
            with self.assertRaises(self.replay.ReplayError), self.replay.checkpoint_region(ctx, 'recompute'):
                if kind == 'count':
                    continue
                self.select(
                    bounds=(self.bounds[0], self.bounds[1] + 1) if kind == 'bounds' else None,
                    inputs={
                        'q': torch.ones(4, 1, 2, 4),
                        'mask': None
                    } if kind == 'shape' else None,
                    contract=('changed', ) if kind == 'contract' else ('layer', ))
            self.assertEqual(self.replay.replay_stats()['live_bytes'], 0)

    def test_budget_limits_and_configuration(self):
        for name, value in [('MCORE_KPOOL_CHECKPOINT_REPLAY', 'bad'), ('MCORE_KPOOL_REPLAY_MAX_BYTES', '0'),
                            ('MCORE_KPOOL_REPLAY_MAX_ENTRIES', '-1')]:
            with patch.dict(os.environ, {name: value}), self.assertRaises(ValueError):
                self.replay.validate_configuration()
        with patch.dict(os.environ, {'MCORE_KPOOL_REPLAY_MAX_BYTES': '1'}):
            with self.assertRaises(self.replay.ReplayError), self.replay.checkpoint_region(
                    types.SimpleNamespace(), 'original'):
                self.select()
        self.assertEqual(self.replay.replay_stats()['live_bytes'], 0)

    def test_cleanup_exception_no_backward_and_retained_graph(self):
        ctx = types.SimpleNamespace()
        with self.assertRaises(ValueError), self.replay.checkpoint_region(ctx, 'original'):
            self.select()
            raise ValueError('forward')
        ctx = types.SimpleNamespace()
        with self.replay.checkpoint_region(ctx, 'original'):
            self.select()
        with self.assertRaises(ValueError), self.replay.checkpoint_region(ctx, 'recompute'):
            self.select()
            raise ValueError('backward')
        with self.assertRaises(self.replay.ReplayError), self.replay.checkpoint_region(ctx, 'recompute'):
            pass
        ctx = types.SimpleNamespace()
        with self.replay.checkpoint_region(ctx, 'original'):
            self.select()
        ref = weakref.ref(ctx._kpool_replay_owner)
        del ctx
        gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(self.replay.replay_stats()['live_bytes'], 0)

    def test_nested_and_mode_change_reject(self):
        with self.assertRaises(self.replay.ReplayError), self.replay.checkpoint_region(
                types.SimpleNamespace(), 'original'):
            with self.replay.checkpoint_region(types.SimpleNamespace(), 'original'):
                pass
        ctx = types.SimpleNamespace()
        with self.replay.checkpoint_region(ctx, 'original'):
            self.select()
        with patch.dict(os.environ, {'MCORE_KPOOL_CHECKPOINT_REPLAY': 'off'}):
            with self.assertRaises(self.replay.ReplayError), self.replay.checkpoint_region(ctx, 'recompute'):
                pass
        self.assertEqual(self.replay.replay_stats()['live_bytes'], 0)

    def test_call_identity_and_cache_allocation_failure(self):
        first = torch.nn.Linear(4, 2).requires_grad_(False)
        second = torch.nn.Linear(4, 2).requires_grad_(False)
        ctx = types.SimpleNamespace()
        with self.replay.checkpoint_region(ctx, 'original'):
            self.replay.validate_eligibility(first, loss_coeff=0, index_share=False, selection_mode='tp_dedup')
            self.select()
        with self.assertRaises(self.replay.ReplayError), self.replay.checkpoint_region(ctx, 'recompute'):
            self.replay.validate_eligibility(second, loss_coeff=0, index_share=False, selection_mode='tp_dedup')
            self.select()
        with patch.dict(os.environ, {'MCORE_KPOOL_CHECKPOINT_REPLAY': 'observe'}):
            with patch.object(self.replay, '_fingerprint', side_effect=ValueError('fingerprint')):
                with self.assertRaises(self.replay.ReplayError), self.replay.checkpoint_region(
                        types.SimpleNamespace(), 'original'):
                    self.select()
        self.assertEqual(self.replay.replay_stats()['live_bytes'], 0)

    def test_recompute_scope_ends_before_inner_backward(self):
        first = types.SimpleNamespace()
        second = types.SimpleNamespace()
        for ctx in (first, second):
            with self.replay.checkpoint_region(ctx, 'original'):
                self.select()
        with self.replay.checkpoint_region(first, 'recompute'):
            self.select()
            self.replay.finish_recompute()
            self.assertEqual(self.replay.current_contract(), ('off', None, 0))
            with self.replay.checkpoint_region(second, 'recompute'):
                self.select()
                self.replay.finish_recompute()
        self.assertEqual(self.replay.replay_stats()['live_bytes'], 0)

    def test_eligibility(self):
        indexer = torch.nn.Linear(4, 2)
        with self.replay.checkpoint_region(types.SimpleNamespace(), 'original'):
            with self.assertRaises(self.replay.ReplayError):
                self.replay.validate_eligibility(indexer, loss_coeff=0, index_share=False, selection_mode='tp_dedup')
            indexer.requires_grad_(False)
            self.replay.validate_eligibility(indexer, loss_coeff=0, index_share=False, selection_mode='tp_dedup')
            for options in (dict(loss_coeff=1, index_share=False, selection_mode='tp_dedup'),
                            dict(loss_coeff=0, index_share=True, selection_mode='tp_dedup'),
                            dict(loss_coeff=0, index_share=False, selection_mode='reference')):
                with self.assertRaises(self.replay.ReplayError):
                    self.replay.validate_eligibility(indexer, **options)

    @unittest.skipUnless(bundle.BASE_WHEEL, 'requires MEGATRON_PATCH_BASE_WHEEL')
    def test_bundled_checkpoint_outputs_gradients_rng_and_lifecycle(self):
        r = patched_random()
        graphs = types.ModuleType('megatron.core.transformer.cuda_graphs')
        graphs.is_graph_capturing = lambda: False

        def trial(mode):
            torch.manual_seed(123)
            calls = []
            pending = []
            xs = []
            with patch.dict(os.environ, {'MCORE_KPOOL_CHECKPOINT_REPLAY': mode}):
                for _ in range(4):
                    x = torch.randn(3, 4, requires_grad=True)
                    xs.append(x)

                    def fn(x):
                        with torch.no_grad():
                            ids = self.select(lambda: calls.append(1) or self.value.clone())
                        return x.sin() + ids[0].float() + torch.rand_like(x)

                    pending.append(r.CheckpointFunction.apply(fn, False, x))
                out = [x.detach().clone() for x in pending]
                for y in reversed(pending):
                    y.sum().backward()
                self.assertFalse(r.is_checkpointing())
                self.assertEqual(self.replay.replay_stats()['live_bytes'], 0)
                return out, [x.grad for x in xs], torch.get_rng_state(), len(calls)

        with patch.dict(sys.modules, {'megatron.core.transformer.cuda_graphs': graphs}):
            baseline = trial('off')
            observed = trial('observe')
            replayed = trial('replay')
        for actual in (observed, replayed):
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline[0], actual[0])))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline[1], actual[1])))
            self.assertTrue(torch.equal(baseline[2], actual[2]))
        self.assertEqual([baseline[3], observed[3], replayed[3]], [8, 8, 4])

    @unittest.skipUnless(bundle.BASE_WHEEL, 'requires MEGATRON_PATCH_BASE_WHEEL')
    def test_dependent_checkpoint_backward_does_not_consume_outer_cache(self):
        r = patched_random()
        graphs = types.ModuleType('megatron.core.transformer.cuda_graphs')
        graphs.is_graph_capturing = lambda: False

        def trial(mode):
            torch.manual_seed(123)
            x = torch.randn(3, 4, requires_grad=True)
            z = torch.randn(3, 4, requires_grad=True)
            calls = []

            def first(value):
                with torch.no_grad():
                    ids = self.select(lambda: calls.append(1) or self.value.clone(), contract=('first', ))
                return value.sin() + ids[0].float()

            with patch.dict(os.environ, {'MCORE_KPOOL_CHECKPOINT_REPLAY': mode}):
                y = r.CheckpointFunction.apply(first, False, x)

                def second(value):
                    with torch.no_grad():
                        ids = self.select(lambda: calls.append(1) or self.value.clone(), contract=('second', ))
                    return value.cos() + y + ids[0].float()

                output = r.CheckpointFunction.apply(second, False, z)
                output.sum().backward()
                self.assertFalse(r.is_checkpointing())
                self.assertEqual(self.replay.replay_stats()['live_bytes'], 0)
                return output.detach(), x.grad, z.grad, len(calls)

        with patch.dict(sys.modules, {'megatron.core.transformer.cuda_graphs': graphs}):
            baseline = trial('off')
            actual = trial('replay')
        for expected, value in zip(baseline[:3], actual[:3]):
            self.assertTrue(torch.equal(expected, value))
        self.assertEqual([baseline[3], actual[3]], [4, 2])

    @unittest.skipUnless(bundle.BASE_WHEEL, 'requires MEGATRON_PATCH_BASE_WHEEL')
    def test_two_rank_local_selection_and_symmetric_failure(self):
        sources = bundle.bundled_sources()
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                distributed_worker, args=('file://' + str(Path(directory) / 'store'), sources), nprocs=2, join=True)

    @unittest.skipUnless(bundle.BASE_WHEEL, 'requires MEGATRON_PATCH_BASE_WHEEL')
    def test_patched_sources_bind_os(self):
        import symtable

        def unbound_os(text, name):
            table = symtable.symtable(text, name, 'exec')
            bindings = {
                symbol.get_name()
                for symbol in table.get_symbols() if symbol.is_imported() or symbol.is_assigned()
            }
            scopes, missing = list(table.get_children()), []
            while scopes:
                scope = scopes.pop()
                scopes.extend(scope.get_children())
                for symbol in scope.get_symbols():
                    if (symbol.get_name() == 'os' and symbol.is_referenced() and symbol.is_global()
                            and 'os' not in bindings):
                        missing.append(scope.get_name())
            return missing

        self.assertEqual(unbound_os('def f():\n    return os.environ\n', 'missing'), ['f'])
        self.assertEqual(unbound_os('import os\ndef f():\n    return os.environ\n', 'global'), [])
        self.assertEqual(unbound_os('def f():\n    import os\n    return os.environ\n', 'local'), [])
        self.assertEqual(unbound_os('def g():\n    import os\ndef f():\n    return os.environ\n', 'wrong_scope'), ['f'])
        spec = importlib.util.spec_from_file_location('replay_installer',
                                                      ROOT / 'src/mcore_bridge/tools/apply_megatron_patch.py')
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        files = installer.patch_files(False)
        for name in ('megatron/core/transformer/experimental_attention_variant/dsa.py',
                     'megatron/core/tensor_parallel/random.py'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root / name
                target.parent.mkdir(parents=True)
                with zipfile.ZipFile(bundle.BASE_WHEEL) as wheel:
                    target.write_bytes(wheel.read(name))
                result = installer.run_patch(root, files[name], dry_run=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(unbound_os(target.read_text(), name), [])

    @unittest.skipUnless(bundle.BASE_WHEEL, 'requires MEGATRON_PATCH_BASE_WHEEL')
    def test_patched_dsa_replay_guard(self):
        text, _, _ = bundle.bundled_sources()
        attention = next(
            node for node in ast.parse(text).body if isinstance(node, ast.ClassDef) and node.name == 'DSAttention')
        forward = next(node for node in attention.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
        region = next(
            node for node in ast.walk(forward) if isinstance(node, ast.With) and any(
                isinstance(child, ast.Constant) and child.value == 'MCORE_KPOOL_CHECKPOINT_REPLAY'
                for child in ast.walk(node)))
        guard = next(node for node in region.body if isinstance(node, ast.If))
        body = region.body[:region.body.index(guard) + 1]
        code = compile(ast.Module(body=body, type_ignores=[]), '<dsa-replay-guard>', 'exec')
        indexer = object()
        attention = types.SimpleNamespace(
            indexer=indexer,
            index_share=False,
            _kpool_selection_mode='tp_dedup',
            config=types.SimpleNamespace(dsa_indexer_loss_coeff=0.0))
        with patch.dict(
                os.environ,
                MCORE_KPOOL_CHECKPOINT_REPLAY='off'), patch.dict(sys.modules,
                                                                 {'mcore_bridge.kpool_checkpoint_replay': None}):
            exec(code, {'self': attention})
        replay = load_replay()
        for mode in ('observe', 'replay'):
            with self.subTest(mode=mode), bridge_stub(replay), patch.dict(
                    os.environ, MCORE_KPOOL_CHECKPOINT_REPLAY=mode), patch.object(
                        replay, 'eligibility_contract', return_value=None) as eligibility:
                exec(code, {'self': attention})
                eligibility.assert_called_once_with(
                    indexer, loss_coeff=0.0, index_share=False, selection_mode='tp_dedup')


if __name__ == '__main__':
    unittest.main()
