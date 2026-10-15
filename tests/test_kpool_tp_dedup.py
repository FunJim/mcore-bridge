# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU regressions of the bundled KPool code; no installed Megatron imports.

Set MEGATRON_PATCH_BASE_WHEEL to the pinned, unpatched Megatron wheel. Real
helper functions and the constructor/selection branch are AST-isolated from
the patched source. This does not exercise full attention forward/backward.
"""
import ast
import hashlib
import importlib.util
import logging
import os
import pathlib
import sys
import tempfile
import time
import types
import unittest
import zipfile
from datetime import timedelta
from unittest.mock import patch

try:
    import torch
    import torch.distributed as dist
except ImportError:
    torch = None
    dist = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = '_kpool_bundle_test'
BASE_WHEEL = os.environ.get('MEGATRON_PATCH_BASE_WHEEL')
DSA_PATH = 'megatron/core/transformer/experimental_attention_variant/dsa.py'
MASK_PATH = 'megatron/core/transformer/experimental_attention_variant/dsa_masking.py'
ADAPTER_PATH = 'megatron/core/transformer/experimental_attention_variant/dsa_kpool_kernels.py'
HELPERS = {'_compute_index_scores', '_kpool_compress_keys', '_expand_pools_to_tokens',
           '_append_tail_to_topk', 'fused_qk_topk_kpool'}
MASK_HELPERS = {'build_valid_mask_from_starts_ends', 'apply_starts_ends_mask_to_scores',
                'normalize_varlen_bounds'}


def bundled_sources():
    spec = importlib.util.spec_from_file_location(
        '_kpool_installer_test', ROOT / 'src/mcore_bridge/tools/apply_megatron_patch.py')
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    entries = installer.patch_files(False)
    with zipfile.ZipFile(BASE_WHEEL) as wheel:
        original = wheel.read(DSA_PATH)
        mask_text = wheel.read(MASK_PATH).decode()
    expected_base = entries[DSA_PATH].split('index ', 1)[1].split('..', 1)[0]
    digest = hashlib.sha1(b'blob ' + str(len(original)).encode() + b'\0' + original).hexdigest()
    if digest != expected_base:
        raise AssertionError('MEGATRON_PATCH_BASE_WHEEL is not the pinned DSA baseline')
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        target = root / DSA_PATH
        target.parent.mkdir(parents=True)
        target.write_bytes(original)
        result = installer.run_patch(root, entries[DSA_PATH], dry_run=False)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        dsa_text = target.read_text()
    adapter_text = installer.new_file_content(entries[ADAPTER_PATH]).decode()
    return dsa_text, mask_text, adapter_text


def compile_functions(text, names, module):
    tree = ast.parse(text)
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in body} != names:
        raise AssertionError('Missing isolated helper')
    future = ast.parse('from __future__ import annotations').body
    exec(compile(ast.fix_missing_locations(ast.Module(body=future + body, type_ignores=[])),
                 '<bundled-helpers>', 'exec'), module.__dict__)


def load_sources(sources):
    dsa_text, mask_text, adapter_text = sources
    package = types.ModuleType(PACKAGE)
    package.__path__ = []
    sys.modules[PACKAGE] = package
    masking = types.ModuleType(PACKAGE + '.dsa_masking')
    masking.torch = torch
    sys.modules[masking.__name__] = masking
    package.dsa_masking = masking
    compile_functions(mask_text, MASK_HELPERS, masking)
    dsa = types.ModuleType(PACKAGE + '.dsa')
    dsa.torch, dsa.dsa_masking = torch, masking
    sys.modules[dsa.__name__] = dsa
    package.dsa = dsa
    compile_functions(dsa_text, HELPERS, dsa)
    adapter = types.ModuleType(PACKAGE + '.dsa_kpool_kernels')
    for module in (masking, dsa, adapter):
        module.__package__ = PACKAGE
    sys.modules[adapter.__name__] = adapter
    package.dsa_kpool_kernels = adapter
    exec(compile(adapter_text, '<bundled-kpool>', 'exec'), adapter.__dict__)
    cls = next(node for node in ast.parse(dsa_text).body
               if isinstance(node, ast.ClassDef) and node.name == 'DSAttention')
    constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')

    class Base:
        def __init__(self, config):
            self.config = config

    isolated = ast.ClassDef(name='DSAttention', bases=[ast.Name(id='MegatronModule', ctx=ast.Load())],
                            keywords=[], body=[constructor], decorator_list=[])
    namespace = dsa.__dict__
    namespace.update(MegatronModule=Base, build_module=lambda *args, **kwargs: kwargs['config'].indexer,
                     dsa_layout=types.SimpleNamespace(normalize_cp_comm_type=lambda value: value),
                     logger=logging.getLogger(__name__))
    future = ast.parse('from __future__ import annotations').body
    exec(compile(ast.fix_missing_locations(ast.Module(body=future + [isolated], type_ignores=[])),
                 '<bundled-constructor>', 'exec'), namespace)
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    branch = next(node for node in ast.walk(forward)
                  if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'is_kpool')
    wrapper = ast.parse('def dispatch(self, q, k, weights, tp_group, sequence_parallel_query_is_local, '
                        'float_mask, varlen_starts, varlen_ends, key_positions, cu_seqlens_kv, '
                        'packed_thd, kpool_gate_score):\n    pass\n').body[0]
    wrapper.body = branch.body + [ast.Return(value=ast.Name(id='topk_indices', ctx=ast.Load()))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                 '<bundled-selection-branch>', 'exec'), namespace)
    return adapter, dsa


def sample(case='packed', dtype=None, batch=1, lengths=(7, 13, 21)):
    dtype = dtype or torch.bfloat16
    generator = torch.Generator().manual_seed(7)
    rows = sum(lengths)
    q = torch.randn(rows, batch, 4, 8, generator=generator).to(dtype)
    k = torch.randn(rows, batch, 8, generator=generator).to(dtype)
    weights = torch.randn(rows, batch, 4, generator=generator)
    gate = torch.randn(rows, batch, 8, generator=generator).to(dtype)
    ape = torch.randn(4, 8, generator=generator)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
    positions = torch.arange(rows)
    starts = cu[torch.searchsorted(cu[1:].long(), positions, right=True)].long()
    ends = positions + 1
    if case == 'unpacked':
        cu, starts = None, torch.zeros_like(ends)
    elif case == 'zigzag':
        order = torch.cat((positions[::2], positions[1::2].flip(0)))
        starts, ends = starts[order], ends[order]
    elif case == 'ties':
        q.zero_()
    elif case == 'near_ties':
        q.mul_(1e-4)
    elif case == 'empty_rows':
        ends = starts.clone()
    kwargs = dict(q=q, k=k, weights=weights, index_topk=16, pool_size=4,
                  gate_score=gate, ape=ape, varlen_starts=starts, varlen_ends=ends,
                  cu_seqlens_kv=cu, use_relu=True, always_select_tail=True)
    if case == 'explicit_positions':
        kwargs.update(key_positions=positions * 2, varlen_starts=starts * 2, varlen_ends=ends * 2)
    return kwargs


def new_attention(dsa, mode='tp_dedup', policy='first', group=None):
    indexer = types.SimpleNamespace(index_kpool=4, index_kpool_compress_ape=sample()['ape'],
                                    index_kpool_always_select_tail=True)
    config = types.SimpleNamespace(num_layers=1, dsa_indexer_topk=16, dsa_indexer_topk_freq=1,
                                   dsa_indexer_skip_topk_offset=0, dsa_indexer_loss_coeff=0,
                                   dsa_indexer_kpool=4, kv_channels=8, dsa_indexer_scoring_relu=True,
                                   indexer=indexer)
    groups = types.SimpleNamespace(tp=group, cp=None)
    with patch.dict(os.environ, {'MCORE_KPOOL_SELECTION_MODE': mode, 'MCORE_KPOOL_VALIDATE': policy}):
        return dsa.DSAttention(config, types.SimpleNamespace(indexer=object()), 1, None, 'self',
                               softmax_scale=1.0, pg_collection=groups)


def dispatch(dsa, attention, kwargs, group=None, local=False):
    return dsa.dispatch(attention, kwargs['q'], kwargs['k'], kwargs['weights'], group, local,
                        kwargs.get('mask'), kwargs.get('varlen_starts'), kwargs.get('varlen_ends'),
                        kwargs.get('key_positions'), kwargs.get('cu_seqlens_kv'),
                        kwargs.get('cu_seqlens_kv') is not None, kwargs['gate_score'])


def distributed_worker(rank, rendezvous, sources):
    torch.set_num_threads(1)
    adapter, dsa = load_sources(sources)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=45))
    try:
        group = dist.group.WORLD
        for case in ('packed', 'unpacked', 'zigzag', 'ties', 'near_ties', 'explicit_positions', 'empty_rows'):
            kwargs = sample(case)
            _, expected = dsa.fused_qk_topk_kpool(**kwargs)
            with patch.object(dsa, '_compute_index_scores', wraps=dsa._compute_index_scores) as scores:
                actual = adapter.run_selection(**kwargs, mode='tp_dedup', tp_group=group, check_replicas=False)
                rows = sum(call.args[0].shape[0] for call in scores.call_args_list)
            assert actual.executed_mode == 'tp_dedup' and torch.equal(actual.indices, expected)
            count = torch.tensor(rows, dtype=torch.int64)
            dist.all_reduce(count, group=group)
            assert count.item() == kwargs['q'].shape[0]
            assert actual.local_rows == (kwargs['q'].shape[0] if rank == 0 else 0)
        kwargs = sample()
        attention = new_attention(dsa, group=group)
        with patch.object(adapter, 'verify_tp_replicas', wraps=adapter.verify_tp_replicas) as verify:
            expected = adapter.select_and_validate(**kwargs, mode='tp_dedup', tp_group=group).indices
            verify.reset_mock()
            assert torch.equal(dispatch(dsa, attention, kwargs, group, local=True), expected)
            assert not attention._kpool_validated and verify.call_count == 0
            assert torch.equal(dispatch(dsa, attention, kwargs, group), expected)
            assert attention._kpool_validated and verify.call_count == 1
            assert torch.equal(dispatch(dsa, attention, kwargs, group), expected)
            assert verify.call_count == 1
        attention = new_attention(dsa, policy='all', group=group)
        with patch.object(adapter, 'verify_tp_replicas', wraps=adapter.verify_tp_replicas) as verify:
            dispatch(dsa, attention, kwargs, group)
            dispatch(dsa, attention, kwargs, group)
            assert verify.call_count == 2
        original_reference = dsa.fused_qk_topk_kpool

        def mismatched_reference(*args, **kwargs):
            scores, indices = original_reference(*args, **kwargs)
            if rank == 1:
                indices = indices.clone()
                indices[0, 0, 0] = 123456
            return scores, indices

        with patch.object(dsa, 'fused_qk_topk_kpool', side_effect=mismatched_reference):
            try:
                adapter.select_and_validate(**kwargs, mode='tp_dedup', tp_group=group)
            except RuntimeError as error:
                assert 'ordered indices differ' in str(error)
            else:
                raise AssertionError('Index mismatch did not fail on every rank')
        changed = sample()
        if rank == 1:
            changed['q'][0, 0, 0, 0] += 1
        try:
            adapter.run_selection(**changed, mode='tp_dedup', tp_group=group)
        except RuntimeError as error:
            assert 'replicas differ' in str(error)
        else:
            raise AssertionError('Unequal replicas accepted')
        with patch.dict(os.environ, {'MCORE_KPOOL_SELECTION_MODE': 'reference' if rank == 0 else 'tp_dedup'}):
            try:
                adapter.initialize_mode(group, loss_coeff=0, pool_size=4, index_share=False)
            except RuntimeError as error:
                assert 'contract mismatch' in str(error)
            else:
                raise AssertionError('Mixed modes accepted')
        with patch.object(dist, 'all_gather_object', side_effect=AssertionError('object collective')):
            with patch.dict(os.environ, {'MCORE_KPOOL_SELECTION_MODE': 'reference'}):
                assert adapter.initialize_mode(group, loss_coeff=0, pool_size=4, index_share=False) == 'reference'
        subgroup = dist.new_group([1, 3], backend='gloo', timeout=timedelta(seconds=45))
        if rank in (1, 3):
            actual = adapter.select_and_validate(**kwargs, mode='tp_dedup', tp_group=subgroup)
            assert torch.equal(actual.indices, expected)
            assert actual.local_rows == (41 if rank == 1 else 0)
        dist.barrier()
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(torch is not None and BASE_WHEEL, 'requires Torch and MEGATRON_PATCH_BASE_WHEEL')
class KPoolTPDedupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.sources = bundled_sources()
        cls.adapter, cls.dsa = load_sources(cls.sources)

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name == PACKAGE or name.startswith(PACKAGE + '.'):
                del sys.modules[name]

    def test_reference_tp1_and_boundaries(self):
        for dtype in (torch.float32, torch.bfloat16):
            for case in ('packed', 'unpacked', 'zigzag', 'ties', 'near_ties', 'explicit_positions', 'empty_rows'):
                with self.subTest(dtype=dtype, case=case):
                    kwargs = sample(case, dtype)
                    _, expected = self.dsa.fused_qk_topk_kpool(**kwargs)
                    for mode in ('reference', 'tp_dedup'):
                        actual = self.adapter.select_and_validate(**kwargs, mode=mode)
                        self.assertEqual(actual.executed_mode, 'reference')
                        self.assertTrue(torch.equal(actual.indices, expected))
                        self.assertEqual(actual.indices.dtype, torch.int32)
                        self.assertFalse(actual.indices.requires_grad)

    def test_partition_preserves_reference_chunks(self):
        for queries in (0, 1, 13, 41, 16387):
            ranges = self.adapter.shard_ranges(queries, 256, 4)
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], queries)
            self.assertEqual(sum(b - a for a, b in ranges), queries)
            self.assertTrue(all(b == ranges[i + 1][0] for i, (_, b) in enumerate(ranges[:-1])))
            self.assertTrue(all(a in (0, queries) or a % 256 == 0 for a, _ in ranges))
        self.assertEqual(self.adapter.shard_ranges(16384, 256, 4),
                         [(0, 4096), (4096, 8192), (8192, 12288), (12288, 16384)])
        for args in ((-1, 8, 4), (1, 0, 4), (1, 8, 0)):
            with self.assertRaises(ValueError):
                self.adapter.shard_ranges(*args)

    def test_multiple_chunks_select_same_order(self):
        for case in ('packed', 'zigzag', 'ties', 'explicit_positions'):
            kwargs = sample(case)
            pooled, bases = self.adapter._prepare(kwargs['k'], kwargs['gate_score'], kwargs['ape'],
                                                  4, kwargs['cu_seqlens_kv'])
            starts, ends, positions = self.adapter._bounds(kwargs['q'], bases, 4, kwargs['varlen_starts'],
                                                          kwargs['varlen_ends'], kwargs.get('key_positions'),
                                                          kwargs['cu_seqlens_kv'])
            pieces = []
            for lo, hi in self.adapter.shard_ranges(41, 8, 4):
                scores = self.dsa._compute_index_scores(kwargs['q'][lo:hi], kwargs['weights'][lo:hi], pooled)
                scores = self.dsa.dsa_masking.apply_starts_ends_mask_to_scores(
                    scores, starts[lo:hi], ends[lo:hi], positions)
                pieces.append(self.adapter._select(scores, bases, starts[lo:hi], ends[lo:hi], 16, 4, True))
            _, expected = self.dsa.fused_qk_topk_kpool(**kwargs)
            self.assertTrue(torch.equal(torch.cat(pieces, dim=1), expected))

    def test_empty_pools_tail_and_output_lifetime(self):
        for lengths in ((1, 2), (3, 5), (4,), (7, 9)):
            for tail in (False, True):
                kwargs = sample(lengths=lengths)
                kwargs['always_select_tail'] = tail
                actual = self.adapter.run_selection(**kwargs)
                saved = actual.indices.clone()
                self.adapter.run_selection(**sample())
                self.assertTrue(torch.equal(saved, actual.indices))
                self.assertEqual(actual.indices.shape[-1], 19 if tail else 16)
                valid = actual.indices >= 0
                selected = actual.indices.clamp_min(0).long()
                self.assertFalse(torch.any(valid & ((selected < kwargs['varlen_starts'][None, :, None])
                                                    | (selected >= kwargs['varlen_ends'][None, :, None]))))

    def test_batch_mask_and_local_fallback(self):
        for kwargs in (sample(batch=2), sample()):
            if kwargs['q'].shape[1] == 1:
                kwargs['varlen_starts'] = kwargs['varlen_ends'] = None
                kwargs['mask'] = torch.zeros(41, 41)
            _, expected = self.dsa.fused_qk_topk_kpool(**kwargs)
            actual = self.adapter.select_and_validate(**kwargs, mode='tp_dedup', query_is_tp_local=True)
            self.assertEqual(actual.executed_mode, 'reference')
            self.assertTrue(torch.equal(actual.indices, expected))

    def test_mode_contract_and_default_constructor(self):
        for field, value in (('MCORE_KPOOL_SELECTION_MODE', 'bad'), ('MCORE_KPOOL_VALIDATE', 'none')):
            with patch.dict(os.environ, {field: value}):
                with self.assertRaises(ValueError):
                    self.adapter.initialize_mode(None, loss_coeff=0, pool_size=4, index_share=False)
        with patch.dict(os.environ, {'MCORE_KPOOL_SELECTION_MODE': 'tp_dedup', 'MCORE_KPOOL_VALIDATE': 'first'}):
            for loss, pool, sharing in ((1, 4, False), (0, 1, False), (0, 4, True)):
                with self.assertRaises(ValueError):
                    self.adapter.initialize_mode(None, loss_coeff=loss, pool_size=pool, index_share=sharing)
        attention = new_attention(self.dsa, mode='reference')
        kwargs = sample()
        with patch.object(self.adapter, 'select_and_validate', side_effect=AssertionError('optimized path')):
            actual = dispatch(self.dsa, attention, kwargs)
        _, expected = self.dsa.fused_qk_topk_kpool(**kwargs)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(attention._kpool_validated)
        self.assertIsNone(attention._kpool_last_selection)
        self.assertFalse(torch.cuda.is_initialized())

    @unittest.skipUnless(dist is not None and dist.is_available() and dist.is_gloo_available(), 'requires Gloo')
    def test_real_tp4_and_nonzero_subgroup(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = pathlib.Path(directory, 'rendezvous').as_uri()
            context = torch.multiprocessing.spawn(distributed_worker,
                                                  args=(rendezvous, self.sources), nprocs=4, join=False)
            deadline = time.monotonic() + 120
            try:
                while not context.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail('Gloo regression exceeded 120 seconds')
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
