# Copyright (c) ModelScope Contributors. All rights reserved.
"""Standard-library installer and packed-entry tests; no Torch/CUDA imports."""
import ast
import hashlib
import importlib.util
import os
import pathlib
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / 'src/mcore_bridge/tools/apply_swift_patch.py'
spec = importlib.util.spec_from_file_location('apply_swift_patch', TOOL)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)

# Minimal source fixture from supported Swift 07d4ecdaa, preserving its packed entry.
FIXTURE = '''# Copyright (c) ModelScope Contributors. All rights reserved.
import re
import torch
import torch.distributed as dist

def get_packed_seq_params(args, position_ids: torch.Tensor) -> PackedSeqParams:
    params = _get_packed_seq_params(position_ids)
    # max_seqlen must be a Python int rather than a 0-dim CUDA tensor.
    # flash-attn 4 (CuTe DSL) embeds booleans derived from max_seqlen into its
    # backward-kernel compile-cache key; a tensor there hashes by object identity,
    # so the cache never hits and every micro-batch backward triggers a full JIT
    # recompilation (~30s each), slowing training by >10x.
    max_seqlen_q = params['max_length_q']
    max_seqlen_kv = params['max_length_k']
    if isinstance(max_seqlen_q, torch.Tensor):
        max_seqlen_q = int(max_seqlen_q.item())
    if isinstance(max_seqlen_kv, torch.Tensor):
        max_seqlen_kv = int(max_seqlen_kv.item())
    packed = PackedSeqParams(
        cu_seqlens_q=params['cu_seq_lens_q'],
        cu_seqlens_kv=params['cu_seq_lens_k'],
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        qkv_format='thd',
    )
    if hasattr(packed, 'total_tokens'):
        packed.total_tokens = position_ids.numel()
    if hasattr(packed, 'cp_partition_mode'):
        packed.cp_partition_mode = args.cp_partition_mode

    if is_torch_npu_available():
        packed.cu_seqlens_q_padded = params['cu_seq_lens_q']
        packed.cu_seqlens_kv_padded = params['cu_seq_lens_k']

    return packed


'''


class SwiftRoutePatchTest(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name).resolve()
        self.target = self.root / installer.TARGET
        self.target.parent.mkdir(parents=True)
        self.target.write_text(FIXTURE)

    def patched(self):
        installer.apply_patch(self.root)
        return self.target.read_bytes()

    def test_apply_check_repeat_and_metadata_parity(self):
        expected = self.patched()
        installer.apply_patch(self.root, check_only=True)
        installer.apply_patch(self.root)
        self.assertEqual(expected, self.target.read_bytes())
        function = self.load_entry()
        os_stub = types.SimpleNamespace(environ={})
        function.__globals__['os'] = os_stub
        for enabled, cp, padded in [(None, 8, False), (False, 8, False), (True, 1, False), (True, 8, False),
                                    (True, 8, True)]:
            with self.subTest(enabled=enabled, cp=cp, padded=padded):
                os_stub.environ = {} if enabled is None else {'MCORE_CP_ROUTE_PREBUILD': str(int(enabled))}
                function.__globals__['is_torch_npu_available'] = lambda: padded
                value = types.SimpleNamespace(numel=lambda: 64)
                module = types.ModuleType('megatron.core.context_parallel_layout.utils')
                finalize = Mock()
                module.finalize_packed_seq_params = finalize
                with patch.dict('sys.modules', {'megatron.core.context_parallel_layout.utils': module}):
                    packed = function(
                        types.SimpleNamespace(context_parallel_size=cp, cp_partition_mode='zigzag'), value)
                self.assertEqual(packed.total_tokens, 64)
                self.assertEqual(packed.cp_partition_mode, 'zigzag')
                self.assertEqual(packed.max_seqlen_q, 32)
                self.assertEqual(packed.cu_seqlens_q, [0, 32, 64])
                if enabled and cp > 1:
                    finalize.assert_called_once_with(packed)
                    self.assertEqual(packed.qkv_format, 'thd')
                    if padded:
                        self.assertIs(packed.cu_seqlens_q_padded, packed.cu_seqlens_q)
                else:
                    finalize.assert_not_called()

    def load_entry(self):
        tree = ast.parse(self.target.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_packed_seq_params')
        function.args.args[1].annotation = None
        function.returns = None
        namespace = {
            '_get_packed_seq_params':
            lambda value: dict(max_length_q=32, max_length_k=32, cu_seq_lens_q=[0, 32, 64], cu_seq_lens_k=[0, 32, 64]),
            'torch':
            types.SimpleNamespace(Tensor=type('Tensor', (), {})),
            'PackedSeqParams':
            lambda **kwargs: types.SimpleNamespace(total_tokens=0, cp_partition_mode=None, **kwargs),
            'is_torch_npu_available':
            lambda: False,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(self.target), 'exec'), namespace)
        return namespace['get_packed_seq_params']

    def test_check_unpatched_does_not_write(self):
        original = self.target.read_bytes()
        self.assertEqual(installer.main(['--root', str(self.root), '--check']), 1)
        self.assertEqual(original, self.target.read_bytes())

    def test_partial_import_rejected(self):
        self.target.write_text(FIXTURE.replace('import re', 'import os\nimport re', 1))
        original = self.target.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'Partially'):
            installer.apply_patch(self.root)
        self.assertEqual(original, self.target.read_bytes())

    def test_partial_call_rejected(self):
        patched = self.patched().decode().replace('import os\n', '', 1)
        self.target.write_text(patched)
        with self.assertRaisesRegex(RuntimeError, 'Partially'):
            installer.apply_patch(self.root)
        self.assertEqual(patched, self.target.read_text())

    def test_conflict_preserves_original(self):
        self.target.write_text(FIXTURE.replace('    return packed', '    return other'))
        original = self.target.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'conflicts'):
            installer.apply_patch(self.root)
        self.assertEqual(original, self.target.read_bytes())

    def test_unrelated_source_changes_preserved(self):
        self.target.write_text(FIXTURE + '\n# unrelated caller\n')
        self.patched()
        self.assertTrue(self.target.read_text().endswith('# unrelated caller\n'))

    def test_source_checkout_index_unchanged(self):
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', 'add', installer.TARGET], cwd=self.root, check=True)
        index = (self.root / '.git/index').read_bytes()
        self.patched()
        installer.apply_patch(self.root, check_only=True)
        self.assertEqual(index, (self.root / '.git/index').read_bytes())

    def test_symlink_target_rejected(self):
        other = self.root / 'other.py'
        other.write_text(FIXTURE)
        self.target.unlink()
        self.target.symlink_to(other)
        with self.assertRaisesRegex(RuntimeError, 'symlinked'):
            installer.apply_patch(self.root)
        self.assertEqual(FIXTURE, other.read_text())

    def test_failed_publish_restores_original(self):
        original = self.target.read_bytes()
        writer = installer._installer.write_bytes
        failed = False

        def fail_once(descriptor, content):
            nonlocal failed
            if not failed:
                failed = True
                writer(descriptor, b'partial')
                raise OSError('simulated write failure')
            return writer(descriptor, content)

        with patch.object(installer._installer, 'write_bytes', fail_once):
            with self.assertRaisesRegex(OSError, 'simulated'):
                installer.apply_patch(self.root)
        self.assertEqual(original, self.target.read_bytes())

    def test_only_packaged_target_allowed(self):
        text = installer.PATCH.read_text().replace(installer.TARGET, 'other/file.py')
        malicious = self.root / 'other.patch'
        malicious.write_text(text)
        with patch.object(installer, 'PATCH', malicious):
            with self.assertRaisesRegex(RuntimeError, 'target or metadata'):
                installer.apply_patch(self.root)
        self.assertEqual(FIXTURE, self.target.read_text())

    def test_locate_swift_without_importing_training(self):
        with patch.object(
                installer.importlib.util,
                'find_spec',
                return_value=types.SimpleNamespace(submodule_search_locations=[str(self.root / 'swift')])) as find:
            self.assertEqual(installer.swift_root(), self.root)
        find.assert_called_once_with('swift')

    @unittest.skipUnless(os.environ.get('SWIFT_ROUTE_BASE_FILE'), 'SWIFT_ROUTE_BASE_FILE required for pinned source')
    def test_pinned_full_source(self):
        original = pathlib.Path(os.environ['SWIFT_ROUTE_BASE_FILE']).read_bytes()
        self.target.write_bytes(original)
        expected = self.patched()
        installer.apply_patch(self.root, check_only=True)
        self.assertEqual(expected, self.target.read_bytes())
        blob = b'blob ' + str(len(expected)).encode() + b'\0' + expected
        self.assertEqual(hashlib.sha1(blob).hexdigest(), '8908f11a29e40f66cc077ed7476eb796755cab91')
        body = ast.get_source_segment(
            expected.decode(),
            next(
                n for n in ast.parse(expected).body
                if isinstance(n, ast.FunctionDef) and n.name == 'get_packed_seq_params'))
        self.assertEqual(body.count('finalize_packed_seq_params(packed)'), 1)


if __name__ == '__main__':
    unittest.main()
