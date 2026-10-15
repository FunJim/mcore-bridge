# Copyright (c) ModelScope Contributors. All rights reserved.
"""Installer regressions using real patch/git executables, without Torch or CUDA."""
import difflib
import hashlib
import importlib.util
import os
import pathlib
import subprocess
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

TOOL = pathlib.Path(__file__).resolve().parents[1] / 'src/mcore_bridge/tools/apply_megatron_patch.py'
spec = importlib.util.spec_from_file_location('apply_megatron_patch', TOOL)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def blob(data: str) -> str:
    content = data.encode()
    return hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()


def make_diff(name: str, before: str, after: str) -> str:
    body = ''.join(
        difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=f'a/{name}', tofile=f'b/{name}'))
    return f'diff --git a/{name} b/{name}\nindex {blob(before)}..{blob(after)} 100644\n{body}'


def make_new_diff(name: str, content: str) -> str:
    body = ''.join(difflib.unified_diff([], content.splitlines(True), fromfile='/dev/null', tofile=f'b/{name}'))
    if not content.endswith('\n'):
        body += '\n\\ No newline at end of file\n'
    return (f'diff --git a/{name} b/{name}\nnew file mode 100644\n'
            f'index {"0" * 40}..{blob(content)}\n{body}')


class ApplyPatchTest(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name).resolve()
        self.names = ['megatron/core/transformer_config.py', 'megatron/core/optimizer.py']
        self.before = 'before_context = 1\nvalue = 1\ncontext_a = 1\ncontext_b = 1\nafter_context = 1\n'
        self.after = self.before.replace('value = 1', 'value = 2')
        self.test_name = 'tests/unit_tests/test_example.py'
        for name in self.names:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.before)
        self.patch_file = self.root / 'fixture.patch'
        self.patch_file.write_text(''.join(make_diff(name, self.before, self.after)
                                           for name in self.names) + make_diff(self.test_name, self.before, self.after))
        mock = patch.object(installer, 'PATCH', self.patch_file)
        mock.start()
        self.addCleanup(mock.stop)

    def git(self, *args):
        return subprocess.run(['git', *args], cwd=self.root, check=True, capture_output=True).stdout

    def init_git(self):
        target = self.root / self.test_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.before)
        self.git('init', '-q')
        self.git('add', '--', *self.names, self.test_name)
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'base')

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def test_wheel_apply_check_repeat_without_test_tree(self):
        self.assertFalse(installer.is_applied(self.root))
        installer.apply_patch(self.root)
        for name in self.names:
            self.assertEqual((self.root / name).read_text(), self.after)
        self.assertFalse((self.root / 'tests').exists())
        expected = self.snapshot()
        installer.apply_patch(self.root, check_only=True)
        installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_marker_alone_is_not_a_complete_installation(self):
        (self.root / self.names[0]).write_text('kda_two_stage_gates = False\n')
        (self.root / self.names[1]).unlink()
        expected = self.snapshot()
        self.assertFalse(installer.is_applied(self.root))
        with self.assertRaisesRegex(RuntimeError, 'Missing'):
            installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_partial_patch_rejected(self):
        (self.root / self.names[0]).write_text(self.after)
        expected = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'Partially'):
            installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_partial_hunks_in_one_file_rejected(self):
        before = self.before + '\n' * 10 + 'second = 1\n'
        after = self.after + '\n' * 10 + 'second = 2\n'
        self.patch_file.write_text(make_diff(self.names[0], before, after))
        (self.root / self.names[0]).write_text(before.replace('value = 1', 'value = 2'))
        expected = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'Partially'):
            installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_partial_source_rejected_without_index_changes(self):
        self.init_git()
        (self.root / self.names[0]).write_text(self.after)
        expected = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'Partially'):
            installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_missing_source_test_file_rejected(self):
        self.init_git()
        (self.root / self.test_name).unlink()
        expected = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'Missing'):
            installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_failed_application_never_writes_target(self):
        expected = self.snapshot()
        original = installer.run_patch

        def fail_apply(root, text, reverse=False, dry_run=True):
            if not dry_run:
                (root / self.names[0]).write_text('partially applied')
                return subprocess.CompletedProcess(['patch'], 1, '', 'simulated failure')
            return original(root, text, reverse, dry_run)

        with patch.object(installer, 'run_patch', fail_apply):
            with self.assertRaisesRegex(RuntimeError, 'simulated failure'):
                installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_unrelated_wheel_edits_preserved(self):
        for name in self.names:
            with (self.root / name).open('a') as target:
                target.write('\n# unrelated upstream addition\n')
        installer.apply_patch(self.root)
        self.assertTrue(installer.is_applied(self.root))
        for name in self.names:
            self.assertEqual((self.root / name).read_text(), self.after + '\n# unrelated upstream addition\n')

    def test_conflict_leaves_no_partial_files_or_rejects(self):
        (self.root / self.names[1]).write_text('conflicting content\n')
        expected = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'conflicts'):
            installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_symlink_rejected(self):
        (self.root / self.names[0]).unlink()
        (self.root / self.names[0]).symlink_to(self.root / self.names[1])
        with self.assertRaisesRegex(RuntimeError, 'symlinked'):
            installer.apply_patch(self.root)

    def test_failed_publish_restores_originals(self):
        expected = self.snapshot()
        original = installer.write_bytes
        calls = 0

        def fail_once(descriptor, data):
            nonlocal calls
            calls += 1
            if calls == 2:
                original(descriptor, b'partial write')
                raise OSError('simulated write failure')
            return original(descriptor, data)

        with patch.object(installer, 'write_bytes', fail_once):
            with self.assertRaisesRegex(OSError, 'simulated'):
                installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_source_applies_test_diff_without_staging(self):
        self.init_git()
        index = (self.root / '.git/index').read_bytes()
        installer.apply_patch(self.root)
        for name in self.names + [self.test_name]:
            self.assertEqual((self.root / name).read_text(), self.after)
        self.assertEqual(index, (self.root / '.git/index').read_bytes())
        self.assertEqual(self.git('diff', '--cached'), b'')
        self.assertTrue(installer.is_applied(self.root))

    def test_three_way_preserves_context_changes_and_index(self):
        self.init_git()
        # Change a context line: patch --fuzz=0 fails, while a three-way merge is clean.
        target = self.root / self.names[0]
        target.write_text(self.before.replace('after_context = 1', 'after_context = 9'))
        self.git('add', '--', self.names[0])
        index = (self.root / '.git/index').read_bytes()
        installer.apply_patch(self.root)
        self.assertEqual(target.read_text(), self.after.replace('after_context = 1', 'after_context = 9'))
        self.assertEqual(index, (self.root / '.git/index').read_bytes())
        expected = self.snapshot()
        installer.apply_patch(self.root, check_only=True)
        installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_three_way_conflict_preserves_worktree_and_index(self):
        self.init_git()
        (self.root / self.names[0]).write_text(self.before.replace('value = 1', 'value = 99'))
        expected = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'conflicts'):
            installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_wheel_nested_in_git_does_not_require_tests(self):
        self.init_git()
        site = self.root / 'venv/lib/site-packages'
        for name in self.names:
            target = site / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.before)
        self.assertFalse(installer.git_root(site))
        installer.apply_patch(site)
        self.assertFalse((site / 'tests').exists())

    def test_find_package_without_importing_core(self):
        location = self.root / 'megatron'
        (location / 'core/__init__.py').write_text('raise RuntimeError("must not import")\n')
        with patch.object(
                installer.importlib.util,
                'find_spec',
                return_value=SimpleNamespace(submodule_search_locations=[str(location)])) as find:
            self.assertEqual(installer.megatron_root(), self.root)
        find.assert_called_once_with('megatron')

    def test_readonly_check_returns_nonzero_for_unpatched_root(self):
        expected = self.snapshot()
        self.assertEqual(installer.main(['--root', str(self.root), '--check']), 1)
        self.assertEqual(expected, self.snapshot())

    def test_packaged_runtime_patch_parses(self):
        with patch.object(installer, 'PATCH', TOOL.parent.parent / 'patches/megatron_glm53_dev.patch'):
            runtime = installer.patch_files(False)
            source = installer.patch_files(True)
        self.assertTrue(all(name.startswith('megatron/core/') for name in runtime))
        self.assertIn('megatron/core/transformer/transformer_config.py', runtime)
        self.assertEqual(source, runtime)


class NewFilePatchTest(ApplyPatchTest):

    def setUp(self):
        super().setUp()
        self.new_name = 'megatron/core/dsa_kpool_kernels.py'
        self.new_content = '"""KPool kernels."""\n\nvalue = 3\n'
        self.patch_file.write_text(self.patch_file.read_text() + make_new_diff(self.new_name, self.new_content))

    def assert_new_file(self):
        target = self.root / self.new_name
        self.assertEqual(target.read_bytes(), self.new_content.encode())
        self.assertEqual(target.stat().st_mode & 0o777, 0o644)

    def test_new_file_wheel_and_source_lifecycle(self):
        for source in (False, True):
            with self.subTest(source=source):
                if source:
                    self.init_git()
                for name in self.names:
                    (self.root / name).write_text(self.before)
                (self.root / self.new_name).unlink(missing_ok=True)
                expected = self.snapshot()
                with self.assertRaisesRegex(RuntimeError, 'not fully applied'):
                    installer.apply_patch(self.root, check_only=True)
                self.assertEqual(expected, self.snapshot())
                installer.apply_patch(self.root)
                self.assert_new_file()
                expected = self.snapshot()
                self.assertTrue(installer.is_applied(self.root))
                installer.apply_patch(self.root)
                self.assertEqual(expected, self.snapshot())

    def test_new_file_only_lifecycle(self):
        self.patch_file.write_text(make_new_diff(self.new_name, self.new_content))
        installer.apply_patch(self.root)
        self.assert_new_file()
        self.assertTrue(installer.is_applied(self.root))

    def test_new_file_without_final_newline(self):
        self.new_content = 'value = 3'
        self.patch_file.write_text(make_new_diff(self.new_name, self.new_content))
        installer.apply_patch(self.root)
        self.assert_new_file()
        self.assertTrue(installer.is_applied(self.root))

    def test_new_file_collision_in_fresh_or_applied_tree(self):
        for applied in (False, True):
            for content in ('', self.new_content + '# extra\n', self.new_content.replace('3', '4')):
                with self.subTest(applied=applied, content=content):
                    for name in self.names:
                        (self.root / name).write_text(self.after if applied else self.before)
                    (self.root / self.new_name).write_text(content)
                    expected = self.snapshot()
                    self.assertFalse(installer.is_applied(self.root))
                    with self.assertRaisesRegex(RuntimeError, 'collision'):
                        installer.apply_patch(self.root)
                    self.assertEqual(expected, self.snapshot())

    def test_partial_new_file_only_or_existing_files_only(self):
        for source in (False, True):
            if source:
                self.init_git()
            for module_only in (False, True):
                with self.subTest(source=source, module_only=module_only):
                    for name in self.names + ([self.test_name] if source else []):
                        (self.root / name).write_text(self.before if module_only else self.after)
                    target = self.root / self.new_name
                    target.unlink(missing_ok=True)
                    if module_only:
                        target.write_text(self.new_content)
                    expected = self.snapshot()
                    with self.assertRaisesRegex(RuntimeError, 'Partially'):
                        installer.apply_patch(self.root)
                    self.assertEqual(expected, self.snapshot())

    def test_three_way_with_new_target_preserves_context_and_index(self):
        self.init_git()
        target = self.root / self.names[0]
        target.write_text(self.before.replace('after_context = 1', 'after_context = 9'))
        self.git('add', '--', self.names[0])
        index = (self.root / '.git/index').read_bytes()
        installer.apply_patch(self.root)
        self.assert_new_file()
        self.assertEqual(target.read_text(), self.after.replace('after_context = 1', 'after_context = 9'))
        self.assertEqual(index, (self.root / '.git/index').read_bytes())
        expected = self.snapshot()
        installer.apply_patch(self.root, check_only=True)
        installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_new_file_is_not_added_to_temporary_index_while_absent(self):
        self.init_git()
        (self.root / self.names[0]).write_text(self.before.replace('after_context = 1', 'after_context = 9'))
        original = subprocess.run
        additions = []

        def inspect(command, *args, **kwargs):
            if command[:2] == ['git', 'add']:
                additions.append(command)
                self.assertNotIn(self.new_name, command)
            return original(command, *args, **kwargs)

        with patch.object(installer.subprocess, 'run', inspect):
            installer.apply_patch(self.root)
        self.assertTrue(additions)
        self.assert_new_file()

    def test_publish_failure_removes_new_file_and_restores_existing(self):
        self.patch_file.write_text(make_new_diff(self.new_name, self.new_content) + self.patch_file.read_text().split(
            f'diff --git a/{self.new_name}', 1)[0])
        expected = self.snapshot()
        original = installer.write_bytes
        calls = 0

        def fail_once(descriptor, data):
            nonlocal calls
            calls += 1
            if calls == 3:
                original(descriptor, b'partial write')
                raise OSError('simulated publish failure')
            return original(descriptor, data)

        with patch.object(installer, 'write_bytes', fail_once):
            with self.assertRaisesRegex(OSError, 'simulated'):
                installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())
        self.assertFalse((self.root / self.new_name).exists())

    def test_double_failure_continues_all_other_rollbacks(self):
        self.patch_file.write_text(make_new_diff(self.new_name, self.new_content) + self.patch_file.read_text().split(
            f'diff --git a/{self.new_name}', 1)[0])
        expected = self.snapshot()
        original = installer.write_bytes
        publish_error = OSError('simulated publish failure')
        rollback_error = OSError('simulated restore failure')
        calls = 0

        def fail_twice(descriptor, data):
            nonlocal calls
            calls += 1
            if calls == 3:
                original(descriptor, b'partial publish')
                raise publish_error
            if calls == 4:
                raise rollback_error
            return original(descriptor, data)

        with patch.object(installer, 'write_bytes', fail_twice):
            with self.assertRaisesRegex(RuntimeError, 'Rollback incomplete') as raised:
                installer.apply_patch(self.root)
        self.assertIs(raised.exception.__cause__, publish_error)
        self.assertIn(str(self.root / self.names[1]), str(raised.exception))
        self.assertIn(str(rollback_error), str(raised.exception))
        self.assertIn('manual recovery required', str(raised.exception))
        self.assertEqual(calls, 5)
        expected[self.names[1]] = b'partial publish'
        self.assertEqual(expected, self.snapshot())
        self.assertEqual((self.root / self.names[0]).read_text(), self.before)
        self.assertFalse((self.root / self.new_name).exists())

    def test_rollback_reports_multiple_failures_and_continues(self):
        self.patch_file.write_text(make_new_diff(self.new_name, self.new_content) + self.patch_file.read_text().split(
            f'diff --git a/{self.new_name}', 1)[0])
        expected = self.snapshot()
        original = installer.write_bytes
        publish_error = OSError('simulated publish failure')
        calls = 0

        def fail_restores(descriptor, data):
            nonlocal calls
            calls += 1
            if calls == 3:
                original(descriptor, b'partial publish')
                raise publish_error
            if data == self.before.encode():
                raise OSError(f'simulated restore failure {calls}')
            return original(descriptor, data)

        with patch.object(installer, 'write_bytes', fail_restores):
            with self.assertRaisesRegex(RuntimeError, 'Rollback incomplete') as raised:
                installer.apply_patch(self.root)
        self.assertIs(raised.exception.__cause__, publish_error)
        for name in self.names:
            self.assertIn(str(self.root / name), str(raised.exception))
        self.assertEqual(calls, 5)
        expected[self.names[0]] = self.after.encode()
        expected[self.names[1]] = b'partial publish'
        self.assertEqual(expected, self.snapshot())
        self.assertFalse((self.root / self.new_name).exists())

    def test_failed_new_file_cleanup_does_not_stop_old_file_restore(self):
        expected = self.snapshot()
        original = installer.write_bytes
        publish_error = OSError('simulated new-file publish failure')
        calls = 0

        def fail_new(descriptor, data):
            nonlocal calls
            calls += 1
            if data == self.new_content.encode():
                original(descriptor, b'partial new file')
                raise publish_error
            return original(descriptor, data)

        original_unlink = os.unlink

        def fail_cleanup(path, *args, **kwargs):
            if path == pathlib.Path(self.new_name).name and kwargs.get('dir_fd') is not None:
                parent = os.fstat(kwargs['dir_fd'])
                target_parent = (self.root / self.new_name).parent.stat()
                if (parent.st_dev, parent.st_ino) == (target_parent.st_dev, target_parent.st_ino):
                    raise OSError('simulated unlink failure')
            return original_unlink(path, *args, **kwargs)

        with patch.object(installer, 'write_bytes', fail_new), patch.object(installer.os, 'unlink', fail_cleanup):
            with self.assertRaisesRegex(RuntimeError, 'Rollback incomplete') as raised:
                installer.apply_patch(self.root)
        self.assertIs(raised.exception.__cause__, publish_error)
        self.assertIn(str(self.root / self.new_name), str(raised.exception))
        self.assertIn('simulated unlink failure', str(raised.exception))
        self.assertEqual(calls, 5)
        expected[self.new_name] = b'partial new file'
        self.assertEqual(expected, self.snapshot())
        for name in self.names:
            self.assertEqual((self.root / name).read_text(), self.before)

    def test_failure_during_new_file_write_removes_created_file(self):
        expected = self.snapshot()
        original = installer.write_bytes

        def fail_new(descriptor, data):
            if data == self.new_content.encode():
                original(descriptor, b'partial new file')
                raise OSError('simulated new-file failure')
            return original(descriptor, data)

        with patch.object(installer, 'write_bytes', fail_new):
            with self.assertRaisesRegex(OSError, 'simulated'):
                installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_concurrent_target_before_publish_is_not_overwritten(self):
        original = installer.prepare_patch

        def create_target(*args):
            original(*args)
            (self.root / self.new_name).write_text('concurrent owner\n')

        expected = self.snapshot()
        with patch.object(installer, 'prepare_patch', create_target):
            with self.assertRaisesRegex(RuntimeError, 'changed during'):
                installer.apply_patch(self.root)
        expected[self.new_name] = b'concurrent owner\n'
        self.assertEqual(expected, self.snapshot())

    def test_concurrent_target_after_preflight_is_not_overwritten(self):
        original = os.open
        fired = False

        def race(path, flags, *args, **kwargs):
            nonlocal fired
            if path == pathlib.Path(self.new_name).name and flags & os.O_EXCL and not fired:
                fired = True
                (self.root / self.new_name).write_text('concurrent owner\n')
            return original(path, flags, *args, **kwargs)

        expected = self.snapshot()
        with patch.object(installer.os, 'open', race):
            with self.assertRaises(FileExistsError):
                installer.apply_patch(self.root)
        self.assertTrue(fired)
        expected[self.new_name] = b'concurrent owner\n'
        self.assertEqual(expected, self.snapshot())

    def test_rollback_does_not_delete_concurrent_replacement(self):
        original = installer.write_bytes

        def replace_new(descriptor, data):
            if data == self.new_content.encode():
                target = self.root / self.new_name
                target.unlink()
                target.write_text('replacement owner\n')
                raise OSError('simulated concurrent replacement')
            return original(descriptor, data)

        expected = self.snapshot()
        with patch.object(installer, 'write_bytes', replace_new):
            with self.assertRaisesRegex(OSError, 'simulated'):
                installer.apply_patch(self.root)
        expected[self.new_name] = b'replacement owner\n'
        self.assertEqual(expected, self.snapshot())

    def test_source_applied_new_module_requires_exact_content(self):
        self.init_git()
        installer.apply_patch(self.root)
        target = self.root / self.new_name
        target.write_text(self.new_content + '# incompatible edit\n')
        expected = self.snapshot()
        self.assertFalse(installer.is_applied(self.root))
        with self.assertRaisesRegex(RuntimeError, 'collision'):
            installer.apply_patch(self.root)
        self.assertEqual(expected, self.snapshot())

    def test_only_new_file_three_way_has_no_git_add_for_absent_target(self):
        self.init_git()
        self.patch_file.write_text(make_new_diff(self.new_name, self.new_content))
        original = installer.run_patch
        forced = False

        def force_merge(root, text, reverse=False, dry_run=True):
            nonlocal forced
            if not reverse and dry_run and not forced:
                forced = True
                return subprocess.CompletedProcess(['patch'], 1, '', 'forced context drift')
            return original(root, text, reverse, dry_run)

        index = (self.root / '.git/index').read_bytes()
        with patch.object(installer, 'run_patch', force_merge):
            installer.apply_patch(self.root)
        self.assert_new_file()
        self.assertEqual(index, (self.root / '.git/index').read_bytes())

    def test_existing_target_changed_after_preflight_is_not_overwritten(self):
        original = os.open
        fired = False

        def race(path, flags, *args, **kwargs):
            nonlocal fired
            if path == pathlib.Path(self.names[1]).name and flags & os.O_RDWR and not fired:
                fired = True
                (self.root / self.names[1]).write_text('concurrent edit\n')
            return original(path, flags, *args, **kwargs)

        expected = self.snapshot()
        with patch.object(installer.os, 'open', race):
            with self.assertRaisesRegex(RuntimeError, 'changed during'):
                installer.apply_patch(self.root)
        self.assertTrue(fired)
        expected[self.names[1]] = b'concurrent edit\n'
        self.assertEqual(expected, self.snapshot())

    def test_existing_target_symlink_race_is_not_followed(self):
        external = self.root / 'external.py'
        external.write_text('external content\n')
        original = os.open
        fired = False

        def race(path, flags, *args, **kwargs):
            nonlocal fired
            if path == pathlib.Path(self.names[1]).name and flags & os.O_RDWR and not fired:
                fired = True
                target = self.root / self.names[1]
                target.unlink()
                target.symlink_to(external)
            return original(path, flags, *args, **kwargs)

        with patch.object(installer.os, 'open', race):
            with self.assertRaises(OSError):
                installer.apply_patch(self.root)
        self.assertEqual(external.read_text(), 'external content\n')
        self.assertEqual((self.root / self.names[0]).read_text(), self.before)
        self.assertFalse((self.root / self.new_name).exists())

    def test_post_publish_replacement_is_detected_and_not_overwritten(self):
        original = installer.write_bytes
        replaced = False

        def replace_existing(descriptor, data):
            nonlocal replaced
            original(descriptor, data)
            if data == self.after.encode() and not replaced:
                replaced = True
                target = self.root / self.names[0]
                target.unlink()
                target.write_text('concurrent replacement\n')

        expected = self.snapshot()
        with patch.object(installer, 'write_bytes', replace_existing):
            with self.assertRaisesRegex(RuntimeError, 'changed during patch publication'):
                installer.apply_patch(self.root)
        expected[self.names[0]] = b'concurrent replacement\n'
        self.assertEqual(expected, self.snapshot())

    def test_new_file_exact_utf8_content_and_abbreviated_index(self):
        self.new_content = 'description = "内核"\n'
        text = make_new_diff(self.new_name, self.new_content)
        self.patch_file.write_text(text.replace(blob(self.new_content), blob(self.new_content)[:12]))
        installer.apply_patch(self.root)
        self.assert_new_file()
        self.assertTrue(installer.is_applied(self.root))

    def test_new_target_symlink_dangling_directory_and_ancestor_rejected(self):
        target = self.root / self.new_name
        for destination in (self.root / self.names[0], self.root / 'missing'):
            with self.subTest(destination=destination):
                target.symlink_to(destination)
                with self.assertRaisesRegex(RuntimeError, 'symlinked'):
                    installer.apply_patch(self.root)
                target.unlink()
        target.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'symlinked'):
            installer.apply_patch(self.root)
        target.rmdir()
        real = self.root / 'real-parent'
        real.mkdir()
        linked = self.root / 'megatron/core/linked'
        linked.symlink_to(real, target_is_directory=True)
        self.patch_file.write_text(make_new_diff('megatron/core/linked/kernel.py', self.new_content))
        with self.assertRaisesRegex(RuntimeError, 'symlinked'):
            installer.apply_patch(self.root)
        self.assertFalse((real / 'kernel.py').exists())

    def test_missing_new_target_parent_is_not_created(self):
        self.patch_file.write_text(make_new_diff('megatron/core/missing/kernel.py', self.new_content))
        expected = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'parent'):
            installer.apply_patch(self.root)
        self.assertFalse((self.root / 'megatron/core/missing').exists())
        self.assertEqual(expected, self.snapshot())

    def test_ancestor_symlink_race_is_rejected_without_external_writes(self):
        parent = self.root / 'megatron/core/module'
        parent.mkdir()
        external = self.root / 'external'
        external.mkdir()
        self.patch_file.write_text(make_new_diff('megatron/core/module/kernel.py', self.new_content))
        original = installer.parent_fd

        def replace_parent(root, name):
            parent.rmdir()
            parent.symlink_to(external, target_is_directory=True)
            return original(root, name)

        with patch.object(installer, 'parent_fd', replace_parent):
            with self.assertRaises(OSError):
                installer.apply_patch(self.root)
        self.assertFalse((external / 'kernel.py').exists())

    def test_malicious_paths_rejected_in_both_modes(self):
        names = ['megatron/core/../escape.py', 'megatron/core/./kernel.py', 'megatron/core//kernel.py',
                 '/megatron/core/kernel.py', 'megatron/core/kernel.py/', 'megatron/core/kernel\\x.py',
                 'megatron/core/kernel\x00.py', 'megatron/core/kernel name.py', 'other/kernel.py']
        for name in names:
            for source in (False, True):
                with self.subTest(name=name, source=source):
                    self.patch_file.write_text(make_new_diff(name, self.new_content))
                    with self.assertRaisesRegex(RuntimeError, 'Unsupported'):
                        installer.patch_files(source)

    def test_unsupported_diff_metadata_and_invalid_new_content_rejected(self):
        valid = make_new_diff(self.new_name, self.new_content)
        mutations = [valid.replace('100644', mode) for mode in ('100755', '120000', '160000')]
        mutations += [valid.replace('new file mode', 'deleted file mode'),
                      valid.replace('--- /dev/null', f'--- a/{self.new_name}'),
                      valid.replace(f'+++ b/{self.new_name}', '+++ /dev/null'),
                      valid.replace(f'b/{self.new_name}\n', 'b/megatron/core/renamed.py\n', 1),
                      valid.replace('new file mode 100644\n', 'similarity index 100%\nrename from a\nrename to b\n'),
                      valid.replace('@@ -0,0 +1,3 @@', '@@ -1,0 +1,3 @@'),
                      valid.replace('@@ -0,0 +1,3 @@', '@@ -0,0 +1,4 @@'),
                      valid.replace('+value = 3', ' value = 3'),
                      valid.replace(blob(self.new_content), '1' * 40),
                      valid + 'GIT binary patch\nliteral 1\nA\n',
                      valid + 'old mode 100644\nnew mode 100755\n']
        for text in mutations:
            with self.subTest(text=text):
                self.patch_file.write_text(text)
                with self.assertRaises(RuntimeError):
                    installer.patch_files(False)

    def test_duplicate_ignored_test_targets_are_still_rejected(self):
        text = make_new_diff(self.test_name, self.new_content)
        self.patch_file.write_text(make_new_diff(self.new_name, self.new_content) + text + text)
        with self.assertRaisesRegex(RuntimeError, 'Unsupported'):
            installer.patch_files(False)

    def test_existing_file_unsupported_modes_and_headers_rejected(self):
        valid = make_diff(self.names[0], self.before, self.after)
        texts = [valid.replace('100644', '100755'), valid.replace('100644', '120000'),
                 valid.replace('index ', 'old mode 100644\nnew mode 100755\nindex '),
                 valid.replace(f'+++ b/{self.names[0]}', '+++ /dev/null'),
                 valid + 'Binary files a/file and b/file differ\n']
        for text in texts:
            with self.subTest(text=text):
                self.patch_file.write_text(text)
                with self.assertRaises(RuntimeError):
                    installer.patch_files(False)


class PackagedPatchIntegrationTest(unittest.TestCase):
    """Optionally exercise the entire bundled patch against real upstream artifacts."""

    @unittest.skipUnless(
        os.environ.get('MEGATRON_PATCH_BASE_WHEEL'), 'set MEGATRON_PATCH_BASE_WHEEL to the baseline wheel')
    def test_baseline_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with zipfile.ZipFile(os.environ['MEGATRON_PATCH_BASE_WHEEL']) as wheel:
                wheel.extractall(root)
            self.verify_complete_patch(root, source=False)

    @unittest.skipUnless(
        os.environ.get('MEGATRON_PATCH_BASE_SOURCE'), 'set MEGATRON_PATCH_BASE_SOURCE to baseline sources')
    def test_baseline_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            base = pathlib.Path(os.environ['MEGATRON_PATCH_BASE_SOURCE'])
            for name, diff in installer.patch_files(True).items():
                if installer.new_file_content(diff) is not None:
                    self.assertFalse((base / name).exists(), f'New module must be absent in baseline: {name}')
                    continue
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((base / name).read_bytes())
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            self.verify_complete_patch(root, source=True)

    def verify_complete_patch(self, root, source):
        entries = installer.patch_files(source)
        for name, diff in entries.items():
            if installer.new_file_content(diff) is not None:
                self.assertFalse((root / name).exists(), f'New module must be absent in baseline: {name}')
        self.assertFalse(installer.is_applied(root))
        installer.apply_patch(root)
        installer.apply_patch(root, check_only=True)
        installer.apply_patch(root)
        for name, diff in entries.items():
            expected = diff.split('index ', 1)[1].split('..', 1)[1].split()[0]
            self.assertEqual(len(expected), 40, f'Bundle must use full-index Git blobs: {name}')
            self.assertEqual(blob((root / name).read_text()), expected, name)
            content = installer.new_file_content(diff)
            if content is not None:
                self.assertEqual((root / name).read_bytes(), content, name)


if __name__ == '__main__':
    unittest.main()
