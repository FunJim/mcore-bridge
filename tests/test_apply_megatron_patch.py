# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU-only installer safety checks; no framework imports or GPUs required."""
import difflib
import importlib.util
import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch

TOOL = pathlib.Path(__file__).resolve().parents[1] / 'src/mcore_bridge/tools/apply_megatron_patch.py'
spec = importlib.util.spec_from_file_location('apply_megatron_patch', TOOL)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class ApplyPatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name).resolve()
        self.names = ['megatron/core/a.py', 'megatron/core/b.py']
        self.before = b'value = 1\n'
        self.after = b'value = 2\n'
        diffs = []
        for name in self.names:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.after)
            after_hash = installer.file_hash(target)
            target.write_bytes(self.before)
            before_hash = installer.file_hash(target)
            diff = ''.join(difflib.unified_diff(self.before.decode().splitlines(True),
                                               self.after.decode().splitlines(True),
                                               fromfile=f'a/{name}', tofile=f'b/{name}'))
            diffs.append(f'diff --git a/{name} b/{name}\nindex {before_hash}..{after_hash} 100644\n{diff}')
        self.patch_file = self.root / 'test.patch'
        self.patch_file.write_text(''.join(diffs))
        mock = patch.object(installer, 'PATCH', self.patch_file)
        mock.start()
        self.addCleanup(mock.stop)

    def test_apply_check_and_repeat(self) -> None:
        with self.assertRaisesRegex(RuntimeError, 'not applied'):
            installer.apply_patch(self.root, check_only=True)
        installer.apply_patch(self.root)
        installer.apply_patch(self.root, check_only=True)
        installer.apply_patch(self.root)
        self.assertTrue(all((self.root / name).read_bytes() == self.after for name in self.names))

    def test_partial_patch_rejected_without_writes(self) -> None:
        (self.root / self.names[0]).write_bytes(self.after)
        with self.assertRaisesRegex(RuntimeError, 'Partial or incompatible'):
            installer.apply_patch(self.root)
        self.assertEqual((self.root / self.names[1]).read_bytes(), self.before)

    def test_unknown_version_rejected_without_writes(self) -> None:
        (self.root / self.names[1]).write_bytes(b'upstream changed\n')
        with self.assertRaisesRegex(RuntimeError, 'Partial or incompatible'):
            installer.apply_patch(self.root)
        self.assertEqual((self.root / self.names[0]).read_bytes(), self.before)

    def test_symlink_rejected(self) -> None:
        target = self.root / self.names[0]
        target.unlink()
        target.symlink_to(self.root / self.names[1])
        with self.assertRaisesRegex(RuntimeError, 'symlinked'):
            installer.apply_patch(self.root)

    def test_failed_apply_restores_originals(self) -> None:
        run = subprocess.run

        def fail_after_write(command: list, **kwargs: object) -> subprocess.CompletedProcess:
            if '--dry-run' in command:
                return run(command, **kwargs)
            (self.root / self.names[0]).write_bytes(self.after)
            raise subprocess.CalledProcessError(1, command)

        with patch.object(installer.subprocess, 'run', side_effect=fail_after_write):
            with self.assertRaises(subprocess.CalledProcessError):
                installer.apply_patch(self.root)
        self.assertTrue(all((self.root / name).read_bytes() == self.before for name in self.names))

    def test_missing_wheel_file_rejected_before_writes(self) -> None:
        (self.root / self.names[1]).unlink()
        with self.assertRaisesRegex(RuntimeError, 'Missing'):
            installer.apply_patch(self.root)
        self.assertEqual((self.root / self.names[0]).read_bytes(), self.before)


if __name__ == '__main__':
    unittest.main()
