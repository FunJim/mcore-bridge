# Copyright (c) ModelScope Contributors. All rights reserved.
"""Apply the Megatron-LM patch that GLM-5.3 needs to the installed megatron.

    pip install git+https://github.com/NVIDIA/Megatron-LM.git@dev
    python -m mcore_bridge.tools.apply_megatron_patch

The patch (`patches/megatron_glm53_dev.patch`, generated against dev ee743d3ef) backports the
unmerged Megatron-LM #7054 GLM numerics -- KDA two-stage gates, mHC precision, KPool DSA -- plus
Megatron fixes that are not model specific (expert shards deduplicated over the expert TP group when
the gradient norm is computed, `allreduce` preserved on fp32 master parameters, and #6099's TP-group
propagation to LayerWise child optimizers and zero counting). Drop each backport when its fix is
present in the supported baseline.

#7054 is tracked by content, not by branch: the numerics come from its `be805e55`, and the chunked
index-score computation from its current head `3fceb0715`. The chunking matters at real scale --
with `index_n_heads=32` the un-chunked fp32 `[seqlen_q, batch, heads, seqlen_k]` tensor is 8 GiB at
sequence 8192 and 128 GiB when packed to 32768 -- and is bit-identical to the un-chunked version.

Prepare an unused environment; do not apply from training ranks. Source checkouts retain
Git three-way merge support. Wheels only receive runtime files, not Megatron's unit tests.
Application is prepared and checked in a temporary directory before any target is written;
conflicts leave the checkout, including its index, unchanged. --check only verifies the patch.
"""
import argparse
import hashlib
import importlib.util
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Dict, Iterator, List, Optional

PATCH = pathlib.Path(__file__).resolve().parent.parent / 'patches' / 'megatron_glm53_dev.patch'


def megatron_root() -> pathlib.Path:
    """Locate the package without importing megatron.core or initializing CUDA dependencies."""
    spec = importlib.util.find_spec('megatron')
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            if (pathlib.Path(location) / 'core' / '__init__.py').is_file():
                return pathlib.Path(location).resolve().parent
    raise RuntimeError('Megatron is not installed; install the supported Megatron version first.')


def git_root(root: pathlib.Path) -> bool:
    """A wheel inside another repository is not a Megatron source checkout."""
    if shutil.which('git') is None:
        return False
    result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=root, capture_output=True, text=True)
    return result.returncode == 0 and pathlib.Path(result.stdout.strip()).resolve() == root


def new_file_content(section: str) -> Optional[bytes]:
    """Decode the complete contents of a restricted regular-file addition."""
    if '\nnew file mode 100644\n' not in section:
        return None
    header, body = section.split('\n@@ ', 1)
    match = re.fullmatch(r'-0,0 \+1(?:,(\d+))? @@[^\n]*\n(.*)', body, flags=re.DOTALL)
    if match is None:
        raise RuntimeError('Unsupported new-file hunk')
    lines = match[2].splitlines(keepends=True)
    content = []
    for line in lines:
        if line == '\\ No newline at end of file\n' and content and content[-1].endswith('\n'):
            content[-1] = content[-1][:-1]
        elif line.startswith('+') and line.endswith('\n'):
            content.append(line[1:])
        else:
            raise RuntimeError('Unsupported new-file content')
    if len(content) != int(match[1] or '1') or not content:
        raise RuntimeError('Invalid new-file line count')
    result = ''.join(content).encode('utf-8')
    digest = hashlib.sha1(b'blob ' + str(len(result)).encode() + b'\0' + result).hexdigest()
    expected = re.search(r'\nindex 0+\.\.([0-9a-f]+)\n', header)
    if expected is None or not digest.startswith(expected[1]):
        raise RuntimeError('New-file Git blob does not match its complete contents')
    return result


def validate_hunks(body: str) -> None:
    """Reject malformed hunks and any unsupported metadata outside their counted lines."""
    lines = iter(body.splitlines(keepends=True))
    line = next(lines, None)
    while line is not None:
        match = re.fullmatch(r'@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@[^\n]*\n', line)
        if match is None:
            raise RuntimeError('Unsupported patch hunk')
        old, new = int(match[1] or '1'), int(match[2] or '1')
        previous = False
        line = next(lines, None)
        while line is not None and not line.startswith('@@ '):
            if line == '\\ No newline at end of file\n' and previous:
                previous = False
            elif line.endswith('\n') and line[:1] in (' ', '+', '-'):
                old -= line[0] in (' ', '-')
                new -= line[0] in (' ', '+')
                previous = True
            else:
                raise RuntimeError('Unsupported patch content')
            if old < 0 or new < 0:
                raise RuntimeError('Invalid patch hunk line count')
            line = next(lines, None)
        if old or new:
            raise RuntimeError('Invalid patch hunk line count')


def patch_files(source_checkout: bool) -> Dict[str, str]:
    """Select regular-file edits/additions, preserving tests only for source checkouts."""
    entries = {}
    seen = set()
    for section in re.split(r'(?=^diff --git )', PATCH.read_text(encoding='utf-8'), flags=re.MULTILINE):
        if not section.strip():
            continue
        match = re.match(r'diff --git a/([A-Za-z0-9_./-]+) b/\1\n', section)
        if match is None:
            raise RuntimeError('Unsupported patch header')
        name = match[1]
        if (not name.startswith(('megatron/core/', 'tests/'))
                or any(part in ('', '.', '..') for part in name.split('/')) or name in seen):
            raise RuntimeError(f'Unsupported patch target: {name}')
        seen.add(name)
        path = re.escape(name)
        header = re.match(
            rf'(?:index [0-9a-f]{{7,40}}\.\.[0-9a-f]{{7,40}} 100644\n--- a/{path}\n\+\+\+ b/{path}\n'
            rf'|new file mode 100644\nindex 0{{7,40}}\.\.[0-9a-f]{{7,40}}\n--- /dev/null\n\+\+\+ b/{path}\n)',
            section[match.end():])
        if header is None:
            raise RuntimeError(f'Unsupported patch metadata or target: {name}')
        body = section[match.end() + header.end():]
        if not body.startswith('@@ '):
            raise RuntimeError('Patch target has no hunks')
        validate_hunks(body)
        new_file_content(section)
        if source_checkout or name.startswith('megatron/core/'):
            entries[name] = section
    if not entries:
        raise RuntimeError('Empty runtime patch')
    return entries


def run_patch(root: pathlib.Path,
              text: str,
              reverse: bool = False,
              dry_run: bool = True) -> subprocess.CompletedProcess:
    command = ['patch', '--batch', '--force', '--fuzz=0', '--no-backup-if-mismatch', '-p1']
    command += ['--reverse'] if reverse else ['--forward']
    if dry_run:
        command.append('--dry-run')
    return subprocess.run(command, cwd=root, input=text, capture_output=True, text=True)


def patch_hunks(entries: Dict[str, str]) -> Iterator[str]:
    """Check each hunk so a partially applied single file cannot pass as a fresh base."""
    for section in entries.values():
        parts = re.split(r'(?=^@@ )', section, flags=re.MULTILINE)
        if len(parts) < 2:
            raise RuntimeError('Patch target has no hunks')
        for hunk in parts[1:]:
            yield parts[0] + hunk


def prepare_patch(root: pathlib.Path, snapshot: pathlib.Path, entries: Dict[str, str], source_checkout: bool) -> None:
    """Apply off-target, allowing a clean Git three-way merge when context has drifted."""
    pending = {name: section for name, section in entries.items()
               if new_file_content(section) is None or not (snapshot / name).exists()}
    text = ''.join(pending.values())
    merged = False
    result = run_patch(snapshot, text)
    if result.returncode == 0:
        result = run_patch(snapshot, text, dry_run=False)
    elif source_checkout:
        merged = True
        # Use a private index and worktree; borrow only objects needed for the merge base.
        common = subprocess.run(['git', 'rev-parse', '--git-common-dir'],
                                cwd=root,
                                capture_output=True,
                                text=True,
                                check=True)
        objects = (root / common.stdout.strip() / 'objects').resolve()
        subprocess.run(['git', 'init', '-q', str(snapshot)], check=True)
        existing = [name for name in pending if (snapshot / name).is_file()]
        if existing:
            subprocess.run(['git', 'add', '--', *existing], cwd=snapshot, check=True)
        env = dict(os.environ, GIT_ALTERNATE_OBJECT_DIRECTORIES=str(objects))
        result = subprocess.run(['git', 'apply', '--3way', '--whitespace=nowarn', '-'],
                                cwd=snapshot,
                                input=text,
                                capture_output=True,
                                text=True,
                                env=env)
    if result.returncode:
        raise RuntimeError(f'Patch conflicts; no target files changed.\n{result.stdout}{result.stderr}')
    # Git verifies a three-way result itself; its merged context may differ from the diff.
    if not merged and run_patch(snapshot, text, reverse=True).returncode:
        raise RuntimeError('Incomplete patch result; no target files changed.')


def target_content(target: pathlib.Path, allow_missing: bool = False) -> Optional[bytes]:
    """Require an existing real parent and a regular, non-symlinked target."""
    if not target.parent.is_dir() or target.parent.resolve() != target.parent:
        raise RuntimeError(f'Missing or symlinked patch target parent: {target.parent}')
    try:
        mode = target.lstat().st_mode
    except FileNotFoundError:
        if allow_missing:
            return None
        raise RuntimeError(f'Missing or symlinked patch target: {target}') from None
    if not stat.S_ISREG(mode) or target.resolve() != target:
        raise RuntimeError(f'Missing or symlinked patch target: {target}')
    return target.read_bytes()


def parent_fd(root: pathlib.Path, name: str) -> int:
    """Pin each ancestor without following symlinks, including during publication."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(root, flags)
    try:
        for part in pathlib.PurePosixPath(name).parts[:-1]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def write_bytes(descriptor: int, content: bytes) -> None:
    """Write completely through a pinned regular-file descriptor."""
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    remaining = memoryview(content)
    while remaining:
        count = os.write(descriptor, remaining)
        if count <= 0:
            raise OSError('Incomplete patch target write')
        remaining = remaining[count:]


def write_files(root: pathlib.Path, snapshot: pathlib.Path, originals: Dict[str, Optional[bytes]]) -> None:
    """Publish in an unused environment with a single installer process.

    Rollback and external-replacement detection are best effort, not atomic guarantees
    against arbitrary concurrent writers (including the stat-to-unlink window).
    """
    for name, content in originals.items():
        if target_content(root / name, allow_missing=content is None) != content:
            raise RuntimeError(f'Target changed during patch preparation: {name}')
    written = []
    try:
        for name, content in originals.items():
            prepared = (snapshot / name).read_bytes()
            parent = parent_fd(root, name)
            leaf = pathlib.PurePosixPath(name).name
            descriptor = None
            try:
                flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
                if content is None:
                    flags |= os.O_CREAT | os.O_EXCL
                descriptor = os.open(leaf, flags, 0o644, dir_fd=parent)
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode):
                    raise RuntimeError(f'Target changed during patch preparation: {name}')
                if content is not None:
                    with os.fdopen(os.dup(descriptor), 'rb') as current:
                        if current.read() != content:
                            raise RuntimeError(f'Target changed during patch preparation: {name}')
                written.append((name, content, parent, descriptor, info.st_dev, info.st_ino))
                parent, descriptor = None, None
                if content is None:
                    os.fchmod(written[-1][3], 0o644)
                write_bytes(written[-1][3], prepared)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                if parent is not None:
                    os.close(parent)
        for name, _, _, _, device, inode in written:
            target = root / name
            current = target.lstat()
            if ((current.st_dev, current.st_ino) != (device, inode)
                    or target_content(target) != (snapshot / name).read_bytes()):
                raise RuntimeError(f'Target changed during patch publication: {name}')
    except BaseException as error:
        failures = []
        for name, content, parent, descriptor, device, inode in reversed(written):
            leaf = pathlib.PurePosixPath(name).name
            try:
                try:
                    current = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if (current.st_dev, current.st_ino) != (device, inode):
                    continue
                if content is None:
                    os.unlink(leaf, dir_fd=parent)
                else:
                    write_bytes(descriptor, content)
            except BaseException as rollback_error:
                failures.append(f'{root / name}: {type(rollback_error).__name__}: {rollback_error}')
        if failures:
            details = '\n'.join(failures)
            raise RuntimeError(
                f'Patch publication failed: {type(error).__name__}: {error}\n'
                f'Rollback incomplete; manual recovery required for:\n{details}') from error
        raise
    finally:
        for _, _, parent, descriptor, _, _ in written:
            os.close(descriptor)
            os.close(parent)


def apply_patch(root: pathlib.Path, check_only: bool = False) -> None:
    root = root.resolve()
    source_checkout = git_root(root)
    entries = patch_files(source_checkout)
    originals = {}
    additions = {}
    for name, section in entries.items():
        expected = new_file_content(section)
        content = target_content(root / name, allow_missing=expected is not None)
        if expected is not None:
            additions[name] = expected
            if content is not None and content != expected:
                raise RuntimeError(f'New patch target collision: {name}')
        originals[name] = content
    with tempfile.TemporaryDirectory(prefix='megatron-patch-') as directory:
        snapshot = pathlib.Path(directory)
        for name, content in originals.items():
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if content is not None:
                target.write_bytes(content)
        if run_patch(snapshot, ''.join(entries.values()), reverse=True).returncode == 0:
            print(f'already applied (all {len(entries)} files verified): {root}')
            return
        partial = any(run_patch(snapshot, hunk, reverse=True).returncode == 0 for hunk in patch_hunks(entries))
        if not source_checkout:
            if check_only:
                raise RuntimeError(f'Patch not fully applied: {root}')
            if partial:
                raise RuntimeError(f'Partially applied patch; no target files changed: {root}')
        prepare_patch(root, snapshot, entries, source_checkout)
        if any((snapshot / name).read_bytes() != expected for name, expected in additions.items()):
            raise RuntimeError('Incomplete new-file patch result; no target files changed.')
        # A three-way application to an already merged source tree is a no-op, even when
        # unrelated edits changed the patch context and reverse dry-run could not match it.
        if all(content is not None and (snapshot / name).read_bytes() == content
               for name, content in originals.items()):
            print(f'already applied (all {len(entries)} files verified): {root}')
            return
        if check_only:
            raise RuntimeError(f'Patch not fully applied: {root}')
        if partial:
            raise RuntimeError(f'Partially applied patch; no target files changed: {root}')
        write_files(root, snapshot, originals)
    print(f'applied and verified all {len(entries)} files: {root}')


def is_applied(root: pathlib.Path) -> bool:
    try:
        apply_patch(pathlib.Path(root), check_only=True)
    except (RuntimeError, OSError, subprocess.CalledProcessError):
        return False
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=pathlib.Path, help='Megatron source root; defaults to installed package')
    parser.add_argument('--check', action='store_true', help='Verify all selected patch hunks without writing')
    args = parser.parse_args(argv)
    try:
        apply_patch(args.root if args.root is not None else megatron_root(), args.check)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
