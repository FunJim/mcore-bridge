# Copyright (c) ModelScope Contributors. All rights reserved.
"""Apply the GLM-5.3 runtime patch once while preparing an unused environment.

The full-index patch targets NVIDIA/Megatron-LM dev ee743d3ef228. It includes
upstream GLM numerics (#7054), expert gradient norm / FP32 attribute fixes,
and the FP32 shard, TileLang shape and KPool context-parallel fixes. Tests and
provenance live in tests/glm53_patch_regression and patches/README.md.

Every affected file must match the recorded base or the complete patched state.
Partial/unknown installations are rejected before writing; prepare a fresh
baseline instead. Do not invoke this tool from distributed training ranks.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import pathlib
import re
import subprocess
import sys

PATCH = pathlib.Path(__file__).resolve().parent.parent / 'patches' / 'megatron_glm53_dev.patch'
BASE_COMMIT = 'ee743d3ef228f546287dc3835c2cf56011d5136b'


def megatron_root() -> pathlib.Path:
    """Locate installed runtime without importing megatron.core or CUDA libraries."""
    spec = importlib.util.find_spec('megatron')
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            if (pathlib.Path(location) / 'core' / '__init__.py').is_file():
                return pathlib.Path(location).resolve().parent
    raise RuntimeError(f'Install NVIDIA/Megatron-LM at {BASE_COMMIT} first.')


def patch_files() -> dict:
    """Read exact before/after Git blob hashes from the bundled full-index diff."""
    entries = re.findall(
        r'^diff --git a/(\S+) b/\1\nindex ([0-9a-f]{40})\.\.([0-9a-f]{40}) 100644$',
        PATCH.read_text(), re.MULTILINE)
    if not entries or len(entries) != PATCH.read_text().count('diff --git '):
        raise RuntimeError('Invalid full-index runtime patch')
    if any(not name.startswith('megatron/core/') or '..' in pathlib.PurePosixPath(name).parts
           for name, _, _ in entries):
        raise RuntimeError('Patch contains paths outside megatron/core')
    return {name: (before, after) for name, before, after in entries}


def file_hash(path: pathlib.Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def patch_state(root: pathlib.Path, entries: dict) -> str:
    """Reject mixed patches, missing files, source drift and symlinked targets."""
    states = []
    for name, (before, after) in entries.items():
        target = root / name
        if not target.is_file() or target.resolve() != target:
            raise RuntimeError(f'Missing or symlinked patch target: {target}')
        digest = file_hash(target)
        states.append('base' if digest == before else 'applied' if digest == after else 'unknown')
    if all(state == 'applied' for state in states):
        return 'applied'
    if all(state == 'base' for state in states):
        return 'base'
    raise RuntimeError(f'Partial or incompatible Megatron patch at {root}; no files changed. '
                       f'Prepare a fresh unused environment from {BASE_COMMIT}.')


def apply_patch(root: pathlib.Path, check_only: bool = False) -> None:
    """Preflight all files, apply without fuzz, and restore originals on failure."""
    root = root.resolve()
    entries = patch_files()
    if check_only:
        if patch_state(root, entries) != 'applied':
            raise RuntimeError(f'Patch not applied: {root}')
        print(f'verified all {len(entries)} patched files: {root}')
        return
    with (root / '.glm53-patch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if patch_state(root, entries) == 'applied':
            print(f'already applied (all {len(entries)} files verified): {root}')
            return
        command = ['patch', '--batch', '--forward', '--fuzz=0', '--no-backup-if-mismatch',
                   '-p1', '-i', str(PATCH)]
        subprocess.run([*command, '--dry-run'], cwd=root, check=True)
        originals = {name: (root / name).read_bytes() for name in entries}
        try:
            subprocess.run(command, cwd=root, check=True)
            if patch_state(root, entries) != 'applied':
                raise RuntimeError('Post-apply verification failed')
        except BaseException:
            for name, data in originals.items():
                (root / name).write_bytes(data)
            raise
        print(f'applied and verified all {len(entries)} files: {root}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=pathlib.Path, help='Megatron source root; defaults to installed package')
    parser.add_argument('--check', action='store_true', help='Verify the complete patch without writing')
    args = parser.parse_args()
    try:
        apply_patch(args.root if args.root is not None else megatron_root(), args.check)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
