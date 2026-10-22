# Copyright (c) ModelScope Contributors. All rights reserved.
"""Install the opt-in GLM-5.3 CP route prebuild at Swift's packed-batch entry.

Run once in an unused environment, after installing the supported Swift/Megatron stack.
This tool uses only the standard library and the existing patch installer helpers;
locating Swift does not import its training modules or initialize CUDA.
"""
import argparse
import ast
import importlib.util
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import List, Optional

PATCH = pathlib.Path(__file__).resolve().parent.parent / 'patches' / 'swift_glm53_route_prebuild.patch'
TARGET = 'swift/megatron/utils/utils.py'

_spec = importlib.util.spec_from_file_location('_glm53_patch_installer',
                                               pathlib.Path(__file__).with_name('apply_megatron_patch.py'))
_installer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_installer)


def swift_root() -> pathlib.Path:
    """Locate installed Swift without importing its CUDA-dependent training modules."""
    spec = importlib.util.find_spec('swift')
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            if (pathlib.Path(location) / 'megatron/utils/utils.py').is_file():
                return pathlib.Path(location).resolve().parent
    raise RuntimeError('Swift Megatron is not installed; install the supported Swift version first.')


def apply_patch(root: pathlib.Path, check_only: bool = False) -> None:
    """Prepare and verify the single-file patch before publishing through the existing installer."""
    root = root.resolve()
    text = PATCH.read_text(encoding='utf-8')
    name = re.escape(TARGET)
    header = re.match(
        rf'diff --git a/{name} b/{name}\nindex [0-9a-f]{{40}}\.\.[0-9a-f]{{40}} 100644\n'
        rf'--- a/{name}\n\+\+\+ b/{name}\n', text)
    if header is None:
        raise RuntimeError('Unsupported Swift patch target or metadata')
    _installer.validate_hunks(text[header.end():])
    original = _installer.target_content(root / TARGET)
    with tempfile.TemporaryDirectory(prefix='swift-route-patch-') as directory:
        snapshot = pathlib.Path(directory)
        target = snapshot / TARGET
        target.parent.mkdir(parents=True)
        target.write_bytes(original)
        if _installer.run_patch(snapshot, text, reverse=True).returncode == 0:
            print(f'already applied (Swift route prebuild verified): {root}')
            return
        if check_only:
            raise RuntimeError(f'Swift route patch not fully applied: {root}')
        if any(
                _installer.run_patch(snapshot, hunk, reverse=True).returncode == 0
                for hunk in _installer.patch_hunks({TARGET: text})):
            raise RuntimeError('Partially applied Swift route patch; no target files changed.')
        result = _installer.run_patch(snapshot, text)
        if result.returncode:
            raise RuntimeError(f'Swift route patch conflicts; no target files changed.\n'
                               f'{result.stdout}{result.stderr}')
        result = _installer.run_patch(snapshot, text, dry_run=False)
        if result.returncode or _installer.run_patch(snapshot, text, reverse=True).returncode:
            raise RuntimeError('Incomplete Swift route patch; no target files changed.')
        ast.parse(target.read_text(encoding='utf-8'))
        _installer.write_files(root, snapshot, {TARGET: original})
    print(f'applied and verified Swift route prebuild: {root}')


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=pathlib.Path, help='Swift source/package parent; default: installed Swift')
    parser.add_argument('--check', action='store_true', help='Verify the complete patch without writing')
    args = parser.parse_args(argv)
    try:
        apply_patch(args.root if args.root is not None else swift_root(), args.check)
    except (RuntimeError, OSError, SyntaxError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
