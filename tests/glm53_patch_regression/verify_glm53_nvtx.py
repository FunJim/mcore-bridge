"""Verify instrumentation-only AST and CPU output/gradient parity against the prior runtime."""

import argparse
import ast
import copy
import functools
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional, TypeVar
from unittest.mock import patch

DSA = 'megatron/core/transformer/experimental_attention_variant/dsa.py'
CP = 'megatron/core/context_parallel_layout/conversion.py'


class StripNvtx(ast.NodeTransformer):
    def visit_With(self, node):
        self.generic_visit(node)
        if len(node.items) == 1:
            context = node.items[0].context_expr
            if isinstance(context, ast.Call) and isinstance(context.func, ast.Name) and context.func.id == 'nvtx_range':
                return node.body
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.decorator_list = [d for d in node.decorator_list if not (
            isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'nvtx_decorator')]
        return node

    def visit_ImportFrom(self, node):
        if node.module == 'megatron.core.utils':
            node.names = [name for name in node.names if name.name not in ('nvtx_range', 'nvtx_decorator')]
            if not node.names:
                return None
        return node


def load_functions(path: Path, names: set[str], namespace: dict) -> dict:
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--ast-only', action='store_true')
    args = parser.parse_args()
    for relative in (DSA, CP):
        before = StripNvtx().visit(ast.parse((args.before / relative).read_text()))
        after = StripNvtx().visit(ast.parse((args.after / relative).read_text()))
        assert ast.dump(before) == ast.dump(after), f'Non-instrumentation change: {relative}'
    report = {'ast_computation_and_collectives_unchanged': True, 'cases': [], 'gpu_capture': 'not_run'}
    if args.ast_only:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
        return

    import torch
    from megatron.core.transformer.experimental_attention_variant import dsa

    torch.set_num_threads(1)
    # Execute the real baseline NVTX helpers with only their CUDA sink replaced.
    utils_path = args.after / 'megatron/core/utils.py'
    nvtx_names = {'configure_nvtx_profiling', 'nvtx_range_push', 'nvtx_range_pop', 'nvtx_range',
                  'nvtx_decorator', '_nvtx_decorator_get_func_path', '_nvtx_range_get_func_path'}
    namespace = dict(torch=torch, functools=functools, contextmanager=contextmanager,
                     lru_cache=functools.lru_cache, Optional=Optional, Callable=Callable,
                     _Wrapped=TypeVar('_Wrapped'), _nvtx_enabled=False,
                     _nvtx_range_messages=[], _nvtx_range_msg_pool={})
    load_functions(utils_path, nvtx_names, namespace)
    functions = {'_compute_index_scores', '_kpool_compress_keys', '_expand_pools_to_tokens',
                 '_append_tail_to_topk', 'fused_qk_topk_kpool'}
    baseline = load_functions(args.before / DSA, functions, dict(dsa.__dict__))
    candidate = load_functions(args.after / DSA, functions, dict(dsa.__dict__,
                              nvtx_range=namespace['nvtx_range'], nvtx_decorator=namespace['nvtx_decorator']))
    observed, stack = [], []

    def push(message):
        observed.append(message)
        stack.append(message)

    def pop():
        assert stack, 'unmatched pop'
        stack.pop()

    def run(function, relu):
        torch.manual_seed(1729)
        tensors = [torch.randn(16, 1, 2, 4, requires_grad=True),
                   torch.randn(16, 1, 2, requires_grad=True),
                   torch.randn(16, 1, 4, requires_grad=True)]
        out = function(*tensors, use_relu=relu)
        out.square().sum().backward()
        return out.detach(), [tensor.grad.detach().clone() for tensor in tensors]

    with patch.object(torch.cuda.nvtx, 'range_push', push), patch.object(torch.cuda.nvtx, 'range_pop', pop):
        for enabled in (False, True):
            namespace['configure_nvtx_profiling'](enabled)
            for relu in (False, True):
                before, grads_before = run(baseline['_compute_index_scores'], relu)
                after, grads_after = run(candidate['_compute_index_scores'], relu)
                torch.testing.assert_close(after, before, rtol=0, atol=0)
                for actual, expected in zip(grads_after, grads_before):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                report['cases'].append({'scoring': True, 'nvtx_enabled': enabled, 'relu': relu,
                                        'output_and_gradients_bit_equal': True})
            torch.manual_seed(71)
            q, k, weights = torch.randn(16, 1, 2, 4), torch.randn(16, 1, 4), torch.randn(16, 1, 2)
            gate, ape = torch.randn(16, 1, 4), torch.randn(4, 4)
            for packed in (False, True):
                kwargs = {'cu_seqlens_kv': torch.tensor([0, 8, 16], dtype=torch.int32)} if packed else {}
                inputs = (q, k, weights, 8, 4, gate, ape)
                before = baseline['fused_qk_topk_kpool'](*inputs, **kwargs)
                after = candidate['fused_qk_topk_kpool'](*inputs, **kwargs)
                for actual, expected in zip(after, before):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                report['cases'].append({'kpool': True, 'packed': packed, 'nvtx_enabled': enabled,
                                        'scores_and_topk_bit_equal': True})
            assert not stack and not namespace['_nvtx_range_messages']
            if not enabled:
                assert not observed, 'disabled instrumentation reached CUDA NVTX'
        required = {'dsa/index_scores/qk', 'dsa/index_scores/relu', 'dsa/index_scores/weight_reduce',
                    'dsa/kpool/scoring', 'dsa/kpool/mask', 'dsa/kpool/topk', 'dsa/kpool/expand_tokens',
                    'dsa/kpool/append_tail', 'dsa/kpool/compress_keys'}
        assert required.issubset(observed), required - set(observed)
        try:
            with namespace['nvtx_range']('exception_cleanup'):
                raise ValueError('expected')
        except ValueError:
            pass
        assert not stack and not namespace['_nvtx_range_messages']
    report.update(observed_markers=sorted(set(observed)), balanced_ranges=True,
                  validation_scope='Actual scoring/kpool on CPU; actual NVTX helper with mocked CUDA sink. '
                                   'CP collectives checked structurally only; real GPU/NCCL remains unverified.')
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
