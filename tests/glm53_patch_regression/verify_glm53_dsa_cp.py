"""Compare GLM DSA KPool CP8 with CP1 using simulated gathers on CPU."""
import argparse
import ast
import copy
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--before', type=Path, required=True, help='Unfixed bridge-patched dsa.py')
parser.add_argument('--after', type=Path, default=Path(dsa.__file__), help='Fixed dsa.py; defaults to installed version')
parser.add_argument('--report', type=Path, help='Optional new JSON report path')
args = parser.parse_args()
torch.set_num_threads(2)


def load_forward(text: str) -> Callable[..., torch.Tensor]:
    """Extract the real method while replacing process-group sizing for CPU simulation."""
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DSAttention')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    namespace = dict(dsa.__dict__)
    namespace['get_pg_size'] = lambda group: group.size() if group is not None else 1
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(method)], type_ignores=[])),
                 '<DSAttention.forward>', 'exec'), namespace)
    return namespace['forward']


def run_case(forward: Callable[..., torch.Tensor], cp_size: int, documents: tuple[int, ...], packed: bool,
             pool_size: int = 4) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Reassemble CP rank outputs and compare gradients in original token order."""
    total = sum(documents)
    generator = torch.Generator().manual_seed(123)
    x_global = torch.randn(total, 1, 4, generator=generator)
    q_global = torch.randn(total, 1, 2, 4, generator=generator, requires_grad=True)
    k_global = torch.randn(total, 1, 2, 4, generator=generator, requires_grad=True)
    v_global = torch.randn(total, 1, 2, 4, generator=generator, requires_grad=True)
    index_k_global = x_global * 0.7 + 0.3
    gate_global = torch.sin(x_global * 1.7)
    ape = torch.randn(4, 4, generator=generator)
    cu = torch.tensor([0] + list(torch.tensor(documents).cumsum(0).tolist()), dtype=torch.int32)
    local_rows = total // cp_size
    positions = []
    for rank in range(cp_size):
        if cp_size == 1:
            pos = torch.arange(total)
        elif packed:
            pos = dsa.dsa_layout.build_packed_allgather_cp_local_positions(
                cu, cp_size, rank, torch.device('cpu'), output_size=local_rows)
        else:
            chunks = torch.arange(total).reshape(2 * cp_size, -1)
            pos = torch.cat((chunks[rank], chunks[2 * cp_size - rank - 1]))
        positions.append(pos.long())
    assert sorted(torch.cat(positions).tolist()) == list(range(total))
    outputs = []
    gather_calls = []
    for rank, pos in enumerate(positions):
        tp = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
        cp = SimpleNamespace(size=lambda: cp_size, rank=lambda: rank)
        config = SimpleNamespace(kv_lora_rank=0, qk_pos_emb_head_dim=0, sequence_parallel=False,
                                 dsa_indexer_loss_coeff=0.0, calculate_per_token_loss=False,
                                 dsa_indexer_use_sparse_loss=False, dsa_indexer_scoring_relu=False)
        indexer = SimpleNamespace(index_kpool=pool_size, index_kpool_compress_ape=ape,
                                  index_kpool_always_select_tail=True, _kpool_gate_score=None)

        def before_topk(x: torch.Tensor, qr: torch.Tensor,
                        packed_seq_params: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            indexer._kpool_gate_score = torch.sin(x * 1.7) if pool_size > 1 else None
            return x.unsqueeze(2), x * 0.7 + 0.3, torch.ones(x.size(0), 1, 1)

        indexer.forward_before_topk = before_topk
        owner = SimpleNamespace(config=config, pg_collection=SimpleNamespace(tp=tp, cp=cp),
                                cp_comm_type='allgather', training=False, index_share=False,
                                skip_topk=False, indexer=indexer, index_topk=8, softmax_scale=0.5)
        candidates = (k_global, v_global, index_k_global, gate_global)

        def gather(tensor: torch.Tensor, group: object) -> torch.Tensor:
            gather_calls.append(rank)
            for full in candidates:
                local = full.index_select(0, pos)
                if tensor.shape == local.shape and torch.equal(tensor, local):
                    return torch.cat([full.index_select(0, p) for p in positions])
            raise AssertionError('unexpected tensor sent to simulated CP gather')

        forward.__globals__['gather_from_sequence_parallel_region'] = gather
        packed_params = None
        if packed:
            packed_params = PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu,
                                            cu_seqlens_kv=cu, max_seqlen_q=max(documents),
                                            max_seqlen_kv=max(documents))
        result = forward(owner, q_global.index_select(0, pos), k_global.index_select(0, pos),
                         v_global.index_select(0, pos), None, x_global.index_select(0, pos),
                         x_global.index_select(0, pos), attn_mask_type=AttnMaskType.causal,
                         packed_seq_params=packed_params)
        assert torch.isfinite(result).all()
        outputs.append(result)
    expected_gathers = cp_size * (4 if pool_size > 1 else 3) if cp_size > 1 else 0
    assert len(gather_calls) == expected_gathers, (len(gather_calls), expected_gathers)
    concatenated = torch.cat(outputs)
    global_order = torch.argsort(torch.cat(positions))
    output = concatenated.index_select(0, global_order)
    output.square().sum().backward()
    grads = [t.grad.detach().clone() for t in (q_global, k_global, v_global)]
    assert all(torch.isfinite(g).all() for g in grads)
    return output.detach(), grads


original_text = args.before.read_text()
candidate_text = args.after.read_text()
before = load_forward(original_text)
candidate = load_forward(candidate_text)
without_reorder_text = candidate_text.replace(
    '                        kpool_gate_score = kpool_gate_score.index_select(0, kv_reorder_idx)\n', '')
assert without_reorder_text != candidate_text
without_reorder = load_forward(without_reorder_text)
reports = []
with patch.object(dsa.dsa_kernels, 'use_fused_dsa_kernels', return_value=False):
    for packed, documents in [(False, (64,)), (True, (64,)), (True, (32, 32))]:
        reference, ref_grads = run_case(before, 1, documents, packed)
        cp1, cp1_grads = run_case(candidate, 1, documents, packed)
        torch.testing.assert_close(cp1, reference, atol=0, rtol=0)
        for actual, expected in zip(cp1_grads, ref_grads):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        try:
            run_case(before, 8, documents, packed)
        except RuntimeError as error:
            assert 'shape' in str(error), str(error)
            baseline_error = str(error)
        else:
            raise AssertionError('baseline CP8 unexpectedly passed')
        actual, grads = run_case(candidate, 8, documents, packed)
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
        for actual_grad, expected_grad in zip(grads, ref_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=2e-5)
        wrong, _ = run_case(without_reorder, 8, documents, packed)
        assert not torch.allclose(wrong, reference, atol=2e-5, rtol=2e-5), 'reorder negative control passed'
        reports.append({'packed': packed, 'documents': documents, 'baseline_error': baseline_error,
                        'cp1_unchanged': True, 'cp8_matches_cp1_output_and_gradients': True,
                        'max_output_abs_error': (actual-reference).abs().max().item(),
                        'missing_reorder_rejected': True})
    for cp_size in (1, 8):
        reference, ref_grads = run_case(before, cp_size, (64,), False, pool_size=1)
        actual, grads = run_case(candidate, cp_size, (64,), False, pool_size=1)
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        for actual_grad, expected_grad in zip(grads, ref_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
        reports.append({'cp_size': cp_size, 'non_kpool_unchanged': True})
report = {'cases': reports, 'scope': 'Actual DSAttention.forward and sparse attention on CPU; deterministic fake indexer and CP gather. Real NCCL, TP4 and long-sequence kernels require GPU preflight.'}
if args.report:
    with args.report.open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
