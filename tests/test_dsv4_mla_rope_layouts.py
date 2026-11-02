# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real packed Q/K projection layouts preserve per-token RoPE values and gradients."""
import ast
import pytest
import torch
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from pathlib import Path
from types import SimpleNamespace


def _load_callsite():
    path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/model/gpts/deepseek_v4.py'
    tree = ast.parse(path.read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_apply_mla_rope')
    attention = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DSv4HybridSelfAttention')
    qkv = next(n for n in attention.body if isinstance(n, ast.FunctionDef) and n.name == 'get_query_key_value_tensors')
    namespace = {'torch': torch, 'apply_rotary_pos_emb': apply_rotary_pos_emb, 'apply_module': lambda module: module}
    exec(compile(ast.Module(body=[helper, qkv], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_apply_mla_rope'], namespace['get_query_key_value_tensors']


def _config():
    return SimpleNamespace(
        apply_rope_fusion=False,
        rotary_interleaved=False,
        mrope_section=None,
        q_lora_rank=3,
        qk_pos_emb_head_dim=4,
        dsv4_version='v41',
        layernorm_epsilon=1e-6)


def _frequencies():
    positions = torch.tensor([7., 8., 0., 1., 2.])
    angles = positions[:, None] * torch.tensor([.13, .07])
    return torch.cat([angles, angles], dim=-1).view(5, 1, 1, 4)


def _reference(tensor, freqs, inverse=False):
    angles = freqs[..., :freqs.shape[-1] // 2].reshape(freqs.shape[0], *([1] * (tensor.ndim - 2)), freqs.shape[-1] // 2)
    cosine, sine = angles.cos(), angles.sin()
    if inverse:
        sine = -sine
    rotary, tail = tensor[..., :freqs.shape[-1]], tensor[..., freqs.shape[-1]:]
    even, odd = rotary[..., ::2], rotary[..., 1::2]
    rotated = torch.stack([even * cosine - odd * sine, odd * cosine + even * sine], dim=-1).flatten(-2)
    return torch.cat([rotated, tail], dim=-1)


@pytest.mark.parametrize('shape', [(5, 6), (5, 3, 6), (5, 1, 6), (5, 1, 3, 6), (5, 2, 3, 6)])
@pytest.mark.parametrize('inverse', [False, True])
def test_rope_layout_values_and_backward(shape, inverse):
    apply, _ = _load_callsite()
    tokens = torch.linspace(-1, 1, steps=torch.tensor(shape).prod().item()).reshape(shape).requires_grad_()
    reference_tokens = tokens.detach().clone().requires_grad_()
    frequencies = _frequencies()
    actual = apply(
        tokens,
        frequencies,
        config=_config(),
        cu_seqlens=torch.tensor([0, 2, 5]),
        cp_group=SimpleNamespace(size=lambda: 1, rank=lambda: 0),
        inverse=inverse)
    expected = _reference(reference_tokens, frequencies, inverse)
    assert actual.shape == tokens.shape
    torch.testing.assert_close(actual, expected)
    probe = torch.linspace(.2, 1, steps=tokens.numel()).reshape(shape)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(tokens.grad, reference_tokens.grad)


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('boundary', [False, True])
def test_real_qkv_projection_callsite(packed, boundary):
    _, qkv = _load_callsite()
    torch.manual_seed(19)
    weights = [
        torch.randn(3, 8, requires_grad=True),
        torch.randn(12, 3, requires_grad=True),
        torch.randn(6, 8, requires_grad=True)
    ]
    attention = SimpleNamespace(
        config=_config(),
        num_attention_heads_per_partition=2,
        q_head_dim=6,
        q_layernorm=torch.nn.Identity(),
        kv_layernorm=torch.nn.Identity(),
        linear_q_down_proj=lambda x: (torch.nn.functional.linear(x, weights[0]), None),
        linear_q_up_proj=lambda x: (torch.nn.functional.linear(x, weights[1]), None),
        linear_kv_proj=lambda x: (torch.nn.functional.linear(x, weights[2]), None),
        recompute_up_proj=False,
        pg_collection=SimpleNamespace(cp=SimpleNamespace(size=lambda: 1, rank=lambda: 0)))
    hidden = torch.randn(5, 1, 8, requires_grad=True)
    frequencies = _frequencies()
    packed_params = SimpleNamespace(
        qkv_format='thd', cu_seqlens_q=torch.tensor([0, 2, 5]), cu_seqlens_kv=torch.tensor([0, 2, 5
                                                                                            ])) if packed else None
    boundary_hidden = torch.randn(2, 1, 8, requires_grad=True) if boundary else None
    result = qkv(
        attention,
        hidden,
        packed_seq_params=packed_params,
        rotary_pos_emb=frequencies,
        boundary_hidden=boundary_hidden,
        boundary_rotary_pos_emb=frequencies[:2] if boundary else None)
    query, key, value = result[:3]
    assert query.shape == ((5, 2, 6) if packed else (5, 1, 2, 6))
    assert key.shape == ((5, 1, 6) if packed else (5, 1, 1, 6))
    assert value is key
    latent = hidden.squeeze(1) if packed else hidden
    q_reference = torch.nn.functional.linear(torch.nn.functional.linear(latent, weights[0]), weights[1])
    q_reference = q_reference.reshape(*latent.shape[:-1], 2, 6)
    q_reference = torch.cat([q_reference[..., :2], _reference(q_reference[..., 2:], frequencies)], dim=-1)
    kv_reference = torch.nn.functional.linear(latent, weights[2])
    kv_reference = torch.cat([kv_reference[..., :2], _reference(kv_reference[..., 2:], frequencies)], dim=-1)
    torch.testing.assert_close(query, q_reference)
    torch.testing.assert_close(key, kv_reference.unsqueeze(-2))
    terms = [query.float().square().sum(), key.float().square().sum()]
    if boundary:
        assert result[5].shape == ((2, 1, 6) if packed else (2, 1, 1, 6))
        terms.append(result[5].float().square().sum())
    sum(terms).backward()
    for tensor in [hidden, *weights, *([boundary_hidden] if boundary else [])]:
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
