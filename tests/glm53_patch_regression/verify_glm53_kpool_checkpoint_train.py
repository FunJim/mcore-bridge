"""Tiny training fixture through actual Megatron CheckpointFunction and SparseMLA."""
import argparse
import json
import os
import torch
import torch.distributed as dist
import types
from datetime import timedelta
from megatron.core.tensor_parallel import random as rng
from megatron.core.transformer.experimental_attention_variant import dsa, dsa_kernels
from megatron.core.transformer.experimental_attention_variant import dsa_kpool_kernels as adapter

from mcore_bridge import kpool_checkpoint_replay as replay


def trial(selection, mode, checkpointed, inputs, steps):
    os.environ['MCORE_KPOOL_CHECKPOINT_REPLAY'] = mode
    torch.manual_seed(713)
    rng.model_parallel_cuda_manual_seed(713, tp_rank=dist.get_rank(), ep_rank=0, etp_rank=0)
    x = torch.randn(257, 32, device='cuda', dtype=torch.bfloat16).requires_grad_()
    projection = torch.nn.Linear(32, 9 * 512, bias=False, device='cuda', dtype=torch.bfloat16)
    opt = torch.optim.AdamW(projection.parameters(), lr=1e-4)
    outputs = []
    gradients = []
    ids_seen = []
    for step in range(steps):
        opt.zero_grad()
        x.grad = None

        def forward(value):
            with torch.no_grad():
                if selection == 'reference':
                    indices = dsa.fused_qk_topk_kpool(**inputs)[1]
                else:
                    indices = adapter.select_and_validate(**inputs, mode='tp_dedup', tp_group=dist.group.WORLD).indices
                ids_seen.append(indices.clone())
            projected = projection(value)
            q = projected[:, :8 * 512].reshape(257, 1, 8, 512).contiguous()
            k = projected[:, 8 * 512:].reshape(257, 1, 1, 512).contiguous()
            out = dsa_kernels.run_fused_absorbed_sparse_attention(
                types.SimpleNamespace(dsa_kernel_backend='tilelang'), q, k, indices, 512**-0.5, 512)
            assert out is not None
            return torch.nn.functional.dropout(out, p=0.1, training=True)

        y = rng.CheckpointFunction.apply(forward, False, x) if checkpointed else forward(x)
        loss = y.float().square().mean()
        loss.backward()
        outputs.append((y.detach().cpu(), float(loss.detach())))
        gradients.append((projection.weight.grad.detach().float().cpu(), x.grad.detach().float().cpu()))
        opt.step()
        assert replay.replay_stats()['live_bytes'] == replay.replay_stats()['live_entries'] == 0
    return outputs, gradients, projection.weight.detach().float().cpu(), torch.cuda.get_rng_state(), ids_seen


def compare(a, b, label):
    # BF16 dKV atomic accumulation may vary; report relative errors for trainable gradients.
    errors = []
    for step, (ea, eb) in enumerate(zip(a[0], b[0])):
        rel = float((ea[0].float() - eb[0].float()).norm() / ea[0].float().norm().clamp_min(1e-20))
        assert rel < 0.002, (label, 'output', step, rel)
        assert abs(ea[1] - eb[1]) < 2e-5, (label, 'loss', step, ea[1], eb[1])
        for i, (ga, gb) in enumerate(zip(a[1][step], b[1][step])):
            rel = float((ga - gb).norm() / ga.norm().clamp_min(1e-20))
            errors.append(rel)
            assert rel < 0.002, (label, 'gradient', step, i, rel)
    assert torch.equal(a[3], b[3]), (label, 'CUDA RNG state')
    for ids in b[4]:
        assert torch.equal(ids, a[4][0]), (label, 'indices')
    rel = float((a[2] - b[2]).norm() / a[2].norm().clamp_min(1e-20))
    assert rel < 0.002, (label, 'weights', rel)
    return {'gradient_max_relative_l2': max(errors), 'weight_relative_l2': rel}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(911)
        inputs = dict(
            q=torch.zeros(257, 1, 32, 128, device='cuda', dtype=torch.bfloat16),
            k=torch.randn(257, 1, 128, device='cuda', dtype=torch.bfloat16),
            weights=torch.randn(257, 1, 32, device='cuda'),
            gate_score=torch.randn(257, 1, 128, device='cuda', dtype=torch.bfloat16),
            ape=torch.randn(4, 128, device='cuda'),
            pool_size=4,
            index_topk=64,
            use_relu=True,
            always_select_tail=True)
        for fusion in ('0', '1'):
            os.environ['MCORE_KPOOL_HEAD_SUM_FUSION'] = fusion
            reference = trial('reference', 'off', False, inputs, args.steps)
            repeated = trial('reference', 'off', False, inputs, args.steps)
            control = compare(reference, repeated, 'repeat reference')
            for mode in ('off', 'observe', 'replay'):
                actual = trial('tp_dedup', mode, True, inputs, args.steps)
                result = compare(reference, actual, mode)
                print(
                    json.dumps({
                        'CHECKPOINT_TRAIN_PASS': True,
                        'mode': mode,
                        'fusion': fusion,
                        'rank': dist.get_rank(),
                        'reference_repeat': control,
                        'metrics': result,
                        'steps': args.steps,
                        'actual_checkpoint_function': True
                    }),
                    flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
