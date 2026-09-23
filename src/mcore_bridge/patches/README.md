# GLM-5.3 runtime patch

Maintained in `FunJim/mcore-bridge`, branch `feat/glm53-flash-support`, starting
from ModelScope bridge `9d610ffb9c75220cadc3f31922346fa81ca8b456`, with upstream
`main` integrated through `cef925c3dc073fc1627b9afa2af936d1d0779574`.
The integration retains this fork's complete runtime patch and strict installer;
its KPool CP implementation already covers upstream PR #200. Upstream PR #208
contains the contributed installer reliability improvements with source-drift /
three-way-merge support. This pinned-baseline fork retains its strict full-file
hash installer and matching tests instead of adopting that broader policy. The supported
training configuration keeps MTP disabled (`mtp_num_layers=0`); the newly imported
MTP path requires separate validation before use. Swift remains upstream
`ac6651a34bedc0d8786291c558314a49371d811f`.

`megatron_glm53_dev.patch` targets NVIDIA/Megatron-LM **dev** commit
`ee743d3ef228f546287dc3835c2cf56011d5136b`. It contains only runtime files,
so it also applies to the installed `megatron-core` wheel. Full Git blob hashes
record the supported before/after content of all 14 affected files.

Included changes and provenance:

| Change | Reason and source | Regression |
|---|---|---|
| Official bridge GLM patch | Original patch from bridge `9d610ff`; GLM numerics from NVIDIA/Megatron-LM PR #7054 `be805e55`, chunked index scoring from `3fceb0715`, expert norm / FP32 attributes from the original patch. Original NVIDIA source licenses and attribution retained. | Existing `tests/test_glm5_hybrid.py`, `tests/test_hybrid_compat.py`; distributed GLM block preflight |
| FP32 optimizer shards | Detach native FP32 views so CPU offload receives leaf tensors; storage remains shared. Maintained in this repository by commit `eb26203`. | `verify_glm53_fp32_shards.py`: offload 0/0.5/1, AdamW update, optimizer state restore |
| Weights-only optimizer reload | Refresh distributed FP32 masters from loaded model shards before refreshing HybridDeviceOptimizer CPU copies. CPU copies of native FP32 parameters must also be refreshed. Otherwise the first update can overwrite loaded weights with stale or nonfinite values. Reproduced during GLM weights-only continuation with CPU offload. | `verify_glm53_optimizer_reload.py`: real shard builder, partial ranges, FP32/BF16, offload 0/0.5/1, overlap on/off, checkpoint-state input, stale/NaN controls, zero-gradient preservation and AdamW update parity |
| TileLang shapes | Support GLM's zero RoPE channels and pad KPool top-k slots without changing selected tokens. Maintained in this repository by commit `ed3f9c6`. | `verify_glm53_dsa.py`: output/gradient comparison, 73728-token fused forward/backward |
| KPool CP | Gather gate scores and reorder them with indexer keys across context-parallel ranks. Maintained in this repository by commit `f5cca49`. | `verify_glm53_dsa_cp.py`: CPU CP1/CP8 output/gradient parity, packed sequences, negative controls; 4-node CP8 preflight |

The complete runtime patch is maintained on `feat/glm53-flash-support`;
its official GLM portion was introduced at `85f6680`. Installation and
regressions depend only on this repository and the upstream baselines above.
Historical source commits cited in published commit messages are preserved in
the maintainer's `FunJim-Megatron-LM.bundle`; those messages are not rewritten.
Regressions are in `tests/glm53_patch_regression/`. The original patch's
Megatron unit-test diff was excluded from this runtime-only bundle; it remains
available in bridge baseline `9d610ff`.

Prepare a **new, unused** environment; install the three selected wheels with
`pip install --no-deps --no-index` (non-editable). Then run once:

```bash
python -m mcore_bridge.tools.apply_megatron_patch
python -m mcore_bridge.tools.apply_megatron_patch --check
```

Alternatively, before building the Megatron wheel, apply to a source checkout:

```bash
python src/mcore_bridge/tools/apply_megatron_patch.py --root /path/to/Megatron-LM
```

All affected files must match the base, or all must match the patched result.
Mixed/unknown content fails before mutation. A dry run precedes application;
failed application restores original files. An advisory lock serializes tool
invocations, but does **not** make modifying an active training environment safe.
Do not run this from each training rank, combine with older patch scripts, or
upgrade an environment in use. `--check` is read-only and suitable for preflight.

Installer tests (standard library only):

```bash
python -m unittest discover -s tests -p test_apply_megatron_patch.py -v
```

On an allocated GPU, test the installed patch without loading any external plugin:

```bash
python tests/glm53_patch_regression/verify_glm53_optimizer_reload.py --report reload.json
python tests/glm53_patch_regression/verify_glm53_fp32_shards.py
```

For the reload negative control, run the same script with `--expect-stale` in an
isolated environment containing the previous patch from `314c649`. Once a newly
prepared environment passes these checks, omit the experiment-specific
`hybrid_reload_fix.py` plugin. Dataset window plugins remain independent.

For the CPU CP regression's negative control, extract
`src/mcore_bridge/patches/megatron_glm53_dev.patch` from this repository's
`ed3f9c6` commit and apply it to a temporary upstream Megatron `ee743d3ef228`
checkout. Use the resulting `megatron/core/transformer/experimental_attention_variant/dsa.py`
as `--before`; it includes FP32/TileLang fixes but not KPool CP. GPU tests require
an allocated device and the same CUDA libraries as training. No full model is loaded.

Updates: bridge tracks `modelscope/mcore-bridge:main`; Swift tracks
`modelscope/ms-swift:main`; Megatron tracks `NVIDIA/Megatron-LM:dev`. Integrate
selected updates on a temporary branch, regenerate the full-index patch against
the new baseline, remove fixes already absorbed upstream, and rerun regressions
before merging to `feat/glm53-flash-support`. Record the three actual commits.
Do not rebase published history or force-push.

The 2026-09-23 integration passed 50 GLM single-GPU tests (no skips), 36 optimizer
reload cases, FP32 shard regressions, a 73728-token TileLang forward/backward,
and CPU CP1/CP8 output/gradient checks with negative controls. A randomly
initialized small GLM model also passed three packed length cases on all eight
H200 ranks (TP1/PP1/CP8/EP1, MTP=0), with finite nonzero DSA/KDA projection
gradients. This is single-node validation, not full-model or cross-node training.
The final #208 merge changes this documentation only relative to the tested
runtime at `b5679967d7c4ff577c453c4c37685d0af48a32f2`.
