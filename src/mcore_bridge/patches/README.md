# GLM-5.3 runtime patch

Maintained in `FunJim/mcore-bridge`, branch `feat/glm53-flash-support`, starting
from ModelScope bridge `9d610ffb9c75220cadc3f31922346fa81ca8b456` (without the
later workspace MTP changes). Swift remains upstream `ac6651a34bedc0d8786291c558314a49371d811f`.

`megatron_glm53_dev.patch` targets NVIDIA/Megatron-LM **dev** commit
`ee743d3ef228f546287dc3835c2cf56011d5136b`. It contains only runtime files,
so it also applies to the installed `megatron-core` wheel. Full Git blob hashes
record the supported before/after content of all 13 affected files.

Included changes and provenance:

| Change | Reason and source | Regression |
|---|---|---|
| Official bridge GLM patch | Original patch from bridge `9d610ff`; GLM numerics from NVIDIA/Megatron-LM PR #7054 `be805e55`, chunked index scoring from `3fceb0715`, expert norm / FP32 attributes from the original patch. Original NVIDIA source licenses and attribution retained. | Existing `tests/test_glm5_hybrid.py`, `tests/test_hybrid_compat.py`; distributed GLM block preflight |
| FP32 optimizer shards | Detach native FP32 views so CPU offload receives leaf tensors; storage remains shared. Migrated from `FunJim/Megatron-LM` commit `10cf7cf625df57a497a876707b2e270c017e7de6`. | `verify_glm53_fp32_shards.py`: offload 0/0.5/1, AdamW update, optimizer state restore |
| TileLang shapes | Support GLM's zero RoPE channels and pad KPool top-k slots without changing selected tokens. Migrated from `5b2a86f499d4e24bfeceb2e8998686215ec6dccf`. | `verify_glm53_dsa.py`: output/gradient comparison, 73728-token fused forward/backward |
| KPool CP | Gather gate scores and reorder them with indexer keys across context-parallel ranks. Migrated from `cb31d880c796bd0b409826ec8790e5206db56a02`. | `verify_glm53_dsa_cp.py`: CPU CP1/CP8 output/gradient parity, packed sequences, negative controls; 4-node CP8 preflight |

The runtime result is identical to `FunJim/Megatron-LM` commit `cb31d880`.
That fork is a historical migration source, not an installation dependency.
Regressions are in `tests/glm53_patch_regression/`. The original patch's
Megatron unit-test diff was excluded from this runtime-only bundle; it remains
available in bridge baseline `9d610ff` and Megatron commit `c97ca74`.

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

For the CPU CP regression's negative control, reconstruct the original official
patch in a temporary baseline checkout, then apply the FP32/TileLang fixes but
not KPool CP; or use the historical `5b2a86f` dsa.py. GPU tests require an
allocated device and the same CUDA libraries as training. No full model is loaded.

Updates: bridge tracks `modelscope/mcore-bridge:main`; Swift tracks
`modelscope/ms-swift:main`; Megatron tracks `NVIDIA/Megatron-LM:dev`. Integrate
selected updates on a temporary branch, regenerate the full-index patch against
the new baseline, remove fixes already absorbed upstream, and rerun regressions
before merging to `feat/glm53-flash-support`. Record the three actual commits.
Do not rebase published history or force-push.
