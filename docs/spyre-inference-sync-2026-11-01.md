# spyre-inference sync: 6e5ff99 → 0ccdbf6

This sync moves the backend from spyre-inference `6e5ff99` (2026-08-26) to
`0ccdbf6` (2026-10-30, #1395), 372 commits. 174 of them change runtime code;
the other 198 are CI (65), tests and test infrastructure (32), benchmark
infrastructure (17), dependency bumps (57), docs (16) and chores (11).

Each runtime change is listed below by area as **ported**, **not applicable**
(vLLM scheduler, runner or engine plumbing that SGLang replaces with its own) or
**out of scope** (a model type or dtype this backend does not serve: pooling,
multimodal, FP8, speculative decoding, tensor parallelism above 1).

## Attention: ported

The kernels are copied unchanged into `sglang_spyre_backend/ops/`; the builder,
recorder and bucketer around them are ported into `spyre_attention.py` and
`spyre_attn_bucketer.py`.

| Area | PRs |
|---|---|
| Head-major KV layout (the default), LX-resident pages, prefill kernel for wide queries, per-sequence decode kernel with query groups folded into rows | #890, #893, #908, #915, #987, #1038 |
| `for_each_tile` page walks and their Python fallback (`walk_tiles`) | #876, #1009 |
| Batched decode: the kernel, its metadata, chunking to the core count, and a single decoding sequence routed to it | #612, #710, #714, #716, #758, #780, #791, #810, #852, #898, #932, #1068, #1161 |
| Sliding-window masks with arithmetic block skip, padded to a per-bucket maximum | #752, #1021 |
| Attention logit soft-capping in every kernel | #731 |
| Per-layer KV heads and head size for heterogeneous models (gemma-4) | #874, #960, #1317 |
| Query row tables, mask stacks mirrored once per step, staging buffers row-outermost | #647, #751, #824, #859, #869, #879, #941, #977 |
| Bucketer: KV ladder by name, query and batch buckets | #1067, #796 (single-sequence form), #810 |
| Warmup recording of every variant, one recording per layer group | #707, #798, #871, #875, #1381 |
| Context-local core cap (`_run_with_capped_cores`) that leaves the rest of the model uncapped | #1413 |
| KV pages zeroed on the device | #1375 |
| Compiled kernels at module scope | #799 |

Not applicable: jagged multi-bucket batched decode and the mixed-batch and
utilisation gates (#832, #838, #1429) only act with several decoding sequences;
the builder carries the jagged code path's inputs, but the validated
configuration runs one sequence. The token-major layout (#864's other half) is
kept upstream only for ALiBi; it is not ported, and neither is ALiBi.

## MoE: ported

`moe.py` carries `spyre_inference/moe.py`'s kernels and weight preparation
unchanged; only the binding to SGLang's `FusedMoE` is new.

| Area | PRs |
|---|---|
| Gemma-4 MoE: gathered decode kernel, persistent all-expert kernel, Spyre weight layout | #802, #842 |
| fp32 routing softmax when the expert count is stick-aligned | #877 |
| Persistent experts with `for_each_tile`, token split planned by torch-spyre | #1058, #1332 |
| One shared expert-weight copy for prefill and decode, down stack in matmul order | #1120, #1145 |
| Gathered/all-expert switch at `SPYRE_MOE_GATHERED_MAX_TOKENS=8`, batch decode loop | #975, #1212 |
| Gathered index read through a reduction | #1304 |
| MoE interface rework (recipes, opt-in per model) | #1175 |

Not applicable: #1175's in-graph MoE runner (`SpyreMoERunner`) is vLLM
`MoERunner` glue; SGLang calls the experts through an opaque op, the path
spyre-inference takes for token counts that are not a whole stick. Out of scope:
#820 (TP sharding of experts).

## Model adaptations: ported

| Area | PRs |
|---|---|
| Gemma-4 head-dim repair for transformers ≥ 5.16 | #717 |
| Gemma-4 text backbone selection, MoE recipe with the expert scale folded into the down stack | #802, #1175 |
| RoPE cache at the model dtype, split-free RoPE for a sub-stick half head, Gemma-4 proportional RoPE | #1158, #1393 |
| fp32 Gemma RMSNorm; plain RMSNorm on upstream's fp32 native math | #845, #936 |
| Final RMSNorm, embedding and LM head compiled on their own | #652, #1348 |

Out of scope: vision towers and multimodal embedding (#843, #944, #1001, #1130,
#1156, #1165, #1185, #1198, #1261, #1316, #1342, #1359, ...), encoder and pooling
models (#649, #683, #795, #999, #1015, #1065, #1070, #1142, #1172, #1226,
#1319, #1382, ...), Mistral/Pixtral (#736), head-dim padding fixes (#614, #1066).

## Linear and LM head: ported

| Area | PRs |
|---|---|
| Blocked host transpose of weights | #1327 |
| Short decode rows of merged gate/up projections padded to 8 | #651 |
| Transposed-weight base without the 3-D matmul workaround | #823 |
| Row-padded, tie-safe LM head that keeps the gather table on the host, host logits at TP1 | #633, #653, #745, #784 |

Out of scope: FP8 linear (#545, #814, #841, #946, #1133), the TP logits gather
(#939, #1283), vocab reindex tables for TP (#648, #728).

## Compile and startup: ported

| Area | PRs |
|---|---|
| Per-block compile with one artifact shared by identical blocks; attention and MoE as opaque ops named by a hoisted `LayerName` | #1320, #1395 |
| Body token buckets (`compile_sizes` defaults) | #718 |
| Aliased blocks compiled once | #797 |
| Default 128-token KV block size, a multiple of 64 and a power of two | #746 |
| Thread pools clamped to the CPU budget, `OMP_WAIT_POLICY=PASSIVE` | #678, #1254 |
| No `HBM_POOL_PLANNING=0` override | #665 |
| `SPYRE_*` environment knobs under spyre-inference's names and defaults, `SPYRE_KERNEL_CACHE` off | #811, `envs.py` |

Not applicable: the compile guard (#943, #800) watches vLLM's warmup; the
in-process compile-cache reuse (#1374), uni executor (#1329), CPU sampling
kernels (#1111), partial-prefill cap (#891), the Model Runner V2 port (#1249,
#1256, #1277, #1354, #1366, #1382) and the remaining V1-runner refactors move
code inside vLLM's runner, which SGLang does not use. The traced KV write
(#610) is replaced by the same write inside the opaque attention op.

## KV cache and FP8: out of scope

FP8 KV cache and its scales (#1146, #1400, #1395's scale resolution) and hybrid
KV-cache sizing (#738).

## Speculative decoding: out of scope

#1394, #1436.
