# HF-adapters backend for SGLang-Spyre

The HF-adapters backend runs stock Hugging Face Transformers backbones on IBM
Spyre through SGLang. It supports text generation and text embeddings by combining
HF-adapters' model preparation and execution drivers with SGLang's serving
infrastructure. It is an experimental, explicitly selected execution mode.

The integration extends upstream SGLang's `sglang.srt.models.transformers`
wrappers: `TransformersForCausalLM` for generation and
`TransformersEmbeddingModel` for embeddings. The extension lives in
`sglang_spyre_backend/hf_adapters_transformers.py`. It uses stock HF model classes
and loaded weights, with runtime adapter transformations and a backbone driver
in place of the stock HF forward.

## Division of responsibilities

| Component | Responsibility |
|---|---|
| Hugging Face Transformers | Model configuration, stock backbone modules and checkpoint structure. |
| HF-adapters | Model-specific computation, projection transformations, head padding, position handling, block compilation and attention binding points. |
| SGLang | Tokenization, request scheduling, checkpoint loading, generation cache allocation and metadata, logits processing, sampling, embedding pooling and responses. |
| SGLang-Spyre | Spyre placement, attention bridges and kernels, static input shapes, warmup and generation output-head execution. |

Dependencies flow from SGLang-Spyre to HF-adapters. The shared adapter execution
API has no dependency on SGLang or another serving framework. Its attention
descriptors identify each layer's head geometry, scale and attention semantics;
the serving integration binds those attention calls to a concrete implementation.

## Shared construction and preparation

Text generation and text embedding share the same lifecycle up to attention binding:

1. **Select the wrapper.** At the worker's model-loading boundary, the explicit
   `spyre_model_preparation="hf_adapters"` override selects the extended
   Transformers wrapper through SGLang's model registry.
2. **Construct and load.** Construct the stock HF backbone on the meta device,
   materialize its parameters on CPU, and use SGLang's checkpoint loader to load
   weights. Preserve the HF backbone's linear, normalization and embedding
   modules during this phase.
3. **Prepare once.** Call
   `hf_adapters.execution.prepare_adapter_execution(model, max_model_len=..., task=...)`
   with `task="generate"` or `task="embed"`. The adapter prepares projections,
   positions and the backbone driver, and exposes the final attention descriptors.
4. **Bind attention.** Attach the task's attention bridge after preparation, so
   head dimensions include any adapter padding. Preserve the descriptor's
   original attention scale; padding a head must not change that scale. Publish
   the resulting geometry to SGLang before generation cache allocation.
5. **Place and compile.** Move the prepared backbone to Spyre, install deferred
   adapter compilation, and warm the configured input shapes. Generation also
   prepares the retained output head.
6. **Serve.** Run the adapter backbone driver and return hidden states to the
   SGLang wrapper for task-specific output processing.

Upstream SGLang normally replaces HF backbone modules before loading weights to
provide its serving-oriented linear, normalization and embedding implementations,
including hooks for tensor parallelism and quantization. Adapter mode bypasses
these replacements because adapter preparation expects stock HF modules and
performs its own Spyre transformations. It also bypasses the native SGLang-Spyre
body replacement and compilation path. The supported configuration is therefore
single-device, unquantized execution with adapter-controlled body compilation.

Backbones that support HF's custom-attention interface retain upstream
construction with the `"sglang"` attention selection. Only stock backbones that
lack this capability, currently GPT-Neo and MPNet, use `"eager"` during
construction. This exception allows the HF model to be instantiated; it does not
select the attention used during serving. The adapter driver calls the explicitly
bound attention bridge, including for these models. Capability flags are not
changed globally, and ordinary Transformers mode delegates to upstream behavior.

## Generation

Generation uses `TransformersForCausalLM` with
`attention_backend="spyre_paged"`. SGLang continues to own the paged KV cache,
request scheduling, prefix reuse, chunked prefill and sampling.

SGLang supplies packed token IDs and request-relative positions. `AdapterBody`
pads the packed token dimension to a configured static bucket and invokes the
adapter backbone driver. Each layer calls `PagedAdapterAttention`, which converts
adapter Q/K/V tensors from `[1, heads, tokens, head_dim]` into the packed layout
expected by SGLang-Spyre. Registered `RadixAttention` instances carry the final
layer geometry and scale.

The bridge calls the existing opaque Spyre attention operation and its head-major
paged-attention kernels. An opaque operation keeps attention and its separately
compiled kernels outside the compiled adapter block; current SGLang request and
cache metadata are read at execution time. HF-adapters creates no dense KV cache
for this path.

Padding tokens are excluded from real request metadata, and dummy cache writes
use the allocator's reserved null slot. After the backbone runs, padding is
removed and hidden states return to CPU so SGLang can select the required rows.
The retained output head projects those rows on Spyre, and SGLang processes the
logits and samples tokens. Tied output-head weights are resolved against the
loaded input embedding before head packing; model-specific logits scaling, such
as Granite's, is applied once by the logits processor.

## Embeddings

Embeddings use `TransformersEmbeddingModel` with `is_embedding=True` and
`attention_backend="spyre_embedding"`. Both encoder backbones and supported
decoder backbones can produce embeddings. This path returns pooled hidden states
without an LM head or logits processor.

`EmbeddingBody` repacks SGLang's packed requests into padded
`[batch, tokens]` rectangles. Each row has its own positions and padding mask, so
requests remain independent. Encoder attention is bidirectional; decoder
embedding attention is causal when specified by the adapter. Learned position
offsets, including RoBERTa's padding-based numbering, are preserved. Token-type
IDs are forwarded when SGLang supplies them.

`CacheFreeAdapterAttention` passes Q/K/V, the request mask, the original attention
scale and any adapter position bias to an opaque operation backed by separately
compiled scaled dot-product attention. It neither reads nor writes persistent KV
state. SGLang retains CPU allocation and scheduling bookkeeping, but this backend
does not allocate a Spyre paged KV cache.

After execution, real token rows are restored to their original packed order and
converted to FP32 on CPU. SGLang's pooler then reduces each request to a vector
and optionally normalizes it. FP32 pooling avoids FP16 accumulation overflow in
MEAN pooling.

Embedding requests must execute in full: prefix caching and chunked prefill are
disabled. An explicit positive `max_running_requests` bounds the batch shapes.

## Installation and process setup

Use a Linux Spyre host or runtime container with the IBM runtime and compiler
installed, access to a Spyre accelerator, Python 3.11 or later, and `uv`. The
project's dependency set uses SGLang 0.5.21, PyTorch 2.13.0, Transformers 5.16.1,
tokenizers 0.23.3 and the Spyre RPM bundle `main_nightly_2026_10_27`.
`pyproject.toml` and `uv.lock` select the Python dependencies and source revisions;
`spyre-rpms.lock` specifies the system packages. Python installation does not
install those system RPMs.

From a SGLang-Spyre checkout, run:

```bash
source /etc/bashrc
uv sync --frozen --extra hf-adapters
source .venv/bin/activate
python -m sglang_oot_patches.apply
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
```

The `hf-adapters` extra installs the shared adapter package at the integration's
pinned revision. Use this serving environment's dependency resolution: its
Transformers override also applies to HF-adapters.

The compatibility patch command modifies the installed SGLang source. It excludes
the existing Spyre paged backends from generic Triton dispatch, makes unused image
processor registration optional, extends Gemma-4's backend allowlist, and extracts
an overridable Transformers construction method. The construction method preserves
upstream behavior by default and enables the narrow capability exception described
above. Patching is idempotent and reports an error if the expected source anchors
are absent. Apply it again after reinstalling SGLang.

Set the environment variables before starting Python. Disabling backend autoload
keeps Spyre initialization out of the SGLang parent process; the worker initializes
the device. Keep `device="cpu"` in SGLang arguments so scheduling and index tensors
stay on the host. Adapter mode places the backbone on Spyre automatically and
does not require `SGLANG_FORCE_DEVICE_SPYRE=1`.

## Generate text

An example of generative LLM setup:

```python
import json

import sglang as sgl


if __name__ == "__main__":
    engine = sgl.Engine(
        model_path="Qwen/Qwen3-0.6B",
        model_impl="transformers",
        json_model_override_args=json.dumps(
            {"spyre_model_preparation": "hf_adapters"}
        ),
        device="cpu",
        attention_backend="spyre_paged",
        dtype="float16",
        page_size=128,
        context_length=384,
        chunked_prefill_size=128,
        max_total_tokens=1536,
        max_running_requests=2,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
    )
    try:
        output = engine.generate(
            "The capital of France is",
            sampling_params={"max_new_tokens": 16, "temperature": 0.0},
        )
        print(output["text"])
    finally:
        engine.shutdown()
```

`context_length` must accommodate the prompt and generated tokens. A positive
`chunked_prefill_size` bounds each prefill chunk; prefix caching can remain
enabled. Use the checkpoint's intended prompt or chat template when preparing
input text.

The current Python argument is `model_impl`, with CLI equivalent
`--model-impl transformers`. Select adapter mode explicitly through
`json_model_override_args`, or
`--json-model-override-args '{"spyre_model_preparation":"hf_adapters"}'` on the
CLI. A field in a checkpoint's configuration alone does not activate the mode.

## Encode text

This example reads pooling and normalization settings from a SentenceTransformers
checkpoint:

```python
import json

import sglang as sgl


if __name__ == "__main__":
    engine = sgl.Engine(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        model_impl="transformers",
        is_embedding=True,
        json_model_override_args=json.dumps(
            {"spyre_model_preparation": "hf_adapters"}
        ),
        device="cpu",
        attention_backend="spyre_embedding",
        dtype="float16",
        page_size=128,
        context_length=128,
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        max_total_tokens=1024,
        max_running_requests=2,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
    )
    try:
        print(engine.encode(["Hello world.", "A second document."]))
    finally:
        engine.shutdown()
```

For CLI use, the task-specific selectors are `--is-embedding` and
`--attention-backend spyre_embedding`; retain the same adapter override and
configuration limits. The `__main__` guard in both scripts supports SGLang's
multiprocess startup, and `shutdown()` releases the engine's workers.

### Pooling and normalization

| Mode | Result before optional normalization |
|---|---|
| `CLS` | The first token's hidden state. |
| `MEAN` | The mean of the request's token hidden states, excluding added batch padding. |
| `LAST` | The last token's hidden state. |

For SentenceTransformers checkpoints, the supported pipeline is a
`sentence_transformers.models.Transformer` at the checkpoint root, followed by
one `Pooling` module and optionally one `Normalize` module. Metadata must select
exactly one of CLS, MEAN or LAST pooling and must include prompt tokens
(`include_prompt=True`, also the default when absent). A `Normalize` module
enables L2 normalization; its absence disables normalization.

For a bare HF backbone without SentenceTransformers pipeline metadata, specify
`pooling_type` explicitly. Normalization defaults to true. For example, replace
the JSON override in the embedding script with:

```json
{
  "spyre_model_preparation": "hf_adapters",
  "pooling_type": "MEAN",
  "normalize": true
}
```

These overrides can also change pooling and normalization for a supported
SentenceTransformers pipeline. They do not bypass pipeline validation: learned
Dense projections, custom modules, combined pooling modes, a Transformer outside
the checkpoint root, and prompt-excluding pooling are unsupported. `normalize`
must be a JSON boolean.

The integration reads pooling metadata; callers still need to supply the
checkpoint's intended query/document prefixes, instructions and input-length
policy. Selecting a pooling mode does not provide that preprocessing.

## Sizing and compilation

Spyre execution uses static shapes. The adapter body compiles with
`dynamic=False`; attention compiles separately behind the opaque boundary.

For generation, token buckets cover decode batches and the configured prefill
budget. That budget is the smaller of the positive chunk size and context length,
or the context length when chunking is disabled. Startup exercises the body and
output head for each bucket using only the reserved null cache page. Real request
metadata is restored afterward. The paged-attention backend manages its own
kernel recording and reuse; body warmup does not guarantee that every attention
variant has compiled before its first request.

For embeddings, token widths are drawn from 64, 128, 256, 512 and the configured
context length, capped at that context length. Batch sizes use powers of two plus
the explicit maximum request count. Startup exercises each configured batch/width
combination, and requests are padded to the next admitted rectangle.

Choose context and concurrency limits for the workload and available memory.
Larger limits increase compilation work and padding; generation also needs room
for persistent KV pages. `max_total_tokens` controls SGLang's token-pool capacity
and bookkeeping, while `context_length` bounds each request. Embeddings still
require SGLang bookkeeping capacity even though their attention has no persistent
device KV cache. Inputs exceeding the adapter's configured shape limits are
rejected. Change these limits by constructing a new engine.

For diagnosis, set `SGLANG_SPYRE_COMPILE_BLOCKS=0` before startup to disable adapter
body compilation. Attention remains separately compiled. SGLang's generic
`enable_torch_compile` and CUDA graph mechanisms are not used for this path.

## Configuration scope

| Setting | Generation | Embeddings |
|---|---|---|
| `model_impl` | `"transformers"` | `"transformers"` |
| Explicit preparation override | `"hf_adapters"` | `"hf_adapters"` |
| `device` | `"cpu"` for SGLang scheduling | `"cpu"` for SGLang scheduling |
| `attention_backend` | `"spyre_paged"` | `"spyre_embedding"` |
| `is_embedding` | False | True |
| Prefix caching | Allowed | Disabled with `disable_radix_cache=True` |
| Chunked prefill | Allowed | Disabled with `chunked_prefill_size=-1` |
| `max_running_requests` | Set explicitly to size concurrency | Explicit positive integer required |

Both paths require FP16 execution (`dtype="float16"`, or `"auto"` resolved by the
Spyre platform), `kv_cache_dtype="auto"`, and TP=PP=DP=1. Use ordinary unquantized
HF checkpoint loading with `load_format` set to `"auto"`, `"safetensors"` or
`"pt"`. `page_size` must be a positive power of two divisible by 64; 128 is used
above. This shared argument check also applies to embeddings, even though their
attention does not use pages. Supply token IDs through SGLang's tokenization
path; direct `input_embeds` are unsupported.

The model must have a dense text backbone supported by the shared adapter API.
Its architecture, position encoding, head geometry and context-length
restrictions continue to apply. The integration requires uniform, aligned
attention head widths; generation requires causal attention without an additional
position-bias tensor. Support for a Transformers model class does not imply
support for every checkpoint variant.

The current path excludes MoE, multimodal, encoder-decoder, hybrid and MLA models;
embedding scoring/classification heads; quantization; LoRA; speculative decoding;
mixed prefill/decode chunks; hierarchical caches; disaggregated serving; CPU
weight backup or offload; overlapped weight loading; and weight reload. Load and
prepare weights once per engine.
