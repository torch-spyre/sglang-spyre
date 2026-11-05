# SGLang Spyre backend

An experimental [SGLang](https://github.com/sgl-project/sglang) backend for
[IBM Spyre Accelerator](https://www.ibm.com/products/spyre-accelerator), built
on [torch-spyre](https://github.com/torch-spyre/torch-spyre). The device kernel
path is synchronized from
[spyre-inference](https://github.com/torch-spyre/spyre-inference), while SGLang
continues to own scheduling and its RadixAttention cache semantics.

> [!IMPORTANT]
> This repository is a working prototype. It is not yet a production-supported
> backend, but its dependency set and benchmark baseline are intentionally
> reproducible.

## Compatibility baseline

The exact machine-readable baseline is in [`compatibility.toml`](compatibility.toml).

| Component | Pin |
|---|---|
| spyre-inference source | `0ccdbf6ac2da7faacf98db93464b274fb8f40126` (main, 2026-10-30) |
| torch-spyre | `64870622fc39cda7e344a271decb901e7088c6c6` |
| vLLM (spyre-inference's reference leg) | `0.32.0` (`5df63a7203cdb9328d824a27b682e77c7b9928be`) |
| PyTorch | `2.13.0` |
| Transformers | `5.16.1` |
| SGLang | `0.5.21` (`e00930c5489053f26d86b179cee0d087f846acbb`) |
| Spyre system RPMs | bundle `main_nightly_2026_10_27`, [`spyre-rpms.lock`](spyre-rpms.lock) |

These pins match spyre-inference's dependency set. SGLang 0.5.21 is the newest
release that fits it: 0.5.22 and 0.5.23 require torch 2.14.1 and
transformers 5.19.0, outside spyre-inference's `torch==2.13.0` and
`transformers<5.17` pins. `spyre-rpms.lock` is spyre-inference's generated
lock for the same bundle and passes its `scripts/lock_spyre_rpms.py --verify`.

## Supported models

The backend ports spyre-inference's Spyre implementation of every model
mechanism its text decoders use. "Validated" means the model was run end to end
on a Spyre card with this baseline and its greedy output compared with
spyre-inference/vLLM on the same card; "runs" means the architecture loads
through the same code paths but was not part of that comparison.

| Model | spyre-inference registry | SGLang status |
|---|---|---|
| `ibm-granite/granite-3.3-8b-instruct` | TP1 and TP4 | **Validated** at TP1. Same answer as vLLM on 4/5 MMLU items, the fifth a near-tie the fp32 reference resolves SGLang's way. Latency 1.00x vLLM ([BENCHMARK.md](BENCHMARK.md)). |
| `google/gemma-4-26B-A4B-it` (text) | TP4 | **Validated** at TP1. Covers MoE, sliding-window attention, final-logit soft-capping and heterogeneous head sizes. Same answer as vLLM wherever both answer; latency 0.99x vLLM ([BENCHMARK.md](BENCHMARK.md)). |
| `google/gemma-3-1b-it` | TP1 | Runs (dense Gemma path with sliding window); not validated in this sync. |
| `ibm-granite/granite-4.1-8b` | TP4 | Runs (the `GraniteForCausalLM` path validated above); not validated at TP1. |
| `ibm-granite/granite-4.1-8b-fp8`, `granite-3.3-8b-instruct-FP8` | TP1/TP4 | Not supported: spyre-inference's FP8 linear and FP8 KV cache are not ported. |
| `ibm-granite/granite-vision-4.1-4b` | TP1 | Not supported: vision towers are not ported. |
| Embedding, reranker and classifier models | various | Not applicable: SGLang text generation only. |

Tensor parallelism above 1 is not supported; spyre-inference's TP4 configs run
here at TP1 when they fit one card (Gemma-4-26B-A4B uses about 52 GB of 112 GB).

The experimental [HF-adapters backend](docs/hf-adapters.md) extends SGLang's
Transformers wrappers for text generation and embeddings. The unified guide
covers model preparation, paged and cache-free attention, pooling, setup and
usage. Its supported scope is separate from the native-model table above.

## What is synchronized

The device code is spyre-inference's, ported as directly as each framework
boundary allows. Kernels are vendored unchanged; framework glue is rewritten
only where vLLM types are involved.

- **Attention** (`sglang_spyre_backend/ops/`, unchanged copies of
  `spyre_inference/v1/attention/ops/`): the head-major paged KV cache, the
  `for_each_tile` page walks, the per-sequence prefill and LX-resident decode
  kernels, and the batched decode kernel that a single decoding sequence now
  takes. `spyre_attention.py` ports the metadata builder (causal and padding
  masks, sliding-window block skipping with per-bucket padding, logit
  soft-capping, page-index and query-row tables, batched-decode metadata) and
  the warmup recorder; `spyre_attn_bucketer.py` is the KV/query/batch bucketer.
  `attention.py` turns SGLang's `ForwardBatch` into the builder's input and
  keeps one builder per layer group (sliding window, KV heads, head size).
- **MoE** (`moe.py`, from `spyre_inference/moe.py`): the gathered and
  persistent expert kernels, fp32 routing reductions, one shared expert-weight
  copy for both, and the token-count switch between them. `models.py` ports the
  Gemma-4 recipe and its startup repairs.
- **Compile and startup** (`model_wrapper.py`): per-block compile with one graph
  shared by identical blocks (attention and MoE stay opaque ops, named by a
  hoisted `LayerName` as vLLM does), body token buckets, and attention graph
  recording at the first forward.
- **Linear and LM head** (`custom_ops/linear.py`): transposed weights with a
  blocked host transpose, short-row padding of merged gate/up projections, and
  a row-padded, tie-safe LM head returning host logits.
- **Norms, activations and RoPE** (`device_ops.py`): fp32 native RMSNorm,
  Gemma-4 norms, device-resident 2x2 RoPE with its split-free form, and
  Gemma-4's proportional RoPE.

[`docs/spyre-inference-sync-2026-11-01.md`](docs/spyre-inference-sync-2026-11-01.md)
lists every runtime change in the synced spyre-inference range as ported, not
applicable (vLLM scheduler or runner plumbing) or out of scope.

## Current scope

- Decoder-only text generation, tensor parallel size 1, FP16.
- SGLang RadixCache with 128-token pages (spyre-inference's default block size).
- `max_running_requests=1` is the validated configuration; the builder and the
  batched decode kernel accept several sequences, as in spyre-inference.
- No CUDA graphs; decoder blocks and attention kernels are compiled with
  `torch.compile(dynamic=False)`.

## Installation

Use a Spyre runtime image containing the RPM versions in `spyre-rpms.lock`.
The normal development flow uses `uv` so the torch-spyre source revision and
its build-time Torch version remain locked:

```bash
git clone https://github.com/torch-spyre/sglang-spyre.git
cd sglang-spyre
uv sync --frozen
source .venv/bin/activate
python -m sglang_oot_patches.apply
```

`torch-spyre` must be built on a supported Spyre host with the IBM runtime and
compiler available. Set the rank variables before importing torch-spyre:

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
```

## Run

```python
import sglang as sgl

engine = sgl.Engine(
    model_path="ibm-granite/granite-3.3-8b-instruct",
    device="cpu",
    attention_backend="spyre_paged",
    dtype="float16",
    page_size=128,
    context_length=2048,
    chunked_prefill_size=512,
    max_running_requests=1,
    disable_cuda_graph=True,
)

result = engine.generate(
    "What are IBM's main businesses?",
    sampling_params={"max_new_tokens": 32, "temperature": 0.0},
)
print(result["text"])
engine.shutdown()
```

Set `SGLANG_FORCE_DEVICE_SPYRE=1` to place supported model-body operations on
Spyre while SGLang keeps host-side scheduling tensors on CPU.

## Verification and benchmark

CPU tests (no card; two upstream kernel test files are ported unchanged):

```bash
pytest -q tests/test_tile_loop.py tests/test_batched_decode_head_major.py \
  tests/test_spyre_attention_builder.py tests/test_moe_routing.py \
  tests/test_platform_ports.py tests/test_reuse_structure.py
```

On a Spyre pod with spyre-inference installed, the A/B test compares the
vendored decode kernel with spyre-inference's own on identical inputs:

```bash
python tests/spyre_kernel_ab.py
```

The SGLang-vs-vLLM comparison runs each framework in its own environment and
combines the records:

```bash
python tests/mmlu_sglang_vs_vllm_bench.py sglang --model M --output sg.json
python tests/mmlu_sglang_vs_vllm_bench.py vllm --model M --output vl.json
python tests/mmlu_sglang_vs_vllm_bench.py compare sg.json vl.json --output out.json
```

See [`BENCHMARK.md`](BENCHMARK.md) for the measured results and the
machine-readable records.

## License

Licensed under the [Apache License 2.0](LICENSE). Code ported from
spyre-inference retains its upstream copyright and license notice.
