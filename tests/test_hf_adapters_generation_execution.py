"""CPU contracts for the optional adapter provider and opaque attention bridge."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sglang_spyre_backend.hf_adapters import (
    AdapterBody,
    PagedAdapterAttention,
    prepare_execution,
    uses_hf_adapters,
    validate_config,
)


def args(**kwargs):
    values = dict(
        json_model_override_args=json.dumps({"spyre_model_preparation": "hf_adapters"}),
        model_impl="transformers",
        device="cpu",
        attention_backend="spyre_paged",
        dtype="float16",
        page_size=128,
    )
    return SimpleNamespace(**(values | kwargs))


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ("{}", False),
        ('{"spyre_model_preparation":"spyre"}', False),
        ('{"spyre_model_preparation":"hf_adapters","foo":42}', True),
    ],
)
def test_explicit_mode(overrides, expected):
    config = args(json_model_override_args=overrides)
    assert uses_hf_adapters(config) is expected
    assert config.json_model_override_args == overrides


@pytest.mark.parametrize(
    "setting",
    [
        {"model_impl": "auto"},
        {"device": "spyre"},
        {"attention_backend": "torch_native"},
        {"tp_size": 2},
        {"pp_size": 2},
        {"dp_size": 2},
        {"dtype": "bfloat16"},
        {"quantization": "fp8"},
        {"kv_cache_dtype": "fp8_e4m3"},
        {"enable_lora": True},
        {"speculative_algorithm": "EAGLE"},
        {"is_embedding": True},
        {"enable_torch_compile": True},
        {"enable_mixed_chunk": True},
        {"load_format": "dummy"},
        {"page_size": 96},
        {"cpu_offload_gb": 1},
        {"disaggregation_mode": "prefill"},
        {"json_model_override_args": "[]"},
        {"json_model_override_args": '{"spyre_model_preparation":"typo"}'},
    ],
)
def test_reject_unsupported_serving_options(setting):
    with pytest.raises(ValueError):
        validate_config(args(**setting))


def test_native_options_are_untouched():
    validate_config(
        args(
            json_model_override_args="{}",
            tp_size=4,
            dtype="bfloat16",
            quantization="fp8",
        )
    )


class ReferenceBackend:
    """Independent request-local SDPA with persistent per-layer KV state."""

    def __init__(self):
        self.rows = []
        self.cache = {}
        self.calls = []

    def _forward_impl(self, q, k, v, layer, save):
        assert save
        self.calls.append(layer.layer_id)
        q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.reshape(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.reshape_as(k)
        for i, (request, position) in enumerate(self.rows):
            self.cache[layer.layer_id, request, position] = (k[i].clone(), v[i].clone())
        output = torch.zeros_like(q)
        for i, (request, position) in enumerate(self.rows):
            cached = [
                self.cache[layer.layer_id, request, p] for p in range(position + 1)
            ]
            key = torch.stack([item[0] for item in cached]).transpose(0, 1)[None]
            value = torch.stack([item[1] for item in cached]).transpose(0, 1)[None]
            output[i] = F.scaled_dot_product_attention(
                q[i][None, :, None],
                key,
                value,
                scale=layer.scaling,
                enable_gqa=True,
            )[0, :, 0]
        return output.reshape(q.shape[0], -1)


@pytest.fixture
def cpu_adapters(monkeypatch):
    common = pytest.importorskip("hf_adapters.hf_common")
    monkeypatch.setattr(common, "DEVICE", "cpu")
    from sglang_spyre_backend import opaque

    monkeypatch.setattr(opaque, "_ATTN_LAYERS", {})
    monkeypatch.setattr(opaque, "_ATTN_BACKEND", [None])
    yield
    torch._dynamo.reset()


GENERATION_FAMILIES = (
    "gpt2",
    "gpt_neo",
    "gpt_neox",
    "granite",
    "llama",
    "mistral",
    "ministral",
    "olmo",
    "olmo2",
    "opt",
    "qwen2",
    "qwen3",
    "phi3",
    "smollm3",
)


def make_model(family, *, tie_word_embeddings=True):
    from transformers import AutoConfig, AutoModelForCausalLM

    width = {"phi3": 96, "olmo2": 128}.get(family, 64)
    extra = {}
    if family == "granite":
        extra.update(
            embedding_multiplier=1.7,
            residual_multiplier=0.7,
            attention_multiplier=0.11,
            logits_scaling=2.5,
        )
    if family == "gpt_neo":
        extra["attention_types"] = [[["global", "local"], 1]]
    if family == "smollm3":
        extra["no_rope_layers"] = [1, 0]
    if family == "ministral":
        extra.update(
            sliding_window=128, layer_types=["full_attention", "sliding_attention"]
        )
    if family not in ("gpt2", "gpt_neo", "gpt_neox", "opt"):
        extra["num_key_value_heads"] = 1
    config = AutoConfig.for_model(
        family,
        hidden_size=width * 2,
        intermediate_size=256,
        n_inner=256,
        ffn_dim=256,
        word_embed_proj_dim=width * 2,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=width,
        vocab_size=129,
        max_position_embeddings=128,
        original_max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=tie_word_embeddings,
        attn_implementation="eager",
        **extra,
    )
    torch.random.default_generator.manual_seed(17)
    return AutoModelForCausalLM.from_config(config).eval().requires_grad_(False)


def bind_reference(prepared):
    from sglang_spyre_backend import opaque

    backend = ReferenceBackend()
    opaque.set_attention_backend(backend)
    for attention in prepared.runtime.attentions:
        spec = attention.spec
        layer = nn.Module()
        layer.layer_id = spec.layer_idx
        layer.tp_q_head_num = spec.num_heads
        layer.tp_k_head_num = spec.num_kv_heads
        layer.qk_head_dim = layer.v_head_dim = spec.head_size
        layer.scaling = spec.scale
        layer.spyre_layer_name = f"attn.{spec.layer_idx}"
        opaque._ATTN_LAYERS[layer.spyre_layer_name] = layer
        attention.bind(PagedAdapterAttention(layer))
    return backend


@pytest.mark.parametrize("family", GENERATION_FAMILIES)
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_packed_prefill_and_reordered_cache_continuation(
    cpu_adapters, monkeypatch, family, compiled
):
    from hf_adapters.hf_common import get_backbone

    model = make_model(family)
    sequences = [torch.tensor([[3, 7, 11, 13, 19]]), torch.tensor([[5, 9, 12, 15]])]
    expected = [model(ids).logits[0] for ids in sequences]
    candidate = copy.deepcopy(model)
    backbone = get_backbone(candidate)
    prepared = prepare_execution(backbone, max_model_len=64)
    backend = bind_reference(prepared)
    graphs = []
    if compiled:
        original_compile = torch.compile

        def counted_backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        monkeypatch.setattr(
            torch,
            "compile",
            lambda fn, **kw: original_compile(fn, backend=counted_backend, **kw),
        )
        prepared.runtime.compile_body(backbone)
    body = AdapterBody(prepared, [1, 2, 8], torch.device("cpu"))
    scale = prepared.runtime.logits_scale or 1.0
    if compiled:
        for size in body.buckets:
            backend.rows = [(0, p) for p in range(size)]
            body(torch.full((size,), 3), torch.arange(size))
        backend.cache.clear()
    count = len(graphs)
    for rows in (
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)],
        [(1, 2), (0, 3)],
        [(0, 4), (1, 3)],
    ):
        backend.rows = rows
        ids = torch.tensor([sequences[r][0, p] for r, p in rows])
        positions = torch.tensor([p for _, p in rows])
        actual = candidate.get_output_embeddings()(body(ids, positions)) * scale
        target = torch.stack([expected[r][p] for r, p in rows])
        torch.testing.assert_close(actual, target, atol=3e-5, rtol=3e-4)
    if compiled:
        assert len(graphs) == count, "Packed requests must reuse warmed body graphs"
    assert set(backend.calls) == {0, 1}
    assert (
        backbone.get_input_embeddings().weight
        is candidate.get_output_embeddings().weight
    )
    if family in ("llama", "phi3"):
        assert {a.spec.head_size for a in prepared.runtime.attentions} == {128}


def test_body_rejects_unwarmed_size_and_invalid_inputs():
    body = AdapterBody(None, [1, 8], torch.device("cpu"))
    with pytest.raises(ValueError, match="bucket limit"):
        body(torch.zeros(9, dtype=torch.int64), torch.arange(9))
    with pytest.raises(ValueError, match="at least one"):
        body(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))
    with pytest.raises(ValueError, match="packed"):
        body(torch.zeros(1, 3, dtype=torch.int64), torch.arange(3))
