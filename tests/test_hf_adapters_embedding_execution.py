"""CPU embedding contracts: masks, packed requests, positions, and pooling metadata."""

from __future__ import annotations

import json

import pytest
import torch
from test_hf_adapters_generation_execution import GENERATION_FAMILIES, args, make_model
from test_hf_adapters_generation_execution import cpu_adapters as cpu_adapters
from torch.nn import functional as F

from sglang_spyre_backend.embedding import (
    CacheFreeAdapterAttention,
    EmbeddingBody,
    EmbeddingMask,
    pooling_config,
)
from sglang_spyre_backend.hf_adapters import prepare_execution, validate_config

ENCODER_FAMILIES = (
    "bert",
    "distilbert",
    "mpnet",
    "modernbert",
    "roberta",
    "xlm-roberta",
)


def make_encoder(family):
    from hf_adapters.hf_common import get_backbone
    from transformers import AutoConfig, AutoModel

    if family in GENERATION_FAMILIES:
        return get_backbone(make_model(family))
    width = 64 if family == "mpnet" else 32
    config = AutoConfig.for_model(
        family,
        vocab_size=129,
        hidden_size=width * 2,
        intermediate_size=128,
        hidden_dim=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=132,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        cls_token_id=0,
        sep_token_id=2,
        type_vocab_size=2,
        **({"sliding_window": 4} if family == "modernbert" else {}),
        attn_implementation="eager",
    )
    torch.random.default_generator.manual_seed(29)
    return AutoModel.from_config(config).eval().requires_grad_(False)


def write_pooling(path, mode, *, normalize=True):
    (path / "1_Pooling").mkdir(exist_ok=True)
    kinds = ["Transformer", "Pooling"] + (["Normalize"] if normalize else [])
    (path / "modules.json").write_text(
        json.dumps(
            [
                {
                    "type": f"sentence_transformers.models.{kind}",
                    "path": "" if i == 0 else f"{i}_{kind}",
                }
                for i, kind in enumerate(kinds)
            ]
        )
    )
    (path / "1_Pooling/config.json").write_text(
        json.dumps({f"pooling_mode_{mode}": True})
    )


@pytest.mark.parametrize("family", ENCODER_FAMILIES + GENERATION_FAMILIES)
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_rectangles_match_independent_hf_requests(
    cpu_adapters, monkeypatch, family, compiled
):
    model = make_encoder(family)
    requests = [
        torch.tensor([3, 7, 9]),
        torch.tensor([5, 4, 8, 11, 13, 17]),
        torch.arange(3, 76),
    ]
    expected = [model(ids[None]).last_hidden_state[0] for ids in requests]
    prepared = prepare_execution(model, max_model_len=128, task="embed")
    mask = EmbeddingMask()
    for executor in prepared.runtime.attentions:
        assert not executor.spec.uses_cache
        executor.bind(CacheFreeAdapterAttention(executor.spec, mask))
    body = EmbeddingBody(
        prepared, mask, max_model_len=128, max_requests=2, device="cpu"
    )
    graphs = []
    if compiled:
        compile_original = torch.compile

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        monkeypatch.setattr(
            torch,
            "compile",
            lambda fn, **kwargs: compile_original(fn, backend=backend, **kwargs),
        )
        prepared.runtime.compile_body(model)
        body.warmup()
    count = len(graphs)
    for order in ([0, 1], [1, 0], [0], [1], [0, 2], [2], [2, 1], [0, 1]):
        ids = torch.cat([requests[i] for i in order])
        positions = torch.cat([torch.arange(len(requests[i])) for i in order])
        actual = body(ids, positions, [len(requests[i]) for i in order])
        reference = torch.cat([expected[i] for i in order])
        torch.testing.assert_close(actual, reference, atol=4e-5, rtol=4e-4)
    if compiled:
        assert (
            len(graphs) == count
        ), "Changing padding masks must not compile new graphs"


@torch.no_grad()
def test_token_type_ids_are_preserved(cpu_adapters):
    model = make_encoder("bert")
    ids = torch.tensor([3, 4, 7, 5])
    types = torch.tensor([0, 0, 1, 1])
    expected = model(ids[None], token_type_ids=types[None]).last_hidden_state[0]
    prepared = prepare_execution(model, max_model_len=64, task="embed")
    mask = EmbeddingMask()
    for executor in prepared.runtime.attentions:
        executor.bind(CacheFreeAdapterAttention(executor.spec, mask))
    body = EmbeddingBody(prepared, mask, max_model_len=64, max_requests=1, device="cpu")
    actual = body(ids, torch.arange(4), [4], types)
    torch.testing.assert_close(actual, expected, atol=4e-5, rtol=4e-4)


@torch.no_grad()
def test_roberta_padding_tokens_preserve_position_numbering(cpu_adapters):
    model = make_encoder("roberta")
    ids = torch.tensor([3, 1, 7, 5])
    expected = model(
        ids[None], attention_mask=torch.ones_like(ids)[None]
    ).last_hidden_state[0]
    prepared = prepare_execution(model, max_model_len=64, task="embed")
    mask = EmbeddingMask()
    for executor in prepared.runtime.attentions:
        executor.bind(CacheFreeAdapterAttention(executor.spec, mask))
    body = EmbeddingBody(prepared, mask, max_model_len=64, max_requests=1, device="cpu")
    actual = body(ids, torch.arange(4), [4])
    torch.testing.assert_close(actual, expected, atol=4e-5, rtol=4e-4)


@pytest.mark.parametrize(
    "mode,name", [("mean_tokens", "MEAN"), ("cls_token", "CLS"), ("lasttoken", "LAST")]
)
def test_pooling_metadata_and_override(tmp_path, mode, name):
    write_pooling(tmp_path, mode)
    config = args(model_path=str(tmp_path))
    assert pooling_config(config) == (name, True)
    config.json_model_override_args = json.dumps(
        {"pooling_type": "LAST", "normalize": False}
    )
    assert pooling_config(config) == ("LAST", False)


@pytest.mark.parametrize("invalid", ["dense", "multiple", "prompt"])
def test_unsupported_pipeline_is_rejected(tmp_path, invalid):
    write_pooling(tmp_path, "mean_tokens")
    if invalid == "dense":
        modules = json.loads((tmp_path / "modules.json").read_text())
        modules.append(
            {"type": "sentence_transformers.models.Dense", "path": "3_Dense"}
        )
        (tmp_path / "modules.json").write_text(json.dumps(modules))
    else:
        path = tmp_path / "1_Pooling/config.json"
        metadata = json.loads(path.read_text())
        metadata[
            "pooling_mode_cls_token" if invalid == "multiple" else "include_prompt"
        ] = (invalid == "multiple")
        path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        pooling_config(args(model_path=str(tmp_path)))


def test_bare_backbone_requires_explicit_pooling(tmp_path):
    config = args(model_path=str(tmp_path))
    with pytest.raises(ValueError, match="Specify pooling_type"):
        pooling_config(config)
    config.json_model_override_args = json.dumps({"pooling_type": "CLS"})
    assert pooling_config(config) == ("CLS", True)


def test_embedding_requires_full_requests():
    values = dict(
        is_embedding=True,
        max_running_requests=2,
        attention_backend="spyre_embedding",
        chunked_prefill_size=-1,
        disable_radix_cache=True,
    )
    validate_config(args(**values))
    for invalid in (
        {"max_running_requests": None},
        {"chunked_prefill_size": 128},
        {"disable_radix_cache": False},
        {"attention_backend": "spyre_paged"},
    ):
        with pytest.raises(ValueError):
            validate_config(args(**(values | invalid)))


@torch.no_grad()
def test_position_bias_resets_for_each_rectangle_row():
    from types import SimpleNamespace

    from torch import nn

    class Bias(nn.Module):
        def forward(self):
            return -torch.abs(torch.arange(8)[:, None] - torch.arange(8)[None, :])[
                None, None
            ].float()

    spec = SimpleNamespace(uses_cache=False, scale=0.17, position_bias=Bias())
    mask = EmbeddingMask()
    mask.value = torch.zeros(2, 1, 8, 8)
    mask.value[0, :, :, 3:] = -10000
    bridge = CacheFreeAdapterAttention(spec, mask)
    q, k, v = [torch.randn(2, 2, 8, 64) for _ in range(3)]
    actual = bridge(q, k, v)
    for i, length in enumerate((3, 8)):
        expected = F.scaled_dot_product_attention(
            q[i : i + 1, :, :length],
            k[i : i + 1, :, :length],
            v[i : i + 1, :, :length],
            attn_mask=Bias()()[..., :length, :length],
            scale=0.17,
        )
        torch.testing.assert_close(actual[i : i + 1, :, :length], expected)
