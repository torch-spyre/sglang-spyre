"""Real SGLang Transformers construction, loading and binding on CPU."""

from __future__ import annotations

import copy
import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch
from test_hf_adapters_generation_execution import (
    GENERATION_FAMILIES,
    ReferenceBackend,
    args,
    make_model,
)
from test_hf_adapters_generation_execution import cpu_adapters as cpu_adapters
from torch import nn

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sglang") is None, reason="SGLang is not installed"
)


@pytest.fixture(scope="module", autouse=True)
def sglang_context(tmp_path_factory):
    from sglang.srt.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.runtime_context import SpawnRanks, publish, reset_context
    from sglang.srt.server_args import ServerArgs

    directory = tmp_path_factory.mktemp("sglang-config")
    make_model("qwen3").save_pretrained(directory)
    config = ServerArgs(
        model_path=str(directory),
        model_impl="transformers",
        device="cpu",
        attention_backend="torch_native",
        dtype="float32",
        context_length=64,
        chunked_prefill_size=32,
        max_running_requests=2,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
    )
    publish(config, role="scheduler", ranks=SpawnRanks(world_rank=0, gpu_id=0))
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        backend="gloo",
        distributed_init_method=f"file://{directory / 'rendezvous'}",
    )
    initialize_model_parallel(backend="gloo")
    yield
    destroy_model_parallel()
    destroy_distributed_environment()
    reset_context()


@pytest.mark.parametrize("family", GENERATION_FAMILIES)
@pytest.mark.parametrize("tie_word_embeddings", [True, False], ids=["tied", "untied"])
@torch.no_grad()
def test_checkpoint_loading_and_real_radix_binding(
    cpu_adapters, monkeypatch, tmp_path, family, tie_word_embeddings
):
    from safetensors.torch import load_file
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.radix_attention import RadixAttention
    from transformers import AutoConfig

    from sglang_spyre_backend import hf_adapters_transformers, opaque
    from sglang_spyre_backend.hf_adapters import AdapterBody, prepare_execution

    monkeypatch.setattr(hf_adapters_transformers, "get_server_args", lambda: args())
    original = make_model(family, tie_word_embeddings=tie_word_embeddings)
    ids = torch.tensor([[3, 7, 9, 11]])
    expected = original(ids).logits[0]
    original.save_pretrained(tmp_path)
    config = AutoConfig.from_pretrained(tmp_path)
    wrapper = hf_adapters_transformers.TransformersForCausalLM(config).eval()
    assert wrapper.model.config._attn_implementation == (
        "eager" if family == "gpt_neo" else "sglang"
    )
    assert isinstance(wrapper.model.get_input_embeddings(), nn.Embedding)
    assert not any(isinstance(m, LinearBase) for m in wrapper.model.modules())
    assert wrapper.attention_instances == {}
    loaded = wrapper.load_weights(load_file(tmp_path / "model.safetensors").items())
    assert loaded
    torch.testing.assert_close(
        wrapper.model.get_input_embeddings().weight,
        original.get_input_embeddings().weight,
    )
    assert (
        wrapper.lm_head.weight is wrapper.model.get_input_embeddings().weight
    ) is tie_word_embeddings
    torch.testing.assert_close(
        wrapper.lm_head.weight[: config.vocab_size],
        original.get_output_embeddings().weight,
    )
    with pytest.raises(ValueError, match="Reloading"):
        wrapper.load_weights([])

    prepared = prepare_execution(wrapper.model, max_model_len=64)
    wrapper._bind_adapter_execution(prepared)
    assert len([m for m in wrapper.modules() if isinstance(m, RadixAttention)]) == 2
    assert set(opaque._ATTN_LAYERS) == {"attn.0", "attn.1"}
    backend = ReferenceBackend()
    backend.rows = [(0, p) for p in range(ids.numel())]
    opaque.set_attention_backend(backend)
    wrapper._adapter_body = AdapterBody(prepared, [1, 8], torch.device("cpu"))
    hidden = wrapper._forward_hidden_states(ids[0], torch.arange(ids.numel()), None)
    actual = torch.nn.functional.linear(hidden, wrapper.lm_head.weight) * (
        prepared.runtime.logits_scale or 1.0
    )
    torch.testing.assert_close(
        actual[:, : config.vocab_size], expected, atol=3e-5, rtol=3e-4
    )
    assert not wrapper._can_torch_compile


def test_native_wrapper_keeps_module_replacements(monkeypatch):
    import sglang.srt.models.transformers as upstream
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    from sglang_spyre_backend import hf_adapters_transformers

    monkeypatch.setattr(upstream, "get_device", lambda: "cpu")
    # 5.16 adds embedding_rowwise to HF plans; the pinned upstream wrapper
    # does not understand that style. Exercise its supported linear TP plan.
    monkeypatch.setattr(
        upstream.TransformersBase,
        "_get_model_tp_plan",
        lambda self: {"layers.*.self_attn.q_proj": "colwise"},
    )

    monkeypatch.setattr(
        hf_adapters_transformers,
        "get_server_args",
        lambda: args(json_model_override_args="{}"),
    )
    config = copy.deepcopy(make_model("qwen3").config)
    config.architectures = ["Qwen3ForCausalLM"]
    wrapper = hf_adapters_transformers.TransformersForCausalLM(config)
    assert not wrapper._hf_adapters
    assert wrapper.model.config._attn_implementation == "sglang"
    assert isinstance(wrapper.model.get_input_embeddings(), VocabParallelEmbedding)
    assert any(isinstance(m, LinearBase) for m in wrapper.model.modules())
    assert len(wrapper.attention_instances) == 2
    assert wrapper._can_torch_compile


def test_registry_install_is_idempotent(monkeypatch):
    from sglang.srt.models.registry import ModelRegistry

    from sglang_spyre_backend import hf_adapters_transformers

    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    native = ModelRegistry.models.get("Qwen3ForCausalLM")
    hf_adapters_transformers.install()
    hf_adapters_transformers.install()
    assert (
        ModelRegistry.resolve_model_cls(["TransformersForCausalLM"])[0]
        is hf_adapters_transformers.TransformersForCausalLM
    )
    assert ModelRegistry.models.get("Qwen3ForCausalLM") is native


def test_worker_load_installs_wrapper_before_model_resolution(monkeypatch):
    from sglang.srt.models.registry import ModelRegistry

    from sglang_spyre_backend import custom_ops, hf_adapters_transformers

    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    ModelRegistry.models["TransformersForCausalLM"] = (
        hf_adapters_transformers.SGLangCausalLM
    )
    events = []

    class Runner:
        server_args = args()
        model_config = None

        def load_model(self):
            resolved, _ = ModelRegistry.resolve_model_cls(["TransformersForCausalLM"])
            assert resolved is hf_adapters_transformers.TransformersForCausalLM
            events.append("load")
            self.model = SimpleNamespace(
                _hf_adapters=True,
                prepare_hf_adapters=lambda runner: events.append("prepare"),
            )
            return "loaded"

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.model_runner",
        SimpleNamespace(ModelRunner=Runner),
    )
    monkeypatch.setattr(custom_ops, "_PATCHED_LOAD_MODEL", False)
    custom_ops._install_post_load_hook()
    assert Runner().load_model() == "loaded"
    assert events == ["load", "prepare"]


@pytest.mark.parametrize("fail", [False, True])
def test_warmup_uses_null_page_and_restores_metadata(fail):
    from sglang_spyre_backend.attention import SpyreAttentionBackend

    backend = SpyreAttentionBackend.__new__(SpyreAttentionBackend)
    saved_batch, saved_metadata = object(), {"original": object()}
    backend._batch, backend._metadata = saved_batch, saved_metadata
    backend._adapter_warmed = False
    backend.max_num_batched_tokens = 128
    backend.max_num_seqs = 2
    backend._max_model_len = 384
    backend.block_size = 128
    seen = []

    class Body:
        buckets = [1, 2, 128]

        def __call__(self, ids, positions):
            batch = backend._batch
            assert not batch.block_table.any()
            assert not batch.slot_mapping.any()
            assert batch.query_start_loc[-1] == len(ids)
            assert backend._metadata == {}
            seen.append(len(ids))
            if fail:
                raise RuntimeError("compile failed")
            return torch.zeros(len(ids), 8)

    backend.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            _adapter_body=Body(),
            lm_head=SimpleNamespace(
                quant_method=SimpleNamespace(apply=lambda head, hidden: hidden)
            ),
        )
    )
    if fail:
        with pytest.raises(RuntimeError, match="compile failed"):
            backend._warmup_adapter()
        assert not backend._adapter_warmed
    else:
        backend._warmup_adapter()
        backend._warmup_adapter()
        assert seen == [1, 2, 128]
        assert backend._adapter_warmed
    assert backend._batch is saved_batch
    assert backend._metadata is saved_metadata


@pytest.mark.parametrize(
    "family,mode",
    [(family, "lasttoken") for family in GENERATION_FAMILIES]
    + [
        ("bert", "mean_tokens"),
        ("distilbert", "mean_tokens"),
        ("mpnet", "mean_tokens"),
        ("modernbert", "mean_tokens"),
        ("roberta", "cls_token"),
        ("xlm-roberta", "cls_token"),
    ],
)
@torch.no_grad()
def test_embedding_checkpoint_and_sglang_pooler(
    cpu_adapters, monkeypatch, tmp_path, family, mode
):
    from safetensors.torch import load_file
    from test_hf_adapters_embedding_execution import make_encoder, write_pooling
    from transformers import AutoConfig

    from sglang_spyre_backend import hf_adapters_transformers
    from sglang_spyre_backend.embedding import (
        CacheFreeAdapterAttention,
        EmbeddingBody,
        EmbeddingMask,
    )
    from sglang_spyre_backend.hf_adapters import prepare_execution

    original = make_encoder(family)
    ids = torch.tensor([3, 7, 9, 5, 4])
    hidden = [original(part[None]).last_hidden_state[0] for part in (ids[:3], ids[3:])]
    vectors = [
        h[0] if mode == "cls_token" else h[-1] if mode == "lasttoken" else h.mean(0)
        for h in hidden
    ]
    expected = torch.nn.functional.normalize(torch.stack(vectors), dim=-1)
    original.save_pretrained(tmp_path)
    write_pooling(tmp_path, mode)
    config_args = args(
        model_path=str(tmp_path),
        is_embedding=True,
        attention_backend="spyre_embedding",
        chunked_prefill_size=-1,
        disable_radix_cache=True,
    )
    monkeypatch.setattr(
        hf_adapters_transformers, "get_server_args", lambda: config_args
    )
    wrapper = hf_adapters_transformers.TransformersEmbeddingModel(
        AutoConfig.from_pretrained(tmp_path)
    ).eval()
    assert wrapper.model.config._attn_implementation == (
        "eager" if family in ("gpt_neo", "mpnet") else "sglang"
    )
    assert wrapper.lm_head is None and wrapper.logits_processor is None
    wrapper.load_weights(load_file(tmp_path / "model.safetensors").items())
    prepared = prepare_execution(wrapper.model, max_model_len=64, task="embed")
    mask = EmbeddingMask()
    for executor in prepared.runtime.attentions:
        executor.bind(CacheFreeAdapterAttention(executor.spec, mask))
    wrapper._adapter_body = EmbeddingBody(
        prepared, mask, max_model_len=64, max_requests=2, device="cpu"
    )
    batch = SimpleNamespace(
        extend_seq_lens_cpu=[3, 2],
        extend_seq_lens=torch.tensor([3, 2]),
        extend_prefix_lens_cpu=[0, 0],
        token_type_ids=None,
        dimensions=None,
    )
    actual = wrapper(ids, torch.tensor([0, 1, 2, 0, 1]), batch, get_embedding=True)
    torch.testing.assert_close(actual.embeddings, expected, atol=4e-5, rtol=4e-4)
    batch.extend_prefix_lens_cpu = [1, 0]
    with pytest.raises(ValueError, match="complete requests"):
        wrapper(ids, torch.tensor([0, 1, 2, 0, 1]), batch, get_embedding=True)


@pytest.mark.parametrize("family", ["qwen3", "granite", "llama", "gpt2", "gpt_neo"])
@pytest.mark.parametrize("adapter_mode", [True, False])
def test_construction_delegates_unless_adapter_needs_eager(
    monkeypatch, family, adapter_mode
):
    from sglang.srt.models.transformers import TransformersBase

    from sglang_spyre_backend import hf_adapters_transformers

    original = make_model(family)
    config = copy.deepcopy(original.config)
    config.architectures = None
    sentinel = nn.Identity()
    calls = []

    def upstream_create(wrapper):
        calls.append(wrapper.config)
        return sentinel

    monkeypatch.setattr(TransformersBase, "_create_transformers_model", upstream_create)
    # Exercise only construction, including configs without architecture metadata.
    wrapper = hf_adapters_transformers.TransformersForCausalLM.__new__(
        hf_adapters_transformers.TransformersForCausalLM
    )
    nn.Module.__init__(wrapper)
    wrapper.config = wrapper.text_config = config
    wrapper._hf_adapters = adapter_mode
    created = wrapper._create_transformers_model()
    if adapter_mode and family == "gpt_neo":
        assert not calls
        assert created.config._attn_implementation == "eager"
        assert all(parameter.is_meta for parameter in created.parameters())
        assert not type(created)._supports_attention_backend
    else:
        assert calls == [config]
        assert created is sentinel
        assert config._attn_implementation == original.config._attn_implementation


def test_attention_recording_and_serving_share_inference_context(cpu_adapters):
    from sglang_spyre_backend.attention import SpyreAttentionBackend

    graphs = []

    def compiler(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def kernel(query, output):
        return output.copy_(query + 1)

    compiled = torch.compile(kernel, backend=compiler, fullgraph=True, dynamic=False)

    class Impl:
        buffers = None

        def staging_buffers(self, device):
            assert torch.is_inference_mode_enabled()
            if self.buffers is None:
                self.buffers = (torch.zeros(3, 1, 64), torch.zeros(3, 1, 64))
            return self.buffers

        def forward(self, query, output, cache, metadata):
            compiled(query, output)

    impl = Impl()
    backend = SpyreAttentionBackend.__new__(SpyreAttentionBackend)
    backend.device = torch.device("cpu")
    backend.model_dtype = torch.float32
    backend._impl_of = lambda layer: impl
    backend._layer_groups = {
        0: SimpleNamespace(num_heads=1, num_kv_heads=1, head_size=64)
    }
    backend._kv_caches = {0: None}
    backend._metadata_for = lambda spec: None
    layer = SimpleNamespace(layer_id=0)
    query = torch.arange(128, dtype=torch.float32).reshape(2, 64)
    for inference in (True, False, True, False):
        with torch.inference_mode(inference), torch.no_grad():
            actual = backend._forward_impl(query, None, None, layer, False)
            torch.testing.assert_close(actual, query + 1)
    assert (
        len(graphs) == 1
    ), "Scheduler gradient-mode changes must reuse the attention graph"
