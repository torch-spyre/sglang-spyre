# SPDX-License-Identifier: Apache-2.0
"""Optional HF-adapters preparation and SGLang attention binding."""

from __future__ import annotations

import bisect
import json
import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from hf_adapters.execution import PreparedExecution


def uses_hf_adapters(args) -> bool:
    overrides = json.loads(getattr(args, "json_model_override_args", "{}") or "{}")
    if not isinstance(overrides, dict):
        raise ValueError("json_model_override_args must contain an object")
    mode = overrides.get("spyre_model_preparation", "spyre")
    if mode not in ("spyre", "hf_adapters"):
        raise ValueError(f"Unknown spyre_model_preparation={mode!r}")
    return mode == "hf_adapters"


def validate_config(args, model_config=None) -> None:
    if not uses_hf_adapters(args):
        return
    if getattr(args, "model_impl", None) != "transformers":
        raise ValueError("hf_adapters requires model_impl='transformers'")
    if getattr(args, "device", None) != "cpu":
        raise ValueError("hf_adapters requires device='cpu' for SGLang scheduling")
    embedding = bool(getattr(args, "is_embedding", False))
    backend = "spyre_embedding" if embedding else "spyre_paged"
    if getattr(args, "attention_backend", None) != backend:
        raise ValueError(f"hf_adapters requires attention_backend={backend!r}")
    if embedding:
        count = getattr(args, "max_running_requests", None)
        if type(count) is not int or count <= 0:
            raise ValueError(
                "hf_adapters embeddings require an explicit positive max_running_requests"
            )
        if getattr(args, "chunked_prefill_size", None) != -1:
            raise ValueError("hf_adapters embeddings require chunked_prefill_size=-1")
        if not getattr(args, "disable_radix_cache", False):
            raise ValueError("hf_adapters embeddings require disable_radix_cache=True")
    if any(getattr(args, name, 1) != 1 for name in ("tp_size", "pp_size", "dp_size")):
        raise ValueError("hf_adapters currently requires TP=PP=DP=1")
    if getattr(args, "dtype", "auto") not in ("auto", "float16"):
        raise ValueError("hf_adapters serving currently requires float16")
    if getattr(args, "kv_cache_dtype", "auto") != "auto":
        raise ValueError("hf_adapters requires kv_cache_dtype='auto'")
    for name in (
        "quantization",
        "speculative_algorithm",
        "enable_lora",
        "enable_torch_compile",
        "enable_piecewise_cuda_graph",
        "enable_hierarchical_cache",
        "enable_mixed_chunk",
        "enable_weights_cpu_backup",
    ):
        if getattr(args, name, None):
            raise ValueError(f"hf_adapters does not support {name}")
    if getattr(args, "disaggregation_mode", "null") not in (None, "null"):
        raise ValueError("hf_adapters does not support disaggregated serving")
    if getattr(args, "cpu_offload_gb", 0):
        raise ValueError("hf_adapters does not support CPU weight offload")
    if getattr(args, "load_format", "auto") not in ("auto", "safetensors", "pt"):
        raise ValueError("hf_adapters requires ordinary HF checkpoint loading")
    page = getattr(args, "page_size", 128)
    if not isinstance(page, int) or page <= 0 or page % 64 or page & (page - 1):
        raise ValueError(
            "hf_adapters requires a power-of-two page_size divisible by 64"
        )
    if model_config is not None:
        cfg = model_config.hf_config
        if any(
            getattr(cfg, name, 0) or 0
            for name in (
                "num_local_experts",
                "num_experts",
                "n_routed_experts",
                "moe_intermediate_size",
            )
        ):
            raise ValueError("hf_adapters serving does not support MoE")
        if (
            model_config.is_generation == embedding
            or model_config.is_multimodal
            or cfg is not model_config.hf_text_config
            or getattr(cfg, "is_encoder_decoder", False)
            or getattr(model_config, "is_hybrid", False)
            or getattr(model_config, "is_mla", False)
            or getattr(model_config, "quantization", None)
        ):
            raise ValueError("hf_adapters requires an unquantized dense text backbone")
        if embedding and any(
            "classification" in name.lower() or "reward" in name.lower()
            for name in (getattr(cfg, "architectures", None) or [])
        ):
            raise ValueError("hf_adapters embeddings do not support scoring models")


def prepare_execution(
    model: nn.Module, *, max_model_len: int, task: str = "generate"
) -> PreparedExecution:
    try:
        from hf_adapters.execution import prepare_adapter_execution
    except ImportError as exc:
        raise ImportError(
            "hf_adapters execution requires hf-adapters-spyre with execution.py; "
            "install the integration checkout in the SGLang worker environment"
        ) from exc

    cfg = model.config
    dimensions = (
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        cfg.hidden_size,
        cfg.vocab_size,
    )
    prepared = prepare_adapter_execution(model, max_model_len=max_model_len, task=task)
    if model.config is not cfg or dimensions != (
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        cfg.hidden_size,
        cfg.vocab_size,
    ):
        raise ValueError(
            "Adapter preparation changed model dimensions or config identity"
        )
    attentions = prepared.runtime.attentions
    if [a.spec.layer_idx for a in attentions] != list(range(cfg.num_hidden_layers)):
        raise ValueError("Adapter attention bindings must cover every layer in order")
    sizes = {a.spec.head_size for a in attentions}
    if len(sizes) != 1:
        raise ValueError("Adapter serving requires uniform attention head widths")
    for attention in attentions:
        spec = attention.spec
        if (
            spec.num_heads != dimensions[1]
            or spec.num_kv_heads != dimensions[2]
            or type(spec.head_size) is not int
            or spec.head_size <= 0
            or spec.head_size % 64
            or not math.isfinite(spec.scale)
            or spec.scale <= 0
            or spec.uses_cache != (task == "generate")
            or (
                task == "generate"
                and (not spec.is_causal or spec.position_bias is not None)
            )
        ):
            raise ValueError("Unsupported adapter attention geometry or semantics")
    scale = prepared.runtime.logits_scale
    if scale is not None and (not math.isfinite(scale) or scale <= 0):
        raise ValueError("Adapter logits scale must be finite and positive")
    return prepared


class PagedAdapterAttention(nn.Module):
    """Use Spyre's opaque attention boundary without capturing a ForwardBatch."""

    def __init__(self, attention: nn.Module):
        super().__init__()
        self.attn = attention

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        from sglang_spyre_backend.opaque import spyre_radix_attention_forward

        batch, heads, tokens, width = q.shape
        query = q.transpose(1, 2).reshape(batch * tokens, heads * width)
        key = k.transpose(1, 2).reshape(batch * tokens, -1)
        value = v.transpose(1, 2).reshape(batch * tokens, -1)
        output = spyre_radix_attention_forward(self.attn, query, key, value)
        return output.reshape(batch, tokens, heads, width).transpose(1, 2)


class AdapterBody:
    """Bucket the adapter driver, whose calls bypass the stock HF forward."""

    def __init__(
        self, prepared: PreparedExecution, buckets: list[int], device: torch.device
    ):
        if not buckets or any(type(n) is not int or n <= 0 for n in buckets):
            raise ValueError("Adapter token buckets must be positive integers")
        self.prepared = prepared
        self.buckets = sorted(set(buckets))
        self.device = device

    def __call__(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        if input_ids.ndim != 1 or positions.shape != input_ids.shape:
            raise ValueError(
                "Adapter serving expects packed 1-D token IDs and positions"
            )
        tokens = input_ids.numel()
        if tokens == 0:
            raise ValueError("Adapter serving requires at least one token")
        index = bisect.bisect_left(self.buckets, tokens)
        if index == len(self.buckets):
            raise ValueError(
                f"{tokens} tokens exceed adapter bucket limit {self.buckets[-1]}"
            )
        padding = self.buckets[index] - tokens
        ids = torch.nn.functional.pad(input_ids, (0, padding)).to(
            self.device, torch.int64
        )
        pos = torch.nn.functional.pad(positions, (0, padding)).to(
            self.device, torch.int64
        )
        hidden = self.prepared.forward(ids.unsqueeze(0), pos.unsqueeze(0))[0]
        # Match the native boundary: SGLang selects rows on CPU before the Spyre head.
        return hidden.to("cpu")[:tokens]
