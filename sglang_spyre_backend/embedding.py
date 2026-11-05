# SPDX-License-Identifier: Apache-2.0
"""Cache-free adapter attention, rectangular input packing, and pooling metadata."""

from __future__ import annotations

import bisect
import json
from pathlib import Path, PurePosixPath

import torch
from torch import nn
from torch.nn import functional as F

from sglang_spyre_backend.custom_ops.utils import direct_register_custom_op


def _dense_attention(q, k, v, mask, scale: float):
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, enable_gqa=True
    ).contiguous()


_compiled_attention = torch.compile(_dense_attention, dynamic=False, fullgraph=True)


def _embedding_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float
) -> torch.Tensor:
    kernel = _compiled_attention if q.device.type == "spyre" else _dense_attention
    return kernel(q, k, v, mask, scale)


def _embedding_attention_fake(q, k, v, mask, scale):
    return q.new_empty(q.shape)


direct_register_custom_op(
    op_name="embedding_attention",
    op_func=_embedding_attention,
    fake_impl=_embedding_attention_fake,
)


def pooling_config(args) -> tuple[str, bool]:
    from transformers.utils.hub import cached_file

    def read_json(name):
        path = cached_file(
            args.model_path,
            name,
            revision=getattr(args, "revision", None),
            _raise_exceptions_for_missing_entries=False,
        )
        return json.loads(Path(path).read_text()) if path is not None else None

    overrides = json.loads(args.json_model_override_args or "{}")
    modules = read_json("modules.json")
    pooling, normalize = None, True
    if modules is not None:
        kinds = [module["type"] for module in modules]
        expected = [
            "sentence_transformers.models.Transformer",
            "sentence_transformers.models.Pooling",
        ]
        if kinds not in (
            expected,
            [*expected, "sentence_transformers.models.Normalize"],
        ):
            raise ValueError(
                "Embedding adapters support Transformer/Pooling/Normalize pipelines only"
            )
        if modules[0]["path"] != "":
            raise ValueError(
                "Embedding adapters require the Transformer at the checkpoint root"
            )
        path = PurePosixPath(modules[1]["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid pooling metadata path")
        metadata = read_json(str(path / "config.json"))
        if not isinstance(metadata, dict):
            raise ValueError("Missing SentenceTransformers pooling metadata")
        modes = [
            key
            for key, value in metadata.items()
            if key.startswith("pooling_mode_") and value is True
        ]
        supported = {
            "pooling_mode_cls_token": "CLS",
            "pooling_mode_mean_tokens": "MEAN",
            "pooling_mode_lasttoken": "LAST",
        }
        if (
            len(modes) != 1
            or modes[0] not in supported
            or metadata.get("include_prompt", True) is not True
        ):
            raise ValueError(
                "Embedding adapters require one CLS/MEAN/LAST mode with include_prompt=True"
            )
        pooling = supported[modes[0]]
        normalize = len(modules) == 3
    pooling = overrides.get("pooling_type", pooling)
    normalize = overrides.get("normalize", normalize)
    if pooling not in ("CLS", "MEAN", "LAST") or type(normalize) is not bool:
        raise ValueError(
            "Specify pooling_type=CLS/MEAN/LAST and a boolean normalize override for a bare backbone"
        )
    return pooling, normalize


class EmbeddingMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("value", torch.zeros(1, 1, 1, 1), persistent=False)


class CacheFreeAdapterAttention(nn.Module):
    def __init__(self, spec, mask: EmbeddingMask):
        super().__init__()
        if spec.uses_cache:
            raise ValueError("Embedding attention cannot use a persistent KV cache")
        self.scale = spec.scale
        self.mask = mask
        self.position_bias = spec.position_bias

    def forward(self, q, k, v):
        mask = self.mask.value
        if self.position_bias is not None:
            width = q.shape[-2]
            mask = mask + self.position_bias()[..., :width, :width]
        return torch.ops.sglang_spyre.embedding_attention(q, k, v, mask, self.scale)


class EmbeddingBody:
    """Pack independent requests into padded rectangles for the shared adapter driver."""

    def __init__(self, prepared, mask, *, max_model_len, max_requests, device):
        if max_model_len <= 0 or max_requests <= 0:
            raise ValueError("Embedding context and request limits must be positive")
        self.prepared = prepared
        self.mask = mask
        self.device = torch.device(device)
        self.dtype = next(prepared.model.parameters()).dtype
        self.max_model_len = max_model_len
        self.token_buckets = sorted(
            {min(n, max_model_len) for n in (64, 128, 256, 512, max_model_len)}
        )
        self.batch_buckets = sorted(
            {
                1,
                max_requests,
                *(
                    2**n
                    for n in range(max_requests.bit_length())
                    if 2**n <= max_requests
                ),
            }
        )
        self.position_offset = 0
        if prepared.encoder_only:
            embeddings = getattr(prepared.model, "embeddings", None)
            position_table = getattr(embeddings, "position_embeddings", None)
            padding_idx = getattr(position_table, "padding_idx", None)
            if padding_idx is not None:
                self.position_offset = padding_idx + 1

    @torch.no_grad()
    def __call__(self, input_ids, positions, lengths, token_type_ids=None):
        lengths = [int(n) for n in lengths]
        if not lengths or any(n <= 0 or n > self.max_model_len for n in lengths):
            raise ValueError(
                "Embedding requests must have positive lengths within the context limit"
            )
        if len(lengths) > self.batch_buckets[-1]:
            raise ValueError("Embedding batch exceeds max_running_requests")
        total = sum(lengths)
        if (
            input_ids.ndim != 1
            or input_ids.numel() != total
            or positions.shape != input_ids.shape
        ):
            raise ValueError("Embedding inputs must be complete, packed 1-D requests")
        if token_type_ids is not None and token_type_ids.shape != input_ids.shape:
            raise ValueError("token_type_ids must match packed input_ids")
        batch = self.batch_buckets[bisect.bisect_left(self.batch_buckets, len(lengths))]
        width = self.token_buckets[bisect.bisect_left(self.token_buckets, max(lengths))]
        ids = torch.zeros(batch, width, dtype=torch.int64)
        pos = torch.full_like(ids, max(self.position_offset - 1, 0))
        types = torch.zeros_like(ids) if token_type_ids is not None else None
        real_lengths = torch.ones(batch, dtype=torch.int64)
        start = 0
        for row, length in enumerate(lengths):
            ids[row, :length] = input_ids[start : start + length].cpu()
            pos[row, :length] = (
                positions[start : start + length].cpu() + self.position_offset
            )
            if self.position_offset:
                padding_idx = self.position_offset - 1
                padding = ids[row, :length] == padding_idx
                pos[row, :length] -= padding.cumsum(0)
                pos[row, :length].masked_fill_(padding, padding_idx)
            if types is not None:
                types[row, :length] = token_type_ids[start : start + length].cpu()
            real_lengths[row] = length
            start += length
        allowed = torch.arange(width)[None, :] < real_lengths[:, None]
        allowed = allowed[:, None, None, :].expand(batch, 1, width, width)
        if self.prepared.runtime.is_causal:
            allowed = allowed & (
                torch.arange(width)[:, None] >= torch.arange(width)[None, :]
            )
        self.mask.value = torch.where(allowed, 0.0, -10000.0).to(
            device=self.device, dtype=self.dtype
        )
        hidden = (
            self.prepared.forward(
                ids.to(self.device),
                pos.to(self.device),
                types.to(self.device) if types is not None else None,
            )
            .to("cpu")
            .float()
        )
        return torch.cat(
            [hidden[row, :length] for row, length in enumerate(lengths)], dim=0
        )

    @torch.no_grad()
    def warmup(self):
        for batch in self.batch_buckets:
            for width in self.token_buckets:
                ids = torch.zeros(batch * width, dtype=torch.int64)
                positions = torch.arange(width).repeat(batch)
                self(ids, positions, [width] * batch)
