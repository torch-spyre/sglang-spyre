# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SGLang adapter for spyre-inference's head-major paged attention.

The device kernels are ported verbatim in :mod:`sglang_spyre_backend.ops` and
driven by :mod:`sglang_spyre_backend.spyre_attention`; this module only turns
SGLang's ``ForwardBatch`` into the four tensors spyre-inference's builder reads
(query start offsets, sequence lengths, the page table and the slot mapping).
Baseline: spyre-inference 0ccdbf6 (2026-10-30).

Layers are grouped as spyre-inference groups them (#874, #960, #1317): one
metadata builder per distinct (sliding window, KV heads, head size, query
heads), and one KV page set per layer. Gemma-4 therefore gets one group for its
sliding-window layers and one for its full-attention layers, each with its own
masks, buckets and recorded kernels. SGLang's page size is the KV block size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

from sglang_spyre_backend import envs
from sglang_spyre_backend.custom_ops.utils import convert
from sglang_spyre_backend.spyre_attention import (
    GroupSpec,
    SeqBatch,
    SpyreAttentionMetadata,
    SpyreAttentionMetadataBuilder,
    SpyreHeadMajorAttention,
    SpyrePagedKVCache,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# SGLang reserves page 0 for padded tokens' dummy writes (PagedTokenToKVPoolAllocator
# .clear), the same role vLLM's null block plays for spyre-inference.
_NULL_SLOT = 0


@dataclass(frozen=True)
class _BucketerConfig:
    block_size: int
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    min_num_kv_heads: int


def _ceil_to(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class SpyreAttentionBackend(AttentionBackend):
    """spyre-inference head-major paged attention behind SGLang's backend API.

    Decoder self-attention, TP=1. Batches of several sequences are accepted by the
    builder and the batched decode kernel, as in spyre-inference; the engine
    defaults keep ``max_running_requests=1`` for the validated configuration.
    """

    needs_cpu_seq_lens: bool = True

    def __init__(self, model_runner):
        super().__init__()
        if envs.SPYRE_ATTN_KV_LAYOUT != "head_major":
            raise NotImplementedError(
                "The SGLang Spyre backend ports only the head-major KV layout, "
                "spyre-inference's default; unset SPYRE_ATTN_KV_LAYOUT."
            )
        self.model_runner = model_runner
        self.device = torch.device("spyre")
        self.model_dtype = torch.float16
        pool = model_runner.token_to_kv_pool
        self.block_size = int(pool.page_size)
        if self.block_size % 64:
            raise ValueError(
                "Spyre paged attention requires page_size to be a multiple of 64; "
                f"got {self.block_size}."
            )
        self.num_pages = _ceil_to(int(pool.size) + self.block_size, self.block_size) // (
            self.block_size
        )
        server_args = model_runner.server_args
        max_model_len = int(model_runner.model_config.context_len)
        chunk = getattr(server_args, "chunked_prefill_size", None)
        budget = chunk if chunk and chunk > 0 else max_model_len
        self.max_num_batched_tokens = int(min(budget, max_model_len))
        self.max_num_seqs = int(getattr(server_args, "max_running_requests", None) or 1)
        self._max_model_len = max_model_len
        # Constant for the run so the kernels never see the graph's token count;
        # +1 keeps every gather a strict subset (torch-spyre#4033).
        self.staging_rows = max(self.max_num_batched_tokens, self.max_num_seqs) + 1
        self._groups: dict[GroupSpec, SpyreAttentionMetadataBuilder] = {}
        self._impls: dict[int, SpyreHeadMajorAttention] = {}
        self._layer_groups: dict[int, GroupSpec] = {}
        self._kv_caches: dict[int, SpyrePagedKVCache] = {}
        self._metadata: dict[GroupSpec, SpyreAttentionMetadata] = {}
        self._batch: SeqBatch | None = None
        self._recorded: set[tuple] = set()
        self._adapter_warmed = False
        from sglang_spyre_backend import opaque

        opaque.set_attention_backend(self)
        self._warmup_adapter()

    # ------------------------------------------------------------------
    # Per-step metadata
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens_cpu
        if seq_lens is None:
            seq_lens = forward_batch.seq_lens.to("cpu")
        seq_lens = seq_lens[:bs].to(torch.int32)
        if forward_batch.forward_mode.is_decode():
            query_lens = torch.ones(bs, dtype=torch.int32)
        else:
            extend = forward_batch.extend_seq_lens_cpu
            if extend is None:
                extend = forward_batch.extend_seq_lens.to("cpu").tolist()
            query_lens = torch.tensor(list(extend)[:bs], dtype=torch.int32)
        query_start_loc = torch.zeros(bs + 1, dtype=torch.int32)
        query_start_loc[1:] = torch.cumsum(query_lens, 0)

        # As wide as vLLM's block table (ceil(max_model_len / block_size)), so the
        # builder's padded block counts index real columns; padding is page 0.
        max_blocks = _ceil_to(self._max_model_len, self.block_size) // self.block_size
        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        req_idx = forward_batch.req_pool_indices.to("cpu")
        block_table = torch.zeros(bs, max_blocks, dtype=torch.int32)
        for s in range(bs):
            n = _ceil_to(int(seq_lens[s]), self.block_size) // self.block_size
            slots = req_to_token[int(req_idx[s]), : int(seq_lens[s]) : self.block_size]
            block_table[s, :n] = (slots.to("cpu")[:n] // self.block_size).to(torch.int32)
        num_tokens = int(query_start_loc[-1])
        slot_mapping = forward_batch.out_cache_loc[:num_tokens].to(
            device="cpu", dtype=torch.int64
        )
        self._batch = SeqBatch(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )
        self._metadata = {}

    @torch.no_grad()
    def _warmup_adapter(self) -> None:
        """Exercise adapter body/head buckets using only the allocator's null page."""
        model = getattr(self.model_runner, "model", None)
        body = getattr(model, "_adapter_body", None)
        if body is None or self._adapter_warmed:
            return
        saved_batch, saved_metadata = self._batch, self._metadata
        try:
            for tokens in body.buckets:
                if tokens > self.max_num_batched_tokens and tokens > self.max_num_seqs:
                    raise ValueError("Adapter token bucket exceeds attention staging capacity")
                # Decode buckets may contain more requests than the context length.
                decode = tokens <= self.max_num_seqs
                lengths = torch.ones(tokens, dtype=torch.int32) if decode else torch.tensor([tokens], dtype=torch.int32)
                if not decode and tokens > self._max_model_len:
                    raise ValueError("Adapter prefill bucket exceeds context length")
                starts = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).to(torch.int32)))
                self._batch = SeqBatch(
                    query_start_loc=starts,
                    seq_lens=lengths,
                    block_table=torch.zeros(
                        len(lengths), _ceil_to(self._max_model_len, self.block_size) // self.block_size,
                        dtype=torch.int32,
                    ),
                    slot_mapping=torch.zeros(tokens, dtype=torch.int64),
                )
                self._metadata = {}
                ids = torch.zeros(tokens, dtype=torch.int64)
                positions = torch.zeros_like(ids) if decode else torch.arange(tokens, dtype=torch.int64)
                hidden = body(ids, positions)
                model.lm_head.quant_method.apply(model.lm_head, hidden)
            self._adapter_warmed = True
        finally:
            self._batch, self._metadata = saved_batch, saved_metadata

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ------------------------------------------------------------------
    # Layer groups
    # ------------------------------------------------------------------

    def _group_of(self, layer: RadixAttention) -> GroupSpec:
        spec = self._layer_groups.get(layer.layer_id)
        if spec is not None:
            return spec
        if layer.qk_head_dim != layer.v_head_dim:
            raise NotImplementedError("Spyre requires equal QK and V head dimensions")
        window = getattr(layer, "sliding_window_size", -1)
        # SGLang stores an inclusive window minus one (HF's convention); the Spyre
        # builder takes the window length a query sees, spyre-inference's convention.
        sliding_window = None if window is None or window < 0 else int(window) + 1
        spec = GroupSpec(
            num_heads=layer.tp_q_head_num,
            num_kv_heads=layer.tp_k_head_num,
            head_size=layer.qk_head_dim,
            sliding_window=sliding_window,
        )
        if spec not in self._groups:
            self._groups[spec] = SpyreAttentionMetadataBuilder(
                spec,
                _BucketerConfig(
                    block_size=self.block_size,
                    max_model_len=self._max_model_len,
                    max_num_batched_tokens=self.max_num_batched_tokens,
                    max_num_seqs=self.max_num_seqs,
                    min_num_kv_heads=spec.num_kv_heads,
                ),
                self.model_dtype,
            )
        self._layer_groups[layer.layer_id] = spec
        return spec

    def _impl_of(self, layer: RadixAttention) -> SpyreHeadMajorAttention:
        impl = self._impls.get(layer.layer_id)
        if impl is None:
            spec = self._group_of(layer)
            impl = SpyreHeadMajorAttention(
                num_heads=spec.num_heads,
                num_kv_heads=spec.num_kv_heads,
                head_size=spec.head_size,
                scale=float(layer.scaling),
                block_size=self.block_size,
                staging_rows=self.staging_rows,
                logits_soft_cap=float(getattr(layer, "logit_cap", 0.0) or 0.0),
                model_dtype=self.model_dtype,
            )
            self._kv_caches[layer.layer_id] = SpyreHeadMajorAttention.allocate_pages(
                self.num_pages,
                spec.num_kv_heads,
                self.block_size,
                spec.head_size,
                self.model_dtype,
                self.device,
            )
            impl.staging_buffers(self.device)
            self._impls[layer.layer_id] = impl
            self._maybe_record(layer, impl, spec)
        return impl

    def _maybe_record(self, layer, impl: SpyreHeadMajorAttention, spec: GroupSpec) -> None:
        """Record every attention variant of a group once, at its first layer (#707, #1381)."""
        if not envs.SPYRE_ATTN_RECORD:
            return
        key = (spec, impl.graph_key())
        if key in self._recorded:
            return
        self._recorded.add(key)
        logger.info("Recording Spyre attention graphs for layer %d (%s)...", layer.layer_id, spec)
        impl.record_graphs(self._kv_caches[layer.layer_id], self._groups[spec])

    def _metadata_for(self, spec: GroupSpec) -> SpyreAttentionMetadata:
        md = self._metadata.get(spec)
        if md is None:
            assert self._batch is not None, "init_forward_metadata must run first"
            md = self._groups[spec].build(self._batch)
            self._metadata[spec] = md
        return md

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        return self._forward_impl(q, k, v, layer, save_kv_cache)

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        return self._forward_impl(q, k, v, layer, save_kv_cache)

    @torch.inference_mode()
    def _forward_impl(self, q, k, v, layer, save_kv_cache: bool) -> torch.Tensor:
        # Recording runs during the first opaque attention call. Normalize the
        # context here so warmup and serving use the same tensor dispatch keys.
        impl = self._impl_of(layer)
        spec = self._layer_groups[layer.layer_id]
        md = self._metadata_for(spec)
        kv_cache = self._kv_caches[layer.layer_id]
        num_heads, num_kv_heads, head_size = spec.num_heads, spec.num_kv_heads, spec.head_size
        original_device = q.device
        # The body pads its tokens to a bucket (model_wrapper._BodyBuckets); rows past
        # the batch are written to the null slot and their outputs are discarded, as
        # spyre-inference pads its slot mapping to the body bucket.
        rows = int(q.shape[0])

        def to_dev(t, heads):
            return convert(t.reshape(rows, heads, head_size), device=self.device, dtype=self.model_dtype)

        if save_kv_cache and k is not None and v is not None:
            if md.kv_write_index is None or md.kv_write_index[0].shape[0] != rows:
                assert self._batch is not None
                slots = self._batch.slot_mapping.clamp(min=_NULL_SLOT)
                if slots.shape[0] < rows:
                    slots = torch.nn.functional.pad(slots, (0, rows - slots.shape[0]), value=_NULL_SLOT)
                md.kv_write_index = impl.kv_write_index(slots[:rows], self.device)
            impl.do_kv_cache_update(
                to_dev(k, num_kv_heads), to_dev(v, num_kv_heads), kv_cache, md.kv_write_index
            )

        q_staging, out_staging = impl.staging_buffers(self.device)
        q_staging[:rows] = to_dev(q, num_heads)
        impl.forward(q_staging, out_staging, kv_cache, md)
        # An offset-0 prefix cloned on the device, as spyre-inference reads its
        # staging buffer back (spyre_attn._run_batched_decode_dispatch); a host
        # copy of the sliced row-outermost view itself does not read back reliably.
        out = out_staging[:rows].clone().reshape(rows, num_heads * head_size)
        if original_device.type != "spyre":
            out = convert(out, device=original_device, dtype=q.dtype)
        return out


__all__ = ["SpyreAttentionBackend"]
