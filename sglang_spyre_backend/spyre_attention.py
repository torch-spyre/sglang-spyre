# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Head-major paged attention on Spyre, framework-neutral half.

Ported from ``spyre_inference/v1/attention/backends/spyre_attn.py`` and
``backends/spyre_head_major_attn.py`` at
``0ccdbf6ac2da7faacf98db93464b274fb8f40126``. The kernels themselves live,
unchanged, in :mod:`sglang_spyre_backend.ops`; this module carries the code
around them that does not depend on vLLM:

- ``SpyreAttentionMetadataBuilder``: the host-side mask tiles (causal, KV
  padding and the sliding-window block skip of #752/#1021), the page-index
  tables, the query row tables (#859, #879) and the batched-decode precomputes
  (#612, #932, #1009, #1068) that bs1 decode now takes. Its input is
  ``SeqBatch``, the four tensors vLLM's ``CommonAttentionMetadata`` contributes.
- ``SpyreHeadMajorAttention``: one layer's KV pages, the traced KV write and the
  kernel dispatch, with the core-cap context fix of #1413.

Each layer group (same sliding window, KV-head count and head size; #874/#960)
gets its own builder, so gemma-4's sliding and full layers keep their own masks
and buckets.
"""

from __future__ import annotations

import logging
import time
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, TypeVar

import torch

from sglang_spyre_backend import envs
from sglang_spyre_backend.custom_ops.utils import convert, row_outermost_layout
from sglang_spyre_backend.ops import tile_loop
from sglang_spyre_backend.ops.batched_decode_head_major import (
    batched_decode_head_major_kernel,
)
from sglang_spyre_backend.ops.layout import (
    INT32_ELEMS_PER_STICK,
    head_major_kv_layout,
    temporary_chunk_major_page_index_layout,
)
from sglang_spyre_backend.ops.page_attn_head_major_decode import (
    page_attn_head_major_decode_kernel,
)
from sglang_spyre_backend.ops.page_attn_head_major_prefill import (
    page_attn_head_major_prefill_kernel,
)
from sglang_spyre_backend.ops.reshape_and_cache_head_major import (
    reshape_and_cache_head_major_kernel,
)
from sglang_spyre_backend.ops.tile_loop import USE_FOR_EACH_TILE
from sglang_spyre_backend.spyre_attn_bucketer import (
    BucketerConfig,
    SpyreAttnBatchedDecodeBucket,
    SpyreAttnBucket,
    SpyreAttnBucketer,
    batched_decode_chunking,
)

logger = logging.getLogger(__name__)

# Compiled at module scope so every layer of a group shares one Dynamo entry per
# shape; the kernels that walk pages tiled need fullgraph for ``for_each_tile``.
_page_attn_prefill_compiled = torch.compile(
    page_attn_head_major_prefill_kernel, dynamic=False, fullgraph=USE_FOR_EACH_TILE
)
_page_attn_decode_compiled = torch.compile(
    page_attn_head_major_decode_kernel, dynamic=False, fullgraph=USE_FOR_EACH_TILE
)
_batched_decode_compiled = torch.compile(
    batched_decode_head_major_kernel, dynamic=False, fullgraph=USE_FOR_EACH_TILE
)
# Always compiled: eager index_copy_ rejects an int32 index and falls back to CPU.
_reshape_compiled = torch.compile(reshape_and_cache_head_major_kernel, dynamic=False)

_SPYRE_CORES = 32
_LX_ATTN_CORES = 8
_T = TypeVar("_T")


def _lx_max_cores(output_units: int) -> int:
    """Core cap for one LX attention compile, 0 for uncapped (upstream #893)."""
    override = envs.SPYRE_ATTN_MAX_CORES
    if override:
        return override
    return 0 if output_units >= _SPYRE_CORES else _LX_ATTN_CORES


def _run_with_capped_cores(output_units: int, fn: Callable[..., _T], *args) -> _T:
    # work_division reads config.sencores per compile, which is what keeps the rest of
    # the model uncapped. Context-local, so the caller's unset state survives (#1413).
    max_cores = _lx_max_cores(output_units)
    if not max_cores:
        return fn(*args)
    from torch_spyre._inductor import config as ts_config

    def run():
        ts_config.sencores = max_cores
        return fn(*args)

    return copy_context().run(run)


class SpyrePagedKVCache(NamedTuple):
    """One layer's paged K/V tensors, ``[num_pages, num_kv_heads, block_size, D]``."""

    k_pages: torch.Tensor
    v_pages: torch.Tensor


@dataclass
class SeqBatch:
    """What the builder reads from the scheduler, in vLLM's naming.

    ``query_start_loc`` [num_seqs + 1] and ``seq_lens`` [num_seqs] are host int32;
    ``block_table`` [num_seqs, max_blocks] holds physical page ids; ``slot_mapping``
    [num_tokens] holds the KV slot of every new token (page * block_size + offset).
    """

    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True

    @property
    def num_seqs(self) -> int:
        return int(self.seq_lens.shape[0])


@dataclass
class SpyreAttentionMetadata:
    """Per-step metadata for one layer group; see upstream ``SpyreAttentionMetadata``."""

    num_seqs: int
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    block_size: int
    apply_causal_mask: bool
    attention_mask_stacks: list[torch.Tensor]
    active_block_indices: list[list[int]] | None
    aligned_query_lens: list[int]
    padded_num_blocks: list[int] | None
    page_index_tables_cpu: list[torch.Tensor]
    num_decode_seqs: int = 0
    padded_num_seqs: int | None = None
    padded_batch_blocks: int | None = None
    blocks_per_chunk: int | None = None
    rep_row_ids_cpu: torch.Tensor | None = None
    chunk_page_ids_cpu: torch.Tensor | None = None
    mask_by_chunk_cpu: torch.Tensor | None = None
    # Device mirrors, filled by the step's first layer and shared by the group.
    rep_row_ids_dev: torch.Tensor | None = None
    chunk_page_ids_dev: torch.Tensor | None = None
    mask_by_chunk_dev: torch.Tensor | None = None
    kernel_index_tables: list[torch.Tensor] | None = None
    query_row_tables: list[torch.Tensor] | None = None
    attention_mask_stacks_device: list[torch.Tensor | None] | None = None
    # One per step: the KV write index, published for the whole group.
    kv_write_index: list[torch.Tensor] | None = None
    extra: dict = field(default_factory=dict)


def _build_query_row_tables(
    attn_metadata: SpyreAttentionMetadata, device: torch.device
) -> list[torch.Tensor]:
    """Per-sequence query gather/store rows, each converted on its own (torch-spyre#3851)."""
    num_seqs = attn_metadata.num_seqs
    starts = attn_metadata.query_start_loc[:num_seqs].cpu()
    lens = attn_metadata.query_start_loc[1 : num_seqs + 1].cpu() - starts
    tables = []
    for s, aligned in enumerate(attn_metadata.aligned_query_lens):
        last_real = max(int(lens[s]) - 1, 0)
        row = (starts[s] + torch.arange(aligned).clamp(max=last_real)).to(torch.int32)
        tables.append(convert(row, device=device))
    return tables


def _mirror_mask_stack(
    attn_metadata: SpyreAttentionMetadata, seq_idx: int, device: torch.device
) -> torch.Tensor:
    """One sequence's device mask stack, mirrored at most once per step (#977)."""
    stacks_device = attn_metadata.attention_mask_stacks_device
    if stacks_device is None:
        stacks_device = [None] * attn_metadata.num_seqs
        attn_metadata.attention_mask_stacks_device = stacks_device
    stack_device = stacks_device[seq_idx]
    if stack_device is None:
        stack_device = convert(attn_metadata.attention_mask_stacks[seq_idx], device=device)
        stacks_device[seq_idx] = stack_device
    return stack_device


@dataclass(frozen=True)
class GroupSpec:
    """The layer-group key: what the builder's masks and buckets depend on."""

    num_heads: int
    num_kv_heads: int
    head_size: int
    sliding_window: int | None


class SpyreAttentionMetadataBuilder:
    """Host-side metadata for one layer group (upstream ``SpyreAttentionMetadataBuilder``)."""

    def __init__(
        self,
        spec: GroupSpec,
        bucketer_config: BucketerConfig,
        model_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.spec = spec
        self.block_size = bucketer_config.block_size
        if self.block_size % 64 != 0:
            raise ValueError(
                "block_size must be a multiple of 64 for the Spyre paged attention "
                f"backend. Got block_size={self.block_size}."
            )
        self.num_heads = spec.num_heads
        self.num_kv_heads = spec.num_kv_heads
        self.head_size = spec.head_size
        self.sliding_window = spec.sliding_window
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {self.sliding_window}")
        self.model_dtype = model_dtype
        self._zero_tiles: dict[int, torch.Tensor] = {}
        self._masked_tiles: dict[int, torch.Tensor] = {}
        self._attn_bucketer = SpyreAttnBucketer(bucketer_config)

    @property
    def attn_bucketer(self) -> SpyreAttnBucketer:
        return self._attn_bucketer

    def _get_zero_tile(self, aligned_query_len: int) -> torch.Tensor:
        tile = self._zero_tiles.get(aligned_query_len)
        if tile is None:
            tile = torch.zeros((aligned_query_len, self.block_size), dtype=self.model_dtype)
            self._zero_tiles[aligned_query_len] = tile
        return tile

    def _get_masked_tile(self, aligned_query_len: int) -> torch.Tensor:
        tile = self._masked_tiles.get(aligned_query_len)
        if tile is None:
            tile = torch.full(
                (aligned_query_len, self.block_size),
                torch.finfo(self.model_dtype).min,
                dtype=self.model_dtype,
            )
            self._masked_tiles[aligned_query_len] = tile
        return tile

    def _pad_num_blocks(self, num_blocks: int) -> int:
        if num_blocks == 0:
            return 0
        padded = self._attn_bucketer.find_blocks_bucket(num_blocks)
        assert padded is not None, (
            f"num_blocks={num_blocks} exceeds the largest recorded bucket "
            f"{self._attn_bucketer.num_blocks_buckets[-1]}, which should cover "
            "ceil(max_model_len / block_size)."
        )
        return padded

    def _max_active_blocks(self, num_blocks: int, aligned_query_len: int) -> int:
        if num_blocks == 0:
            return 0
        assert self.sliding_window is not None
        span = self.sliding_window + aligned_query_len - 1
        bound = min(num_blocks, (span + self.block_size - 1) // self.block_size + 1)
        assert bound >= 1
        return bound

    def _build_attention_mask(
        self,
        seq_lens: torch.Tensor,
        query_lens: torch.Tensor,
        apply_causal_mask: bool,
        aligned_query_len: int,
        aligned_max_seq_len: int,
    ) -> torch.Tensor:
        """Additive mask for the non-sliding-window path, vectorized over sequences."""
        assert self.sliding_window is None
        q_pos = torch.arange(aligned_query_len)
        kv_pos = torch.arange(aligned_max_seq_len)
        q_pos = torch.minimum(q_pos.unsqueeze(0), (query_lens - 1).clamp(min=0).unsqueeze(1))
        kv_valid = kv_pos.unsqueeze(0) < seq_lens.unsqueeze(1)
        attend = kv_valid.unsqueeze(1).expand(-1, aligned_query_len, -1)
        if apply_causal_mask:
            context_lens = seq_lens - query_lens
            causal_limit = (context_lens.unsqueeze(1) + q_pos).unsqueeze(2)
            attend = attend & (kv_pos.unsqueeze(0).unsqueeze(0) <= causal_limit)
        return torch.where(
            ~attend,
            torch.tensor(torch.finfo(self.model_dtype).min, dtype=self.model_dtype),
            torch.tensor(0.0, dtype=self.model_dtype),
        )

    def _build_single_tile(
        self,
        block_idx: int,
        kv_len: int,
        query_len: int,
        context_len: int,
        aligned_query_len: int,
        apply_causal_mask: bool,
    ) -> torch.Tensor:
        """The mask tile of one boundary block on the sliding-window path."""
        block_size = self.block_size
        kv_start = block_idx * block_size
        q_pos = torch.arange(aligned_query_len).clamp(max=max(query_len - 1, 0))
        kv_pos = torch.arange(kv_start, kv_start + block_size)
        attend = (kv_pos < kv_len).unsqueeze(0).expand(aligned_query_len, -1)
        if apply_causal_mask:
            attend = attend & (kv_pos.unsqueeze(0) <= (context_len + q_pos).unsqueeze(1))
        assert self.sliding_window is not None
        window_start = (context_len + q_pos - self.sliding_window + 1).clamp(min=0)
        attend = attend & (kv_pos.unsqueeze(0) >= window_start.unsqueeze(1))
        return torch.where(
            ~attend,
            torch.tensor(torch.finfo(self.model_dtype).min, dtype=self.model_dtype),
            torch.tensor(0.0, dtype=self.model_dtype),
        )

    def _build_active_tiles_with_skip(
        self,
        kv_len: int,
        query_len: int,
        context_len: int,
        aligned_query_len: int,
        apply_causal_mask: bool,
    ) -> tuple[list[int], list[torch.Tensor]]:
        """(active block indices, mask tiles) under a sliding window, by arithmetic skip."""
        assert self.sliding_window is not None
        block_size = self.block_size
        num_blocks = (kv_len + block_size - 1) // block_size
        earliest_window_start = max(0, context_len - self.sliding_window + 1)
        latest_window_start = max(0, kv_len - self.sliding_window)
        first_active = earliest_window_start // block_size
        last_lower_boundary = latest_window_start // block_size
        if apply_causal_mask:
            last_causal_interior = (context_len + 1) // block_size - 1
        else:
            last_causal_interior = num_blocks - 1
        last_block = num_blocks - 1
        active_bs = list(range(first_active, num_blocks))
        if not active_bs:
            return [], []
        zero_tile = self._get_zero_tile(aligned_query_len)
        tiles: list[torch.Tensor] = []
        for b in active_bs:
            is_lower_boundary = b <= last_lower_boundary
            is_upper_boundary = (b == last_block) and not is_lower_boundary
            is_causal_boundary = apply_causal_mask and b > last_causal_interior and b != last_block
            if is_lower_boundary or is_upper_boundary or is_causal_boundary:
                tiles.append(
                    self._build_single_tile(
                        b, kv_len, query_len, context_len, aligned_query_len, apply_causal_mask
                    )
                )
            else:
                tiles.append(zero_tile)
        return active_bs, tiles

    def build(self, batch: SeqBatch, *, per_seq_only: bool = False) -> SpyreAttentionMetadata:
        """Build one step's metadata (upstream ``build``, decoder path)."""
        seq_lens = batch.seq_lens.to(torch.int64)
        query_start_loc = batch.query_start_loc.to(torch.int64)
        block_table = batch.block_table
        num_seqs = batch.num_seqs
        query_lens = query_start_loc[1 : num_seqs + 1] - query_start_loc[:num_seqs]
        max_query_len = int(query_lens.max()) if num_seqs else 0
        apply_causal_mask = batch.causal and max_query_len > 1

        aligned_query_lens: list[int] = []
        for query_len in query_lens.tolist():
            if query_len <= 1:
                aligned_query_lens.append(1)
                continue
            aligned = self._attn_bucketer.find_query_bucket(query_len)
            assert aligned is not None, (
                f"no query bucket for query_len={query_len}; top bucket is "
                f"{self._attn_bucketer.query_buckets[-1]}, which should be at least "
                "the chunked-prefill token budget."
            )
            aligned_query_lens.append(aligned)

        block_size = self.block_size
        attention_mask_stacks: list[torch.Tensor] = []
        active_block_indices: list[list[int]] | None = None
        padded_num_blocks: list[int] | None = None
        real_num_blocks: list[int] = []

        if self.sliding_window is None:
            for s in range(num_seqs):
                real_num_blocks.append((int(seq_lens[s]) + block_size - 1) // block_size)
            padded_num_blocks = [self._pad_num_blocks(n) for n in real_num_blocks]
            attention_mask_stacks = [
                torch.empty(0, aligned, block_size, dtype=self.model_dtype)
                for aligned in aligned_query_lens
            ]
            for aligned_query_len in sorted(set(aligned_query_lens)):
                group = [s for s in range(num_seqs) if aligned_query_lens[s] == aligned_query_len]
                mask_cpu = self._build_attention_mask(
                    seq_lens[group],
                    query_lens[group],
                    apply_causal_mask and aligned_query_len > 1,
                    aligned_query_len,
                    max(padded_num_blocks[s] for s in group) * block_size,
                )
                if aligned_query_len == 1:
                    tiles_by_block = mask_cpu.reshape(
                        len(group), mask_cpu.shape[-1] // block_size, 1, block_size
                    )
                    for row, s in enumerate(group):
                        attention_mask_stacks[s] = tiles_by_block[row][: padded_num_blocks[s]]
                    continue
                for row, s in enumerate(group):
                    attention_mask_stacks[s] = (
                        mask_cpu[row]
                        .reshape(aligned_query_len, -1, block_size)
                        .transpose(0, 1)[: padded_num_blocks[s]]
                        .contiguous()
                    )
        else:
            active_block_indices = []
            for s in range(num_seqs):
                kv_len_s = int(seq_lens[s])
                query_len_s = int(query_lens[s])
                context_len_s = kv_len_s - query_len_s
                active_bs, tiles = self._build_active_tiles_with_skip(
                    kv_len_s,
                    query_len_s,
                    context_len_s,
                    aligned_query_lens[s],
                    apply_causal_mask and aligned_query_lens[s] > 1,
                )
                if active_bs:
                    max_active_s = self._max_active_blocks(
                        self._pad_num_blocks((kv_len_s + block_size - 1) // block_size),
                        aligned_query_lens[s],
                    )
                    assert len(active_bs) <= max_active_s, (
                        f"{len(active_bs)} active blocks exceed the bucket bound "
                        f"{max_active_s} (kv_len={kv_len_s}, query_len={query_len_s})"
                    )
                    if len(active_bs) < max_active_s:
                        pad = max_active_s - len(active_bs)
                        active_bs = active_bs + [active_bs[-1]] * pad
                        tiles = tiles + [self._get_masked_tile(aligned_query_lens[s])] * pad
                active_block_indices.append(active_bs)
                attention_mask_stacks.append(
                    torch.stack(tiles)
                    if tiles
                    else torch.empty(0, aligned_query_lens[s], block_size, dtype=self.model_dtype)
                )

        num_active = [int(stack.shape[0]) for stack in attention_mask_stacks]
        page_index_tables_cpu = []
        for s, n in enumerate(num_active):
            blocks_s = slice(n) if active_block_indices is None else active_block_indices[s]
            table = torch.zeros(n, INT32_ELEMS_PER_STICK, dtype=torch.int32)
            table[:, 0] = block_table[s, blocks_s]
            page_index_tables_cpu.append(table)

        num_decode_seqs = 0
        for q in query_lens.tolist():
            if q != 1:
                break
            num_decode_seqs += 1

        metadata = SpyreAttentionMetadata(
            num_seqs=num_seqs,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            block_size=block_size,
            apply_causal_mask=apply_causal_mask,
            attention_mask_stacks=attention_mask_stacks,
            active_block_indices=active_block_indices,
            aligned_query_lens=aligned_query_lens,
            padded_num_blocks=padded_num_blocks,
            page_index_tables_cpu=page_index_tables_cpu,
            num_decode_seqs=num_decode_seqs,
        )
        if not per_seq_only and envs.SPYRE_BATCHED_DECODE and num_decode_seqs > 0:
            blocks_per_seq = real_num_blocks if active_block_indices is None else num_active
            decode_blocks = blocks_per_seq[:num_decode_seqs]
            b_seqs = self._attn_bucketer.find_sequence_bucket(num_decode_seqs)
            b_blocks = self._attn_bucketer.find_blocks_bucket(max(decode_blocks))
            if b_seqs is not None and b_blocks is not None:
                (
                    metadata.padded_num_seqs,
                    metadata.blocks_per_chunk,
                    metadata.padded_batch_blocks,
                    metadata.rep_row_ids_cpu,
                    metadata.chunk_page_ids_cpu,
                    metadata.mask_by_chunk_cpu,
                ) = self._batched_decode_tensors(
                    list(range(num_decode_seqs)),
                    b_seqs,
                    b_blocks,
                    blocks_per_seq,
                    block_table,
                    query_start_loc,
                    active_block_indices,
                    attention_mask_stacks,
                )
        return metadata

    def _batched_decode_tensors(
        self,
        members: list[int],
        b_seqs: int,
        b_blocks: int,
        blocks_per_seq: list[int],
        block_table: torch.Tensor,
        query_start_loc: torch.Tensor,
        active_block_indices: list | None,
        attention_mask_stacks: list[torch.Tensor],
    ):
        """Upstream ``_batched_decode_tensors``, unchanged."""
        block_size = self.block_size
        n = len(members)
        decode_blocks = [blocks_per_seq[m] for m in members]
        blocks_per_chunk, num_chunks = batched_decode_chunking(b_seqs, b_blocks)
        padded_batch_blocks = num_chunks * blocks_per_chunk
        assert padded_batch_blocks >= b_blocks
        query_row_ids = torch.zeros(b_seqs, dtype=torch.int32)
        query_row_ids[:n] = query_start_loc[members].to(torch.int32)
        rep_row_ids_cpu = query_row_ids.repeat(blocks_per_chunk)

        block_ids_padded = torch.zeros(padded_batch_blocks, b_seqs, dtype=torch.int32)
        bt = block_table[members].to(torch.int32)
        if active_block_indices is None:
            n_use_list = [min(nb, b_blocks) for nb in decode_blocks]
            w = min(b_blocks, bt.shape[1])
            cols = torch.arange(w)
            in_range = cols.unsqueeze(0) < torch.tensor(n_use_list, dtype=torch.int64).unsqueeze(1)
            block_ids_padded[:w, :n] = (bt[:, :w] * in_range).t()
        else:
            for s, m in enumerate(members):
                abs_blocks = active_block_indices[m]
                n_use = min(len(abs_blocks), b_blocks)
                for b, abs_b in enumerate(abs_blocks[:n_use]):
                    block_ids_padded[b, s] = bt[s, abs_b]
        chunk_page_ids_cpu = block_ids_padded

        mask_bs_bb = torch.full(
            (b_seqs, padded_batch_blocks, block_size), float("-inf"), dtype=self.model_dtype
        )
        for s, m in enumerate(members):
            n_use = min(blocks_per_seq[m], b_blocks)
            if n_use:
                mask_bs_bb[s, :n_use] = attention_mask_stacks[m][:n_use, 0]
        mask_bs_bb[n:, 0] = torch.finfo(self.model_dtype).min
        kv_extent = self.num_kv_heads if tile_loop.USE_FOR_EACH_TILE else 1
        mask_by_chunk_cpu = (
            mask_bs_bb.transpose(0, 1)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(padded_batch_blocks, b_seqs, kv_extent, 1, block_size)
            .contiguous()
        )
        return (
            b_seqs,
            blocks_per_chunk,
            padded_batch_blocks,
            rep_row_ids_cpu,
            chunk_page_ids_cpu,
            mask_by_chunk_cpu,
        )

    def build_for_variant(self, bucket: SpyreAttnBucket, causal: bool = True):
        """Metadata for the one-sequence batch that runs ``bucket`` (warmup recording)."""
        query_len = self._attn_bucketer.min_real_query_len(bucket.padded_query_len)
        kv_len = bucket.num_blocks * self.block_size
        assert query_len <= kv_len, f"{bucket} pairs a query length no sequence can reach"
        return self.build(
            SeqBatch(
                query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
                seq_lens=torch.tensor([kv_len], dtype=torch.int32),
                # Every block points at page 0, the reserved padding page.
                block_table=torch.zeros(1, bucket.num_blocks, dtype=torch.int32),
                slot_mapping=torch.zeros(query_len, dtype=torch.int64),
                causal=causal,
            ),
            per_seq_only=True,
        )

    def build_for_batched_decode_variant(self, bucket: SpyreAttnBatchedDecodeBucket):
        """Metadata for the all-decode batch that dispatches to ``bucket``."""
        num_seqs = bucket.num_seqs
        kv_len = bucket.num_blocks * self.block_size
        metadata = self.build(
            SeqBatch(
                query_start_loc=torch.arange(num_seqs + 1, dtype=torch.int32),
                seq_lens=torch.full((num_seqs,), kv_len, dtype=torch.int32),
                block_table=torch.zeros(num_seqs, bucket.num_blocks, dtype=torch.int32),
                slot_mapping=torch.zeros(num_seqs, dtype=torch.int64),
            )
        )
        assert metadata.padded_num_seqs is not None, (
            f"build() declined the batched path for {bucket}"
        )
        return metadata


class SpyreHeadMajorAttention:
    """One attention layer on a head-major paged KV cache (upstream
    ``SpyreHeadMajorAttentionImpl``).

    ``staging_rows`` bounds the batch's token count: the kernels gather from and
    store into constant-shaped buffers so the batch's row count never becomes a
    Dynamo guard (#941).
    """

    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        scale: float,
        block_size: int,
        staging_rows: int,
        logits_soft_cap: float = 0.0,
        model_dtype: torch.dtype = torch.float16,
    ) -> None:
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.block_size = block_size
        self.logits_soft_cap = float(logits_soft_cap or 0.0)
        self.model_dtype = model_dtype
        self.staging_rows = staging_rows
        self._staging: tuple[torch.Tensor, torch.Tensor] | None = None
        self._kv_rows: SpyrePagedKVCache | None = None
        self._folded: SpyrePagedKVCache | None = None
        self._kv_row_pool_device: torch.Tensor | None = None

    def graph_key(self) -> tuple:
        """The kernel arguments that may differ between layers of one group."""
        return (self.num_heads, self.scale, self.logits_soft_cap)

    @staticmethod
    def allocate_pages(
        num_pages: int,
        num_kv_heads: int,
        block_size: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> SpyrePagedKVCache:
        layout = head_major_kv_layout(num_pages * num_kv_heads, block_size, head_size, dtype)
        shape = (num_pages, num_kv_heads, block_size, head_size)
        pages = [
            torch.empty(shape, dtype=dtype, device=device, device_layout=layout)
            for _ in range(2)
        ]
        # Zeroed on the device (#1375); through a byte view for a float8 cache.
        for p in pages:
            p.view(torch.uint8).zero_()
        return SpyrePagedKVCache(*pages)

    def staging_buffers(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._staging is None:
            shape = (self.staging_rows, self.num_heads, self.head_size)
            layout = (
                row_outermost_layout(shape, self.model_dtype) if device.type == "spyre" else None
            )
            self._staging = (
                convert(torch.zeros(shape, dtype=self.model_dtype), device, device_layout=layout),
                convert(torch.zeros(shape, dtype=self.model_dtype), device, device_layout=layout),
            )
        return self._staging

    def kv_write_index(self, slot_mapping: torch.Tensor, device: torch.device) -> list[torch.Tensor]:
        """Head h of slot ``block * block_size + offset`` lives at row
        ``(block * num_kv_heads + h) * block_size + offset``; one tensor per head (#3851)."""
        block = torch.div(slot_mapping, self.block_size, rounding_mode="floor")
        base = block * self.num_kv_heads * self.block_size + slot_mapping % self.block_size
        return [
            convert(base + h * self.block_size, device=device) for h in range(self.num_kv_heads)
        ]

    def kv_slot_views(self, kv_cache: SpyrePagedKVCache) -> SpyrePagedKVCache:
        if self._kv_rows is None:
            k_pages, v_pages = kv_cache
            shape = (-1, k_pages.shape[3])
            self._kv_rows = SpyrePagedKVCache(k_pages.view(shape), v_pages.view(shape))
        return self._kv_rows

    def do_kv_cache_update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: SpyrePagedKVCache,
        write_index: list[torch.Tensor],
    ) -> None:
        assert key.device.type == kv_cache[0].device.type, (
            f"kv cache update source is on {key.device.type}, pages on {kv_cache[0].device.type}"
        )
        k_rows, v_rows = self.kv_slot_views(kv_cache)
        _reshape_compiled(key, value, k_rows, v_rows, write_index)

    def _folded_pages(self, k_pages: torch.Tensor, v_pages: torch.Tensor) -> SpyrePagedKVCache:
        if self._folded is None:
            shape = (k_pages.shape[0] * k_pages.shape[1], k_pages.shape[2], k_pages.shape[3])
            self._folded = SpyrePagedKVCache(k_pages.view(shape), v_pages.view(shape))
        return self._folded

    def _kv_row_pool(self, num_pages: int, device: torch.device) -> torch.Tensor:
        if self._kv_row_pool_device is None:
            rows = torch.arange(num_pages * self.num_kv_heads, dtype=torch.int32)
            self._kv_row_pool_device = convert(
                rows.reshape(num_pages, self.num_kv_heads, 1), device=device
            )
        return self._kv_row_pool_device

    def _batched_decode_supported(self) -> bool:
        return envs.SPYRE_BATCHED_DECODE

    def _batched_decode_preconditions_met(
        self, attn_metadata: SpyreAttentionMetadata, num_pages: int
    ) -> bool:
        if not self._batched_decode_supported() or attn_metadata.padded_num_seqs is None:
            return False
        assert attn_metadata.blocks_per_chunk is not None
        # A gather of the whole cache faults the device (torch-spyre#4033).
        return attn_metadata.padded_num_seqs * attn_metadata.blocks_per_chunk < num_pages

    def _mirror_batched_decode_indices(
        self, attn_metadata: SpyreAttentionMetadata, device: torch.device
    ) -> None:
        """Mirror the batched-decode precomputes once per step; each page id in its own
        stick under the tiled walk (TEMPORARY until torch-spyre#4603)."""
        assert attn_metadata.rep_row_ids_cpu is not None
        assert attn_metadata.chunk_page_ids_cpu is not None
        assert attn_metadata.mask_by_chunk_cpu is not None
        if not USE_FOR_EACH_TILE:
            attn_metadata.rep_row_ids_dev = convert(attn_metadata.rep_row_ids_cpu, device=device)
            attn_metadata.chunk_page_ids_dev = convert(
                attn_metadata.chunk_page_ids_cpu, device=device
            )
            attn_metadata.mask_by_chunk_dev = convert(
                attn_metadata.mask_by_chunk_cpu, device=device
            )
            return
        b = attn_metadata.padded_num_seqs
        bpc = attn_metadata.blocks_per_chunk
        assert b is not None and bpc is not None
        e = bpc * b
        c = attn_metadata.chunk_page_ids_cpu.shape[0] // bpc
        idx = (
            attn_metadata.chunk_page_ids_cpu.reshape(c, bpc, b)
            .reshape(c, e, 1)
            .to(torch.int32)
            .contiguous()
        )
        attn_metadata.chunk_page_ids_dev = idx.to(
            device, device_layout=temporary_chunk_major_page_index_layout(c, e)
        )
        attn_metadata.rep_row_ids_dev = convert(attn_metadata.rep_row_ids_cpu, device=device)
        attn_metadata.mask_by_chunk_dev = convert(
            attn_metadata.mask_by_chunk_cpu.reshape(
                c, bpc, b, *attn_metadata.mask_by_chunk_cpu.shape[2:]
            ),
            device=device,
        )

    def _run_batched_decode(self, query_dev, k_pages, v_pages, attn_metadata, out):
        b_seqs = attn_metadata.padded_num_seqs
        blocks_per_chunk = attn_metadata.blocks_per_chunk
        return _run_with_capped_cores(
            b_seqs * blocks_per_chunk * self.num_kv_heads,
            _batched_decode_compiled,
            query_dev,
            attn_metadata.rep_row_ids_dev,
            k_pages,
            v_pages,
            attn_metadata.chunk_page_ids_dev,
            attn_metadata.mask_by_chunk_dev,
            self.scale,
            b_seqs,
            blocks_per_chunk,
            self.num_kv_heads,
            self.num_queries_per_kv,
            attn_metadata.block_size,
            self.head_size,
            self.logits_soft_cap,
            out,
        )

    def _run_page_attn(
        self,
        query,
        row_table,
        k_pages,
        v_pages,
        index_table,
        mask_stack,
        num_blocks,
        padded_query_len,
        out,
    ):
        assert row_table.shape == (padded_query_len,)
        assert index_table.shape[0] == num_blocks
        if padded_query_len > 1:
            return _run_with_capped_cores(
                self.num_kv_heads * padded_query_len,
                _page_attn_prefill_compiled,
                query,
                row_table,
                k_pages,
                v_pages,
                index_table,
                mask_stack,
                self.scale,
                num_blocks,
                padded_query_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                self.logits_soft_cap,
                out,
            )
        k_folded, v_folded = self._folded_pages(k_pages, v_pages)
        kv_row_pool = self._kv_row_pool(k_pages.shape[0], query.device)
        return _run_with_capped_cores(
            self.num_kv_heads * padded_query_len,
            _page_attn_decode_compiled,
            query,
            row_table,
            k_folded,
            v_folded,
            index_table,
            kv_row_pool,
            mask_stack,
            self.scale,
            num_blocks,
            padded_query_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.block_size,
            self.logits_soft_cap,
            out,
        )

    def forward(
        self,
        q_staging: torch.Tensor,
        out_staging: torch.Tensor,
        kv_cache: SpyrePagedKVCache,
        attn_metadata: SpyreAttentionMetadata,
    ) -> None:
        """Attention over the staged query rows into ``out_staging`` (upstream
        ``_online_softmax_attention``, pre-staged path)."""
        k_pages, v_pages = kv_cache
        device = k_pages.device
        num_seqs = attn_metadata.num_seqs
        if self._batched_decode_preconditions_met(attn_metadata, k_pages.shape[0]):
            if attn_metadata.rep_row_ids_dev is None:
                self._mirror_batched_decode_indices(attn_metadata, device)
            self._run_batched_decode(q_staging, k_pages, v_pages, attn_metadata, out_staging)
            if attn_metadata.num_decode_seqs == num_seqs:
                return
            seq_start = attn_metadata.num_decode_seqs
        else:
            seq_start = 0

        if attn_metadata.kernel_index_tables is None:
            attn_metadata.kernel_index_tables = [
                convert(t, device=device) for t in attn_metadata.page_index_tables_cpu
            ]
        if attn_metadata.query_row_tables is None:
            attn_metadata.query_row_tables = _build_query_row_tables(attn_metadata, device)
        assert max(attn_metadata.aligned_query_lens, default=1) < self.staging_rows

        for seq_idx in range(seq_start, num_seqs):
            num_blocks = int(attn_metadata.attention_mask_stacks[seq_idx].shape[0])
            if num_blocks == 0:
                q_start = int(attn_metadata.query_start_loc[seq_idx])
                q_end = int(attn_metadata.query_start_loc[seq_idx + 1])
                out_staging[q_start:q_end] = 0.0
                continue
            mask_stack = _mirror_mask_stack(attn_metadata, seq_idx, device)[:num_blocks]
            self._run_page_attn(
                q_staging,
                attn_metadata.query_row_tables[seq_idx],
                k_pages,
                v_pages,
                attn_metadata.kernel_index_tables[seq_idx],
                mask_stack,
                num_blocks,
                attn_metadata.aligned_query_lens[seq_idx],
                out_staging,
            )

    def record_graphs(
        self,
        kv_cache: SpyrePagedKVCache,
        builder: SpyreAttentionMetadataBuilder,
        *,
        causal: bool = True,
    ) -> int:
        """Compile every variant the builder's bucketer enumerates (upstream
        ``record_graphs``); a failing variant is logged and compiles on first use."""
        num_pages = kv_cache[0].shape[0]
        device = kv_cache[0].device
        q_staging, out_staging = self.staging_buffers(device)
        bucketer = builder.attn_bucketer
        t0 = time.time()
        prev_limit = torch._dynamo.config.accumulated_recompile_limit
        variants = bucketer.variants()
        decode_variants = (
            bucketer.batched_decode_variants() if self._batched_decode_supported() else []
        )
        torch._dynamo.config.accumulated_recompile_limit = max(
            prev_limit, 4 * (len(variants) + len(decode_variants)) + 64
        )
        recorded: set = set()
        try:
            for bucket in variants:
                if bucket.num_blocks > num_pages:
                    continue
                try:
                    md = builder.build_for_variant(bucket, causal=causal)
                    realized = SpyreAttnBucket(
                        num_blocks=int(md.attention_mask_stacks[0].shape[0]),
                        padded_query_len=md.aligned_query_lens[0],
                    )
                    if realized in recorded:
                        continue
                    self.forward(q_staging, out_staging, kv_cache, md)
                    recorded.add(realized)
                except Exception:
                    logger.warning(
                        "Attention variant %s failed to record; it will compile on first "
                        "use instead.",
                        bucket,
                        exc_info=True,
                    )
            assert self.staging_rows >= max((v.num_seqs for v in decode_variants), default=0)
            for bucket in decode_variants:
                try:
                    md = builder.build_for_batched_decode_variant(bucket)
                    key = (
                        md.padded_num_seqs,
                        md.blocks_per_chunk,
                        md.chunk_page_ids_cpu.shape[0] // md.blocks_per_chunk,
                    )
                    if md.padded_num_seqs * md.blocks_per_chunk >= num_pages or key in recorded:
                        continue
                    self.forward(q_staging, out_staging, kv_cache, md)
                    recorded.add(key)
                except Exception:
                    logger.warning(
                        "Batched decode variant %s failed to record; it will compile on "
                        "first use instead.",
                        bucket,
                        exc_info=True,
                    )
        finally:
            torch._dynamo.config.accumulated_recompile_limit = prev_limit
        logger.info(
            "Recorded %d attention variants (%d per-seq, %d batched decode) in %.2fs.",
            len(recorded),
            len(variants),
            len(decode_variants),
            time.time() - t0,
        )
        return len(recorded)


__all__ = [
    "GroupSpec",
    "SeqBatch",
    "SpyreAttentionMetadata",
    "SpyreAttentionMetadataBuilder",
    "SpyreHeadMajorAttention",
    "SpyrePagedKVCache",
]
