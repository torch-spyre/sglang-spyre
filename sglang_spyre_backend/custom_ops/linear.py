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

"""Spyre linear fast path: store ``Wᵀ`` and run ``x @ Wᵀ``.

Ported from ``spyre_inference/custom_ops/linear.py`` and
``custom_ops/parallel_lm_head.py`` at ``0ccdbf6ac2da7faacf98db93464b274fb8f40126``:
the blocked host transpose (#1327), the short-row padding of merged gate/up
projections (#651, torch-spyre#4032) and the row-padded LM head (#784, #633).
SGLang's ``LinearBase`` carries the same ``quant_method.apply`` seam vLLM's
does, so the method objects are installed on the layer after weight loading.
"""

from __future__ import annotations

import logging
from typing import cast

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang_spyre_backend import envs

logger = logging.getLogger(__name__)

_PAD_ROWS = 8
_MAX_PAD_WEIGHT = 200_000_000
_TRANSPOSE_BLOCK_ROWS = 128
# torch-spyre matmul work-division limit on the LM head's output rows.
LM_HEAD_ROW_ALIGN = 64 * 32


def _transpose_contiguous(w: torch.Tensor) -> torch.Tensor:
    """`w.t().contiguous()`, bit-identical. PyTorch copies a 2-D transpose on one thread;
    with SPYRE_BLOCKED_TRANSPOSE a 2-D CPU `w` is copied 128 source rows at a time, which
    runs on torch's CPU threads and keeps each block's reads and writes in cache."""
    if not (
        envs.SPYRE_BLOCKED_TRANSPOSE
        and w.dim() == 2
        and w.device.type == "cpu"
        and not w.t().is_contiguous()
    ):
        return w.t().contiguous()
    out = torch.empty((w.shape[1], w.shape[0]), dtype=w.dtype, device=w.device)
    for i in range(0, w.shape[0], _TRANSPOSE_BLOCK_ROWS):
        out[:, i : i + _TRANSPOSE_BLOCK_ROWS].copy_(w[i : i + _TRANSPOSE_BLOCK_ROWS].t())
    return out


def spyre_linear_t(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    bias: torch.Tensor | None,
    pad_rows: bool = False,
):
    """Linear forward with a pre-transposed weight: `x @ Wᵀ (+ bias)`.

    `weight_t` is the physically-transposed weight of shape `[in, out]`, so the
    matmul is a plain `x @ A` (the Spyre-fast layout), not `F.linear`'s `x @ Aᵀ`.

    ``pad_rows`` pads a short 2-D row block up to ``_PAD_ROWS`` and slices it
    back (torch-spyre#4032). Callers that must not pad leave it off.
    """
    rows = x.shape[0] if pad_rows and x.dim() == 2 else 0
    if 0 < rows < _PAD_ROWS:
        x = F.pad(x, (0, 0, 0, _PAD_ROWS - rows))
    out = torch.matmul(x, weight_t)
    if bias is not None:
        out = out + bias
    if 0 < rows < _PAD_ROWS:
        out = out[:rows]
    return out


class SpyreTransposedWeightMethod:
    """Store `Wᵀ` (optionally row-padded) under ``WEIGHT_T_ATTR`` and matmul it.

    ``WEIGHT_T_ATTR="weight"`` replaces the weight in place; a distinct name keeps
    the source ``weight``, required for a tied LM head whose ``weight`` is the
    embedding table and keeps its gather layout. ``ROW_ALIGN`` pads the output
    rows to a multiple before transposing.
    """

    WEIGHT_T_ATTR: str = "weight"
    ROW_ALIGN: int | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.build_weight_t(layer, cast(torch.Tensor, layer.weight).data)

    def build_weight_t(self, layer: torch.nn.Module, w: torch.Tensor) -> None:
        padding = (-w.shape[0]) % self.ROW_ALIGN if self.ROW_ALIGN else 0
        layer.spyre_row_padding = padding
        if padding:
            w = F.pad(w, (0, 0, 0, padding))
        setattr(
            layer,
            self.WEIGHT_T_ATTR,
            Parameter(_transpose_contiguous(w), requires_grad=False),
        )

    def _pad_short_rows(self, layer: torch.nn.Module) -> bool:
        return False

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = spyre_linear_t(
            x,
            getattr(layer, self.WEIGHT_T_ATTR),
            bias,
            pad_rows=self._pad_short_rows(layer),
        )
        padding = cast(int, layer.spyre_row_padding)
        if padding:
            out = out[..., :-padding]
        return out


class SpyreUnquantizedLinearMethod(SpyreTransposedWeightMethod):
    """Unquantized linear method: store `Wᵀ` in place (torch-spyre #3512)."""


class SpyrePaddedRowsLinearMethod(SpyreUnquantizedLinearMethod):
    """Pads a partial row block to `_PAD_ROWS`; set on every merged-column layer."""

    def _pads(self, layer: torch.nn.Module) -> bool:
        weight = cast(torch.Tensor, getattr(layer, self.WEIGHT_T_ATTR))
        return weight.numel() <= _MAX_PAD_WEIGHT

    def _pad_short_rows(self, layer: torch.nn.Module) -> bool:
        return self._pads(layer)


class SpyreLMHeadMethod(SpyreTransposedWeightMethod):
    """LM head: a row-padded ``Wᵀ`` in ``padded_weight_t``, the source left alone.

    The source ``weight`` stays where it is, which for a tied head is the
    embedding table's row-gathered layout; the gather and the matmul layouts
    differ, so both stay resident, as in spyre-inference (#633).
    """

    WEIGHT_T_ATTR = "padded_weight_t"
    ROW_ALIGN = LM_HEAD_ROW_ALIGN


__all__ = [
    "LM_HEAD_ROW_ALIGN",
    "SpyreLMHeadMethod",
    "SpyrePaddedRowsLinearMethod",
    "SpyreUnquantizedLinearMethod",
    "spyre_linear_t",
]
