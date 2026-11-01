"""Move model weights to Spyre and install the Spyre linear methods.

Called after weights load on CPU. Every unquantized ``LinearBase`` gets
spyre-inference's transposed-weight method (``custom_ops/linear.py``): merged
gate/up projections pad short row blocks (#651), every weight is transposed on
the host in blocks (#1327) before it moves to the device, and the LM head keeps
a separate row-padded ``Wᵀ`` (#633, #784). The LM head returns its logits on the
host, as spyre-inference's ``SpyreLogitsProcessor._apply_head`` does at TP1,
so SGLang's logits post-processing and sampling stay on CPU.

Mirrors spyre-inference's ``move_model_to_spyre`` and LM-head placement at
``0ccdbf6ac2da7faacf98db93464b274fb8f40126``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from sglang_spyre_backend.custom_ops.linear import (
    SpyreLMHeadMethod,
    SpyrePaddedRowsLinearMethod,
    SpyreUnquantizedLinearMethod,
)
from sglang_spyre_backend.custom_ops.utils import convert

logger = logging.getLogger(__name__)

_SPYRE = torch.device("spyre")
_DTYPE = torch.float16


class _SpyreLinear:
    """Wrap a transposed-weight method with the host-side placement of its input."""

    def __init__(self, method):
        self.method = method

    def create_weights(self, *args, **kwargs):
        pass

    def process_weights_after_loading(self, layer):
        weight = layer.weight.data.to(device="cpu", dtype=_DTYPE)
        self.method.build_weight_t(layer, weight)
        layer.weight = nn.Parameter(layer.weight.data.to(_SPYRE), requires_grad=False)
        if getattr(layer, "bias", None) is not None:
            layer.bias = nn.Parameter(
                layer.bias.data.to(device=_SPYRE, dtype=_DTYPE), requires_grad=False
            )

    def apply(self, layer, x, bias=None):
        if x.device.type != "spyre" or x.dtype != _DTYPE:
            x = convert(x, device=_SPYRE, dtype=_DTYPE)
        return self.method.apply(layer, x, bias)


class SpyreLMHead:
    """LM-head method for SGLang's logits processor, tie-safe.

    For a tied head the module is also the input embedding, so ``embedding``
    stays the original method's host gather while ``apply`` projects through the
    row-padded ``Wᵀ`` on Spyre and returns host logits.
    """

    def __init__(self, original):
        self.original = original
        self.method = SpyreLMHeadMethod()

    def __getattr__(self, name):
        return getattr(self.original, name)

    def embedding(self, layer, input_):
        return self.original.embedding(layer, input_)

    def process_weights_after_loading(self, layer):
        weight = layer.weight.data.to(device="cpu", dtype=_DTYPE)
        self.method.build_weight_t(layer, weight)
        layer.padded_weight_t = nn.Parameter(
            layer.padded_weight_t.data.to(_SPYRE), requires_grad=False
        )

    def apply(self, layer, x, bias=None):
        if x.device.type != "spyre" or x.dtype != _DTYPE:
            x = convert(x, device=_SPYRE, dtype=_DTYPE)
        logits = self.method.apply(layer, x, bias)
        return convert(logits, device="cpu")


def _is_merged_column(module) -> bool:
    from sglang.srt.layers.linear import MergedColumnParallelLinear

    return isinstance(module, MergedColumnParallelLinear)


def move_model_to_spyre(model: nn.Module) -> None:
    """Install the Spyre linear methods and move linear weights to the device."""
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    n_patched = 0
    n_skipped_quant = 0
    for _name, module in model.named_modules():
        if isinstance(module, LinearBase):
            qm = getattr(module, "quant_method", None)
            if isinstance(qm, UnquantizedLinearMethod):
                method = (
                    SpyrePaddedRowsLinearMethod()
                    if _is_merged_column(module)
                    else SpyreUnquantizedLinearMethod()
                )
                module.quant_method = _SpyreLinear(method)
                module.quant_method.process_weights_after_loading(module)
                n_patched += 1
            elif qm is not None:
                n_skipped_quant += 1

    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight"):
        head_method = SpyreLMHead(getattr(lm_head, "quant_method", None))
        head_method.process_weights_after_loading(lm_head)
        lm_head.quant_method = head_method

    logger.info(
        "patched %d LinearBase layers, skipped %d quantized", n_patched, n_skipped_quant
    )


__all__ = ["SpyreLMHead", "move_model_to_spyre"]
