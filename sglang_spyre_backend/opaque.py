"""Opaque attention and MoE ops at the block-graph boundary.

spyre-inference compiles each transformer block as one graph and keeps two
things out of it: the attention core, behind vLLM's
``unified_attention_with_output`` custom op (its per-sequence loop cannot be
captured with ``fullgraph=True``), and the MoE dispatch, behind
``moe_forward`` for token counts that are not a whole stick (#1175). This module
gives SGLang the same two boundaries as ``torch.ops.sglang_spyre.*`` ops: the
block graph sees one opaque node, and the op body runs spyre-inference's own
compiled kernels eagerly.

Layers are named the way vLLM names them: the op takes a ``LayerName``, vLLM's
hoisted value-type opaque object (``vllm/utils/torch_utils.py``), built inside
the traced frame from the layer's string name. Dynamo lifts it to a graph input
instead of baking it in as a constant, so identical blocks share one compiled
graph rather than one per layer (spyre-inference's block-sharing precondition,
``block_sharing_defeated_by``).
"""

import torch
from torch._library.opaque_object import register_opaque_type
from torch._opaque_base import OpaqueBase

from sglang_spyre_backend.custom_ops.utils import direct_register_custom_op

_ATTN_LAYERS: dict[str, torch.nn.Module] = {}
_MOE_LAYERS: dict[str, torch.nn.Module] = {}
_ATTN_BACKEND: list = [None]


class LayerName(OpaqueBase):
    """A module name as a hoisted torch opaque value (vLLM's ``LayerName``)."""

    def __init__(self, value: str):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, LayerName) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __fx_repr__(self):
        return (f"LayerName({self.value!r})", {"LayerName": LayerName})


register_opaque_type(LayerName, typ="value", hoist=True)


def set_attention_backend(backend) -> None:
    """The backend whose per-step metadata the attention op reads."""
    _ATTN_BACKEND[0] = backend


def register_attention_layers(model: torch.nn.Module) -> int:
    from sglang.srt.layers.radix_attention import RadixAttention

    count = 0
    for module in model.modules():
        if isinstance(module, RadixAttention):
            module.spyre_layer_name = f"attn.{module.layer_id}"
            _ATTN_LAYERS[module.spyre_layer_name] = module
            count += 1
    return count


def register_moe_layer(name: str, experts: torch.nn.Module) -> None:
    _MOE_LAYERS[name] = experts


def _unified_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer_name: LayerName,
    out_dim: int,
    save_kv_cache: bool,
) -> torch.Tensor:
    backend = _ATTN_BACKEND[0]
    assert backend is not None, "no Spyre attention backend is active"
    layer = _ATTN_LAYERS[layer_name.value]
    has_kv = k.numel() > 0
    return backend._forward_impl(
        q, k if has_kv else None, v if has_kv else None, layer, save_kv_cache and has_kv
    )


def _unified_attention_fake(q, k, v, layer_name, out_dim, save_kv_cache):
    return q.new_empty((q.shape[0], out_dim))


def _moe_forward(
    x: torch.Tensor, router_logits: torch.Tensor, layer_name: LayerName
) -> torch.Tensor:
    from sglang_spyre_backend.moe import spyre_moe_forward

    return spyre_moe_forward(_MOE_LAYERS[layer_name.value], x, router_logits)


def _moe_forward_fake(x, router_logits, layer_name):
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="unified_attention",
    op_func=_unified_attention,
    fake_impl=_unified_attention_fake,
)
direct_register_custom_op(
    op_name="moe_forward",
    op_func=_moe_forward,
    fake_impl=_moe_forward_fake,
)


def spyre_radix_attention_forward(
    self,
    q,
    k,
    v,
    forward_batch=None,
    save_kv_cache: bool = True,
    key_value_num_tokens=None,
    **kwargs,
):
    """``RadixAttention.forward`` on Spyre: one opaque op per layer call."""
    if kwargs:
        raise NotImplementedError(
            f"Spyre attention does not support {sorted(kwargs)} (sinks, rope fusion, ...)"
        )
    if k is None:
        k = v = q.new_empty(0)
    out_dim = self.tp_q_head_num * self.v_head_dim
    return torch.ops.sglang_spyre.unified_attention(
        q, k, v, LayerName(self.spyre_layer_name), out_dim, bool(save_kv_cache)
    )


__all__ = [
    "LayerName",
    "register_attention_layers",
    "register_moe_layer",
    "set_attention_backend",
    "spyre_radix_attention_forward",
]
