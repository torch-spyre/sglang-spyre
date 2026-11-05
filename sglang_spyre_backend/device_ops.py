"""Register Spyre on-device ops ported from current spyre-inference.

When SGLang runs with the model body on Spyre, RMSNorm, activations, RoPE and
linear run on the device and the residual stream stays resident across the
forward pass, so the CPU<->Spyre boundary is crossed once per forward.

Reuse strategy: SGLang's ops are ``BaseFusedOp`` subclasses (the 0.5.21 name of
``MultiPlatformOp``) whose dispatch consults
``_oot_forward_registry[get_dispatch_key_name()]`` first. We register a Spyre
forward per op class via ``BaseFusedOp.register_oot_forward(op_cls, fn,
"spyre")``. The bodies follow spyre-inference's custom ops at
``0ccdbf6ac2da7faacf98db93464b274fb8f40126``:

- RMSNorm runs upstream's fp32-promoted native body, as spyre-inference does
  since torch-spyre#4490 kept the fp32 reduction in the compiled graph (#936).
- Gemma-4's norms run SGLang's ``Gemma4RMSNorm`` CUDA semantics in fp32 native
  math, the same math vLLM's ``RMSNorm`` uses for Gemma-4 (#845 for the fp32
  Gemma form), including the router's folded scale.
- RoPE is the device-resident 2x2 rotation, with the cache cast to the model
  dtype at build time (#1393) and the split-free form for a sub-stick half head
  (#1158). Gemma-4's proportional RoPE is a full neox rotation over an
  identity-padded cache, so it takes the same path.
- GELU-and-mul is ``F.gelu(..., approximate)``, the upstream native body;
  spyre-inference removed its NewGELU workaround once torch-spyre lowered
  ``pow`` correctly (#923).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sglang.kernels.fused_op import BaseFusedOp

SPYRE_KEY = "spyre"
# One Spyre stick of fp16 elements.
_STICK = 64


# ---------------------------------------------------------------------------
# RMSNorm: upstream fp32 native math (vllm/ir/ops/layernorm.py), residual first.
# ---------------------------------------------------------------------------
def _rmsnorm_forward_spyre(self, x, residual=None, post_residual_addition=None, **_):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        if post_residual_addition is not None:
            x = x + post_residual_addition.to(torch.float32)
        residual = x.to(orig_dtype)
    elif post_residual_addition is not None:
        x = x + post_residual_addition.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + self.variance_epsilon)
    if getattr(self, "has_weight", True) and getattr(self, "weight", None) is not None:
        x = x.to(self.weight.dtype) * self.weight
    x = x.to(orig_dtype)
    if residual is None:
        return x
    return x, residual


def _gemma4_rmsnorm_forward_spyre(self, x):
    """SGLang's ``Gemma4RMSNorm.forward_cuda`` semantics in fp32 native math.

    ``forward_native`` applies the weight only when ``with_scale``; the CUDA path
    always does, which is what ``Gemma4Router.fuse_scale`` relies on: it folds
    ``scale * hidden_size**-0.5`` into a ``with_scale=False`` norm's weight. The
    weight of every other ``with_scale=False`` norm (``v_norm``) is ones.
    """
    orig_dtype = x.dtype
    xf = x.float()
    xf = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
    weight = self.weight.float()
    if self.with_scale:
        weight = weight + self.scale_shift
    return (xf * weight).to(orig_dtype)


# ---------------------------------------------------------------------------
# Activations.
# ---------------------------------------------------------------------------
def _silu_and_mul_forward_spyre(self, x):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def _gelu_and_mul_forward_spyre(self, x):
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d], approximate=getattr(self, "approximate", "none")) * x[..., d:]


# ---------------------------------------------------------------------------
# RoPE: spyre-inference custom_ops/rotary_embedding.py.
# ---------------------------------------------------------------------------
def _rotate_neox_2x2(x, rot, head_size):
    num_tokens = x.shape[0]
    inner = head_size // 2
    x_pairs = x.view(num_tokens, -1, 2, inner)
    out = (rot.unsqueeze(1) * x_pairs.unsqueeze(-3)).sum(dim=-2)
    return out.flatten(-2).view(x.shape)


def _rotate_half_matrix(head_size: int, dtype: torch.dtype) -> torch.Tensor:
    half = head_size // 2
    idx = torch.arange(half)
    r = torch.zeros(head_size, head_size, dtype=dtype)
    r[idx + half, idx] = -1.0
    r[idx, idx + half] = 1.0
    return r


def _rotate_neox_split_free(x, cos_sin, rotate_half, head_size):
    num_tokens = x.shape[0]
    heads = x.view(num_tokens, -1, head_size)
    cos, sin = cos_sin[:, 0].unsqueeze(1), cos_sin[:, 1].unsqueeze(1)
    rotated = (heads.reshape(-1, head_size) @ rotate_half).view(heads.shape)
    return (heads * cos + rotated * sin).view(x.shape)


def rope_device_cache(module, device: torch.device, dtype: torch.dtype) -> None:
    """Build the device rotation cache once, before any graph traces the module.

    The cache is the host ``cos_sin_cache`` cast to the model dtype (#1393),
    rearranged to ``[[cos, -sin], [sin, cos]]`` per position and placed rows
    outermost so a position gather lowers (torch-spyre#3418).
    """
    from sglang_spyre_backend.custom_ops.utils import place_row_gathered

    if getattr(module, "_spyre_rotation_cache", None) is not None:
        return
    if not (module.is_neox_style and module.rotary_dim == module.head_size):
        raise NotImplementedError(
            "Spyre RoPE supports only neox-style full rotary (rotary_dim == head_size); "
            f"got is_neox_style={module.is_neox_style}, rotary_dim={module.rotary_dim}, "
            f"head_size={module.head_size}."
        )
    cos_sin = module.cos_sin_cache.to(device="cpu", dtype=dtype)
    inner = cos_sin.shape[-1] // 2
    cos, sin = cos_sin.chunk(2, dim=-1)
    half = module.head_size // 2
    if half % _STICK:
        module._spyre_rotate_half = _rotate_half_matrix(module.head_size, dtype).to(device)
        cache = torch.cat([cos, cos, sin, sin], dim=-1)
    else:
        module._spyre_rotate_half = None
        cache = torch.stack([cos, -sin, sin, cos], dim=1).view(cos_sin.shape[0], 2, 2, inner)
        cache = cache.flatten(1)
    module._spyre_rotation_cache = place_row_gathered(
        cache.contiguous(), lambda t: t.to(device=device, dtype=dtype), "RoPE rotation cache"
    )


def _rope_forward_spyre(
    self,
    positions,
    query,
    key=None,
    offsets=None,
    fused_set_kv_buffer_arg=None,
    **_,
):
    assert fused_set_kv_buffer_arg is None, "Spyre RoPE: fused KV write unsupported"
    if offsets is not None:
        positions = positions + offsets
    if getattr(self, "_spyre_rotation_cache", None) is None:
        rope_device_cache(self, query.device, query.dtype)
    cache = self._spyre_rotation_cache
    pos = positions.flatten()
    if pos.device.type != cache.device.type:
        pos = pos.to(cache.device)
    rotate_half = self._spyre_rotate_half
    if rotate_half is not None:
        cos_sin = cache.index_select(0, pos).view(-1, 2, self.head_size)
        out_query = _rotate_neox_split_free(query, cos_sin, rotate_half, self.head_size)
        out_key = (
            _rotate_neox_split_free(key, cos_sin, rotate_half, self.head_size)
            if key is not None
            else None
        )
        return out_query, out_key
    rot = cache.index_select(0, pos).view(-1, 2, 2, self.head_size // 2)
    out_query = _rotate_neox_2x2(query, rot, self.head_size)
    out_key = _rotate_neox_2x2(key, rot, self.head_size) if key is not None else None
    return out_query, out_key


ROPE_CLASSES = (
    ("sglang.srt.layers.rotary_embedding.base", "RotaryEmbedding"),
    ("sglang.srt.layers.rotary_embedding.rope_variant", "Llama3RotaryEmbedding"),
    ("sglang.srt.layers.rotary_embedding.rope_variant", "Gemma4RotaryEmbedding"),
    ("sglang.srt.layers.rotary_embedding.yarn", "YaRNScalingRotaryEmbedding"),
)


# ---------------------------------------------------------------------------
# Registration. Each registration is guarded so a class that does not exist in
# this SGLang version is skipped, not fatal.
# ---------------------------------------------------------------------------
def register_spyre_ops() -> list[str]:
    registered: list[str] = []

    def _reg(module_path, cls_name, fn):
        try:
            mod = __import__(module_path, fromlist=[cls_name])
            op_cls = getattr(mod, cls_name)
            BaseFusedOp.register_oot_forward(op_cls, fn, SPYRE_KEY)
            registered.append(cls_name)
        except Exception:
            pass

    _reg("sglang.srt.layers.layernorm", "RMSNorm", _rmsnorm_forward_spyre)
    _reg("sglang.srt.layers.layernorm", "Gemma4RMSNorm", _gemma4_rmsnorm_forward_spyre)
    _reg("sglang.srt.layers.activation", "SiluAndMul", _silu_and_mul_forward_spyre)
    _reg("sglang.srt.layers.activation", "GeluAndMul", _gelu_and_mul_forward_spyre)
    # BaseFusedOp dispatch keys on the concrete class name.
    for module_path, cls_name in ROPE_CLASSES:
        _reg(module_path, cls_name, _rope_forward_spyre)
    return registered


__all__ = ["ROPE_CLASSES", "SPYRE_KEY", "register_spyre_ops", "rope_device_cache"]
