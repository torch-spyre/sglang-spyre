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

"""Model-specific Spyre adaptations for SGLang's model classes.

Ported from ``spyre_inference/models/gemma4.py`` at
``0ccdbf6ac2da7faacf98db93464b274fb8f40126``:

- ``repair_head_dim_access`` (#717): transformers >= 5.16 makes gemma-4's bare
  ``config.head_dim`` / ``config.num_key_value_heads`` raise, and SGLang reads
  them in its config normalization. Applied when the config is parsed.
- ``force_text_backbone``: a gemma-4 checkpoint is served through its text
  decoder ``Gemma4ForCausalLM``, so the vision tower's weights are skipped.
  spyre-inference applies this to tower-free checkpoints by default; SGLang
  text serving on Spyre has no vision path, so it is applied to every gemma-4
  checkpoint here and an image request is rejected by the engine.
- ``configure_gemma4_moe_layers`` (#802): full-softmax routing over the raw
  router logits and a ``gelu_tanh`` expert recipe, with Gemma's per-expert output
  scale folded into the down-projection stack (``_fold_gemma4_expert_scale``)
  instead of into the routing weights.

Router (#877): SGLang's ``Gemma4Router`` already returns logits in the weight
dtype, which is what spyre-inference's ``SpyreGateLinear`` restores on vLLM, so
it needs no change.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import torch

from sglang_spyre_backend.moe import (
    SpyreMoERecipe,
    configure_spyre_moe_layer,
    prepare_moe_layer,
    spyre_moe_forward,
)

logger = logging.getLogger(__name__)

_GEMMA4_FULL_ATTENTION_ATTRS = {
    "global_head_dim": "head_dim",
    "num_global_key_value_heads": "num_key_value_heads",
}
_GEMMA4_TEXT_MODEL_TYPES = {"gemma4", "gemma4_text"}
GEMMA4_TEXT_BACKBONE = "Gemma4ForCausalLM"


def repair_head_dim_access(config: Any) -> None:
    """Upstream ``repair_head_dim_access``: restore the <=5.14 bare-attribute view."""
    if getattr(config, "model_type", None) not in _GEMMA4_TEXT_MODEL_TYPES:
        return
    text_config = getattr(config, "text_config", config)
    for cfg in {id(config): config, id(text_config): text_config}.values():
        cfg.allow_global_per_layer_attribute_access = True
    per_layer = getattr(text_config, "per_layer_config", None)
    layer_types = getattr(text_config, "layer_types", None)
    if per_layer is not None and layer_types:
        full_idx = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
        for global_attr, src_attr in _GEMMA4_FULL_ATTENTION_ATTRS.items():
            if hasattr(text_config, global_attr) or not full_idx:
                continue
            values = {getattr(per_layer[i], src_attr) for i in full_idx}
            if len(values) == 1:
                setattr(text_config, global_attr, values.pop())


def force_text_backbone(config: Any) -> Any:
    """Serve a gemma-4 checkpoint through its text decoder (upstream
    ``GEMMA4_TEXT_BACKBONE_OVERRIDE``): the text config becomes the model config,
    so SGLang builds ``Gemma4ForCausalLM`` and skips the towers' weights."""
    if getattr(config, "model_type", None) not in _GEMMA4_TEXT_MODEL_TYPES:
        return config
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return config
    if getattr(config, "audio_config", None) is not None and getattr(
        config, "vision_config", None
    ) is None:
        raise NotImplementedError(
            "gemma-4 audio is not supported on Spyre; use a text-only or vision checkpoint."
        )
    text_config.architectures = [GEMMA4_TEXT_BACKBONE]
    # SGLang's own gemma-4 normalization (hf_transformers/config.py) keys on the
    # multimodal model_type; apply its mapping to the text config here. Base
    # attributes describe the full-attention layers, ``swa_*`` the sliding ones.
    swa_head_dim = text_config.head_dim
    swa_kv_heads = text_config.num_key_value_heads
    text_config.swa_head_dim = swa_head_dim
    text_config.swa_v_head_dim = swa_head_dim
    text_config.swa_num_key_value_heads = swa_kv_heads
    if getattr(text_config, "global_head_dim", None) is not None:
        text_config.head_dim = text_config.global_head_dim
    if getattr(text_config, "num_global_key_value_heads", None) is not None:
        text_config.num_key_value_heads = text_config.num_global_key_value_heads
    if not hasattr(text_config, "v_head_dim"):
        text_config.v_head_dim = text_config.head_dim
    for key in ("_name_or_path", "torch_dtype", "dtype"):
        if hasattr(config, key) and getattr(config, key, None) is not None:
            try:
                setattr(text_config, key, getattr(config, key))
            except Exception:
                pass
    logger.info("gemma-4: loading text-only backbone %s.", GEMMA4_TEXT_BACKBONE)
    return text_config


def _fold_gemma4_expert_scale(down_weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Fold Gemma's output scale into the source down-projection stack."""
    return down_weight * scale.detach().to(down_weight.dtype).view(-1, 1, 1)


def configure_gemma4_moe_layers(layers) -> int:
    """Opt every gemma-4 MoE block into Spyre (upstream ``configure_gemma4_moe_layers``)."""
    configured = 0
    for decoder in layers:
        moe = getattr(decoder, "moe", None)
        if moe is None:
            continue
        configure_spyre_moe_layer(
            moe.experts,
            SpyreMoERecipe(
                activation="gelu_tanh",
                routing="full_softmax",
                prepare_down_weight=partial(_fold_gemma4_expert_scale, scale=moe.per_expert_scale),
            ),
            top_k=moe.topk.topk_config.top_k,
        )
        moe._spyre_moe = True
        moe._spyre_moe_key = f"moe.{moe.layer_id}"
        configured += 1
    if configured:
        logger.info("Spyre: configured %d Gemma-4 MoE layers.", configured)
    return configured


def _gemma4_moe_forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
    """``Gemma4MoE.forward`` on Spyre: experts from raw router logits, scale folded.

    The per-layer key reaches the op only as a hoisted ``LayerName``; branching on
    it here would make Dynamo guard on its value and compile one graph per layer.
    """
    if not getattr(self, "_spyre_moe", False):
        return self._sglang_forward(hidden_states, router_logits)
    from sglang_spyre_backend.opaque import LayerName

    num_tokens, hidden_dim = hidden_states.shape
    out = torch.ops.sglang_spyre.moe_forward(
        hidden_states, router_logits, LayerName(self._spyre_moe_key)
    )
    return out.view(num_tokens, hidden_dim)


_INSTALLED = False


def _patch_final_logit_softcap() -> None:
    """Take SGLang's host softcap path, not its Triton kernel.

    ``LogitsProcessor`` applies ``final_logit_softcapping`` (gemma-4: 30.0) with a
    Triton kernel unless the engine is the CPU one. On Spyre the LM head returns
    host logits, as spyre-inference's ``SpyreLogitsProcessor._apply_head`` does,
    so the softcap is the same ``cap * tanh(logits / cap)`` vLLM applies.
    """
    from sglang.srt.layers import logits_processor as lp

    lp._is_cpu = True


def install_config_hooks() -> None:
    """The config-parse repair alone: runs in every process, from activate()."""
    _patch_config_parse()


def install() -> None:
    """Patch SGLang's config parse, final-logit softcap and gemma-4 MoE forward."""
    global _INSTALLED
    if _INSTALLED:
        return
    _patch_config_parse()
    _patch_final_logit_softcap()
    try:
        from sglang.srt.models.gemma4_causal import Gemma4MoE

        Gemma4MoE._sglang_forward = Gemma4MoE.forward
        Gemma4MoE.forward = _gemma4_moe_forward
    except ImportError:
        logger.debug("SGLang has no gemma4_causal; skipping the gemma-4 MoE adaptation")
    _INSTALLED = True


def _patch_config_parse() -> None:
    from sglang.srt.utils.hf_transformers import config as hf_config_mod

    parser_cls = hf_config_mod.HfModelConfigParser
    if getattr(parser_cls, "_spyre_patched", False):
        return
    original_parse = parser_cls.parse

    def parse(self, model, trust_remote_code, revision=None, **kwargs):
        from transformers import AutoConfig

        # Load once ourselves to repair the gemma-4 view before SGLang reads it.
        real_from_pretrained = AutoConfig.from_pretrained

        def from_pretrained(*args, **kw):
            cfg = real_from_pretrained(*args, **kw)
            repair_head_dim_access(cfg)
            return force_text_backbone(cfg)

        AutoConfig.from_pretrained = from_pretrained
        try:
            return original_parse(self, model, trust_remote_code, revision, **kwargs)
        finally:
            AutoConfig.from_pretrained = real_from_pretrained

    parser_cls.parse = parse
    parser_cls._spyre_patched = True


def post_load(model: torch.nn.Module) -> None:
    """Load-time adaptations that need the loaded weights (upstream load fix-ups)."""
    body = getattr(model, "model", None)
    layers = getattr(body, "layers", None) if body is not None else None
    if layers is not None and type(model).__name__ == GEMMA4_TEXT_BACKBONE:
        is_moe = getattr(model.config, "enable_moe_block", False)
        if not configure_gemma4_moe_layers(layers) and is_moe:
            raise RuntimeError(
                "gemma-4 MoE checkpoint, but no decoder layer exposes `moe`: SGLang has "
                "moved it; update configure_gemma4_moe_layers."
            )
        for decoder in layers:
            moe = getattr(decoder, "moe", None)
            if moe is not None and getattr(moe, "_spyre_moe", False):
                prepare_moe_layer(moe.experts)
                # Folded into the down stack; the routing weights must not apply it twice.
                moe.per_expert_scale.data.fill_(1.0)


__all__ = [
    "configure_gemma4_moe_layers",
    "force_text_backbone",
    "install",
    "post_load",
    "repair_head_dim_access",
]
