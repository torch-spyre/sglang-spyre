# SPDX-License-Identifier: Apache-2.0
"""SGLang Transformers wrapper with opt-in HF-adapters body execution."""

from __future__ import annotations

import copy
import os

import torch
from sglang.srt.models.transformers import (
    TransformersEmbeddingModel as SGLangEmbeddingModel,
)
from sglang.srt.models.transformers import TransformersForCausalLM as SGLangCausalLM
from sglang.srt.runtime_context import get_server_args
from torch import nn

from sglang_spyre_backend.hf_adapters import (
    AdapterBody,
    PagedAdapterAttention,
    prepare_execution,
    uses_hf_adapters,
    validate_config,
)


class AdapterTransformersMixin:
    def __init__(self, config, quant_config=None, prefix=""):
        self._hf_adapters = uses_hf_adapters(get_server_args())
        self._adapter_body = None
        self._adapter_loaded = False
        if self._hf_adapters and quant_config is not None:
            raise ValueError("hf_adapters requires unquantized checkpoint weights")
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        if self._hf_adapters and self.text_config.model_type == "distilbert":
            # This is an internal encoder stack, unlike GPT's backbone prefix.
            self.weight_mapper = copy.deepcopy(self.weight_mapper)
            self.weight_mapper.orig_to_new_prefix["transformer."] = "model.transformer."

    def _create_transformers_model(self):
        if not self._hf_adapters:
            return super()._create_transformers_model()
        from sglang.srt.models.transformers import _init_on_device_without_buffers
        from transformers import AutoModel
        from transformers.models.auto.auto_factory import _get_model_class

        # As in spyre-inference, resolve the actual stock backbone even when
        # config.architectures is absent. Unknown/custom models stay upstream.
        if type(self.config) not in AutoModel._model_mapping:
            return super()._create_transformers_model()
        model_class = _get_model_class(self.config, AutoModel._model_mapping)
        # Match SGLang's construction capability check. Compatible models keep
        # its attention selection and all other construction behavior.
        if getattr(model_class, "_supports_attention_backend", True):
            return super()._create_transformers_model()

        # GPT-Neo and MPNet cannot use HF's custom-attention interface. Their
        # adapter drivers supply attention independently of the HF forward.
        self.text_config._attn_implementation = "eager"
        with _init_on_device_without_buffers(torch.device("meta")):
            return AutoModel.from_config(
                self.config, dtype=torch.get_default_dtype(), trust_remote_code=False
            )

    def recursive_replace(self):
        if not self._hf_adapters:
            super().recursive_replace()

    def _init_parameters(self, module):
        if not self._hf_adapters:
            return super()._init_parameters(module)
        # SGLang's get_device() returns the platform device even with CPU scheduling.
        # Adapters must transform fully loaded CPU weights before placement.
        for child in module.modules():
            for name, param in child.named_parameters(recurse=False):
                if param.is_meta:
                    setattr(
                        child,
                        name,
                        nn.Parameter(
                            torch.empty_like(param, device="cpu"),
                            requires_grad=param.requires_grad,
                        ),
                    )

    def replace_vocab_embed_class(self, module):
        if not self._hf_adapters:
            super().replace_vocab_embed_class(module)

    def _create_attention_instances(self):
        if self._hf_adapters:
            return {}
        return super()._create_attention_instances()

    @property
    def _can_torch_compile(self):
        return False if self._hf_adapters else super()._can_torch_compile

    def load_weights(self, weights):
        if self._hf_adapters and self._adapter_loaded:
            raise ValueError("Reloading HF-adapters weights is not supported")
        result = super().load_weights(weights)
        if self._hf_adapters:
            if self.lm_head is not None and getattr(
                self.text_config, "tie_word_embeddings", False
            ):
                # Resolve tying against the materialized, loaded embedding.
                self.lm_head.weight = self.model.get_input_embeddings().weight
            self._adapter_loaded = True
        return result

    def _prepare_execution(self, runner, task="generate"):
        if not self._hf_adapters or self._adapter_body is not None:
            raise RuntimeError("Prepare HF-adapters exactly once in adapter mode")
        if not self._adapter_loaded:
            raise RuntimeError("Load checkpoint weights before adapter preparation")
        if runner.startup_weight_load is not None:
            raise ValueError("hf_adapters does not support overlapped weight loading")
        validate_config(runner.server_args, runner.model_config)
        prepared = prepare_execution(
            self.model, max_model_len=runner.model_config.context_len, task=task
        )
        width = prepared.runtime.attentions[0].spec.head_size
        runner.model_config.head_dim = width
        for name in ("v_head_dim", "swa_head_dim", "swa_v_head_dim"):
            if hasattr(runner.model_config, name):
                setattr(runner.model_config, name, width)
        return prepared

    def prepare_hf_adapters(self, runner):
        from sglang_spyre_backend.custom_ops.weight_placement import SpyreLMHead
        from sglang_spyre_backend.model_wrapper import default_token_buckets

        prepared = self._prepare_execution(runner)
        self._bind_adapter_execution(prepared)
        # Adapter preparation owns HF geometry/scale; publish cache dimensions on the runner.
        if prepared.runtime.logits_scale is not None:
            self.logits_processor.logit_scale = prepared.runtime.logits_scale

        device = torch.device("spyre")
        head = SpyreLMHead(getattr(self.lm_head, "quant_method", None))
        head.process_weights_after_loading(self.lm_head)
        self.lm_head.quant_method = head
        self.model.to(device)
        chunk = getattr(runner.server_args, "chunked_prefill_size", None)
        budget = (
            min(chunk, runner.model_config.context_len)
            if chunk and chunk > 0
            else runner.model_config.context_len
        )
        count = getattr(runner.server_args, "max_running_requests", None) or 1
        buckets = default_token_buckets(budget, count)
        # Include the full admitted prefill budget, even when it exceeds 512.
        buckets.append(max(budget, count))
        self._adapter_body = AdapterBody(prepared, buckets, device)
        if os.environ.get("SGLANG_SPYRE_COMPILE_BLOCKS", "1") != "0":
            prepared.runtime.compile_body(self.model)

    def _bind_adapter_execution(self, prepared):
        from sglang.srt.layers.radix_attention import RadixAttention

        from sglang_spyre_backend import opaque

        self.adapter_attentions = nn.ModuleList()
        for executor in prepared.runtime.attentions:
            spec = executor.spec
            attention = RadixAttention(
                num_heads=spec.num_heads,
                num_kv_heads=spec.num_kv_heads,
                head_dim=spec.head_size,
                scaling=spec.scale,
                layer_id=spec.layer_idx,
                prefix=f"adapter_attentions.{spec.layer_idx}.attn",
            )
            bridge = PagedAdapterAttention(attention)
            self.adapter_attentions.append(bridge)
            self.attention_instances[spec.layer_idx] = attention
            executor.bind(bridge)
        opaque.register_attention_layers(self)

    def _forward_hidden_states(
        self, input_ids, positions, forward_batch, input_embeds=None
    ):
        if not self._hf_adapters:
            return super()._forward_hidden_states(
                input_ids, positions, forward_batch, input_embeds
            )
        if self._adapter_body is None:
            raise RuntimeError("HF-adapters body has not been prepared")
        if input_ids is None or input_embeds is not None:
            raise ValueError("HF-adapters serving requires token IDs")
        return self._adapter_body(input_ids, positions)


class TransformersForCausalLM(AdapterTransformersMixin, SGLangCausalLM):
    pass


class TransformersEmbeddingModel(AdapterTransformersMixin, SGLangEmbeddingModel):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        if self._hf_adapters:
            from sglang.srt.layers.pooler import Pooler, PoolingType

            from sglang_spyre_backend.embedding import pooling_config

            pooling, normalize = pooling_config(get_server_args())
            self.pooler = Pooler(PoolingType[pooling], normalize=normalize)

    def prepare_hf_adapters(self, runner):
        from sglang_spyre_backend.embedding import (
            CacheFreeAdapterAttention,
            EmbeddingBody,
            EmbeddingMask,
        )

        prepared = self._prepare_execution(runner, task="embed")
        mask = EmbeddingMask()
        self.adapter_attentions = nn.ModuleList()
        for executor in prepared.runtime.attentions:
            bridge = CacheFreeAdapterAttention(executor.spec, mask)
            self.adapter_attentions.append(bridge)
            executor.bind(bridge)
        device = torch.device("spyre")
        self.model.to(device)
        self.adapter_attentions.to(device)
        self._adapter_body = EmbeddingBody(
            prepared,
            mask,
            max_model_len=runner.model_config.context_len,
            max_requests=runner.server_args.max_running_requests or 1,
            device=device,
        )
        if os.environ.get("SGLANG_SPYRE_COMPILE_BLOCKS", "1") != "0":
            prepared.runtime.compile_body(self.model)

    def _forward_hidden_states(
        self, input_ids, positions, forward_batch, input_embeds=None
    ):
        if not self._hf_adapters:
            return super()._forward_hidden_states(
                input_ids, positions, forward_batch, input_embeds
            )
        if input_ids is None or input_embeds is not None:
            raise ValueError("HF-adapters embeddings require token IDs")
        if self._adapter_body is None:
            raise RuntimeError("HF-adapters embedding body has not been prepared")
        prefixes = forward_batch.extend_prefix_lens_cpu
        if prefixes is not None and any(prefixes):
            raise ValueError(
                "HF-adapters embeddings require complete requests without cached prefixes"
            )
        return self._adapter_body(
            input_ids,
            positions,
            forward_batch.extend_seq_lens_cpu,
            getattr(forward_batch, "token_type_ids", None),
        )


def install() -> None:
    from sglang.srt.models.registry import ModelRegistry

    ModelRegistry.models["TransformersForCausalLM"] = TransformersForCausalLM
    ModelRegistry.models["TransformersEmbeddingModel"] = TransformersEmbeddingModel
