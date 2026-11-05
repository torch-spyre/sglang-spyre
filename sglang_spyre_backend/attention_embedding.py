# SPDX-License-Identifier: Apache-2.0
"""SGLang lifecycle for cache-free HF-adapters embedding attention."""

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


class SpyreEmbeddingAttentionBackend(AttentionBackend):
    needs_cpu_seq_lens = True

    def __init__(self, model_runner):
        super().__init__()
        model = model_runner.model
        if not getattr(model, "_hf_adapters", False) or model._adapter_body is None:
            raise ValueError(
                "spyre_embedding requires the HF-adapters embedding wrapper"
            )
        model._adapter_body.warmup()

    def init_forward_metadata(self, forward_batch):
        if forward_batch.forward_mode.is_decode():
            raise ValueError("spyre_embedding does not support decoding")

    def forward_extend(self, *args, **kwargs):
        raise RuntimeError("Embedding attention must execute inside adapter blocks")

    def forward_decode(self, *args, **kwargs):
        raise RuntimeError("Embedding attention does not use a KV cache")

    def init_cuda_graph_state(self, *args, **kwargs):
        raise NotImplementedError("spyre_embedding does not use CUDA graphs")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1
