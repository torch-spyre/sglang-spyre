"""Register the Spyre attention backends with SGLang's attention registry.

The registry uses a decorator pattern: @register_attention_backend("name")
wraps a factory that takes (model_runner) and returns the backend instance.

Two backends:
  - "spyre"        : SpyreAttnBackend — a thin subclass of SGLang's
                     TorchNativeAttnBackend that overrides only the inner
                     attention math (bucketed _attn_4d on Spyre). KV stays in
                     SGLang's pool (CPU). Works today.
  - "spyre_paged"  : SpyreAttentionBackend (BMM scatter/gather, KV-on-Spyre).
                     WIP — torch_spyre layout constraints cause crashes on
                     some shapes.
"""

from sglang.srt.layers.attention.attention_registry import register_attention_backend


@register_attention_backend("spyre")
def _create_spyre_attention_backend(model_runner):
    from sglang_spyre_backend.attention_native import SpyreAttnBackend
    return SpyreAttnBackend(model_runner)


@register_attention_backend("spyre_paged")
def _create_spyre_paged_attention_backend(model_runner):
    from sglang_spyre_backend.attention import SpyreAttentionBackend
    return SpyreAttentionBackend(model_runner)


@register_attention_backend("spyre_embedding")
def _create_spyre_embedding_attention_backend(model_runner):
    from sglang_spyre_backend.attention_embedding import SpyreEmbeddingAttentionBackend

    return SpyreEmbeddingAttentionBackend(model_runner)


__all__ = [
    "_create_spyre_attention_backend",
    "_create_spyre_paged_attention_backend",
    "_create_spyre_embedding_attention_backend",
]
