"""Structural tests for the reuse-refactor (no Spyre card required).

These assert the Phase-A reuse decisions hold:
  - SpyreAttnBackend is a *subclass* of SGLang's TorchNativeAttnBackend
    (we inherit the gather/save/forward plumbing, override only the SDPA seam).
  - SpyreMHATokenToKVPool is a real *subclass* of MHATokenToKVPool, not an
    identity re-export.
  - The platform no longer ships a bespoke eager graph runner (core runs eager
    with graph_runner=None when support_cuda_graph() is False), and there is no
    graph_runner module left to import.

Requires SGLang importable (torch + sglang). Skips cleanly if not present, so
it is safe to collect anywhere; run it on the dev pod for real coverage.
"""

import importlib
import importlib.util

import pytest

sglang_available = importlib.util.find_spec("sglang") is not None
pytestmark = pytest.mark.skipif(
    not sglang_available, reason="sglang not importable in this environment"
)


def test_paged_attention_uses_framework_neutral_port():
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

    from sglang_spyre_backend.attention import SpyreAttentionBackend
    from sglang_spyre_backend.ops import (
        batched_decode_head_major,
        page_attn_head_major_decode,
        page_attn_head_major_prefill,
        reshape_and_cache_head_major,
    )

    assert issubclass(SpyreAttentionBackend, AttentionBackend)
    assert callable(page_attn_head_major_prefill.page_attn_head_major_prefill_kernel)
    assert callable(page_attn_head_major_decode.page_attn_head_major_decode_kernel)
    assert callable(batched_decode_head_major.batched_decode_head_major_kernel)
    assert callable(reshape_and_cache_head_major.reshape_and_cache_head_major_kernel)


def test_kv_pool_is_real_subclass():
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    from sglang_spyre_backend.memory_pool import SpyreMHATokenToKVPool

    assert issubclass(SpyreMHATokenToKVPool, MHATokenToKVPool)
    # Must be a distinct subclass, not the identity re-export it used to be.
    assert SpyreMHATokenToKVPool is not MHATokenToKVPool


def test_no_bespoke_graph_runner():
    # The eager graph runner module was deleted; nothing should import it.
    assert importlib.util.find_spec("sglang_spyre_backend.graph_runner") is None

    from sglang_spyre_backend.platform import SpyreSRTPlatform

    plat = SpyreSRTPlatform.__new__(SpyreSRTPlatform)
    assert plat.support_cuda_graph() is False
    # get_graph_runner_cls is dead code under eager execution and must refuse.
    with pytest.raises(NotImplementedError):
        plat.get_graph_runner_cls()


def test_attention_registry_contains_both_paths():
    # ``spyre_paged`` is the synchronized default; ``spyre`` remains as the
    # legacy dense fallback while downstream users migrate.
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS

    import sglang_spyre_backend.register  # noqa: F401  (registers as side effect)

    assert "spyre" in ATTENTION_BACKENDS
    assert "spyre_paged" in ATTENTION_BACKENDS
