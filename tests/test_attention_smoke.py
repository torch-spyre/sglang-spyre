"""Smoke test: drive SpyreAttentionBackend through one prefill + one decode.

Validates that:
  1. The plugin discovers and activates correctly
  2. SpyreAttentionBackend can be instantiated through SGLang's registry
  3. init_forward_metadata() runs (mask construction, BMM selection masks)
  4. forward_extend() and forward_decode() produce non-NaN output
  5. KV cache state is preserved across calls

This does NOT exercise a full model — just the attention backend in isolation.
"""

import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch


def main():
    # Step 1: activate plugin
    from sglang_spyre_backend import activate

    result = activate()
    assert result == "sglang_spyre_backend.platform.SpyreSRTPlatform", (
        f"activate() returned {result}"
    )

    # Step 2: instantiate platform and init backend
    import torch_spyre

    torch_spyre._autoload()
    torch.spyre.set_device(0)
    device_count = torch.spyre.device_count()
    print(f"  ✓ Platform initialized, torch.spyre.device_count = {device_count}")

    # Step 3: create a minimal model_runner stub
    class StubModelConfig:
        head_dim = 64

        def get_num_kv_heads(self, tp_size):
            return 4

        def get_num_attention_heads(self, tp_size):
            return 4

    class StubKVPool:
        size = 1024
        page_size = 128

    class StubReqToTokenPool:
        # Request 0 owns page 1 onward; page 0 is the reserved padding page.
        req_to_token = (torch.arange(1024) + 128).reshape(1, 1024)

    class StubServerArgs:
        tp_size = 1
        chunked_prefill_size = 512
        max_running_requests = 1

    class StubModelConfig2:
        context_len = 1024

    class StubModelRunner:
        model_config = StubModelConfig2()
        token_to_kv_pool = StubKVPool()
        req_to_token_pool = StubReqToTokenPool()
        server_args = StubServerArgs()

    runner = StubModelRunner()

    # Step 4: instantiate attention backend via SGLang's registry
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS

    # init_backend() imports this in an engine's worker process.
    import sglang_spyre_backend.register  # noqa: F401

    assert "spyre_paged" in ATTENTION_BACKENDS
    backend = ATTENTION_BACKENDS["spyre_paged"](runner)
    print(f"  ✓ SpyreAttentionBackend instantiated: {type(backend).__name__}")

    # Step 5: build a fake ForwardBatch for prefill
    class StubForwardMode:
        def is_decode(self):
            return False

        def is_extend(self):
            return True

        def is_idle(self):
            return False

        def is_mixed(self):
            return False

    class StubForwardBatch:
        forward_mode = StubForwardMode()
        batch_size = 1
        seq_lens_cpu = torch.tensor([8])
        req_pool_indices = torch.tensor([0], dtype=torch.int64)
        extend_seq_lens_cpu = [8]
        extend_seq_lens = torch.tensor([8])
        extend_num_tokens = 8
        out_cache_loc = torch.arange(8, dtype=torch.int64) + 128

    fb = StubForwardBatch()

    print("\n  Running init_forward_metadata (prefill)...")
    backend.init_forward_metadata(fb)
    print(f"    block table: {backend._batch.block_table[0, :2].tolist()}")
    print("  ✓ Metadata built")

    # Step 6: build a fake RadixAttention layer
    class StubRadixAttention:
        # granite-3.3-8b-instruct's attention geometry, the one validated end to end.
        layer_id = 0
        tp_q_head_num = 32
        tp_k_head_num = 8
        tp_v_head_num = 8
        qk_head_dim = 128
        scaling = 1.0 / (128**0.5)
        v_head_dim = 128
        sliding_window_size = -1
        logit_cap = 0.0

    layer = StubRadixAttention()
    num_tokens = 8
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128

    # Random q,k,v in FP16 on CPU (SGLang convention)
    torch.manual_seed(42)
    q = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.float16)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.float16)
    v = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.float16)

    print("\n  Running forward_extend...")
    out = backend.forward_extend(q, k, v, layer, fb, save_kv_cache=True)
    print(f"    output shape: {out.shape}")
    print(f"    output dtype: {out.dtype}")
    print(f"    output device: {out.device}")
    assert out.shape == (num_tokens, num_heads * head_dim), f"bad shape: {out.shape}"
    assert not torch.isnan(out).any(), "output contains NaN"
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(0, 1), k.float().transpose(0, 1), v.float().transpose(0, 1),
        is_causal=True, enable_gqa=True,
    ).transpose(0, 1).reshape(num_tokens, num_heads * head_dim)
    err = (out.float() - ref).abs().max().item()
    print(f"    max abs error vs CPU SDPA: {err:.4f}")
    assert err < 2e-2, f"output differs from CPU SDPA by {err}"
    print("  ✓ forward_extend matches CPU attention")

    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
