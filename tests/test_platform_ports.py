"""Card-free checks for the SGLang 0.5.21 glue around the ported Spyre code.

Requires SGLang importable; skips cleanly otherwise, like test_reuse_structure.py.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

sglang_available = importlib.util.find_spec("sglang") is not None
pytestmark = pytest.mark.skipif(
    not sglang_available, reason="sglang not importable in this environment"
)


def _allocator(num_pages=8, page=4):
    from sglang_spyre_backend.paged_allocator import SpyrePagedAllocator

    alloc = SpyrePagedAllocator.__new__(SpyrePagedAllocator)
    alloc.page_size = page
    alloc.device = "cpu"
    alloc.need_sort = False
    alloc.num_pages = num_pages
    alloc.free_pages = torch.arange(1, num_pages + 1, dtype=torch.int64)
    return alloc


def test_alloc_extend_matches_sglang_naive_layout():
    alloc = _allocator()
    out = alloc.alloc_extend(
        prefix_lens=torch.tensor([0]),
        prefix_lens_cpu=torch.tensor([0]),
        seq_lens=torch.tensor([10]),
        seq_lens_cpu=torch.tensor([10]),
        last_loc=torch.tensor([-1]),
        extend_num_tokens=10,
    )
    # Pages 1, 2, 3 in order: slots 4..15 cover tokens 0..9.
    assert out.tolist() == list(range(4, 14))
    assert alloc.free_pages.tolist()[0] == 4


def test_alloc_decode_stays_in_page_then_opens_the_next():
    alloc = _allocator()
    # Token at position 9 (seq_len 10) continues the page holding slot 13.
    out = alloc.alloc_decode(torch.tensor([10]), torch.tensor([10]), torch.tensor([12]))
    assert out.tolist() == [13]
    # Token at position 12 (seq_len 13) is a page's first: it takes the next free page.
    out = alloc.alloc_decode(torch.tensor([13]), torch.tensor([13]), torch.tensor([15]))
    assert out.tolist() == [1 * 4]


def test_gemma4_head_dim_repair_restores_bare_reads():
    from transformers import Gemma4TextConfig

    from sglang_spyre_backend.models import repair_head_dim_access

    cfg = Gemma4TextConfig(
        head_dim=256,
        global_head_dim=512,
        num_key_value_heads=8,
        num_global_key_value_heads=2,
        # As in gemma-4-26B-A4B: the global KV-head count applies under k_eq_v.
        attention_k_eq_v=True,
        num_hidden_layers=4,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
    )
    repair_head_dim_access(cfg)
    assert cfg.head_dim == 256
    assert cfg.num_key_value_heads == 8
    assert cfg.global_head_dim == 512
    assert cfg.num_global_key_value_heads == 2


def test_spyre_ops_register_on_sglang_fused_op_classes():
    from sglang.kernels.fused_op import BaseFusedOp
    from sglang.srt.layers.layernorm import Gemma4RMSNorm, RMSNorm
    from sglang.srt.layers.rotary_embedding.rope_variant import Gemma4RotaryEmbedding

    from sglang_spyre_backend.device_ops import SPYRE_KEY, register_spyre_ops

    registered = register_spyre_ops()
    assert {"RMSNorm", "Gemma4RMSNorm", "GeluAndMul", "Gemma4RotaryEmbedding"} <= set(
        registered
    )
    table = BaseFusedOp._oot_forward_registry[SPYRE_KEY]
    for cls in (RMSNorm, Gemma4RMSNorm, Gemma4RotaryEmbedding):
        assert cls in table


def test_rmsnorm_spyre_body_matches_sglang_native():
    from sglang.srt.layers.layernorm import RMSNorm

    from sglang_spyre_backend.device_ops import _rmsnorm_forward_spyre

    torch.manual_seed(0)
    norm = RMSNorm(64, eps=1e-5)
    norm.weight.data.normal_()
    x = torch.randn(3, 64, dtype=torch.float16)
    residual = torch.randn(3, 64, dtype=torch.float16)
    got, got_res = _rmsnorm_forward_spyre(norm, x.clone(), residual.clone())
    exp, exp_res = norm.forward_native(x.clone(), residual.clone())
    torch.testing.assert_close(got, exp, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(got_res, exp_res)


def test_rope_2x2_matches_sglang_native():
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    from sglang_spyre_backend.device_ops import _rope_forward_spyre

    torch.manual_seed(0)
    rope = RotaryEmbedding(128, 128, 256, 10000.0, True, torch.float32)
    positions = torch.tensor([0, 5, 200])
    q = torch.randn(3, 2 * 128)
    k = torch.randn(3, 128)
    got_q, got_k = _rope_forward_spyre(rope, positions, q.clone(), k.clone())
    exp_q, exp_k = rope.forward_native(positions, q.clone(), k.clone())
    torch.testing.assert_close(got_q, exp_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got_k, exp_k, atol=1e-5, rtol=1e-5)


def test_platform_defaults_follow_spyre_inference():
    from sglang_spyre_backend.platform import SpyreSRTPlatform

    class _Args:
        disable_cuda_graph = False
        dtype = "auto"
        attention_backend = None
        page_size = None
        tp_size = 1
        watchdog_timeout = 300
        disable_overlap_schedule = False
        disable_hybrid_swa_memory = False

    args = _Args()
    SpyreSRTPlatform.__new__(SpyreSRTPlatform).apply_server_args_defaults(args)
    assert (args.dtype, args.attention_backend, args.page_size) == ("float16", "spyre_paged", 128)
    assert args.disable_cuda_graph and args.watchdog_timeout >= 3600
    assert args.disable_overlap_schedule and args.disable_hybrid_swa_memory


def test_gemma4_router_norm_keeps_the_folded_scale():
    """The router folds scale * hidden**-0.5 into a with_scale=False norm's weight,
    which SGLang applies on its CUDA path; the Spyre body must apply it too."""
    from sglang.srt.layers.layernorm import Gemma4RMSNorm

    from sglang_spyre_backend.device_ops import _gemma4_rmsnorm_forward_spyre

    torch.manual_seed(0)
    norm = Gemma4RMSNorm(64, eps=1e-6, with_scale=False)
    norm.weight.data.copy_(torch.rand(64) + 0.5)
    x = torch.randn(3, 64)
    xf = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(_gemma4_rmsnorm_forward_spyre(norm, x), xf * norm.weight)

    scaled = Gemma4RMSNorm(64, eps=1e-6)
    scaled.weight.data.normal_()
    torch.testing.assert_close(
        _gemma4_rmsnorm_forward_spyre(scaled, x), scaled.forward_native(x)
    )
