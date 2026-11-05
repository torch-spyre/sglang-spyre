"""Card-free: the ported MoE routing and Gemma-4 recipe match SGLang's reference.

The Spyre kernels themselves need a card; their routing math and the per-expert
scale fold are host code and are checked here against SGLang's own Gemma-4
routing (softmax over all experts, top-k, renormalize, per-expert scale).
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from sglang_spyre_backend import moe

sglang_available = importlib.util.find_spec("sglang") is not None


def _gemma4_reference(logits, top_k, per_expert_scale, x_experts):
    probs = torch.softmax(logits.float(), dim=-1)
    w, ids = torch.topk(probs, top_k, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    w = w * per_expert_scale[ids]
    return (w.unsqueeze(-1) * x_experts.gather(1, ids.unsqueeze(-1).expand(-1, -1, x_experts.shape[-1]))).sum(1)


def test_full_softmax_routing_matches_gemma4_reference():
    torch.manual_seed(0)
    tokens, experts, top_k = 5, 16, 4
    logits = torch.randn(tokens, experts)
    weights, ids = moe._routing_weights(logits, top_k, "full_softmax", torch.float32)
    probs = torch.softmax(logits, dim=-1)
    ref_w, ref_ids = torch.topk(probs, top_k, dim=-1)
    torch.testing.assert_close(ids, ref_ids)
    torch.testing.assert_close(weights, ref_w / ref_w.sum(-1, keepdim=True))


def test_single_row_topk_pads_to_two_rows():
    logits = torch.randn(1, 8)
    weights, ids = moe._topk(logits, 2)
    assert weights.shape == (1, 2) and ids.shape == (1, 2)


def test_folding_expert_scale_into_down_weight_equals_scaled_routing():
    """Folding ``per_expert_scale`` into the down stack (spyre-inference) equals
    folding it into the routing weights (SGLang's Gemma4MoE)."""
    from sglang_spyre_backend.models import _fold_gemma4_expert_scale

    torch.manual_seed(1)
    tokens, experts, hidden, inter, top_k = 3, 8, 16, 12, 2
    x = torch.randn(tokens, hidden)
    logits = torch.randn(tokens, experts)
    gate_up = torch.randn(experts, 2 * inter, hidden)
    down = torch.randn(experts, hidden, inter)
    scale = torch.rand(experts) + 0.5

    def expert_outputs(down_w):
        g = torch.einsum("th,eih->tei", x, gate_up[:, :inter])
        u = torch.einsum("th,eih->tei", x, gate_up[:, inter:])
        act = torch.nn.functional.gelu(g, approximate="tanh") * u
        return torch.einsum("tei,ehi->teh", act, down_w)

    weights, ids = moe._routing_weights(logits, top_k, "full_softmax", torch.float32)
    folded = expert_outputs(_fold_gemma4_expert_scale(down, scale))
    got = (weights.unsqueeze(-1) * folded.gather(1, ids.unsqueeze(-1).expand(-1, -1, hidden))).sum(1)
    expected = _gemma4_reference(logits, top_k, scale, expert_outputs(down))
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("hidden,top_k,stick,cores", [(2816, 8, 64, 32), (4096, 8, 64, 32)])
def test_moe_chunks_divide_hidden_into_whole_sticks(hidden, top_k, stick, cores):
    chunks = moe._derive_moe_chunks(hidden, top_k, stick, cores)
    assert hidden % (chunks * stick) == 0


def test_moe_chunks_reject_unaligned_hidden():
    with pytest.raises(ValueError, match="multiple of 64"):
        moe._derive_moe_chunks(100, 8, 64, 32)


@pytest.mark.skipif(not sglang_available, reason="sglang not importable in this environment")
def test_configure_rejects_quantized_experts():
    class _Quantized:
        pass

    layer = torch.nn.Module()
    layer.quant_method = _Quantized()
    with pytest.raises(NotImplementedError, match="quantized"):
        moe.configure_spyre_moe_layer(
            layer, moe.SpyreMoERecipe("gelu_tanh", "full_softmax"), top_k=8
        )
