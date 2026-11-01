"""Model-body placement and per-block compile for Spyre.

Mirrors spyre-inference's ``_SpyreModelWrapper`` boundary and its per-block
compile (``SPYRE_COMPILE_GRANULARITY=block``, ``compile_granularity.py`` at
``0ccdbf6ac2da7faacf98db93464b274fb8f40126``, #1320/#1395). SGLang keeps
``device="cpu"``, so the KV page table, sampling indices and scheduling stay on
the host. The model body moves to Spyre and the CPU<->Spyre boundary is crossed
once per forward:

  input_ids (CPU) -> embed_tokens (CPU) --hook--> hidden on Spyre
                  -> [decoder blocks on Spyre, each its own compiled graph]
                  -> final norm (Spyre) --hook--> hidden on CPU
                  -> lm_head projection on Spyre (custom_ops.weight_placement)

Each decoder block is wrapped so that its dense part (norms, projections, RoPE,
MLP) compiles into one graph per block class and token bucket, while attention
and MoE stay opaque custom ops that run spyre-inference's own compiled kernels.
That is the graph split spyre-inference makes: its attention core is a vLLM
custom op outside the block graph, and its MoE dispatch takes the opaque op for
token counts that are not a whole stick (#1175).
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

SPYRE = torch.device("spyre")
CPU = torch.device("cpu")


def _move(obj, dev):
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, (tuple, list)):
        return type(obj)(_move(x, dev) for x in obj)
    return obj


def _find(model, *paths):
    for p in paths:
        cur, ok = model, True
        for part in p.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and cur is not model:
            return cur
    return None


def _text_body(model):
    return _find(model, "model", "language_model", "transformer")


def default_token_buckets(max_num_batched_tokens: int, max_num_seqs: int) -> list[int]:
    """spyre-inference's default decoder ``compile_sizes`` (platform.py, #718).

    Decode packs one token per running sequence and a prefill lands on the
    largest bucket, so the buckets are the powers of two below ``max_num_seqs``,
    ``max_num_seqs`` itself, and the prefill bucket ``min(budget, 512)``.
    """
    prefill = 512
    max_capture = min(max_num_batched_tokens, prefill)
    num_seqs = min(max_num_seqs, max_capture)
    sizes = {max_capture, num_seqs}
    size = 1
    while size < num_seqs:
        sizes.add(size)
        size *= 2
    return sorted(sizes)


class _BodyBuckets:
    """Pad every body forward to a token bucket (spyre-inference's
    ``SpyreShapeBucketer.warmed_bucket``) and slice the output back.

    Integer inputs cross to Spyre here once per forward, as int64, as in
    ``_SpyreModelWrapper.__call__``, so no block graph starts with an H2D copy of
    ``positions``. ``input_ids`` stay on the host for the CPU embedding.
    """

    def __init__(self, buckets: list[int]):
        self.buckets = sorted(buckets)
        self.real_tokens: int | None = None

    def _bucket(self, n: int) -> int:
        import bisect

        i = bisect.bisect_left(self.buckets, n)
        return self.buckets[i] if i < len(self.buckets) else n

    def pre(self, module, args, kwargs):
        args = list(args)

        def get(i, name):
            if name in kwargs:
                return kwargs[name], ("kw", name)
            if len(args) > i:
                return args[i], ("pos", i)
            return None, None

        def put(where, value):
            if where[0] == "kw":
                kwargs[where[1]] = value
            else:
                args[where[1]] = value

        kwargs = dict(kwargs)
        input_ids, ids_at = get(0, "input_ids")
        positions, pos_at = get(1, "positions")
        n = int(positions.shape[0]) if isinstance(positions, torch.Tensor) else None
        self.real_tokens = n
        if n is not None:
            pad = self._bucket(n) - n
            if pad > 0:
                positions = torch.nn.functional.pad(positions, (0, pad))
                if isinstance(input_ids, torch.Tensor):
                    put(ids_at, torch.nn.functional.pad(input_ids, (0, pad)))
            put(pos_at, positions.to(dtype=torch.int64).to(SPYRE))
        return tuple(args), kwargs

    def post(self, module, args, kwargs, out):
        n = self.real_tokens
        out = _move(out, CPU)
        if n is not None and isinstance(out, torch.Tensor) and out.shape[0] != n:
            out = out[:n]
        return out


def bodify_model_for_spyre(model, token_buckets: list[int] | None = None):
    """Move the model body onto Spyre, keep boundaries on CPU, install hooks."""
    from sglang_spyre_backend.device_ops import ROPE_CLASSES, rope_device_cache

    rope_types = tuple(
        getattr(__import__(m, fromlist=[c]), c) for m, c in ROPE_CLASSES
    )
    # 1. Everything to Spyre, then pull back what must stay on CPU. RoPE modules keep
    # their host cos_sin_cache: the device rotation cache is built from it below.
    rope_modules = [m for m in model.modules() if isinstance(m, rope_types)]
    for m in rope_modules:
        m._spyre_skip_move = True
    _orig_apply = torch.nn.Module._apply

    def _apply_skip_rope(self, fn, recurse=True):
        if getattr(self, "_spyre_skip_move", False):
            return self
        return _orig_apply(self, fn, recurse)

    torch.nn.Module._apply = _apply_skip_rope
    try:
        model.to(SPYRE)
    finally:
        torch.nn.Module._apply = _orig_apply

    body = _text_body(model)
    embed = _find(body, "embed_tokens") if body is not None else None
    if embed is not None:
        # The gather stays on the host; a tied head's row-padded Wᵀ stays on Spyre,
        # as spyre-inference's SpyreParallelLMHead._apply keeps it (#653).
        head_t = embed._parameters.pop("padded_weight_t", None)
        embed.to(CPU)
        if head_t is not None:
            embed._parameters["padded_weight_t"] = head_t
    for m in rope_modules:
        rope_device_cache(m, SPYRE, torch.float16)

    # 2. Re-dispatch fused ops: dispatch is resolved lazily on first call, but any
    # op already called (or pinned in __init__) cached a CPU path.
    import sglang_spyre_backend.register  # noqa: F401

    redispatched = 0
    for m in model.modules():
        if hasattr(m, "dispatch_forward") and hasattr(m, "_forward_method"):
            try:
                m._forward_method = m.dispatch_forward()
                redispatched += 1
            except Exception:
                logger.debug("re-dispatch failed for %s", type(m).__name__, exc_info=True)
    logger.info("bodify: re-dispatched %d fused ops", redispatched)

    # 3. Boundary hooks: embed output -> Spyre, body output -> CPU.
    if embed is not None:
        embed.register_forward_hook(
            lambda mod, inp, out: _move(out, SPYRE).to(torch.float16)
            if isinstance(out, torch.Tensor)
            else _move(out, SPYRE)
        )
    if body is not None:
        buckets = _BodyBuckets(token_buckets or [])
        body.register_forward_pre_hook(buckets.pre, with_kwargs=True)
        body.register_forward_hook(buckets.post, with_kwargs=True)

    # 4. Attention and MoE behind opaque ops, so a block graph holds neither.
    import types

    from sglang.srt.layers.radix_attention import RadixAttention

    from sglang_spyre_backend import opaque

    opaque.register_attention_layers(model)
    for m in model.modules():
        if isinstance(m, RadixAttention):
            m.forward = types.MethodType(opaque.spyre_radix_attention_forward, m)
        moe = getattr(m, "moe", None)
        if moe is not None and getattr(moe, "_spyre_moe", False):
            opaque.register_moe_layer(moe._spyre_moe_key, moe.experts)

    if os.environ.get("SGLANG_SPYRE_COMPILE_BLOCKS", "1") != "0":
        compile_blocks(body if body is not None else model)
    return model


def _decoder_layers(body):
    layers = getattr(body, "layers", None)
    if isinstance(layers, torch.nn.ModuleList):
        return list(layers)
    return []


def compile_blocks(body) -> int:
    """Compile every decoder block in place (upstream ``compile_blocks``).

    ``dynamic=False`` is mandatory: the Spyre backend rejects SymInt shapes.
    Blocks are compiled in place so parameter names, and with them weight reload,
    stay as loaded; aliased blocks are compiled once (#797).
    """
    seen: set[int] = set()
    count = 0
    for block in _decoder_layers(body):
        if id(block) in seen:
            continue
        seen.add(id(block))
        block.compile(backend="inductor", fullgraph=False, dynamic=False)
        count += 1
    logger.info("Wrapped %d transformer blocks for per-block compile on Spyre.", count)
    return count


__all__ = ["bodify_model_for_spyre", "compile_blocks", "default_token_buckets"]
