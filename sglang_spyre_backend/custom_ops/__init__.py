"""Spyre custom-op installation: weight placement + model body wrapping.

This package is the "max-on-Spyre" path. When the user runs the Engine with
``device="spyre"`` and ``attention_backend="spyre"``, calling ``install()``
during init_backend() arranges for:

  1. ``register_spyre_ops()`` — register Spyre forward bodies for RMSNorm /
     SiluAndMul / GeluAndMul / RotaryEmbedding via SGLang's MultiPlatformOp
     OOT registry. (Defined in ``sglang_spyre_backend.device_ops``.)

  2. A ``ModelRunner.load_model`` monkeypatch that, after weights load:
       - swaps every Linear's quant_method to SpyreLinearMethod and moves
         each weight onto the Spyre device (``move_model_to_spyre``)
       - moves the rest of the body onto Spyre, pulls embed/lm_head/RoPE
         back to CPU, installs boundary hooks (``bodify_model_for_spyre``)

The cost of this path is per-op CPU<->Spyre transfer for any op that didn't
get a forward_spyre. The benefit is that Linear / RMSNorm / SiluAndMul /
RoPE all run on Spyre, with the residual stream resident on the device
between ops. This is the configuration that maximises Spyre coverage; it is
not always the fastest configuration on small models (Granite-1B at hidden
2048: per-op transfer can cost more than the compute saved). The user has
explicitly asked for max-coverage regardless of TPS.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def install() -> None:
    """Wire up max-on-Spyre. Idempotent."""
    _install_oot_op_forwards()
    _force_rotary_init_on_cpu()
    _install_post_load_hook()
    from sglang_spyre_backend import models

    models.install()


def _install_oot_op_forwards() -> None:
    from sglang_spyre_backend.device_ops import register_spyre_ops

    register_spyre_ops()


_PATCHED_ROTARY = False


def _force_rotary_init_on_cpu() -> None:
    """Force RotaryEmbedding's cos_sin_cache to be built on CPU.

    On device="spyre", SGLang's ``_compute_inv_freq`` runs ``torch.arange(...)
    / self.rotary_dim`` on the current device (Spyre), and the divide triggers
    the Spyre compiler's ``DtException: 4Bsplat`` because the resulting FIFO
    can't be lowered to a register. Building on CPU and letting downstream
    code transfer (or in our case, ``bodify_model_for_spyre`` keeps RoPE on
    CPU permanently) avoids the bad lowering.
    """
    global _PATCHED_ROTARY
    if _PATCHED_ROTARY:
        return

    import torch
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    _orig_compute_inv_freq = RotaryEmbedding._compute_inv_freq

    def _spyre_compute_inv_freq(self, base):
        # Inline the same body but pin device="cpu" on the arange.
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device="cpu")
                / self.rotary_dim
            )
        )
        return inv_freq

    RotaryEmbedding._compute_inv_freq = _spyre_compute_inv_freq

    _orig_compute_cos_sin = RotaryEmbedding._compute_cos_sin_cache

    def _spyre_compute_cos_sin_cache(self):
        # Mirror SGLang's body verbatim but pin every tensor on CPU so the
        # einsum/cos/sin/cat chain doesn't lower on Spyre.
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float, device="cpu")
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    RotaryEmbedding._compute_cos_sin_cache = _spyre_compute_cos_sin_cache
    _PATCHED_ROTARY = True


_PATCHED_LOAD_MODEL = False


def _install_post_load_hook() -> None:
    """Monkeypatch ModelRunner.load_model so weights land on Spyre after load.

    SGLang doesn't expose a clean post-load callback — load_model() builds the
    model on whatever ``device_config.device`` says and stops. We wrap it so
    that, on Spyre, after weights are in CPU (via the safetensors loader), we
    then push them onto the device and install body hooks.

    We patch the module function (``modules[...].ModelRunner.load_model``)
    by referencing the module object — the class binding doesn't yet exist
    when this is called from within the module's own import chain. We defer
    the actual class lookup until first call.
    """
    global _PATCHED_LOAD_MODEL
    if _PATCHED_LOAD_MODEL:
        return

    # Resolve to the *module* (which may be partially initialised) — this is
    # safe whether or not the module has finished importing. The class lookup
    # below happens lazily on first call to load_model, by which time the
    # module is fully initialised.
    # If this import fails the environment is truly broken; let it propagate
    # so init_backend logs it rather than silently disabling max-on-Spyre.
    import importlib

    mr_mod = importlib.import_module("sglang.srt.model_executor.model_runner")

    # Capture the original at patch time (works even mid-import: classes are
    # added to the module namespace as their `class` statements execute, so
    # by the time ANY caller imports our package, ModelRunner is bound).
    if not hasattr(mr_mod, "ModelRunner"):
        # Mid-import and the class statement hasn't run yet — defer the patch
        # to the first time someone actually constructs a ModelRunner.
        return _defer_patch_until_class_exists(mr_mod)

    ModelRunner = mr_mod.ModelRunner
    _orig_load_model = ModelRunner.load_model

    def _spyre_load_model(self, *args, **kwargs):
        from sglang_spyre_backend.hf_adapters import uses_hf_adapters, validate_config

        adapter_mode = uses_hf_adapters(self.server_args)
        if adapter_mode:
            validate_config(self.server_args, self.model_config)
            # Platform discovery can run while the registry is already importing,
            # before a deferred import hook can observe it. The worker load boundary
            # is always after registry initialization and before model construction.
            from sglang_spyre_backend.hf_adapters_transformers import install

            install()
        result = _orig_load_model(self, *args, **kwargs)
        if adapter_mode:
            if not getattr(self.model, "_hf_adapters", False):
                raise RuntimeError("HF-adapters Transformers wrapper was not selected")
            self.model.prepare_hf_adapters(self)
            return result
        # Trigger max-on-Spyre when EITHER:
        #   (a) the user passed device="spyre" to the engine, OR
        #   (b) SGLANG_FORCE_DEVICE_SPYRE=1 is set (the "max-on-Spyre while
        #       SGLang sees device=cpu" workaround — needed because the real
        #       device=spyre path crashes during model __init__ on tensor
        #       defaults / rotary inv_freq lowering).
        # Modes 2/3 with device=cpu and the env var unset are unaffected.
        import os as _os

        try:
            dev = str(getattr(self, "device", "cpu"))
        except Exception:
            dev = "cpu"
        force_spyre = _os.environ.get("SGLANG_FORCE_DEVICE_SPYRE") == "1"
        if "spyre" in dev or force_spyre:
            from sglang_spyre_backend import models
            from sglang_spyre_backend.custom_ops.weight_placement import (
                move_model_to_spyre,
            )
            from sglang_spyre_backend.model_wrapper import bodify_model_for_spyre

            # Model adaptations that read loaded weights (MoE expert stacks)
            # run before the device move, as spyre-inference's load fix-ups do.
            models.post_load(self.model)
            # Weights are transposed on the host and moved by their methods; the
            # rest of the body follows, with RoPE and the embedding kept on CPU.
            move_model_to_spyre(self.model)
            from sglang_spyre_backend.model_wrapper import default_token_buckets

            args = self.server_args
            chunk = getattr(args, "chunked_prefill_size", None) or 0
            budget = chunk if chunk > 0 else int(self.model_config.context_len)
            bodify_model_for_spyre(
                self.model,
                default_token_buckets(budget, int(getattr(args, "max_running_requests", 1) or 1)),
            )
        return result

    ModelRunner.load_model = _spyre_load_model
    _PATCHED_LOAD_MODEL = True


def _defer_patch_until_class_exists(mr_mod) -> None:
    """The model_runner module is mid-import and ModelRunner isn't bound yet.

    SGLang 0.5.21 calls ``init_backend()`` from model_runner's own module body,
    before ``class ModelRunner`` runs, and its callers bind the class with
    ``from ... import ModelRunner``, which reads the module dict and never
    reaches a module ``__getattr__``. So a meta-path finder wraps the module's
    loader and installs the patch the moment the module finishes executing.
    """
    import importlib.abc
    import sys

    name = mr_mod.__name__
    spec = getattr(mr_mod, "__spec__", None)
    loader = getattr(spec, "loader", None)
    if loader is None or getattr(loader, "_spyre_wrapped", False):
        return
    original_exec = loader.exec_module

    def exec_module(module):
        original_exec(module)
        if module.__name__ == name:
            _install_post_load_hook()

    # The module is executing under this loader right now; wrapping the bound
    # method on the instance covers this execution's tail as well, because the
    # finder below re-runs the patch at the next import-system touch otherwise.
    loader.exec_module = exec_module
    loader._spyre_wrapped = True

    class _PatchWhenLoaded(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if "ModelRunner" in vars(mr_mod) and not _PATCHED_LOAD_MODEL:
                _install_post_load_hook()
            if _PATCHED_LOAD_MODEL and self in sys.meta_path:
                sys.meta_path.remove(self)
            return None

    sys.meta_path.insert(0, _PatchWhenLoaded())


__all__ = ["install"]
