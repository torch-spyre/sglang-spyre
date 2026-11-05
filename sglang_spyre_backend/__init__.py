"""SGLang backend plugin for IBM Spyre AIU accelerator."""

import contextlib
import os

# CRITICAL: spyreccl env vars must be set at MODULE IMPORT TIME, not in
# init_backend(). libspyre_comms.so reads RANK/WORLD_SIZE/LOCAL_RANK/
# LOCAL_WORLD_SIZE at dlopen — and the scheduler subprocess (where the
# Spyre work actually runs) inherits env from the parent at fork/spawn,
# *before* any of our Python code runs. Setting these in init_backend
# is too late: libspyre_comms is already mapped, and SEGVs deep inside
# libsenlib-dd2.so on the first device op (offset +0x42accf, deterministic).
# os.environ.setdefault is a no-op if the user already set them in the shell,
# so this is also idempotent with the documented `export RANK=0 ...` recipe.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")


def _install_device_module_stubs(spyre_mod) -> None:
    """Give torch.spyre the CUDA-style Stream/Event/stream API SGLang expects.

    CRITICAL: torch_spyre's module ``__getattr__`` lazily opens the (single-tenant)
    Spyre VFIO device on access to *any unknown* attribute. So we must NOT use
    ``getattr``/``hasattr`` here — that would open the card inside activate(), in
    every process (parent Engine + scheduler subprocess), and the second one to
    open it dies with "Device or resource busy". We touch only ``vars(spyre_mod)``
    (the module __dict__), which never triggers __getattr__, so this stays
    completely card-free.

    Self-contained (no sglang import); safe from activate() mid-import. Spyre is
    eager/synchronous, so a stream is a no-op context and an event is always
    'done'. Idempotent.
    """
    d = vars(spyre_mod)
    if d.get("_sglang_stubs_installed"):
        return

    class _NoOpStream:
        def __init__(self, *a, **k): pass
        def synchronize(self): pass
        def wait_stream(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _NoOpEvent:
        def __init__(self, *a, **k): pass
        def record(self, *a, **k): pass
        def wait(self, *a, **k): pass
        def synchronize(self): pass
        def query(self): return True
        def elapsed_time(self, other): return 0.0

    # Direct __dict__ writes only (no getattr/hasattr -> no card open).
    d.setdefault("Stream", _NoOpStream)
    d.setdefault("Event", _NoOpEvent)
    d.setdefault("stream", lambda *a, **k: contextlib.nullcontext())
    d.setdefault("current_stream", lambda *a, **k: _NoOpStream())
    d.setdefault("synchronize", lambda *a, **k: None)
    d["_sglang_stubs_installed"] = True


def _install_when_imported(target: str, module: str, func: str) -> None:
    """Call ``module.func()`` once ``target`` has finished importing.

    activate() runs while ``sglang.srt.*`` is mid-import, so it must not import
    SGLang itself; a meta-path finder wraps the target's loader instead.
    """
    import importlib
    import importlib.abc
    import sys

    def run():
        getattr(importlib.import_module(module), func)()

    if target in sys.modules and getattr(sys.modules[target], "__spec__", None) is not None:
        mod = sys.modules[target]
        if getattr(mod, "_spyre_hooked", False):
            return
        if hasattr(mod, "HfModelConfigParser"):
            mod._spyre_hooked = True
            run()
            return

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target_=None):
            if fullname != target:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return spec
            original = spec.loader.exec_module

            def exec_module(m):
                original(m)
                m._spyre_hooked = True
                run()

            spec.loader.exec_module = exec_module
            return spec

    import importlib.util  # noqa: F401

    sys.meta_path.insert(0, _Finder())


def activate():
    """Return platform class name if Spyre hardware is available.

    Called by SGLang's platform discovery mechanism at startup.
    Registers the Spyre attention backend as a side-effect.
    """
    try:
        import os
        # spyreccl env vars MUST be set before torch_spyre is imported.
        # libspyre_comms.so reads them at dlopen — late-setting causes
        # silent failure modes (e.g., bmm crash with meta_tensor errors).
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("LOCAL_WORLD_SIZE", "1")

        # spyre-inference's configure_threading (#678, #1254): clamp the CPU thread
        # pools and default OMP_WAIT_POLICY=PASSIVE before the scheduler process
        # is spawned, which reads them at its torch import.
        try:
            from sglang_spyre_backend.threading_config import configure_threading

            configure_threading(worker_count=1)
        except Exception as exc:  # never block platform discovery on this
            import logging

            logging.getLogger(__name__).warning("Spyre threading config skipped: %s", exc)

        # Probe: does torch_spyre import? That's enough to claim the platform —
        # device availability will be checked at init_backend() time.
        import torch
        import torch_spyre  # noqa: F401

        # Install the CUDA-style Stream/Event/stream stubs on torch.spyre BEFORE
        # SGLang is imported. SGLang reads get_device_module(device).Event/.Stream
        # at *import* time (class-body annotations in cache_controller /
        # hisparse_coordinator / hybrid_cache_controller), so under device="spyre"
        # these must exist or `import sglang.srt` raises AttributeError. torch_spyre
        # is compile-oriented and ships none of them; Spyre runs eagerly so they are
        # safe no-ops.
        #
        # IMPORTANT: activate() runs *inside* SGLang's platform-resolution, which is
        # itself triggered while `sglang.srt.*` is mid-import. So we must NOT import
        # any `sglang.srt.*`-dependent module here (e.g. register.py / the attention
        # backends / the op layers) — that re-enters a partially-initialised sglang
        # and makes activate() throw -> return None -> "plugin returned None". The
        # attention-backend and op registration is therefore done lazily in
        # SpyreSRTPlatform.init_backend(), not here. Keep activate() minimal.
        try:
            torch_spyre._autoload()
        except Exception:
            pass
        if hasattr(torch, "spyre"):
            _install_device_module_stubs(torch.spyre)

        # Config normalization runs in the Engine (parent) process, before any
        # worker calls init_backend(), so the model-config repairs are installed
        # here, deferred until SGLang's config module finishes importing.
        _install_when_imported(
            "sglang.srt.utils.hf_transformers.config",
            "sglang_spyre_backend.models",
            "install_config_hooks",
        )

        return "sglang_spyre_backend.platform.SpyreSRTPlatform"

    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(
            "SpyreBackend: not activating — %s", e
        )
        return None
