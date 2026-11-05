"""SpyreSRTPlatform — SGLang SRT backend for IBM Spyre AIU."""

from sglang.srt.platforms.interface import SRTPlatform

from sglang_spyre_backend.device import SpyreDeviceMixin


class SpyreSRTPlatform(SRTPlatform, SpyreDeviceMixin):
    """SGLang SRT platform for IBM Spyre AIU accelerator.

    Execution model:
    - No CUDA graph capture/replay
    - FP16 compute, matching the current spyre-inference compatibility contract
    - KV cache lives on Spyre in a dense slot-major layout and uses indirect access
    - TP=1 only (TP>1 requires spyreccl all_reduce, not yet stable)
    - Static shapes only (Spyre Inductor cannot handle SymInt)
    """

    # ------------------------------------------------------------------
    # Configuration lifecycle
    # ------------------------------------------------------------------

    def apply_server_args_defaults(self, server_args) -> None:
        """Apply Spyre-specific defaults after ServerArgs parsing."""
        # Disable CUDA graph — Spyre uses eager execution
        if not server_args.disable_cuda_graph:
            server_args.disable_cuda_graph = True

        # Match spyre-inference: float16 is the supported model/cache dtype.
        if getattr(server_args, "dtype", "auto") == "auto":
            server_args.dtype = "float16"

        # The synchronized paged backend is the production-comparable path.
        if getattr(server_args, "attention_backend", None) is None:
            server_args.attention_backend = "spyre_paged"

        # spyre-inference's default KV block size (platform._DEFAULT_BLOCK_SIZE, #746):
        # a power of two and a multiple of the 64-element stick.
        if getattr(server_args, "page_size", None) is None:
            server_args.page_size = 128

        # spyre-inference keeps every layer's KV in one paged cache and applies a
        # layer's sliding window in its attention mask (#1317, #752); this backend
        # owns the pages the same way, so SGLang's separate SWA pool is not used.
        if hasattr(server_args, "disable_hybrid_swa_memory"):
            server_args.disable_hybrid_swa_memory = True

        # Spyre runs each step synchronously, so SGLang's overlap scheduler has no
        # device time to hide host work behind; it only launches one speculative
        # decode step past a request's last token. vLLM's spyre-inference engine
        # steps synchronously too.
        if hasattr(server_args, "disable_overlap_schedule"):
            server_args.disable_overlap_schedule = True

        # The first request traces and compiles every decoder block; raise the
        # watchdog so that warmup does not count as a hung forward.
        if getattr(server_args, "watchdog_timeout", 300) < 3600:
            server_args.watchdog_timeout = 3600

        # TP=1 only
        if getattr(server_args, "tp_size", 1) > 1:
            raise ValueError(
                "SpyreSRTPlatform: tensor parallelism > 1 is not yet supported. "
                "Set --tp-size 1."
            )

    # ------------------------------------------------------------------
    # Subsystem factory methods
    # ------------------------------------------------------------------

    def get_default_attention_backend(self) -> str:
        return "spyre_paged"

    def get_graph_runner_cls(self) -> type:
        # Spyre runs eager: support_cuda_graph() returns False, so SGLang core
        # sets model_runner.graph_runner = None itself (model_runner.py, the
        # is_out_of_tree() branch) and never calls this. We therefore need no
        # bespoke graph runner — the previous SpyreEagerGraphRunner just
        # re-implemented the eager forward path the core already runs.
        raise NotImplementedError(
            "SpyreSRTPlatform runs eager (support_cuda_graph() is False); "
            "core uses graph_runner=None and never requests a graph runner class."
        )

    def get_mha_kv_pool_cls(self) -> type:
        from sglang_spyre_backend.memory_pool import SpyreMHATokenToKVPool

        return SpyreMHATokenToKVPool

    def get_mla_kv_pool_cls(self) -> type:
        # MLA (DeepSeek) not yet supported on Spyre; fall back to MHA
        from sglang_spyre_backend.memory_pool import SpyreMHATokenToKVPool

        return SpyreMHATokenToKVPool

    def get_dsa_kv_pool_cls(self) -> type:
        # DSA (DeepSeek V3.2) not yet supported
        from sglang_spyre_backend.memory_pool import SpyreMHATokenToKVPool

        return SpyreMHATokenToKVPool

    def get_paged_allocator_cls(self) -> type:
        # SGLang's PagedTokenToKVPoolAllocator.alloc_extend/alloc_decode dispatch
        # Triton kernels, which crash on Spyre ("0 active drivers" — no Triton
        # backend). SpyrePagedAllocator overrides just those two with torch-native
        # equivalents that keep the exact page-aligned slot layout. Only used at
        # page_size>1 (the "spyre_paged" path); page_size=1 uses the non-paged
        # torch allocator, so the dense "spyre" backend is unaffected.
        from sglang_spyre_backend.paged_allocator import SpyrePagedAllocator

        return SpyrePagedAllocator

    def get_piecewise_backend_cls(self) -> type:
        # Piecewise torch.compile not used on Spyre
        raise NotImplementedError("SpyreSRTPlatform does not support piecewise backend")

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def support_cuda_graph(self) -> bool:
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        return False

    def supports_fp8(self) -> bool:
        # Spyre 1.5 supports FP8 at the hardware level, but the current
        # spyre-inference stack does not yet expose FP8 quantization.
        return False

    def is_pin_memory_available(self, device=None) -> bool:
        # Spyre uses its own allocator; pin_memory is a no-op.
        return False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_backend(self) -> None:
        """Initialize the Spyre device backend in this worker.

        Mirrors spyre-inference's TorchSpyreWorker.init_device() sequence:

          1. Set RANK/WORLD_SIZE/LOCAL_RANK/LOCAL_WORLD_SIZE env vars BEFORE
             importing torch_spyre — libspyre_comms.so reads them at dlopen
             time. Skipping this step makes torch.bmm / F.linear crash with
             a meta_tensor no_dispatch AssertionError on the first call.

          2. Call torch_spyre._autoload() which registers the "spyre"
             PrivateUse1 device, exposes torch.spyre.* APIs, registers
             eager kernels for aten.bmm / aten.mm / etc. (each wrapped
             with torch.compile internally), and registers the spyreccl
             distributed backend.

          3. Pin this worker's device.

          4. Register the attention backends + on-device ops (safe here;
             sglang.srt is fully imported by now).

          5. Install the max-on-Spyre custom ops + weight-placement hook
             (no-op unless device="spyre").

          6. Patch torch._inductor.decomposition with the Spyre addmm
             decomposition. Workaround for torch-spyre issue #1420 —
             without it, the first FX graph trace crashes. Runs last: the
             table only needs to exist before the first forward.

        SINGLE-TENANT GUARD: the Spyre AIU is a single-tenant VFIO device — only
        one OS process may open it. SGLang calls init_backend() at *module import*
        of model_executor.model_runner (model_runner.py:244), which runs in BOTH
        the parent Engine process and the spawned scheduler subprocess. If both
        open the card, the second dies with "Device or resource busy". So we open
        the device ONLY in the worker (scheduler) process and make this a no-op in
        the main/parent process — exactly the discipline spyre-inference uses
        (platform.py: `if multiprocessing.current_process().name != "MainProcess"`).
        """
        import multiprocessing
        import os

        # SINGLE-TENANT GUARD: do not open the card in the parent ("MainProcess").
        # The scheduler runs as an mp.Process named "Process-N"; only it should
        # hold the device. (Idempotent across re-imports within the worker.)
        if multiprocessing.current_process().name == "MainProcess":
            return
        if getattr(self, "_spyre_backend_inited", False):
            return

        # 1. spyreccl env vars (must be set before torch_spyre import)
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("LOCAL_WORLD_SIZE", "1")

        # 2. torch_spyre autoload
        import torch
        import torch_spyre

        torch_spyre._autoload()

        # 3. pin device + ensure the device-module stubs are present
        if hasattr(torch, "spyre"):
            torch.spyre.set_device(0)
            from sglang_spyre_backend import _install_device_module_stubs

            _install_device_module_stubs(torch.spyre)
            # Eagerly CLAIM the single-tenant card in THIS worker process. set_device
            # only records the index; the VFIO open is lazy (first tensor op). Pinning
            # it here makes the worker the deterministic owner before any forward runs.
            # NOTE: the real cure for the parent-vs-worker "Device or resource busy"
            # race is keeping torch_spyre's autoload OFF in the parent — set
            # TORCH_DEVICE_BACKEND_AUTOLOAD=0 in the environment before the parent
            # imports anything (activate() sets it, but only if it runs before any
            # other import of torch_spyre / torch.spyre).
            try:
                _ = torch.zeros(1, device="spyre")
            except Exception:
                pass

        # 4. Register the attention backends + on-device ops NOW (safe here:
        #    init_backend runs after sglang.srt is fully imported, unlike
        #    activate() which runs mid-import). Side-effect imports.
        try:
            import sglang_spyre_backend.register  # noqa: F401
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "Spyre op/backend registration failed"
            )

        # 5. Max-on-Spyre wiring: register MultiPlatformOp Spyre forwards
        #    (RMSNorm/SiluAndMul/RoPE) and install a post-load hook that moves
        #    Linear weights onto Spyre and modifies the model body
        #    (see bodify_model_for_spyre). The hook is a no-op unless the user
        #    selected device="spyre" — modes 2/3 are unaffected. Idempotent.
        try:
            from sglang_spyre_backend.custom_ops import install as _install_max_on_spyre

            _install_max_on_spyre()
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "Spyre custom-op installation failed (max-on-Spyre disabled)"
            )

        self._spyre_backend_inited = True

        # 6. addmm decomposition workaround (torch-spyre issue #1420).
        #    Runs last, after registration — the decomposition table only
        #    needs to be in place before the first FX graph trace, which
        #    happens on the first forward, not during init.
        try:
            import torch._inductor.decomposition
            from torch_spyre._inductor.decompositions import spyre_decompositions

            for op, impl in spyre_decompositions.items():
                if "addm" in op.name():
                    torch._inductor.decomposition.decompositions[op] = impl
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # MultiPlatformOp dispatch
    # ------------------------------------------------------------------

    def get_dispatch_key_name(self) -> str:
        return "spyre"
