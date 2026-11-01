"""Spyre device mixin — device identity and basic operations."""

from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum

# Off-chip memory per Spyre chip (128 GB on 1.0, 96 GB per chip on 1.5).
# We report the 1.5 single-chip value as a conservative default; the
# allocator will discover available memory at runtime via the Spyre C runtime.
_SPYRE_TOTAL_MEMORY_BYTES = 96 * (1 << 30)


class SpyreDeviceMixin(DeviceMixin):
    """Device identity and operations for IBM Spyre AIU."""

    _enum = PlatformEnum.OOT
    device_name = "spyre"
    device_type = "spyre"

    # ------------------------------------------------------------------
    # Active methods — called by SGLang core through current_platform
    # ------------------------------------------------------------------

    def get_device_total_memory(self, device_id: int = 0) -> int:
        """Return total off-chip memory in bytes."""
        return _SPYRE_TOTAL_MEMORY_BYTES

    def get_current_memory_usage(self, device: Optional[torch.device] = None) -> float:
        """Return current peak memory usage in bytes (best effort)."""
        return 0.0

    # ------------------------------------------------------------------
    # [Planned] overrides — base DeviceMixin currently raises NotImplementedError
    # for these (core still uses hardcoded torch.cuda.* paths), so they only
    # become load-bearing once SGLang migrates the call sites. We keep the ones
    # that do *real* Spyre work and let the base provide the rest:
    #   - empty_cache():            base default is already a no-op → omitted.
    #   - get_device_capability():  base raise is fine (no Spyre capability) → omitted.
    #   - get_available_memory():   derived from get_device_total_memory above → omitted.
    # ------------------------------------------------------------------

    def get_device(self, local_rank: int) -> torch.device:
        return torch.device("spyre", local_rank)

    def set_device(self, device: torch.device) -> None:
        # After torch_spyre._autoload(), torch.spyre is the device module.
        if hasattr(torch, "spyre"):
            torch.spyre.set_device(device.index or 0)

    def get_device_name(self, device_id: int = 0) -> str:
        return f"IBM Spyre AIU {device_id}"

    def synchronize(self) -> None:
        # Real work (not the base no-op): flush the Spyre command queue.
        if hasattr(torch, "spyre"):
            try:
                torch.spyre.synchronize()
            except Exception:
                pass

    def get_torch_profiler_activity_str(self) -> str:
        # torch-spyre reports device kernels under PrivateUse1.
        return "PrivateUse1"

    def get_torch_profiler_activity(self) -> torch.profiler.ProfilerActivity:
        return torch.profiler.ProfilerActivity.PrivateUse1

    def get_torch_distributed_backend_str(self) -> str:
        # spyreccl for Spyre tensors; gloo for CPU coordination tensors.
        return "cpu:gloo,spyre:spyreccl"
