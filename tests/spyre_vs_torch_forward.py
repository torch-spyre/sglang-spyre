"""Compare one Granite-shaped attention forward on Spyre and native PyTorch CPU."""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")

import torch
import torch_spyre

from sglang_spyre_backend.ops.page_attn import page_attn_kernel


def _elapsed_once(fn, args, *, copy_to_cpu: bool) -> tuple[float, torch.Tensor]:
    started = time.perf_counter()
    output = fn(*args)
    if copy_to_cpu:
        output = output.to("cpu")
    elapsed_ms = (time.perf_counter() - started) * 1000
    return elapsed_ms, output


def main() -> None:
    torch_spyre._autoload()
    torch.spyre.set_device(0)
    torch.manual_seed(11)

    # micro-g3.3-8b-instruct-1b: 32 query heads, 8 KV heads, head size 128.
    num_blocks, block_size = 4, 64
    num_heads, num_kv_heads, head_size = 32, 8, 128
    query_len = 1
    scale = 1.0 / head_size**0.5

    num_kv_heads_q = num_heads // num_kv_heads
    query_cpu = torch.randn(query_len, num_heads, head_size, dtype=torch.float16)
    k_cpu = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.float16)
    v_cpu = torch.randn_like(k_cpu)
    rows_cpu = torch.zeros(query_len, dtype=torch.int64)
    indices_cpu = torch.zeros(num_blocks, 32, dtype=torch.int64)
    indices_cpu[:, 0] = torch.arange(num_blocks)
    mask_cpu = torch.zeros(num_blocks, query_len, block_size, dtype=torch.float16)
    del num_kv_heads_q

    def native(*args):
        return page_attn_kernel(*args)

    compiled_spyre = torch.compile(page_attn_kernel, dynamic=False)
    statics = (scale, num_blocks, query_len, num_heads, num_kv_heads, head_size)
    spyre_args = (
        query_cpu.to("spyre"),
        rows_cpu.to(torch.int32).to("spyre"),
        k_cpu.to("spyre"),
        v_cpu.to("spyre"),
        indices_cpu.to(torch.int32).to("spyre"),
        mask_cpu.to("spyre"),
        *statics,
    )
    cpu_args = (query_cpu, rows_cpu, k_cpu, v_cpu, indices_cpu, mask_cpu, *statics)

    # Compile and warm both paths outside the reported single-pass timing.
    spyre_warm = compiled_spyre(*spyre_args).to("cpu")
    cpu_warm = native(*cpu_args)
    torch.testing.assert_close(spyre_warm, cpu_warm, rtol=2e-2, atol=2e-2)

    cpu_ms, cpu_output = _elapsed_once(native, cpu_args, copy_to_cpu=False)
    spyre_ms, spyre_output = _elapsed_once(compiled_spyre, spyre_args, copy_to_cpu=True)
    torch.testing.assert_close(spyre_output, cpu_output, rtol=2e-2, atol=2e-2)

    print(
        json.dumps(
            {
                "model_geometry": "ibm-ai-platform/micro-g3.3-8b-instruct-1b (token-major paged kernel)",
                "operation": "one decode attention forward over 256 cached tokens",
                "pytorch_cpu_ms": cpu_ms,
                "spyre_ms_including_output_copy": spyre_ms,
                "speedup_over_pytorch_cpu": cpu_ms / spyre_ms,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
