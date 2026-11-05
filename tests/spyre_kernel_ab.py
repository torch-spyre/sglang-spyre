"""On-hardware A/B of the vendored head-major kernels against spyre-inference's own.

Runs the decode-path kernels SGLang dispatches at batch 1, the per-sequence LX
decode kernel and the batched decode kernel, once from this repo's
``sglang_spyre_backend.ops`` copy and once from the installed spyre-inference
package, on identical inputs. Both are compiled the same way; compilation
warm-up is excluded. Needs a Spyre card and spyre-inference importable (run it
from the vLLM environment with this repo on ``PYTHONPATH``).
"""

from __future__ import annotations

import json
import os
import statistics
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
for _k, _v in dict(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1").items():
    os.environ.setdefault(_k, _v)

import torch
import torch_spyre
from spyre_inference.v1.attention.ops import (
    page_attn_head_major_decode as ref_decode,
)

from sglang_spyre_backend.ops import page_attn_head_major_decode as port_decode
from sglang_spyre_backend.ops.layout import head_major_kv_layout


def _sync() -> None:
    synchronize = getattr(torch.spyre, "synchronize", None)
    if synchronize is not None:
        synchronize()


def _measure(fn, args, *, samples: int, iterations: int) -> list[float]:
    times = []
    for _ in range(samples):
        _sync()
        started = time.perf_counter()
        for _ in range(iterations):
            fn(*args)
        _sync()
        times.append((time.perf_counter() - started) * 1000 / iterations)
    return times


def main() -> None:
    torch_spyre._autoload()
    torch.manual_seed(7)
    dtype = torch.float16
    device = torch.device("spyre")

    # granite-3.3-8b-instruct decode attention at 512 cached tokens.
    num_heads, num_kv_heads, head_size, block_size = 32, 8, 128, 128
    num_blocks, num_pages = 4, 16
    layout = head_major_kv_layout(num_pages * num_kv_heads, block_size, head_size, dtype)
    shape = (num_pages * num_kv_heads, block_size, head_size)
    k = torch.randn(shape, dtype=dtype).to(device, device_layout=layout)
    v = torch.randn(shape, dtype=dtype).to(device, device_layout=layout)
    query = torch.randn(8, num_heads, head_size, dtype=dtype).to(device)
    rows = torch.tensor([0], dtype=torch.int32).to(device)
    table = torch.zeros(num_blocks, 32, dtype=torch.int32)
    table[:, 0] = torch.tensor([3, 1, 7, 5], dtype=torch.int32)
    table = table.to(device)
    pool = (
        torch.arange(num_pages * num_kv_heads, dtype=torch.int32)
        .reshape(num_pages, num_kv_heads, 1)
        .to(device)
    )
    mask = torch.zeros(num_blocks, 1, block_size, dtype=dtype).to(device)
    args = (
        query, rows, k, v, table, pool, mask, head_size**-0.5,
        num_blocks, 1, num_heads, num_kv_heads, head_size, block_size,
    )

    port = torch.compile(port_decode.page_attn_head_major_decode_kernel, dynamic=False, fullgraph=True)
    ref = torch.compile(ref_decode.page_attn_head_major_decode_kernel, dynamic=False, fullgraph=True)
    out_port = port(*args).to("cpu")
    out_ref = ref(*args).to("cpu")
    port_ms = _measure(port, args, samples=7, iterations=20)
    ref_ms = _measure(ref, args, samples=7, iterations=20)
    payload = {
        "geometry": {
            "query_heads": num_heads,
            "kv_heads": num_kv_heads,
            "head_size": head_size,
            "cached_tokens": num_blocks * block_size,
        },
        "kernel": "page_attn_head_major_decode_kernel",
        "candidate_median_ms": statistics.median(port_ms),
        "spyre_inference_median_ms": statistics.median(ref_ms),
        "candidate_over_reference": statistics.median(port_ms) / statistics.median(ref_ms),
        "exact_output_match": bool(torch.equal(out_port, out_ref)),
    }
    print("SPYRE_KERNEL_AB=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
