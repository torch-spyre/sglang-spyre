"""Small, reproducible end-to-end latency benchmark for the Spyre backend."""

from __future__ import annotations

import json
import os
import statistics
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
os.environ.setdefault("SGLANG_FORCE_DEVICE_SPYRE", "1")


def main() -> None:
    import sglang as sgl

    model = os.environ.get(
        "SGLANG_TEST_MODEL", "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
    )
    input_len = int(os.environ.get("SGLANG_INPUT_LEN", "64"))
    output_len = int(os.environ.get("SGLANG_OUTPUT_LEN", "16"))
    warmups = int(os.environ.get("SGLANG_WARMUPS", "1"))
    iterations = int(os.environ.get("SGLANG_ITERATIONS", "3"))

    engine = sgl.Engine(
        model_path=model,
        device="cpu",
        attention_backend="spyre_paged",
        dtype="float16",
        page_size=128,
        context_length=1024,
        max_total_tokens=1024,
        max_running_requests=1,
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        log_level="warning",
    )
    input_ids = [42] * input_len
    sampling_params = {
        "max_new_tokens": output_len,
        "temperature": 0.0,
        "ignore_eos": True,
    }

    for _ in range(warmups):
        engine.generate(input_ids=input_ids, sampling_params=sampling_params)

    times = []
    outputs = []
    for _ in range(iterations):
        started = time.perf_counter()
        result = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
        times.append(time.perf_counter() - started)
        outputs.append(result["meta_info"].get("completion_tokens", output_len))

    engine.shutdown()
    payload = {
        "model": model,
        "input_len": input_len,
        "output_len": output_len,
        "batch_size": 1,
        "latency_seconds": times,
        "mean_latency_seconds": statistics.mean(times),
        "median_latency_seconds": statistics.median(times),
        "mean_output_tokens_per_second": statistics.mean(
            count / elapsed for count, elapsed in zip(outputs, times)
        ),
    }
    print("SGLANG_SPYRE_RESULT=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
