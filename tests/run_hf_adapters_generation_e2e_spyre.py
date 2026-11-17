"""Compiled adapter serving probe. Run serially on an otherwise idle Spyre pod."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import time
import warnings
from pathlib import Path


def configure_process():
    for key, value in {
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
    }.items():
        os.environ.setdefault(key, value)
    # Run in the launcher and spawned workers. Python parses -W before the
    # environment's site-packages are available, so PYTHONWARNINGS cannot import
    # this warning class reliably. Register kernels without opening the card.
    import torch_spyre

    torch_spyre._autoload()
    from torch_spyre.ops.fallbacks import FallbackWarning

    warnings.simplefilter("error", FallbackWarning)
    from hf_adapters_compilation import install_worker_probe

    install_worker_probe()


def token_ids(output):
    probabilities = output["meta_info"]["output_token_logprobs"]
    assert probabilities, "No generated-token probabilities were returned"
    assert all(math.isfinite(item[0]) for item in probabilities)
    return [item[1] for item in probabilities]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-running-requests", type=int, default=2)
    parser.add_argument("--compile-counts", action="store_true")
    args = parser.parse_args()
    from hf_adapters_compilation import snapshot, start_report, summarize

    compilation_path = start_report(args.output) if args.compile_counts else None
    import sglang as sgl
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    shared = tokenizer.encode(
        "Read this background carefully. IBM develops computing systems and software. "
        * 32,
        add_special_tokens=True,
    )[:256]
    prompts = [
        shared[: length - len(tail)] + tail
        for length, suffix in zip(
            (120, 200),
            (
                "\nQuestion: What does IBM develop?\nAnswer:",
                "\nQuestion: Name one kind of IBM product.\nAnswer:",
            ),
        )
        for tail in [tokenizer.encode(suffix, add_special_tokens=False)]
    ]
    # Unequal lengths exercise chunking; the short request crosses a KV page in decode.
    assert [len(p) for p in prompts] == [120, 200]
    started = time.perf_counter()
    settings = dict(
        model_path=args.model,
        model_impl="transformers",
        json_model_override_args=json.dumps({"spyre_model_preparation": "hf_adapters"}),
        device="cpu",
        attention_backend="spyre_paged",
        dtype="float16",
        page_size=128,
        context_length=256,
        chunked_prefill_size=128,
        max_total_tokens=1536,
        max_running_requests=args.max_running_requests,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
    )
    engine = sgl.Engine(**settings)
    report = {
        "model": args.model,
        "settings": settings,
        "startup_seconds": time.perf_counter() - started,
        "input_ids": prompts,
    }
    report["packages"] = {
        name: importlib.metadata.version(name)
        for name in ("sglang", "torch", "torch-spyre", "transformers")
    }
    try:
        if compilation_path:
            snapshot(engine, "startup")

        def generate(batch):
            return engine.generate(
                input_ids=batch,
                sampling_params={
                    "temperature": 0.0,
                    "max_new_tokens": 16,
                    "ignore_eos": True,
                },
                return_logprob=True,
            )

        cold = generate(prompts)
        if compilation_path:
            snapshot(engine, "cold")
        repeated = generate(prompts)
        if compilation_path:
            snapshot(engine, "repeated")
        reordered = generate(list(reversed(prompts)))
        if compilation_path:
            snapshot(engine, "reordered")
        engine.flush_cache()
        flushed = generate(prompts)
        if compilation_path:
            snapshot(engine, "flushed")
        individual = []
        for prompt in prompts:
            engine.flush_cache()
            individual.append(generate([prompt])[0])
        if compilation_path:
            snapshot(engine, "individual")
            report["compilation"] = summarize(compilation_path)
            report["compilation_stable"] = all(
                row["new_graphs"]
                == row["new_sdsc_compiles"]
                == row["new_backend_compiles"]
                == 0
                for row in report["compilation"]
            )
        report.update(
            cold=cold,
            repeated=repeated,
            reordered=reordered,
            flushed=flushed,
            individual=individual,
            consistency={
                "repeat": [token_ids(o) for o in cold]
                == [token_ids(o) for o in repeated],
                "order": [token_ids(o) for o in cold]
                == [token_ids(o) for o in reversed(reordered)],
                "flush": [token_ids(o) for o in cold]
                == [token_ids(o) for o in flushed],
                "individual": [token_ids(o) for o in cold]
                == [token_ids(o) for o in individual],
            },
            prefix_cache_reused=any(
                o["meta_info"].get("cached_tokens", 0) >= 128 for o in repeated
            ),
        )
        report["passed"] = (
            all(report["consistency"].values()) and report["prefix_cache_reused"]
        )
        report["passed"] = report["passed"] and report.get("compilation_stable", True)
        assert report["passed"], {
            key: report[key] for key in ("consistency", "compilation") if key in report
        }
    finally:
        engine.shutdown()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ in ("__main__", "__mp_main__"):
    configure_process()

if __name__ == "__main__":
    main()
