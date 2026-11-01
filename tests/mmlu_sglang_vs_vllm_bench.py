"""Compare warmed SGLang/Spyre and vLLM/spyre-inference on five pinned MMLU items.

Extends ``mmlu_speedup_bench.py``: the same pinned ``cais/mmlu`` revision and
prompt template, now five items (test index 0 of five subjects, in a fixed order)
served by both frameworks on Spyre from the exact same token ids, at batch size 1
with greedy decoding. Each framework runs in its own process and virtualenv (the
two pin different SGLang/vLLM stacks over the same torch-spyre), so the script is
run once per framework and once more to combine the two records:

    python tests/mmlu_sglang_vs_vllm_bench.py sglang --model M --output sg.json
    python tests/mmlu_sglang_vs_vllm_bench.py vllm   --model M --output vl.json
    python tests/mmlu_sglang_vs_vllm_bench.py compare sg.json vl.json --output out.json

Per item, each framework gets one warmup request (which also warms that prompt
shape), then ``--iterations`` measured requests. A measured request is timed
twice: once with one output token (TTFT, prefill plus the first sample) and once
with ``--max-new-tokens`` tokens (end-to-end latency); output throughput is
``(max_new_tokens - 1) / (latency - TTFT)``. The answer is the first A-D letter
of the generated text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
CASES = (
    ("abstract_algebra", 0),
    ("anatomy", 0),
    ("high_school_physics", 0),
    ("professional_medicine", 0),
    ("world_religions", 0),
)
ANSWER_LABELS = "ABCD"
PARITY_THRESHOLD = 0.15


def build_prompt(question: str, choices: list[str]) -> str:
    choices_text = "\n".join(
        f"{label}. {choice}" for label, choice in zip(ANSWER_LABELS, choices)
    )
    return (
        "Answer the following multiple-choice question. Reply with only A, B, C, "
        f"or D.\n\n{question}\n{choices_text}\nAnswer:"
    )


def load_cases(tokenizer: Any) -> list[dict[str, Any]]:
    from datasets import load_dataset

    cases = []
    for subject, index in CASES:
        row = load_dataset("cais/mmlu", subject, split="test", revision=MMLU_REVISION)[index]
        prompt = build_prompt(row["question"], row["choices"])
        cases.append(
            {
                "subject": subject,
                "test_index": index,
                "answer": ANSWER_LABELS[row["answer"]],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "input_ids": tokenizer.encode(prompt, add_special_tokens=True),
            }
        )
    return cases


def first_letter(text: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else None


def package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


class _SGLang:
    name = "sglang"

    def __init__(self, model: str, context_length: int):
        for k, v in dict(
            TORCH_DEVICE_BACKEND_AUTOLOAD="0",
            RANK="0",
            WORLD_SIZE="1",
            LOCAL_RANK="0",
            LOCAL_WORLD_SIZE="1",
            SGLANG_FORCE_DEVICE_SPYRE="1",
        ).items():
            os.environ.setdefault(k, v)
        import sglang as sgl

        self.engine = sgl.Engine(
            model_path=model,
            device="cpu",
            attention_backend="spyre_paged",
            dtype="float16",
            page_size=128,
            context_length=context_length,
            max_total_tokens=2 * context_length,
            max_running_requests=1,
            chunked_prefill_size=512,
            disable_radix_cache=True,
            disable_cuda_graph=True,
            log_level="warning",
        )

    def generate(self, input_ids: list[int], max_new_tokens: int) -> str:
        out = self.engine.generate(
            input_ids=input_ids,
            sampling_params={
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        )
        return out["text"]

    def close(self):
        self.engine.shutdown()


class _VLLM:
    name = "vllm"

    def __init__(self, model: str, context_length: int):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM

        self.llm = LLM(
            model=model,
            max_model_len=context_length,
            max_num_seqs=1,
            tensor_parallel_size=1,
            max_num_batched_tokens=512,
        )

    def generate(self, input_ids: list[int], max_new_tokens: int) -> str:
        from vllm import SamplingParams, TokensPrompt

        sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, ignore_eos=True)
        out = self.llm.generate(TokensPrompt(prompt_token_ids=input_ids), sp, use_tqdm=False)
        return out[0].outputs[0].text

    def close(self):
        pass


def run_framework(args) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cases = load_cases(tokenizer)
    runner = (_SGLang if args.framework == "sglang" else _VLLM)(args.model, args.context_length)
    results = []
    try:
        for case in cases:
            ids = case["input_ids"]
            for _ in range(args.warmups):
                runner.generate(ids, 1)
                runner.generate(ids, args.max_new_tokens)
            ttft, latency, text = [], [], ""
            for _ in range(args.iterations):
                t0 = time.perf_counter()
                runner.generate(ids, 1)
                ttft.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                text = runner.generate(ids, args.max_new_tokens)
                latency.append(time.perf_counter() - t0)
            decode = statistics.mean(latency) - statistics.mean(ttft)
            results.append(
                {key: value for key, value in case.items() if key != "input_ids"}
                | {
                    "input_tokens": len(ids),
                    "ttft_seconds": ttft,
                    "latency_seconds": latency,
                    "mean_ttft_seconds": statistics.mean(ttft),
                    "mean_latency_seconds": statistics.mean(latency),
                    "output_tokens_per_second": (args.max_new_tokens - 1) / decode
                    if decode > 0
                    else None,
                    "generated_text": text,
                    "predicted": first_letter(text),
                }
            )
            print(
                f"[{args.framework}] {case['subject']}: ttft={statistics.mean(ttft) * 1e3:.1f} ms "
                f"latency={statistics.mean(latency) * 1e3:.1f} ms text={text!r}",
                flush=True,
            )
    finally:
        runner.close()
    return {
        "framework": args.framework,
        "model": args.model,
        "dataset": "cais/mmlu",
        "dataset_revision": MMLU_REVISION,
        "split": "test",
        "batch_size": 1,
        "greedy": True,
        "max_new_tokens": args.max_new_tokens,
        "warmup_iterations_per_example": args.warmups,
        "measured_iterations_per_example": args.iterations,
        "software": {
            name: package_version(name)
            for name in ("sglang", "vllm", "spyre-inference", "torch", "torch-spyre", "transformers")
            if package_version(name) != "unknown"
        },
        "examples": results,
    }


def compare(sg: dict[str, Any], vl: dict[str, Any]) -> dict[str, Any]:
    assert sg["model"] == vl["model"], "records are for different models"
    rows = []
    for a, b in zip(sg["examples"], vl["examples"], strict=True):
        assert a["prompt_sha256"] == b["prompt_sha256"], "records used different prompts"
        rows.append(
            {
                "subject": a["subject"],
                "test_index": a["test_index"],
                "answer": a["answer"],
                "input_tokens": a["input_tokens"],
                "sglang": {k: a[k] for k in _KEYS},
                "vllm": {k: b[k] for k in _KEYS},
                "same_text": a["generated_text"] == b["generated_text"],
                "same_answer": a["predicted"] == b["predicted"],
            }
        )

    def mean(fw, key):
        values = [r[fw][key] for r in rows if r[fw][key] is not None]
        return statistics.mean(values)

    agg = {}
    for fw in ("sglang", "vllm"):
        agg[fw] = {
            "mean_ttft_seconds": mean(fw, "mean_ttft_seconds"),
            "mean_latency_seconds": mean(fw, "mean_latency_seconds"),
            "mean_output_tokens_per_second": mean(fw, "output_tokens_per_second"),
            "accuracy": sum(r[fw]["predicted"] == r["answer"] for r in rows) / len(rows),
        }
    ratio = agg["sglang"]["mean_latency_seconds"] / agg["vllm"]["mean_latency_seconds"]
    return {
        "model": sg["model"],
        "dataset": sg["dataset"],
        "dataset_revision": sg["dataset_revision"],
        "batch_size": 1,
        "max_new_tokens": sg["max_new_tokens"],
        "software": {"sglang": sg["software"], "vllm": vl["software"]},
        "examples": rows,
        "aggregate": agg
        | {
            "sglang_over_vllm_latency": ratio,
            "sglang_over_vllm_ttft": agg["sglang"]["mean_ttft_seconds"]
            / agg["vllm"]["mean_ttft_seconds"],
            "sglang_over_vllm_output_throughput": agg["sglang"]["mean_output_tokens_per_second"]
            / agg["vllm"]["mean_output_tokens_per_second"],
            "matching_texts": sum(r["same_text"] for r in rows),
            "matching_answers": sum(r["same_answer"] for r in rows),
            "parity_threshold": PARITY_THRESHOLD,
            "parity_pass": abs(ratio - 1.0) <= PARITY_THRESHOLD,
        },
    }


_KEYS = (
    "mean_ttft_seconds",
    "mean_latency_seconds",
    "output_tokens_per_second",
    "generated_text",
    "predicted",
    "ttft_seconds",
    "latency_seconds",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="framework", required=True)
    for name in ("sglang", "vllm"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--warmups", type=int, default=1)
        p.add_argument("--iterations", type=int, default=3)
        p.add_argument("--max-new-tokens", type=int, default=16)
        p.add_argument("--context-length", type=int, default=2048)
        p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("sglang_record", type=Path)
    p.add_argument("vllm_record", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.framework == "compare":
        payload = compare(
            json.loads(args.sglang_record.read_text()), json.loads(args.vllm_record.read_text())
        )
    else:
        if args.warmups < 1 or args.iterations < 1:
            sys.exit("--warmups and --iterations must both be positive")
        payload = run_framework(args)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("MMLU_RESULT=" + json.dumps(payload.get("aggregate", {}), sort_keys=True))


if __name__ == "__main__":
    main()
