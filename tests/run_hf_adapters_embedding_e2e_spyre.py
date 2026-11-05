"""Compiled SGLang embeddings against independent stock-HF FP32 references."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from run_hf_adapters_generation_e2e_spyre import configure_process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pooling", choices=["CLS", "MEAN", "LAST"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-counts", action="store_true")
    parser.add_argument("--pooling-override", action="store_true")
    args = parser.parse_args()
    from hf_adapters_compilation import snapshot, start_report, summarize

    compilation_path = start_report(args.output) if args.compile_counts else None

    import sglang as sgl
    import torch
    from torch.nn import functional as F
    from transformers import AutoModel, AutoTokenizer

    texts = [
        "IBM develops computing systems.",
        "A short sentence about the weather.",
        "This paragraph discusses software, hardware, and language models. " * 7,
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    references, input_ids = [], []
    with torch.inference_mode():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            ids = inputs["input_ids"][0].tolist()
            assert 0 < len(ids) <= 128
            input_ids.append(ids)
            hidden = model(**inputs).last_hidden_state[0].float()
            if args.pooling == "CLS":
                vector = hidden[0]
            elif args.pooling == "MEAN":
                vector = hidden.mean(0)
            else:
                vector = hidden[-1]
            references.append(F.normalize(vector, dim=-1))
    reference = torch.stack(references)
    del model
    overrides = {"spyre_model_preparation": "hf_adapters"}
    if args.pooling_override:
        overrides.update(pooling_type=args.pooling, normalize=True)
    settings = dict(
        model_path=args.model,
        model_impl="transformers",
        is_embedding=True,
        json_model_override_args=json.dumps(overrides),
        device="cpu",
        attention_backend="spyre_embedding",
        dtype="float16",
        page_size=128,
        context_length=128,
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        max_total_tokens=1024,
        max_running_requests=2,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
    )
    report = {
        "model": args.model,
        "pooling": args.pooling,
        "settings": settings,
        "texts": texts,
        "input_ids": input_ids,
    }
    report["packages"] = {
        name: importlib.metadata.version(name)
        for name in ("sglang", "torch", "torch-spyre", "transformers")
    }
    engine = sgl.Engine(**settings)
    try:
        if compilation_path:
            snapshot(engine, "startup")

        def encode(batch):
            output = engine.encode(batch)
            if isinstance(output, dict):
                output = [output]
            return torch.tensor(
                [item["embedding"] for item in output], dtype=torch.float32
            )

        first = encode(texts)
        if compilation_path:
            snapshot(engine, "first")
        repeated = encode(texts)
        if compilation_path:
            snapshot(engine, "repeated")
        reordered = encode(list(reversed(texts))).flip(0)
        if compilation_path:
            snapshot(engine, "reordered")
        individual = torch.cat([encode(text) for text in texts])
        if compilation_path:
            snapshot(engine, "individual")
        if compilation_path:
            report["compilation"] = summarize(compilation_path)
            report["compilation_stable"] = all(
                row["new_graphs"]
                == row["new_sdsc_compiles"]
                == row["new_backend_compiles"]
                == 0
                for row in report["compilation"]
            )
        assert torch.isfinite(first).all()
        cosine = F.cosine_similarity(first, reference)
        consistency = [
            F.cosine_similarity(first, value).min().item()
            for value in (repeated, reordered, individual)
        ]
        report.update(
            shape=list(first.shape),
            reference_cosine=cosine.tolist(),
            consistency_cosine=consistency,
            repeat_max_abs_error=float((first - repeated).abs().max()),
            batch_single_max_abs_error=float((first - individual).abs().max()),
            norms=first.norm(dim=-1).tolist(),
        )
        assert cosine.min() > 0.999, report
        assert min(consistency) > 0.9999, report
        torch.testing.assert_close(
            first.norm(dim=-1), torch.ones(len(texts)), atol=1e-5, rtol=1e-5
        )
        assert report.get("compilation_stable", True), report["compilation"]
        report["passed"] = True
    finally:
        engine.shutdown()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ in ("__main__", "__mp_main__"):
    configure_process()

if __name__ == "__main__":
    main()
