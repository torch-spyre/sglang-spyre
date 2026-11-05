# Spyre parity benchmark

## SGLang vs vLLM on five MMLU items (2026-11-01)

The machine-readable record is
[`benchmarks/2026-11-01-mmlu-sglang-vs-vllm.json`](benchmarks/2026-11-01-mmlu-sglang-vs-vllm.json).
It was produced by [`tests/mmlu_sglang_vs_vllm_bench.py`](tests/mmlu_sglang_vs_vllm_bench.py),
which extends the three-example benchmark below.

**Setup**
- Both frameworks ran on one pod with the same torch-spyre build
  (`64870622`), each on its own card with 32 pinned cores.
- SGLang: 0.5.21 with this backend.
- vLLM: 0.32.0 with spyre-inference `0ccdbf6`, the revision this backend is
  synchronized to.

**Method**
- Items: test index 0 of five `cais/mmlu` subjects, at the pinned revision
  `c30699e8356da336a370243923dbaf21066bb9fe`.
- Prompts: both frameworks receive the same tokenized prompt (the earlier
  benchmark's template).
- Decoding: batch size 1, greedy, 16 new tokens.
- Timing: per item, one warmup, then three measured requests of each length.
  TTFT is a 1-token request; latency is the 16-token request.
- Output throughput is 15 tokens over (latency − TTFT).

Each cell shows SGLang / vLLM; "Answer" is the first A–D letter generated.

### `ibm-granite/granite-3.3-8b-instruct`

| MMLU subject | Input tokens | TTFT (ms) | Latency (ms) | Output tok/s | Gold | Answer | Same text |
|---|---:|---:|---:|---:|:-:|:-:|:-:|
| abstract algebra | 71 | 278 / 262 | 2670 / 2663 | 6.27 / 6.25 | B | D / B | no |
| anatomy | 125 | 265 / 262 | 2578 / 2533 | 6.49 / 6.61 | A | A / A | yes |
| high school physics | 100 | 255 / 263 | 2507 / 2586 | 6.66 / 6.46 | B | C / C | no |
| professional medicine | 320 | 292 / 276 | 2571 / 2384 | 6.58 / 7.11 | C | C / C | yes |
| world religions | 55 | 258 / 284 | 2471 / 2611 | 6.78 / 6.45 | D | D / D | yes |
| **mean** | | **270 / 269** | **2559 / 2555** | **6.56 / 6.57** | | 3/5 / 4/5 correct | 3/5 |

SGLang/vLLM ratios:
- latency **1.001**
- TTFT 1.001
- output throughput 0.997

This is inside the ±15% parity gate.

The two frameworks give the same answer on 4 of 5 items. The fifth,
abstract algebra, is a near tie: the fp32 Transformers CPU reference puts
`D` at 19.50 and `B` at 19.40. SGLang picks `D`, which is the fp32 reference's
choice; vLLM picks `B`. Two of the five texts diverge later, inside the
explanation.

### `google/gemma-4-26B-A4B-it` (text backbone, one card)

| MMLU subject | Input tokens | TTFT (ms) | Latency (ms) | Output tok/s | Gold | Answer | Same text |
|---|---:|---:|---:|---:|:-:|:-:|:-:|
| abstract algebra | 70 | 878 / 841 | 2366 / 2451 | 10.08 / 9.32 | B | — / A | no |
| anatomy | 104 | 875 / 865 | 2388 / 2392 | 9.91 / 9.83 | A | A / A | yes |
| high school physics | 94 | 882 / 864 | 2415 / 2363 | 9.78 / 10.00 | B | A / A | yes |
| professional medicine | 259 | 880 / 969 | 2363 / 2449 | 10.11 / 10.13 | C | A / A | no |
| world religions | 53 | 894 / 841 | 2443 / 2395 | 9.68 / 9.65 | D | A / A | no |
| **mean** | | **882 / 876** | **2395 / 2410** | **9.91 / 9.79** | | 1/5 / 1/5 correct | 2/5 |

SGLang/vLLM ratios:
- latency **0.994**
- TTFT 1.006
- output throughput 1.013

This is inside the ±15% parity gate.

Both frameworks score 1/5. This instruction-tuned model gets the raw template
without its chat template, and often starts with a newline or a restated
option. The bf16 Transformers CPU reference shows the same leading tokens:
`" "` first on abstract algebra, `"\n\n"` first on professional medicine.

The two frameworks give the same answer letter wherever both produce one. On
abstract algebra SGLang's first sixteen tokens contain no letter, so the
"Answer" cell is empty. The texts agree token for token on two items and
diverge at fp16 noise after the first few tokens on the others.

With the model's chat template, the earlier 16-token probe gave the same
first 13 tokens on both frameworks. Two fixes found in this comparison are
part of the change:
- the router's folded scale on the Spyre norm path;
- the tied LM head's device weight.

### Kernel A/B and attention smoke

These are recorded in
[`benchmarks/2026-11-01-spyre-dev.json`](benchmarks/2026-11-01-spyre-dev.json).

- **Kernel A/B.** The vendored head-major decode kernel matches
  spyre-inference's own bit for bit at Granite's geometry (32 query heads,
  8 KV heads, head size 128, 512 cached tokens). Its median is 0.150 ms
  against 0.140 ms; both are compiled the same way.
- **Smoke test.** `tests/test_attention_smoke.py` drives one 8-token prefill
  through the SGLang backend on a card. The result is within 0.0088 of
  CPU SDPA.

Reproduce on a Spyre host with each framework's environment:

```bash
python tests/mmlu_sglang_vs_vllm_bench.py sglang --model ibm-granite/granite-3.3-8b-instruct --output sg.json
python tests/mmlu_sglang_vs_vllm_bench.py vllm --model ibm-granite/granite-3.3-8b-instruct --output vl.json
python tests/mmlu_sglang_vs_vllm_bench.py compare sg.json vl.json --output compare.json
```

## Previous baseline (2026-08-26)

The reproducible result record is
[`benchmarks/2026-08-26-spyre-dev.json`](benchmarks/2026-08-26-spyre-dev.json).
Both framework runs used one refreshed Spyre pod, the same Granite micro 1B
model, FP16, batch size 1, 64 input tokens, 16 output tokens, one warmup, and
three measured requests. The pulled runtime image resolved to digest
`sha256:daa809d6c0487b019bc992e988570be5e358aa9c2b451911074654bbeec2dacd`.

| Path | Mean latency | Output throughput |
|---|---:|---:|
| latest spyre-inference | 427.74 ms | 37.41 tok/s |
| SGLang with this backend | 424.71 ms | 37.67 tok/s |

SGLang is 0.7% faster in this small run (1.007x), satisfying the ±15% parity
gate. The directly ported attention kernel also matches the reference output
exactly and is within 1.3% of its median latency.

For the requested native PyTorch comparison, one decode-attention forward with
the supported model's geometry (32 query heads, 8 KV heads, head size 128, 256
cached tokens) took 1.270 ms on CPU and 0.979 ms on Spyre, including the output
copy: a 1.30x speedup. This is an operator-level forward comparison, not a claim
about end-to-end server speedup. A separate model-level check recorded a
14.57 ms cached-token bare PyTorch CPU forward. A warmed 64-to-16 generation
request took 334.19 ms in bare PyTorch and 424.71 ms through SGLang/Spyre, or
0.79x. In other words, this tiny batch-1 model reaches parity with the reference
Spyre stack but does not beat a 96-thread CPU end to end; the measured 1.30x
speedup is specifically the synchronized attention forward.

### Three-example MMLU comparison

[`tests/mmlu_speedup_bench.py`](tests/mmlu_speedup_bench.py) compares an
end-to-end, one-token greedy `generate` call through SGLang/Spyre with bare
PyTorch CPU. It uses the same model and exact tokenized prompt on both paths,
batch size 1, one warmup per prompt shape, and three measured iterations. The
three cases are test index 0 from the pinned
[`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe)
revision `c30699e8356da336a370243923dbaf21066bb9fe`.

| MMLU subject | Input tokens | PyTorch CPU mean | SGLang/Spyre mean | Speedup vs CPU |
|---|---:|---:|---:|---:|
| abstract algebra | 71 | 101.31 ms | 135.78 ms | 0.746x |
| anatomy | 125 | 125.27 ms | 167.85 ms | 0.746x |
| high-school physics | 100 | 106.76 ms | 147.09 ms | 0.726x |

The total-latency ratio is **0.740x** (geometric mean 0.739x), meaning that
SGLang/Spyre was 35.2% slower than the 96-thread PyTorch CPU baseline for this
small-request workload. SGLang's timing includes scheduler IPC and sampling;
the bare PyTorch timing calls `model.generate` directly. This is a latency-only
comparison: the single generated token matched on the physics case but differed
on the other two, so the result does not establish output equivalence.

Run it on a configured Spyre host with:

```bash
python tests/mmlu_speedup_bench.py \
  --output benchmarks/2026-08-26-mmlu-speedup.json
```

The checked-in machine-readable result is
[`benchmarks/2026-08-26-mmlu-speedup.json`](benchmarks/2026-08-26-mmlu-speedup.json).
