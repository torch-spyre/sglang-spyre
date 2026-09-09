# SGLang vs vLLM on Spyre — a deep dive

A comparison aimed at Spyre engineers deciding which inference framework to
invest in next, and what the upcoming Spyre dynamic-shapes refactor changes
for each side.

The reference points used in this doc:
- **vLLM**: upstream `main`, plus the in-tree `spyre-inference` plugin
  (`github.com/torch-spyre/spyre-inference`). The behaviour described here
  is the post-V1 plugin model. Spyre's `wiki/stack/vllm.md` page documents
  an older snapshot from before the V1 refactor and is being rewritten.
- **SGLang**: upstream `main` (≈v0.5.12), plus this repo
  (`Gal-Bloch/Spyre-SGLang`) — an out-of-tree platform plugin we built to
  prototype Spyre integration. Where this doc cites concrete TPS, those
  are from our prototype on Granite-1B; they are validation numbers, not
  reliable benchmarks.

---

## TL;DR

### What each library does better

| Concern | vLLM is better at | SGLang is better at |
|---|---|---|
| Continuous batching at high request fanout | ✅ More mature scheduler, 3+ years of tuning | comparable |
| Long shared prefixes (system prompts, RAG, agents) | block-level sharing only at 16-tok boundaries | ✅ **RadixCache** — any-length sharing |
| GPU INT-quant breadth (AWQ / GPTQ / GGUF / bitsandbytes) | ✅ Wider matrix upstream, but **none apply to Spyre** (INT support dropped) | n/a |
| Spyre's actual low-precision formats (DL16 / BF16 / FP8 E4M3+E5M2 / FP4) | comparable — both require plugin-side integration | comparable |
| FP8 KV-cache plumbing upstream (scale management, dequant-on-read) | slightly more mature in core | newer, narrower — but on Spyre the KV pool is owned by the plugin anyway, so this gap doesn't carry over |
| Speculative decoding, multi-LoRA, MoE routing | ✅ More mature | newer, narrower |
| Encoder-decoder, sliding-window, prefix quant | ✅ Available today | partial |
| Plugin contract for a new device (less code) | ~3500 lines for `spyre-inference` | ✅ ~1500 lines for this plugin |
| Reusable attention base class | interface only | ✅ `TorchNativeAttnBackend` is real |
| Reusable KV pool | plugin-supplied | ✅ device-agnostic base, override one method |
| Memory allocator maturity | ✅ Battle-tested, fragmentation-aware | younger; we had to override two methods |

### How HW integration differs

The two frameworks ask very different things of a device backend:

- **vLLM** uses a **Worker + ModelRunner subclass** model. You inherit
  `Worker` and `ModelRunner`, override `execute_model()`, manage the
  per-process model + KV pool + sampler yourself, and provide
  CPU<->device staging buffers because V1's sampler is CPU-side and
  expects integer inputs (token IDs, positions) on CPU. The plugin
  carries this dual-buffer plumbing — `_SpyreModelWrapper` in
  `spyre-inference` does the CPU<->Spyre conversion at the model
  boundary on every forward.
- **SGLang** uses a **Platform + factory methods** model. You provide
  factories (KV pool class, paged allocator class, attention backend),
  declare `init_backend()` for per-process device init, and that's it.
  No Worker class to subclass, no separate ModelRunner. Sampling and
  scheduling are device-agnostic at SGLang's layer; with `device="cpu"`
  the model body stays on CPU, attention crosses to the device, and no
  staging is required. With `device="spyre"`, ~10 lines of boundary
  hooks accomplish what `_SpyreModelWrapper` does in vLLM.
- **Attention base class**: vLLM's `AttentionBackend` is an interface —
  every plugin reimplements KV gather, block-table interpretation,
  extend-vs-decode dispatch, GQA / causal handling. SGLang's
  `TorchNativeAttnBackend` is a real subclass with usable defaults; for
  Spyre we override **only the inner SDPA seam** and inherit the rest.
  This is roughly two-thirds of the line-count gap between the plugins.
- **KV pool**: vLLM has no device-agnostic base; the plugin provides
  the whole pool. SGLang's `MHATokenToKVPool` allocates buffers on
  `self.device`, so a Spyre subclass is one method override.
- **Block-table semantics**: vLLM gives the backend
  `block_table[seq, page] → block_id` (page-size-coupled). SGLang gives
  `req_to_token[req, slot] → token_idx` (per-token). On Spyre, the
  per-token form is a closer fit for indexed-gather lowering and works
  cleanly at `page_size=1` without special cases.
- **Core source patches**: vLLM V1's plugin contract is clean — no core
  edits needed. SGLang core is younger and assumes CUDA/Triton in a few
  places; we ship four small idempotent patches in
  `sglang_oot_patches/apply.py` (memory_pool_host CUDA-only guard, vllm
  optional import in rotary base, support_triton allowlist,
  SUPPORTED_DEVICES whitelist). All upstreamable.

The bottom line on integration: **vLLM has a higher floor of plugin
machinery you must implement; SGLang has a lower floor but requires
small core patches today**. The gap doesn't close with the dynamic-
shapes refactor — it's structural to the framework contracts.

### How broad is model support today?

At the SGLang revision pinned by this sync, **12 of 244 model
architectures are validated on the Spyre backend**. Another 13 reach
generation but are not supported because their output diverges or a
correctness-critical path remains unverified. The remaining 219 are
stopped by a known incompatibility. See §5 for the compatibility
breakdown and the methodology behind those numbers.

### When vLLM's attention approach wins; when RadixCache wins

The two attention layouts optimise for different workload shapes:

| Workload pattern | Better fit | Why |
|---|---|---|
| Many requests, mostly **unique** prompts (public-API serving, OpenAI-style chat completions across diverse users) | **vLLM** block-paged | Prefix sharing buys little; mature continuous batching dominates. Block-paged also gets better KV bandwidth on large-KV devices. |
| Many requests sharing **long system prompts** (RAG with template, instruction-following with shared persona) | **SGLang RadixCache** | Block-aligned sharing in vLLM only kicks in at 16-tok boundaries; SGLang shares any-length prefix. On a 200-tok system prompt with 16-tok blocks, vLLM shares 192/200 tokens. SGLang shares all 200. The gap compounds across thousands of requests. |
| **Multi-turn chat** (each turn extends the previous conversation) | **SGLang RadixCache** | Each new turn shares the entire prior conversation as a prefix, at any token boundary. vLLM's block-aligned sharing wastes the tail of the last block. |
| **Agent flows** (many short actions sharing a long tool/system prompt) | **SGLang RadixCache** | High-fanout, deep shared prefixes — RadixCache's strongest case. |
| **Long single-prompt generation** (one user, one long prompt, lots of decode) | comparable | Neither cache helps; throughput is decode-bound. vLLM's broader feature matrix may matter (spec decoding, MoE). |
| **Batch-1 throughput** (single-stream eval, smoke tests) | comparable | Neither cache fires; overhead is per-forward kernel dispatch. |
| **High-concurrency, prompts > KV cache** (eviction-dominated regimes) | **vLLM** | More mature eviction policy, better-tuned allocator. |
| **Large contiguous KV reads** (HBM-equipped GPUs; AIU 1.5 with HBM3e) | **vLLM** block-paged | Block layout is a closer fit for large-block DMA than per-token addressing. |

**Spyre-specific addendum**: today, even where RadixCache *would* help,
Spyre's static-shape constraint forces a BMM-against-one-hot for
runtime indexing — which costs more than the prefix re-transfer it
saves on small models. The dynamic-shapes refactor (KTIR
`construct_indirect_access_tile` lowering) replaces the BMM with real
`cache[block_table[i]]` indirection and **makes RadixCache prefix hits
genuinely free on Spyre** for the first time. That is the regime where
SGLang's structural advantage shows up as actual TPS gains. See §6 for
the differential impact on each framework.

For IBM-internal workloads (RAG with shared system prompts, agent
flows, multi-turn chat), the workload pattern matrix above lines up
heavily on the SGLang side. For broad-fanout serving with diverse
prompts, vLLM is the better fit.

---

## 1. High-level library design

The two libraries solve overlapping problems with very different
orchestration models. Both run a sequence-level scheduler over a paged
KV cache and dispatch attention/MLP kernels to a device backend. They
diverge on what the scheduler optimises for, and how the prefix cache is
represented.

### vLLM — block-paged scheduler, vLLM V1 plugin model

vLLM's scheduler is a continuous-batching engine over a fixed-size pool
of KV blocks (typically 16-token blocks). Block IDs are assigned per
sequence. Prefix sharing is **block-level**: two sequences that share a
prompt up to a block boundary can re-use those physical blocks; anything
shorter than a block isn't shared.

Backend integration is via the **V1 platform plugin**:
- A `Platform` subclass registered through the `vllm.platform_plugins`
  entry point. It dictates dtype defaults, attention backend, worker class.
- A `Worker` (or subclass of `CPUWorker` / `GPUWorker`) that owns the
  process-local model + KV pool + sampler.
- A `ModelRunner` whose `execute_model()` is the per-step entry point.
  vLLM hands it a `ForwardBatch`-equivalent and expects logits back.
- An `AttentionBackend` subclass that exposes `forward_*` methods over
  query / paged-KV indices. Backends own block-table interpretation.
- Optional **`register_oot()`** for op-level overrides (RMSNorm, SiluAndMul,
  RoPE, etc.). vLLM's MLP/norm modules dispatch through these on OOT
  devices; this is the hook spyre-inference uses for on-device kernels.

**Control flow**: scheduler produces a batch → ModelRunner moves inputs
to device → forward → AttentionBackend with paged scatter/gather →
sampler on CPU → response back. Each step is a self-contained graph.

### SGLang — RadixCache scheduler, OOT platform via entry point

SGLang's scheduler centres on a **RadixCache**: prefix sharing is at
arbitrary token granularity, represented as a radix tree of token IDs
keyed to KV-pool slots. Two requests sharing a 51-token prefix get
`cached_tokens=51` even when 51 isn't a block boundary. The primary
optimisation target is high-fanout request patterns (chat completions,
RAG, agent workflows) where many requests share long prefixes.

Backend integration is via SGLang's `sglang.srt.platforms` entry point:
- An `SRTPlatform` subclass — same role as vLLM's `Platform`. Provides
  factories for KV pool class, paged allocator class, graph runner class,
  and a `get_default_attention_backend()` returning a registered name.
- An `AttentionBackend` registered via `@register_attention_backend("name")`.
  Two flavours: a base `AttentionBackend` (full ownership of KV gather +
  metadata), or a `TorchNativeAttnBackend` subclass that inherits the KV
  gather and overrides only the inner SDPA seam.
- Op-level overrides via `MultiPlatformOp.register_oot_forward(op_cls,
  fn, dispatch_key)`. Same mechanism as vLLM's `register_oot()`,
  different name.
- KV pool subclassing (`MHATokenToKVPool`) for device-resident KV cache.
  The base class is already device-agnostic — `_create_buffers` allocates
  on `self.device` — so a Spyre subclass only has to override that one
  method.

**Control flow**: scheduler does a radix-tree lookup on the request
tokens → matches longest existing prefix → assigns logical block-table
entries that point at the cached slots → ModelRunner → AttentionBackend
sees a `forward_batch` whose `req_to_token` already encodes the cache
hit → forward → response. Prefix hits never reach the attention compute
in the first place; the scheduler trims the prefill region before the
forward begins.

### What this means for the device backend

The backends are doing the same job: take q/k/v on whatever device the
platform says, look up KV slots from a block table, run attention math,
return output. The differences:

- **Block-table semantics**. vLLM hands the backend `block_table[seq, page]`
  → physical block ID. SGLang hands the backend `req_to_token[req, slot]`
  → physical token slot. Same idea, different indexing — the SGLang
  variant is **per-token**, which means our gather doesn't need to know
  about page sizes at all (and works for `page_size=1` without special
  cases). In return, vLLM's per-block layout is friendlier to large
  contiguous KV blobs.
- **Where the cache logic lives**. vLLM's RadixCache equivalent is a
  per-batch hashing layer; the backend sees only block IDs. SGLang's
  RadixCache lives in the scheduler proper; the backend sees the
  already-resolved slot list and never participates in cache decisions.
  This means SGLang prefix hits are entirely transparent to a device
  backend — you can re-use slots without lifting a finger.
- **Multi-platform op dispatch**. Both libraries have it; the surface
  area is comparable. vLLM's `register_oot()` is per-op-class with a
  string device key; SGLang's `MultiPlatformOp.register_oot_forward()`
  is similar but resolved at op `__init__` time (cached on
  `_forward_method`), so re-registration after model build needs a
  re-dispatch step — `sglang_spyre_backend/model_wrapper.py` does this.

---

## 2. Performance considerations (general)

We can't reliably benchmark Spyre right now (single-tenant pod, dynamic
shapes still pre-refactor, kernel-cache effects dominate at this scale).
What we *can* do is reason about which framework is structurally better
suited to which workload.

### Throughput at high request fanout

SGLang's RadixCache is the standout feature. For workloads with many
requests sharing long prefixes (system prompts, multi-turn chat,
batched agent calls, RAG-with-template), it skips both **recompute** and
— if KV is on device — **re-transfer** of the prefix tokens. Published
SGLang numbers vs vLLM on multi-turn benchmarks show 2-5× higher
throughput, primarily from this effect. The structural argument:

- vLLM block-level sharing only triggers at multiples of `block_size`
  (16). For 200-token system prompts with 16-token boundaries, you
  effectively share 192 tokens out of 200 — fine, but only once you're
  paying for 16-token-aligned batching.
- SGLang shares at any prefix length. A 51-token system prompt is shared
  in full. For shorter shared regions or non-aligned prompts, this
  difference compounds quickly.

### Throughput at single-request / batch=1

The two are roughly even. Both libraries' overhead is dominated by per-
forward kernel dispatch + sampling, not by cache machinery. SGLang's
radix tree adds a small constant-time lookup per request; vLLM's hash
adds the same. Neither matters at batch=1.

### Memory utilisation

vLLM's block-paged design is mature and well-tuned: page-aligned KV
buffers, predictable allocation, good fragmentation behaviour at scale.
SGLang's KV pool is also paged but the allocator (the "PagedTokenToKV
PoolAllocator") is younger and less battle-tested — we've already had
to override `alloc_extend` / `alloc_decode` for Spyre because the stock
ones dispatch Triton kernels that don't exist on our device. vLLM's
allocator is pure torch-native at this layer.

### Continuous batching

vLLM was the project that popularised continuous batching at Python
level; its scheduler reflects 3+ years of tuning. SGLang has continuous
batching too but the literature treats it as "comparable" rather than
"better" — the bigger win on the SGLang side is RadixCache, not CB.

### Quantization & low precision

This is the area where the upstream story and the Spyre story diverge
most. A naive read of vLLM vs SGLang says "vLLM has more quantization
formats." That's true upstream — vLLM ships AWQ, GPTQ, GGUF,
bitsandbytes, FP8, FP4, MXFP4, blockwise INT8, and more. But almost
all of that breadth is **CUDA-targeting INT-pack formats** with
hand-tuned CUDA / CUTLASS / Marlin kernels:

- **AWQ, GPTQ, GGUF, bitsandbytes**: INT4 / INT3 / INT8 weight packing
  for GPU. Spyre dropped INT support, so these aren't viable on Spyre
  regardless of which library exposes them. They run on GPU by
  decompressing INT-packed weights to BF16 at load time and using
  INT-specific CUDA kernels — none of that helps a Spyre backend.
- **FP8 / FP4 / MXFP4**: these are the formats Spyre actually targets
  natively (FP8 on AIU 1.5 hardware; FP4 is 1.5-only via AIUFP4).
  Upstream both libraries ship CUDA / CUTLASS implementations — neither
  library carries a Spyre-targeting path. The actual FP8/FP4 compute
  on Spyre comes from torch-spyre kernels, registered through the
  plugin's op-forward surface (`register_oot()` on vLLM,
  `MultiPlatformOp.register_oot_forward()` on SGLang). The
  registration shapes are equivalent.

Where vLLM does have a small *real* edge for Spyre: **FP8 KV-cache
plumbing** (per-block scale management, dequant-on-read at attention
time) is more mature in vLLM core than in SGLang. But on Spyre the KV
pool is owned by the plugin's `MHATokenToKVPool` subclass anyway, so
the layout decisions are made on the plugin side either way; vLLM's
upstream plumbing is templates, not directly usable kernels.

**Net for Spyre's low-precision regime**: neither library has a
structural advantage. Quant integration is a plugin-side problem in
both — same op-registration surface, same op-forward surface, same
weight-placement hook (vLLM: `process_weights_after_loading` on
`QuantizationConfig`; SGLang: same name on the equivalent
`QuantizeMethodBase`). vLLM's wider upstream matrix matters only if
the workload demands a CUDA-style INT format, which Spyre doesn't run.

### Other long-context features

vLLM is ahead on speculative decoding, multi-LoRA, MoE routing,
encoder-decoder, and sliding-window attention. SGLang has most of
these now but arrived later. For a Spyre bring-up these matter only
if the workload demands one of them — neither library *forces* the
device backend to support them.

### Bottom line

For Spyre's likely workloads — IBM-internal RAG, agent flows, long
shared system prompts — **SGLang's RadixCache should produce more
realised speedup per Spyre cycle than vLLM's block sharing**, because
prefix lengths are determined by application code (not aligned to
16-token boundaries). For workloads dominated by short or unique
prompts, the two are roughly equivalent — vLLM's CB scheduler is more
battle-tested, and its serving-feature matrix (spec decoding, multi-
LoRA, MoE routing) has fewer gaps. Quantization-format breadth is not
a Spyre-relevant tiebreaker (see "Quantization & low precision" above).

---

## 3. Attention approaches

The two libraries arrive at the same shape of computation but with
materially different layout assumptions, which matters for what's hard
on Spyre.

### vLLM attention

vLLM's `AttentionBackend.forward()` receives:
- `query`: `[num_tokens, num_q_heads, head_dim]`
- `kv_cache`: a single tensor `[num_blocks, block_size, num_kv_heads, head_dim]`
  laid out for contiguous block reads
- `block_table`: `[num_seqs, max_num_blocks]` mapping logical → physical
  block IDs
- `slot_mapping`: `[num_tokens]` flat indices into `kv_cache` for the
  *new* tokens being written this step

The backend writes new k/v at `slot_mapping`, then runs attention reading
from `kv_cache[block_table[req]]`. On GPU, FlashAttention / FlashInfer
handle the indirection in fused kernels. On Spyre — where runtime tensor
indexing is a known constraint and chunked prefill with `KV_LENGTH_
ALIGNMENT=256` and `QUERY_CHUNK_SIZE=32` is the supported shape — the
plugin (`spyre-inference`'s `SpyreAttentionExpBackend`) replaces the
indexing with **BMM against one-hot selection masks**: build an
`[N_blocks, block_size, num_blocks]` one-hot tensor on CPU, transfer it
once, and use BMM as a runtime-indexable gather. Costly, but produces
a Spyre-legal IR.

### SGLang attention

SGLang's `AttentionBackend.forward_extend/decode()` receives a
`forward_batch` containing:
- `q`: `[num_tokens, num_q_heads, head_dim]` (or already-flattened
  `[num_tokens, n_q_heads * head_dim]`)
- `req_to_token`: `[max_running_requests, max_seq_len]` mapping
  `(req, slot) → physical_kv_token_index`
- `seq_lens`, `extend_prefix_lens`, `extend_seq_lens` for per-request
  shape info

The KV cache is a per-layer `[N_kv_slots, num_kv_heads, head_dim]`
tensor. Per-token addressing — no block size. The base `TorchNativeAttn
Backend` does the gather as `k_cache[req_to_token[req, :seq_len]]`
followed by SDPA, all in pure torch.

For Spyre, this is much friendlier than vLLM's layout because **the
block size is whatever you want**. Our `attention_native.py` subclasses
`TorchNativeAttnBackend` and overrides only the inner SDPA seam:
inherit the KV gather, the pool save, the GQA/causal/cross-attention
plumbing — and swap `F.scaled_dot_product_attention` for the bucketed
`_attn_4d` kernel from spyre-inference. This is mode 2 in this repo's
README.

For the on-device-KV path (mode 1), the same BMM-against-one-hot trick
vLLM's experimental backend uses applies — and we ported
`spyre_attn_exp.py` directly, with the only adjustment being that the
"block table" is SGLang's `req_to_token` instead of vLLM's
`block_table`.

### Why the SGLang surface is easier for Spyre

Three concrete reasons:

1. **Inheritance, not reimplementation**. SGLang's `TorchNativeAttn
   Backend` is a real base class with usable defaults. Subclassing it
   for Spyre meant the entire KV gather / pool save / extend-vs-decode
   bookkeeping was inherited verbatim — only the inner SDPA call was
   swapped. vLLM's `AttentionBackend` is more of an interface; every
   plugin reimplements the gather machinery.
2. **Per-token addressing**. No block-size constant has to thread
   through the Spyre kernels. `KV_LENGTH_ALIGNMENT=256` is purely a
   bucketing decision in the kernel dispatcher; it's not coupled to the
   KV pool's storage layout.
3. **Page-size=1 works**. SGLang's non-paged allocator is pure torch.
   vLLM page-size=1 is unusual enough that some kernels assume
   `block_size >= 16`. For a debug or correctness-first port, this
   matters.

The flip side: vLLM's block-paged layout is genuinely better for
contiguous KV reads on devices that *like* large block transfers (any
HBM-equipped GPU, eventually Spyre 1.5 once HBM3e lands). SGLang's
per-token layout makes shared-prefix bookkeeping cleaner but loses some
locality.

---

## 4. Spyre integration — framework-level cost

What follows is a comparison of what each framework's plugin contract
asks of a device backend. It deliberately ignores the current crop of
Spyre-toolchain bugs and shape constraints — those are listed
separately at the end of this section, because they apply equally to
both frameworks and most of them disappear after the dynamic-shapes
refactor.

The framework-level cost is the part that **doesn't** change post-
refactor: the contract you're held to as a plugin author, the surfaces
you have to subclass, the lifecycle hooks you have to honour. This is
what stays.

### vLLM plugin contract

- `Platform` subclass registered via the `vllm.platform_plugins` entry
  point. Owns dtype defaults, attention backend selection, worker class.
- `Worker` subclass (or `CPUWorker` extension): owns the per-process
  model + KV pool + sampler.
- `ModelRunner` whose `execute_model()` is the per-step entry point —
  receives a batch of requests, returns logits. The plugin owns input
  marshalling, sampling glue, and any device staging.
- `AttentionBackend` subclass — an interface, not an inheritable base.
  Each plugin reimplements KV gather, block-table interpretation,
  extend-vs-decode dispatch.
- `register_oot()` for op-level overrides (RMSNorm, SiluAndMul, RoPE,
  Linear weight placement).
- KV pool is plugin-supplied — vLLM doesn't provide a device-agnostic
  base for backends to subclass.

### SGLang plugin contract

- `SRTPlatform` subclass registered via the `sglang.srt.platforms`
  entry point. Same role as vLLM's `Platform`.
- `init_backend()` lifecycle hook on the platform — no separate
  `Worker` class. The platform itself handles per-process device init.
- `AttentionBackend` registered via `@register_attention_backend("name")`.
  Two flavours:
  - Subclass of `AttentionBackend` (full ownership), or
  - Subclass of `TorchNativeAttnBackend` — a real, inheritable base
    class with usable defaults. Override only the inner SDPA seam and
    inherit KV gather, pool save, GQA / causal / cross-attention
    plumbing, extend-vs-decode dispatch.
- `MultiPlatformOp.register_oot_forward(op_cls, fn, dispatch_key)` for
  op-level overrides — same bodies as vLLM's `register_oot()`,
  different registration call.
- `MHATokenToKVPool` is device-agnostic in the base class
  (`_create_buffers` allocates on `self.device`). A device subclass
  only has to override that one method.
- `PagedTokenToKVPoolAllocator` — vLLM has no equivalent split; SGLang's
  paged allocator is a separate factory class on the platform, which
  gives the plugin a clean override point.

### Framework-level comparison

The numbers below are measurements from each plugin's current
codebase, not estimates. They reflect the framework contract, not
which Spyre toolchain bugs each plugin happens to work around.

| Aspect | vLLM (`spyre-inference`) | SGLang (this plugin) |
|---|---|---|
| Plugin discovery | `vllm.platform_plugins` entry point | `sglang.srt.platforms` entry point |
| Worker lifecycle | `Worker` + `ModelRunner` subclasses, with `_SpyreModelWrapper` for CPU↔device staging | `init_backend()` on the platform — no Worker class |
| Attention base class | Interface (you reimplement gather, dispatch, metadata) | `TorchNativeAttnBackend` is inheritable; override only the SDPA seam |
| KV pool reuse | Plugin-supplied | `MHATokenToKVPool` device-agnostic; one method to override |
| Custom ops | `register_oot()` | `MultiPlatformOp.register_oot_forward()` |
| Block-table semantics | `block_table[seq, page] → block_id` (page-size-coupled) | `req_to_token[req, slot] → token_idx` (per-token) |
| Status | Production, stable | Prototype (this repo) |
| Lines of plugin code | ~3500 | ~1500 |

The SGLang plugin is roughly half the code. About two-thirds of that
gap is the attention layer (real base class to subclass vs interface
to reimplement); the rest is the KV pool reuse and the Worker-class
absence. **None of that gap closes post-refactor — it's structural to
the framework contract.**

### Spyre-toolchain workarounds (apply equally to both)

These are bugs and shape constraints in the current Spyre stack, not
properties of either framework. Both `spyre-inference` and this plugin
carry essentially the same list. Including them so the picture is
honest, not because they differentiate the two frameworks.

Most of these go away with the dynamic-shapes refactor (see §6).

| Workaround | What it does | Post-refactor status |
|---|---|---|
| `RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`LOCAL_WORLD_SIZE` env vars set before worker spawn | `libspyre_comms.so` reads at dlopen — SEGV in `libsenlib-dd2.so` if missing | likely persists (env-var contract of `libspyre_comms`); not a refactor target |
| Single-tenant VFIO guard (`init_backend` no-op in parent) | Only one process can open the AIU card | persists until VFIO multi-tenant lands; orthogonal to dynamic shapes |
| `addmm` decomposition patch (torch-spyre #1420) | First FX trace crashes without it | likely fixed by the refactor's lowering rewrite |
| RoPE cache forced to CPU | `1.0 / (base ** (arange / rotary_dim))` triggers `DtException 4Bsplat` | likely fixed (the bad lowering pattern is in scope of the rewrite) |
| `_attn_4d` bucketing (`KV_LENGTH_ALIGNMENT=256`, `QUERY_CHUNK_SIZE=32`) | Static shapes only — every distinct length is a separate compiled graph | **fixed by the refactor — variable-length attention without bucketing** |
| `num_seqs == 1` hard-coded in attention backends | Continuous batching unsupported on the device side | **fixed by the refactor** |
| BMM-against-one-hot scatter/gather (mode 1 / `SpyreAttentionExpBackend`) | Replaces unsupported runtime tensor indexing | **fixed by the refactor** (KTIR `construct_indirect_access_tile` lowering replaces the BMM with real `cache[block_table[i]]` indirection) |
| Triton allocator override (SGLang `alloc_extend`/`alloc_decode` for `page_size > 1`) | Stock allocator dispatches Triton; no Spyre Triton backend | possibly fixed if SGLang adopts a torch-native fallback path; orthogonal to the Spyre refactor |
| Four small SGLang core patches in `sglang_oot_patches/apply.py` (CUDA-only guard in `memory_pool_host`, optional `vllm._custom_ops` in rotary base, `support_triton` allowlist, `SUPPORTED_DEVICES` whitelist) | Teaches SGLang that an OOT, non-Triton, non-CUDA device is a real thing | likely upstreamable to SGLang independently of the Spyre refactor; not a permanent SGLang-specific cost |

The SGLang core patches are the only entry on this list that is
SGLang-specific. They aren't a Spyre-toolchain issue at all — they're
SGLang OOT-readiness gaps. Reasonable to upstream; until then, four
short `sed` lines applied at install time.

The asymmetry to call out: vLLM has a specific Spyre integration cost
that SGLang does not — **the dual-buffer dance**. vLLM's V1 sampler
runs CPU-side and expects integer inputs (token IDs, positions) to
arrive on CPU, while model tensors live on Spyre. `TorchSpyreModelRunner`
maintains parallel CPU staging buffers and `_SpyreModelWrapper` handles
the conversion at the model boundary. SGLang's design is more lenient
here: with `device="cpu"` (where the model body stays on CPU and only
attention crosses to Spyre, our mode 2), no staging is needed at all;
with `device="spyre"`-equivalent (our mode 4), boundary hooks at
embed-output / body-output handle the same role in ~10 lines. The
dynamic-shapes refactor doesn't change this — it's a vLLM V1 design
choice, not a Spyre toolchain limitation.

### Concrete TPS from this repo's prototype (Granite-1B, 200 tok)

Not benchmarks. Validation that the path runs end-to-end. No claim of
representativeness for production.

| Mode | `attention_backend` | TPS | What runs on Spyre |
|---|---|---:|---|
| 3 | `torch_native` | 19.3 | nothing (CPU baseline) |
| 2 | `spyre` | 14.8 | `_attn_4d` only; KV on CPU |
| 1 | `spyre_paged` | wip | `_attn_4d` + KV on device (BMM scatter/gather) |
| 4 | `spyre` + `SGLANG_FORCE_DEVICE_SPYRE=1` | 3.9 | every Linear, RMSNorm, SiluAndMul, RoPE-residency, attention |

The mode-4 number underperforms the baseline because per-op CPU↔Spyre
transfer dominates compute at hidden=2048. This is the same effect
spyre-inference observes on small models — moving the body to Spyre
only wins above hidden=4096+ or longer sequences, or with larger
batches where the transfer amortises.

vLLM on the same pod with `spyre-inference` runs Granite-1B at roughly
12-15 TPS depending on configuration (rough range, taken from
spyre-inference's own bench harness). Approximately the same ballpark
as our mode 2.

---

## 5. Model coverage after this sync

We validated every `EntryClass` registered by SGLang at revision
`71de97b2` with real Hugging Face configurations. The audit covered the
hard failure conditions in this backend: the batch, page, and
head-layout checks in
[`attention.py`](../sglang_spyre_backend/attention.py), the full,
neox-style RoPE requirement and out-of-tree operation registry in
[`device_ops.py`](../sglang_spyre_backend/device_ops.py), and the TP=1
guard in [`platform.py`](../sglang_spyre_backend/platform.py).

We count an architecture as supported only when it runs with correct
output semantics. Clearing the backend's hard blockers is not enough:
architectures with divergent output or an unverified
correctness-critical path remain explicitly unsupported.

SGLang's registry contains 245 entries in this snapshot. One of them,
`MindSpore`, is an NPU wrapper rather than a model architecture. Removing
it leaves **244 architectures: 12 are validated, 13 run but are not
supported, and 219 are blocked**. The blocked table retains `MindSpore`
as a separate row so the raw registry audit remains complete; its rows
therefore sum to 220.

### Validated and supported (12)

`LlamaForCausalLM`, `Qwen3ForCausalLM`, `GraniteForCausalLM`,
`GemmaForCausalLM`, `Olmo2ForCausalLM`, `BaichuanForCausalLM`,
`Exaone4ForCausalLM`, `OrionForCausalLM`, `XverseForCausalLM`,
`TeleFLMForCausalLM`, `HrmTextForCausalLM`, and
`IQuestCoderForCausalLM`.

### Unsupported: runs, but correctness diverges or is unconfirmed (13)

| Architecture | Known gap |
|---|---|
| `Gemma2ForCausalLM` | Sliding-window attention and logit soft-capping are ignored; GELU also differs. |
| `Gemma3ForCausalLM` | Sliding-window attention is ignored; GELU also differs. |
| `Qwen2ForCausalLM` | Sliding-window attention is ignored. |
| `Ministral3ForCausalLM` | Sliding-window attention is ignored. |
| `MiMoForCausalLM` | Sliding-window attention is ignored. |
| `SolarForCausalLM` | Sliding-window attention is ignored. |
| `Phi3ForCausalLM` (4k) | Sliding-window attention is ignored; its head size is 96. |
| `IQuestLoopCoderForCausalLM` | Loop sliding-window attention is ignored. |
| `DFlashLagunaForCausalLM` | Sliding-window attention is ignored; this is also a draft model. |
| `SDARForCausalLM` | `ENCODER_ONLY` block masks are forced to causal masks. |
| `ApertusForCausalLM` | The `xielu` activation is not registered for Spyre. |
| `ArceeForCausalLM` | The `relu2` activation is not registered for Spyre. |
| `MiniCPMForCausalLM` | The normalization and RoPE path still needs correctness confirmation. |

### Blocked by a known incompatibility (219 architectures, plus one wrapper)

| Primary blocker | Count | Representative entries |
|---|---:|---|
| Multimodal architecture | 68 | Qwen-VL, LLaVA, InternVL, Gemma 3 VL |
| Not a generation architecture | 67 | Embedding, reward, and classification models |
| Mixture of experts | 56 | Mixtral, Qwen3-MoE, DeepSeek, GPT-OSS, Granite 4, Mellum2 |
| LayerNorm | 8 | GPT-2, StarCoder2, OPT, Qwen 1, Phi-3-small, EXAONE 3.5, Persimmon, GPTBigCode |
| Non-neox RoPE | 6 | Cohere/Command-R, GLM-4, ChatGLM, GPT-J, DeepSeek-V4, MuseGlimmer |
| Partial rotary embeddings | 4 | Phi-2, StableLM, GLM-4, Gemma 4 Assistant |
| SSM or linear attention | 4 | Falcon-H1, LFM2, Jet-Nemotron, Qwen3.5 |
| Unsupported RoPE scaling | 3 | InternLM2/3 (`dynamic`), Phi-3-128k (`longrope`) |
| Multi-head latent attention | 2 | MiniCPM3, Mistral Large 3 |
| Unregistered custom operations | 1 | Gemma3n (AltUp/Laurel) |
| Not a model architecture | 1 | MindSpore NPU wrapper |

The support claim is therefore deliberately strict: **12 verified
architectures, not 25 architectures that merely execute**. The next 13
are useful bring-up targets because they already run, but they remain
unsupported until their output is correct and every critical path is
validated. The blocked distribution also makes the larger roadmap
clear: MoE and multimodal support account for well over half of the
remaining model architectures.

---

## 6. The dynamic-shapes refactor — what it changes

The Spyre toolchain is moving from static-shape Inductor lowering (the
current world) to a dynamic-shape regime where graphs can accept
varying sequence lengths and batch sizes without recompilation. This
matters more for one of the two frameworks than the other.

### What's currently shape-sensitive on Spyre

- **Query length**. The current `_attn_4d` requires queries bucketed
  to `QUERY_CHUNK_SIZE=32`. Shorter queries get padded; longer get
  chunked. Each distinct bucket is a separate compiled graph in the
  kernel cache.
- **KV length**. Same — padded to multiples of `KV_LENGTH_ALIGNMENT
  =256`. Compile-cache miss on each new bucket.
- **Batch size**. Currently `num_seqs == 1` is the only validated
  shape; both `spyre-inference` and our plugin assert this in the
  attention backend.
- **Block-table size**. The BMM scatter/gather builds an
  `[N_blocks, block_size, num_blocks]` one-hot tensor whose shape is
  fixed at compile time.

### What dynamic shapes unlock

After the refactor:
- Variable-length attention without bucketing. The query and KV
  dimensions become symbolic; one compiled kernel handles the full
  shape range.
- Real continuous batching. `num_seqs > 1` works; the scheduler can
  actually batch across requests.
- Real radix prefix hits. The block table can vary in shape per step
  without triggering recompile.
- KTIR `construct_indirect_access_tile` lowering — the proper fix for
  runtime tensor indexing, replacing the BMM-against-one-hot workaround
  with real `cache[block_table[i]]` indirection.

### What it means differentially for vLLM vs SGLang

**For vLLM**:
- The biggest current pain (`SpyreAttentionExpBackend`'s BMM scatter/
  gather mask rebuild every step) collapses into a single indexed gather.
- Continuous batching, currently unused on Spyre because `num_seqs==1`
  is hard-coded, becomes available — and vLLM's continuous batching is
  the more mature implementation. This is the bigger win on the vLLM
  side: the scheduler is ready, only the backend has been holding back.
- Block-table dynamism aligns naturally with vLLM's existing layout
  (`block_table[seq, page]` already varies per step on GPU; Spyre just
  catches up).

**For SGLang**:
- RadixCache prefix hits become *real* on device — currently the
  prefix-cached path in mode 2 still re-transfers the prefix KV every
  step because it lives on CPU; in mode 1 the BMM mask rebuild eats
  the savings. With dynamic shapes + real indirection, neither cost
  applies. **This is the configuration where SGLang's structural
  advantage (any-length prefix sharing) translates to actual Spyre TPS
  gains.**
- Per-token KV addressing (`req_to_token` instead of block IDs) was
  always a closer fit for indexed-gather lowering than vLLM's block
  layout. Dynamic shapes don't change this; they just make it usable.
- `num_seqs > 1` unlocks the same continuous-batching gain as on the
  vLLM side; SGLang's CB is comparable but not a differentiator.

**Summary**:

> Dynamic shapes help vLLM more on the **continuous-batching** axis (its
> existing strength becomes usable on Spyre) and help SGLang more on the
> **prefix-sharing-on-device** axis (its existing strength becomes
> usable on Spyre). Whichever workload dominates Spyre's target use
> cases is the one that should drive the framework choice.

For IBM-internal workloads (long shared prompts, RAG, agent flows),
the prefix-sharing axis is the bigger lever — favouring SGLang. For
high-fanout serving where every request has a different prompt
(public-API serving), continuous batching is the bigger lever —
favouring vLLM.

---

## 7. Recommendation

There isn't a single "right" choice — they optimise for different
workload shapes. But three structural observations:

1. **The integration cost is lower for SGLang.** ~1500 lines of plugin
   code vs ~3500, and a real attention base class to subclass instead
   of an interface to reimplement. That gap doesn't change post-
   refactor.
2. **The performance ceiling is workload-dependent.** Long shared
   prefixes → SGLang (any-length sharing). High-fanout unique prompts
   → vLLM (mature CB scheduler, serving-feature matrix). Low-precision
   formats — DL16/BF16/FP8/FP4 — are a wash; both libraries delegate
   to plugin-side integration and Spyre's INT path was dropped, so
   vLLM's GPU INT-quant breadth doesn't carry over.
3. **Dual-tracking is feasible.** The plugins share the same Spyre
   primitives (`_attn_4d`, BMM scatter/gather, RoPE-on-CPU, env-var
   ordering, addmm decomp fix). Maintaining both is mostly
   maintaining the integration glue, not duplicating the device-side
   work. The kernels are shared.

Suggested path: keep `spyre-inference` as the production target for
serving workloads and breadth of serving features (spec decoding,
multi-LoRA, MoE); carry the SGLang plugin
forward for IBM-internal RAG/agent workloads where RadixCache is the
differentiator. Re-evaluate after the dynamic-shapes refactor lands,
when both libraries' Spyre numbers will mean something.

---

## References

- `wiki/stack/vllm.md` — Spyre Knowledgebase page on the vLLM plugin
  (pre-V1-refactor snapshot; being rewritten).
- `wiki/concepts/attention-on-spyre.md` — Spyre attention shape
  invariants (chunked prefill, BLOCK_Q / CORES_PER_Q_TILE).
- `wiki/concepts/precision-scaling.md` — DL16 vs BF16 vs FP8 numerics.
- `github.com/torch-spyre/spyre-inference` — the vLLM Spyre plugin.
- This repo (`Gal-Bloch/Spyre-SGLang`) — the SGLang Spyre plugin.
- SGLang main: `github.com/sgl-project/sglang`.
