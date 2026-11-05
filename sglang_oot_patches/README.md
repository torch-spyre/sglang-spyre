# SGLang 0.5.21 compatibility patch

SGLang 0.5.21 supports out-of-tree devices, but three small compatibility gaps
remain for a text-only Spyre environment:

- `support_triton()` treats every custom attention backend as Triton-capable.
- Image-processor registration requires `torchvision`.
- Gemma-4's server-argument hook accepts only in-tree attention backends.

Two 0.5.18 patches are gone. SGLang 0.5.21 imports `decode_jpeg` lazily and
registers the Qwen3-ASR configuration names idempotently itself.

Run this once in the environment where the pinned SGLang is installed:

```bash
python -m sglang_oot_patches.apply
```

The patch is idempotent. It does three things:

- excludes `spyre` and `spyre_paged` from Triton dispatch;
- makes the unused image-processor registration optional;
- adds the two Spyre backend names to Gemma-4's allowlist.

It fails loudly if the pinned SGLang source has drifted, instead of silently
producing a partially patched environment.

A few further SGLang gaps are handled by the backend at runtime instead of by
source patches, because each is a single setting:

- final-logit soft-capping takes SGLang's host path, not its Triton kernel;
- the overlap scheduler and the separate SWA KV pool are disabled on the Spyre
  platform.
