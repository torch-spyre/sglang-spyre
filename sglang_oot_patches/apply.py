#!/usr/bin/env python3
"""Apply the SGLang 0.5.21 compatibility patches needed by the Spyre backend.

SGLang 0.5.21 already handles OOT device admission, CPU-safe host-cache
imports, native RoPE fallback, a lazy ``decode_jpeg`` import and an idempotent
``qwen3_asr`` config registration, so the 0.5.18 patches for the last two are
gone. Its global ``support_triton`` helper still treats unknown attention
backends as Triton-capable, so exclude the two Spyre backend names, and
image-processor registration still requires torchvision. Gemma-4's
server-argument hook accepts only in-tree attention backends, so the Spyre
backend names are added to its allowlist. Transformers construction gains an
overridable factory for adapter-owned attention. Idempotent: an already-patched
file is skipped.

Run inside the venv where sglang is installed:
    python sglang_oot_patches/apply.py
"""

from __future__ import annotations

import json
import os
from importlib.metadata import distribution
from urllib.parse import unquote, urlparse


def _sglang_srt_dir() -> str:
    # Locate the distribution without importing SGLang: its top-level import
    # requires optional CUDA/Triton and torchvision modules before this
    # compatibility patch has had a chance to run.
    dist = distribution("sglang")
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        source_url = json.loads(direct_url).get("url", "")
        parsed = urlparse(source_url)
        if parsed.scheme == "file":
            editable_root = unquote(parsed.path)
            candidate = os.path.join(editable_root, "sglang", "srt")
            if os.path.isdir(candidate):
                return candidate
    return os.fspath(dist.locate_file("sglang/srt"))


# Extract the upstream constructor without changing its default semantics.
_TRANSFORMERS_INIT = """        # Resolve model class for _supports_attention_backend check
        model_cls = _resolve_attention_backend_model_cls(config)

        supports_backend = (
            getattr(model_cls, "_supports_attention_backend", True)
            if model_cls
            else True
        )

        # Initialize on meta device to avoid premature GPU allocation
        self.text_config._attn_implementation = "sglang"
        if supports_backend:
            with _init_on_device_without_buffers(torch.device("meta")):
                self.model: PreTrainedModel = AutoModel.from_config(
                    self.config,
                    torch_dtype=torch.get_default_dtype(),
                    trust_remote_code=True,
                )
        else:
            raise ValueError(
                f"Model {model_cls} does not support custom attention backends "
                "(_supports_attention_backend=False). The Transformers backend "
                "requires custom attention support."
            )"""
_TRANSFORMERS_FACTORY_ANCHOR = '''    @property
    def _can_torch_compile(self) -> bool:
        """Whether this model instance is safe to wrap with torch.compile."""'''
_TRANSFORMERS_FACTORY = '''    def _create_transformers_model(self):
        # Resolve model class for _supports_attention_backend check
        model_cls = _resolve_attention_backend_model_cls(self.config)

        supports_backend = (
            getattr(model_cls, "_supports_attention_backend", True)
            if model_cls
            else True
        )

        # Initialize on meta device to avoid premature GPU allocation
        self.text_config._attn_implementation = "sglang"
        if supports_backend:
            with _init_on_device_without_buffers(torch.device("meta")):
                return AutoModel.from_config(
                    self.config,
                    torch_dtype=torch.get_default_dtype(),
                    trust_remote_code=True,
                )
        else:
            raise ValueError(
                f"Model {model_cls} does not support custom attention backends "
                "(_supports_attention_backend=False). The Transformers backend "
                "requires custom attention support."
            )

    @property
    def _can_torch_compile(self) -> bool:
        """Whether this model instance is safe to wrap with torch.compile."""'''

PATCHES = [
    # (relative path, old, new)
    (
        "utils/common.py",
        'return backend not in ["torch_native", "intel_amx"]',
        'return backend not in ["torch_native", "intel_amx", "spyre", "spyre_paged"]',
    ),
    (
        "configs/utils.py",
        """    AutoImageProcessor.register(
        config, slow_image_processor_class=image_processor, exist_ok=True
    )""",
        """    try:
        AutoImageProcessor.register(
            config, slow_image_processor_class=image_processor, exist_ok=True
        )
    except ImportError:
        # Text-only out-of-tree runtimes need not install torchvision.
        pass""",
    ),
    (
        "arg_groups/model_hook.py",
        """            "intel_amx",
            "aiter",
        )
        assert (
            prefill_backend in accepted_backends and decode_backend in accepted_backends
        ), (
            "Gemma4 only supports""",
        """            "intel_amx",
            "aiter",
            "spyre",
            "spyre_paged",
        )
        assert (
            prefill_backend in accepted_backends and decode_backend in accepted_backends
        ), (
            "Gemma4 only supports""",
    ),
    # HF-adapters bypasses HF attention dispatch, including on GPT-Neo.
    (
        "models/transformers.py",
        _TRANSFORMERS_INIT,
        "        self.model = self._create_transformers_model()",
    ),
    ("models/transformers.py", _TRANSFORMERS_FACTORY_ANCHOR, _TRANSFORMERS_FACTORY),
]


def main() -> int:
    root = _sglang_srt_dir()
    print(f"sglang.srt at: {root}")
    for rel, old, new in PATCHES:
        path = os.path.join(root, rel)
        with open(path) as f:
            src = f.read()
        if new in src:
            print(f"  [skip] {rel} already patched")
            continue
        if old not in src:
            raise RuntimeError(
                f"{rel}: patch anchor not found; expected sglang==0.5.21"
            )
        with open(path, "w") as f:
            f.write(src.replace(old, new, 1))
        print(f"  [ok]   {rel} patched")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
