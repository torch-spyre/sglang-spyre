"""Torch-native paged KV allocator for Spyre (no Triton).

SGLang's ``PagedTokenToKVPoolAllocator.alloc_extend`` / ``alloc_decode`` dispatch
the Triton kernels ``alloc_extend_kernel`` / ``alloc_decode_kernel``
(mem_cache/allocator/paged.py). Triton has no Spyre backend, so on Spyre they
crash with ``RuntimeError: 0 active drivers``.

SGLang 0.5.21 ships ``alloc_extend_naive``, a pure-torch reference of the
extend kernel with the same page-fill layout, so extend reuses it unchanged.
Decode is the one-token special case: the slot after ``last_loc`` when it stays
inside its page, otherwise the first slot of the next free page. Each request's
``req_to_token`` slots therefore stay page-aligned and contiguous (``slot =
page*page_size + offset``), the property SpyreAttentionBackend's page table
relies on.

Everything else (alloc, free, clear, free_pages accounting, need_sort) is
inherited unchanged.
"""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)


class SpyrePagedAllocator(PagedTokenToKVPoolAllocator):
    def alloc_extend(
        self,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
        num_new_pages=None,
    ):
        ps = self.page_size
        if num_new_pages is None:
            num_new_pages = int(
                (
                    (seq_lens_cpu + ps - 1) // ps - (prefix_lens_cpu + ps - 1) // ps
                ).sum()
            )
        if self.need_sort and num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_new_pages > len(self.free_pages):
            return None

        out = torch.empty((extend_num_tokens,), dtype=torch.int64, device=self.device)
        alloc_extend_naive(
            prefix_lens.to(torch.int64),
            seq_lens.to(torch.int64),
            last_loc.to(torch.int64),
            self.free_pages,
            out,
            ps,
            self.device,
        )
        self.free_pages = self.free_pages[num_new_pages:]
        return out

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        ps = self.page_size
        bs = len(seq_lens)
        # A token at position seq_len - 1 opens a page iff it is a page's first.
        needs_page = ((seq_lens_cpu.to(torch.int64) - 1) % ps) == 0
        num_new_pages = int(needs_page.sum())
        if self.need_sort and num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_new_pages > len(self.free_pages):
            return None

        out = last_loc.to(device=self.device, dtype=torch.int64) + 1
        if num_new_pages:
            needs = needs_page.to(self.device)
            rank = torch.cumsum(needs.to(torch.int64), 0) - 1
            fresh = self.free_pages[rank.clamp(min=0)] * ps
            out = torch.where(needs, fresh, out)
        out = out[:bs]
        self.free_pages = self.free_pages[num_new_pages:]
        return out
