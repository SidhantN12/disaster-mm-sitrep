"""Unverified-sentence flagging and text<->tile retrieval.

A generated SITREP sentence is flagged "unverified" when it is *both*
(a) not well aligned with the tile's own visual shared embedding, and
(b) not strongly supported by any single report (per the MIL attention
weights) -- i.e. neither the image nor the text evidence backs it up.
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F

from .models.part1_mmae import Part1MMAE


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


@torch.no_grad()
def embed_sentences(model: Part1MMAE, tokenizer, sentences: list[str], max_len: int, device: torch.device) -> torch.Tensor:
    """Embed each sentence as a lone "report" through E_T -> z_s^T = g(0, h_T)."""
    if not sentences:
        return torch.zeros((0, model.shared_dim), device=device)
    ids = torch.tensor([tokenizer.encode(s, max_len) for s in sentences], dtype=torch.long, device=device)
    ids = ids.unsqueeze(1)  # (n, 1, max_len) -- one report per "sample"
    pad_mask = torch.zeros((ids.shape[0], 1), dtype=torch.bool, device=device)
    text_out = model.text_encoder(ids, pad_mask)
    h_t = text_out["pooled"]
    zero_i = torch.zeros((h_t.shape[0], model.dim_i), device=device)
    return model.shared_encoder(zero_i, h_t)


def flag_unverified(
    sentence_embeds: torch.Tensor,  # (n, shared_dim)
    tile_z_s_i: torch.Tensor,  # (shared_dim,)
    mil_alpha: torch.Tensor,  # (k,)
    sim_threshold: float = 0.2,
    alpha_threshold: float = 0.3,
) -> list[bool]:
    if sentence_embeds.shape[0] == 0:
        return []
    sim = F.cosine_similarity(sentence_embeds, tile_z_s_i.unsqueeze(0).expand_as(sentence_embeds), dim=-1)
    max_alpha = mil_alpha.max().item() if mil_alpha.numel() > 0 else 0.0
    text_supported = max_alpha > alpha_threshold
    flags = [(s.item() < sim_threshold) and (not text_supported) for s in sim]
    return flags


def recall_at_k(similarity: torch.Tensor, ground_truth: torch.Tensor, ks: list[int]) -> dict[int, float]:
    """similarity: (n_queries, n_targets) higher = more similar.
    ground_truth: (n_queries,) index of the correct target for each query."""
    n = similarity.shape[0]
    ranks = similarity.argsort(dim=-1, descending=True)
    out = {}
    for k in ks:
        topk = ranks[:, :k]
        hit = (topk == ground_truth.unsqueeze(1)).any(dim=1)
        out[k] = hit.float().mean().item() if n > 0 else 0.0
    return out


def text_to_tile_retrieval(text_embeds: torch.Tensor, tile_embeds: torch.Tensor, ground_truth: torch.Tensor, ks: list[int] | None = None) -> dict[int, float]:
    ks = ks or [1, 5, 10]
    sim = F.normalize(text_embeds, dim=-1) @ F.normalize(tile_embeds, dim=-1).t()
    return recall_at_k(sim, ground_truth, ks)
