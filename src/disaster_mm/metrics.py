"""Evaluation metrics: classification F1, BLEU, templated count/fact
accuracy, pointing accuracy, retrieval Recall@K, and modality-ablation
evaluation helpers."""
from __future__ import annotations

import re

import numpy as np
import sacrebleu
import torch
from sklearn.metrics import f1_score

from .models.part2_seq2seq import PATCH_GRID

DAMAGE_LEVELS = ["none", "minor", "major", "destroyed"]

_COUNT_RE = re.compile(
    r"(\d+) destroyed, (\d+) major damage, (\d+) minor damage, (\d+) undamaged structures"
)
_BRIDGE_RE = re.compile(r"Bridge is (\w+)")
_LEVEL_RE = re.compile(r"Overall damage level: (\w+)")


def classification_f1(preds: np.ndarray, targets: np.ndarray, n_classes: int) -> dict:
    labels = list(range(n_classes))
    per_class = f1_score(targets, preds, labels=labels, average=None, zero_division=0)
    weighted = f1_score(targets, preds, labels=labels, average="weighted", zero_division=0)
    return {"per_class_f1": per_class.tolist(), "weighted_f1": float(weighted)}


def bleu(hyps: list[str], refs: list[str]) -> float:
    if not hyps:
        return 0.0
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def _parse_template_fields(text: str) -> dict | None:
    m = _COUNT_RE.search(text)
    b = _BRIDGE_RE.search(text)
    lvl = _LEVEL_RE.search(text)
    if not (m and b and lvl):
        return None
    return {
        "counts": tuple(int(x) for x in m.groups()),
        "bridge": b.group(1),
        "level": lvl.group(1),
    }


def count_fact_accuracy(generated: list[str], references: list[str]) -> dict:
    n = len(generated)
    counts_correct = 0
    bridge_correct = 0
    level_correct = 0
    parseable = 0
    for gen, ref in zip(generated, references):
        g = _parse_template_fields(gen)
        r = _parse_template_fields(ref)
        if r is None:
            continue
        if g is None:
            continue
        parseable += 1
        if g["counts"] == r["counts"]:
            counts_correct += 1
        if g["bridge"] == r["bridge"]:
            bridge_correct += 1
        if g["level"] == r["level"]:
            level_correct += 1
    denom = max(n, 1)
    return {
        "parseable_fraction": parseable / denom,
        "count_accuracy": counts_correct / denom,
        "bridge_accuracy": bridge_correct / denom,
        "level_accuracy": level_correct / denom,
    }


def _patch_building_damage_membership(building_masks: torch.Tensor) -> torch.Tensor:
    """building_masks: (S, H, W) bool -> (S, 256) fraction of each patch covered."""
    S, H, W = building_masks.shape
    ph, pw = H // PATCH_GRID, W // PATCH_GRID
    bm = building_masks.float()
    patches = bm.unfold(1, ph, ph).unfold(2, pw, pw).sum(dim=(-1, -2))
    return patches.reshape(S, PATCH_GRID * PATCH_GRID)


def pointing_accuracy(
    argmax_patches: list[int],
    word_damage_classes: list[int],
    building_masks: list[torch.Tensor],
    building_damage: list[torch.Tensor],
) -> float:
    """For each (predicted argmax patch, target damage class) pair, check whether
    that patch lies inside a ground-truth structure with the matching damage class."""
    if not argmax_patches:
        return 0.0
    hits = 0
    for patch_idx, dmg_class, masks, dmg in zip(argmax_patches, word_damage_classes, building_masks, building_damage):
        membership = _patch_building_damage_membership(masks)  # (S, 256)
        covers = membership[:, patch_idx] > 0
        matching = (dmg == dmg_class) & covers
        if matching.any():
            hits += 1
    return hits / len(argmax_patches)


def recall_at_k_from_embeddings(query: torch.Tensor, gallery: torch.Tensor, ground_truth: torch.Tensor, ks: list[int]) -> dict[int, float]:
    import torch.nn.functional as F

    sim = F.normalize(query, dim=-1) @ F.normalize(gallery, dim=-1).t()
    ranks = sim.argsort(dim=-1, descending=True)
    out = {}
    for k in ks:
        topk = ranks[:, :k]
        hit = (topk == ground_truth.unsqueeze(1)).any(dim=1)
        out[k] = hit.float().mean().item()
    return out
