"""Visualization helpers: attention heatmap overlays and image-vs-report
attention bar charts. Pure matplotlib, no display needed (Agg backend)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .models.part2_seq2seq import PATCH_GRID


def _to_numpy_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    return np.clip(image, 0, 1)


def save_attention_heatmap(
    image: torch.Tensor | np.ndarray,
    patch_attn: torch.Tensor | np.ndarray,
    out_path: str,
    word: str = "",
    alpha: float = 0.55,
) -> str:
    """image: (3, H, W) or (H, W, 3) in [0, 1]. patch_attn: (256,) over the 16x16 patch grid."""
    img = _to_numpy_image(image)
    if isinstance(patch_attn, torch.Tensor):
        patch_attn = patch_attn.detach().cpu().numpy()
    patch_attn = patch_attn.reshape(PATCH_GRID, PATCH_GRID)
    heat = np.kron(patch_attn, np.ones((img.shape[0] // PATCH_GRID, img.shape[1] // PATCH_GRID)))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.imshow(heat, cmap="jet", alpha=alpha)
    ax.set_title(f'attention: "{word}"' if word else "attention")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_modality_bar_chart(
    words: list[str], image_mass: list[float], report_mass: list[float], out_path: str
) -> str:
    n = len(words)
    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(4, n * 0.8), 3.5))
    ax.bar(x - width / 2, image_mass, width, label="image")
    ax.bar(x + width / 2, report_mass, width, label="report")
    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=45, ha="right")
    ax.set_ylabel("attention mass")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
