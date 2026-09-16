"""Part 1: multimodal autoencoder (shared + private latent decomposition).

Design choices (see README for the full rationale):
  * D_I reconstructs the full 6-channel (pre+post) input, not just the post
    image -- the encoder sees both frames, so full reconstruction gives a
    stronger self-supervised signal and implicitly encourages a
    change-sensitive shared code.
  * Modality dropout (p=0.3) only zeroes an input to the *shared* encoder
    g; the private encoders always see their own real modality, since
    "private" detail is by definition not recoverable from the other
    modality.
  * Cross-reconstruction reconstructs each modality from the *other*
    modality's shared code alone (private code zeroed, since it cannot be
    inferred cross-modally): D_I([z_s^T, 0]) and D_T([z_s^I, 0]).
  * The orthogonality penalty follows Bousmalis et al. 2016 (Domain
    Separation Networks): soft subspace orthogonality between the batch of
    shared activations and the batch of private activations for the same
    modality, ||S^T P||_F^2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(out_dim, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedEncoder(nn.Module):
    """g(h_I, h_T) -> z_s. Either input may be an all-zero tensor."""

    def __init__(self, dim_i: int, dim_t: int, shared_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.mlp = MLPEncoder(dim_i + dim_t, shared_dim, hidden_dim)

    def forward(self, h_i: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([h_i, h_t], dim=-1))


class ImageDecoder(nn.Module):
    def __init__(self, in_dim: int, base_channels: int = 64, out_chans: int = 6) -> None:
        super().__init__()
        self.base_channels = base_channels
        self.fc = nn.Linear(in_dim, base_channels * 16 * 16)
        c = base_channels
        self.up = nn.Sequential(
            nn.ConvTranspose2d(c, c, 4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, c // 2, 4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(c // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c // 2, c // 2, 4, stride=2, padding=1),  # 64 -> 128
            nn.BatchNorm2d(c // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c // 2, c // 4, 4, stride=2, padding=1),  # 128 -> 256
            nn.BatchNorm2d(c // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 4, out_chans, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.shape[0], self.base_channels, 16, 16)
        return torch.sigmoid(self.up(x))


class TextDecoder(nn.Module):
    """Bag-of-words head: predicts which vocab words occur in the reports."""

    def __init__(self, in_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, vocab_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


class Part1MMAE(nn.Module):
    def __init__(
        self,
        image_variant: str = "tiny",
        text_variant: str = "tiny",
        vocab_size: int = 300,
        sep_id: int = 2,
        pad_id: int = 0,
        img_embed_dim: int = 64,
        img_depth: int = 2,
        img_heads: int = 2,
        txt_embed_dim: int = 64,
        txt_depth: int = 2,
        txt_heads: int = 2,
        shared_dim: int = 32,
        private_dim: int = 16,
        decoder_base_channels: int = 32,
        modality_dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(image_variant, in_chans=6, embed_dim=img_embed_dim, depth=img_depth, num_heads=img_heads)
        self.text_encoder = TextEncoder(
            text_variant, vocab_size, sep_id, pad_id, embed_dim=txt_embed_dim, depth=txt_depth, num_heads=txt_heads
        )
        dim_i, dim_t = self.image_encoder.embed_dim, self.text_encoder.embed_dim
        self.dim_i, self.dim_t = dim_i, dim_t
        self.shared_dim, self.private_dim = shared_dim, private_dim

        self.shared_encoder = SharedEncoder(dim_i, dim_t, shared_dim)
        self.private_i = MLPEncoder(dim_i, private_dim)
        self.private_t = MLPEncoder(dim_t, private_dim)

        self.decoder_i = ImageDecoder(shared_dim + private_dim, decoder_base_channels, out_chans=6)
        self.decoder_t = TextDecoder(shared_dim + private_dim, vocab_size)

        self.modality_dropout_p = modality_dropout_p
        self.vocab_size = vocab_size

    def encode(self, image: torch.Tensor, report_tokens: torch.Tensor, report_pad_mask: torch.Tensor) -> dict:
        patch_tokens, h_i = self.image_encoder(image)
        text_out = self.text_encoder(report_tokens, report_pad_mask)
        h_t = text_out["pooled"]

        z_i_p = self.private_i(h_i)
        z_t_p = self.private_t(h_t)

        zero_t = torch.zeros_like(h_t)
        zero_i = torch.zeros_like(h_i)
        z_s_i = self.shared_encoder(h_i, zero_t)  # g(h_I, 0)
        z_s_t = self.shared_encoder(zero_i, h_t)  # g(0, h_T)

        if self.training and self.modality_dropout_p > 0:
            B = image.shape[0]
            device = image.device
            drop = torch.rand(B, device=device) < self.modality_dropout_p
            drop_image = drop & (torch.rand(B, device=device) < 0.5)
            drop_text = drop & (~drop_image)
            h_i_eff = torch.where(drop_image.unsqueeze(-1), zero_i, h_i)
            h_t_eff = torch.where(drop_text.unsqueeze(-1), zero_t, h_t)
        else:
            h_i_eff, h_t_eff = h_i, h_t

        z_s = self.shared_encoder(h_i_eff, h_t_eff)

        return {
            "patch_tokens": patch_tokens,
            "h_i": h_i,
            "h_t": h_t,
            "z_i_p": z_i_p,
            "z_t_p": z_t_p,
            "z_s": z_s,
            "z_s_i": z_s_i,
            "z_s_t": z_s_t,
            "text_out": text_out,
        }

    def forward(self, image: torch.Tensor, report_tokens: torch.Tensor, report_pad_mask: torch.Tensor) -> dict:
        enc = self.encode(image, report_tokens, report_pad_mask)
        recon_i = self.decoder_i(torch.cat([enc["z_s"], enc["z_i_p"]], dim=-1))
        recon_t_logits = self.decoder_t(torch.cat([enc["z_s"], enc["z_t_p"]], dim=-1))

        zero_p_i = torch.zeros_like(enc["z_i_p"])
        zero_p_t = torch.zeros_like(enc["z_t_p"])
        cross_recon_i = self.decoder_i(torch.cat([enc["z_s_t"], zero_p_i], dim=-1))
        cross_recon_t_logits = self.decoder_t(torch.cat([enc["z_s_i"], zero_p_t], dim=-1))

        enc.update(
            recon_i=recon_i,
            recon_t_logits=recon_t_logits,
            cross_recon_i=cross_recon_i,
            cross_recon_t_logits=cross_recon_t_logits,
        )
        return enc


def bow_target(report_tokens: torch.Tensor, report_pad_mask: torch.Tensor, vocab_size: int, pad_id: int) -> torch.Tensor:
    """Multi-hot bag-of-words target over the reports of each sample."""
    B, k, Lr = report_tokens.shape
    target = torch.zeros(B, vocab_size, device=report_tokens.device)
    if k == 0:
        return target
    valid = ~report_pad_mask.unsqueeze(-1)  # (B, k, 1)
    ids = report_tokens.clamp(min=0, max=vocab_size - 1)
    mask = valid.expand(-1, -1, Lr) & (report_tokens != pad_id)
    flat_ids = ids.view(B, -1)
    flat_mask = mask.view(B, -1)
    target.scatter_reduce_(1, flat_ids, flat_mask.float(), reduce="amax", include_self=True)
    target[:, pad_id] = 0.0
    return target


def orthogonality_loss(s: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """||S^T P||_F^2 / B, the Domain-Separation-Networks soft subspace penalty."""
    B = s.shape[0]
    prod = s.transpose(0, 1) @ p  # (shared_dim, private_dim)
    return (prod**2).sum() / max(B, 1)


def info_nce_loss(z_i: torch.Tensor, z_t: torch.Tensor, paired_mask: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE between paired image/text shared codes, cosine similarity."""
    idx = paired_mask.nonzero(as_tuple=True)[0]
    if idx.numel() < 2:
        return z_i.new_zeros(())
    zi = F.normalize(z_i[idx], dim=-1)
    zt = F.normalize(z_t[idx], dim=-1)
    logits = zi @ zt.t() / tau
    labels = torch.arange(idx.numel(), device=z_i.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)


def part1_loss(
    out: dict,
    image: torch.Tensor,
    report_tokens: torch.Tensor,
    report_pad_mask: torch.Tensor,
    num_reports: torch.Tensor,
    vocab_size: int,
    pad_id: int,
    lambda_r: float = 1.0,
    lambda_c: float = 0.5,
    lambda_orth: float = 0.1,
    lambda_n: float = 0.5,
    tau: float = 0.07,
) -> dict:
    bow = bow_target(report_tokens, report_pad_mask, vocab_size, pad_id)

    mse = F.mse_loss(out["recon_i"], image)
    bce = F.binary_cross_entropy_with_logits(out["recon_t_logits"], bow)
    recon_loss = mse + bce

    cross_mse = F.mse_loss(out["cross_recon_i"], image)
    cross_bce = F.binary_cross_entropy_with_logits(out["cross_recon_t_logits"], bow)
    cross_loss = cross_mse + cross_bce

    orth = orthogonality_loss(out["z_s_i"], out["z_i_p"]) + orthogonality_loss(out["z_s_t"], out["z_t_p"])

    paired_mask = num_reports > 0
    nce = info_nce_loss(out["z_s_i"], out["z_s_t"], paired_mask, tau=tau)

    total = lambda_r * recon_loss + lambda_c * cross_loss + lambda_orth * orth + lambda_n * nce
    return {
        "loss": total,
        "recon_mse": mse.detach(),
        "recon_bce": bce.detach(),
        "cross_loss": cross_loss.detach(),
        "orth_loss": orth.detach(),
        "info_nce": nce.detach(),
    }
