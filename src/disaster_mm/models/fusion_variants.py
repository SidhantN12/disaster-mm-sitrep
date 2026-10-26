"""Early- and late-fusion baselines for the fusion comparison study.

These sit alongside ``part2_seq2seq.Part2Seq2Seq`` (the hybrid / intermediate
cross-attention model) and expose the *same* ``forward`` / ``infer``
interface, so a single training and evaluation loop can drive all three.

The comparison holds everything except the fusion mechanism fixed:

  * identical unimodal encoders (Part 1's ViT + text encoder), loaded from
    the same stage-1 checkpoint and frozen,
  * identical decoder block (``Part2Decoder``), ``d_model``, layer counts,
    head counts and FFN width,
  * identical data, seeds, optimiser, batch size and epoch budget.

What differs is *where* the two modalities meet:

  early  -- the pooled image and text vectors are concatenated and mixed by
            an MLP before any joint modelling; the decoder memory is one
            flat sequence of patch + report tokens processed by ordinary
            self-attention, so the modalities are blended from layer 1 and
            there is no mechanism that can weight one report against
            another (mean pooling stands in for MIL).
  late   -- the two modalities never meet inside the network. Each runs its
            own encoder stack, its own decoder and its own heads; only the
            output distributions are combined, through a learned gate.
  hybrid -- ``Part2Seq2Seq``: bidirectional co-attention in the encoder,
            cross-attention in the decoder, attention-MIL over reports.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .part2_seq2seq import NUM_PATCHES, PATCH_GRID, Part2Decoder, _safe_mha

__all__ = ["EarlyFusionSeq2Seq", "LateFusionSeq2Seq", "build_fusion_model"]


class SelfAttnLayer(nn.Module):
    """Plain pre-norm-free transformer encoder layer over a single sequence."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.ReLU(inplace=True), nn.Linear(ffn_dim, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        sa, _ = _safe_mha(self.attn, x, x, x, key_padding_mask=pad_mask)
        x = self.ln1(x + self.dropout(sa))
        x = self.ln2(x + self.dropout(self.ffn(x)))
        return x


def _patch_pos(row_embed: torch.Tensor, col_embed: torch.Tensor, d_model: int) -> torch.Tensor:
    pos = row_embed.unsqueeze(1) + col_embed.unsqueeze(0)
    return pos.reshape(NUM_PATCHES, d_model)


def _masked_mean(h_k: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """Mean over non-padded reports; all-padded rows (k = 0) return zeros."""
    if h_k.shape[1] == 0:
        return h_k.new_zeros((h_k.shape[0], h_k.shape[-1]))
    keep = (~pad_mask).float().unsqueeze(-1)
    denom = keep.sum(dim=1).clamp(min=1.0)
    return (h_k * keep).sum(dim=1) / denom


# --------------------------------------------------------------------------
# Early fusion
# --------------------------------------------------------------------------
class EarlyFusionSeq2Seq(nn.Module):
    """Feature-level fusion.

    ``h_I`` and ``h_T`` are concatenated and passed through an MLP to a single
    joint code before anything else happens. Patch tokens and report tokens
    are projected into one shared space, concatenated into a single sequence
    with one fused context token, and mixed by ordinary self-attention -- the
    modalities are indistinguishable to the stack from the first layer.
    """

    fusion_name = "early"

    def __init__(
        self,
        dim_i: int,
        dim_t: int,
        joint_dim: int,
        vocab_size: int,
        d_model: int = 128,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 256,
        max_sitrep_len: int = 48,
        dropout: float = 0.1,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> None:
        super().__init__()
        self.joint_dim = joint_dim
        self.fuse = nn.Sequential(
            nn.Linear(dim_i + dim_t, joint_dim), nn.ReLU(inplace=True), nn.Linear(joint_dim, joint_dim)
        )
        self.patch_proj = nn.Linear(dim_i, d_model)
        self.report_proj = nn.Linear(dim_t, d_model)
        self.geo_time_mlp = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model))
        self.ctx_proj = nn.Linear(joint_dim, d_model)
        self.row_embed = nn.Parameter(torch.randn(PATCH_GRID, d_model) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(PATCH_GRID, d_model) * 0.02)
        self.d_model = d_model

        self.layers = nn.ModuleList([SelfAttnLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(n_enc_layers)])
        self.decoder = Part2Decoder(
            vocab_size, joint_dim, d_model, n_dec_layers, n_heads, ffn_dim, max_sitrep_len, dropout, pad_id, bos_id, eos_id
        )
        hidden = max(joint_dim, 16)
        self.damage_head = nn.Sequential(nn.Linear(joint_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 4))
        self.bag_head = nn.Sequential(nn.Linear(joint_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))

    def _encode(self, part1_out, report_geo, report_time, report_pad_mask):
        h_i, h_t = part1_out["h_i"], part1_out["h_t"]
        z = self.fuse(torch.cat([h_i, h_t], dim=-1))

        B = h_i.shape[0]
        device = h_i.device
        patches = self.patch_proj(part1_out["patch_tokens"]) + _patch_pos(
            self.row_embed, self.col_embed, self.d_model
        ).to(device).unsqueeze(0)
        ctx = self.ctx_proj(z).unsqueeze(1)

        h_k = part1_out["text_out"]["h_k"]
        k = h_k.shape[1]
        if k > 0:
            geo_time = torch.cat([report_geo, report_time], dim=-1)
            reports = self.report_proj(h_k) + self.geo_time_mlp(geo_time)
        else:
            reports = h_k.new_zeros((B, 0, self.d_model))

        seq = torch.cat([patches, ctx, reports], dim=1)
        pad = torch.cat(
            [torch.zeros(B, NUM_PATCHES + 1, dtype=torch.bool, device=device), report_pad_mask], dim=1
        )
        for layer in self.layers:
            seq = layer(seq, pad)

        memory = {
            "memory": seq,
            "mem_pad_mask": pad,
            "n_patches": NUM_PATCHES,
            "n_context": 1,
            "n_reports": k,
            "attn_maps": [],
        }
        return z, memory

    def forward(self, part1_out, report_geo, report_time, report_pad_mask, target_tokens=None) -> dict:
        z, memory = self._encode(part1_out, report_geo, report_time, report_pad_mask)
        result = {
            "bag_logit": self.bag_head(z).squeeze(-1),
            "damage_logits": self.damage_head(z),
            "mil_alpha": z.new_zeros((z.shape[0], report_pad_mask.shape[1])),
            "encoder_out": memory,
        }
        if target_tokens is not None:
            result["decoder_out"] = self.decoder.forward_teacher_forcing(z, target_tokens, memory)
        return result

    @torch.no_grad()
    def infer(self, part1_out, report_geo, report_time, report_pad_mask, beam_size: int = 1, max_len=None) -> dict:
        z, memory = self._encode(part1_out, report_geo, report_time, report_pad_mask)
        if beam_size > 1:
            gen = {"tokens": self.decoder.generate_beam(z, memory, beam_size=beam_size, max_len=max_len)}
        else:
            gen = self.decoder.generate_greedy(z, memory, max_len=max_len)
        return {
            "bag_logit": self.bag_head(z).squeeze(-1),
            "damage_logits": self.damage_head(z),
            "mil_alpha": z.new_zeros((z.shape[0], report_pad_mask.shape[1])),
            "encoder_out": memory,
            **gen,
        }


# --------------------------------------------------------------------------
# Late fusion
# --------------------------------------------------------------------------
class _UnimodalBranch(nn.Module):
    """One self-contained branch: project tokens, self-attend, decode, predict."""

    def __init__(
        self, in_dim, code_dim, vocab_size, d_model, n_enc_layers, n_dec_layers, n_heads, ffn_dim,
        max_sitrep_len, dropout, pad_id, bos_id, eos_id, is_image: bool,
    ) -> None:
        super().__init__()
        self.is_image = is_image
        self.d_model = d_model
        self.code = nn.Sequential(nn.Linear(in_dim, code_dim), nn.ReLU(inplace=True), nn.Linear(code_dim, code_dim))
        self.tok_proj = nn.Linear(in_dim, d_model)
        self.ctx_proj = nn.Linear(code_dim, d_model)
        if is_image:
            self.row_embed = nn.Parameter(torch.randn(PATCH_GRID, d_model) * 0.02)
            self.col_embed = nn.Parameter(torch.randn(PATCH_GRID, d_model) * 0.02)
        else:
            self.geo_time_mlp = nn.Sequential(
                nn.Linear(3, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model)
            )
        self.null_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.layers = nn.ModuleList([SelfAttnLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(n_enc_layers)])
        self.decoder = Part2Decoder(
            vocab_size, code_dim, d_model, n_dec_layers, n_heads, ffn_dim, max_sitrep_len, dropout, pad_id, bos_id, eos_id
        )
        hidden = max(code_dim, 16)
        self.damage_head = nn.Sequential(nn.Linear(code_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 4))
        self.bag_head = nn.Sequential(nn.Linear(code_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))

    def encode(self, pooled, tokens, pad_mask, geo_time=None):
        B = pooled.shape[0]
        device = pooled.device
        z = self.code(pooled)
        ctx = self.ctx_proj(z).unsqueeze(1)

        if tokens.shape[1] > 0:
            t = self.tok_proj(tokens)
            if self.is_image:
                t = t + _patch_pos(self.row_embed, self.col_embed, self.d_model).to(device).unsqueeze(0)
            elif geo_time is not None:
                t = t + self.geo_time_mlp(geo_time)
            seq = torch.cat([t, ctx], dim=1)
            pad = torch.cat([pad_mask, torch.zeros(B, 1, dtype=torch.bool, device=device)], dim=1)
        else:
            # k = 0: the branch still needs a non-empty memory, so it gets a
            # learned null token next to its context token.
            null = self.null_token.view(1, 1, -1).expand(B, 1, -1)
            seq = torch.cat([null, ctx], dim=1)
            pad = torch.zeros(B, 2, dtype=torch.bool, device=device)

        for layer in self.layers:
            seq = layer(seq, pad)

        n_tok = seq.shape[1] - 1
        memory = {
            "memory": seq,
            "mem_pad_mask": pad,
            "n_patches": n_tok if self.is_image else 0,
            "n_context": 1,
            "n_reports": 0 if self.is_image else n_tok,
            "attn_maps": [],
        }
        return z, memory


class LateFusionSeq2Seq(nn.Module):
    """Decision-level fusion.

    The image and text branches share no parameters and never exchange
    activations. Each produces its own damage logits, bag logit and SITREP
    distribution; a learned gate (softmax over two scalars, per output head)
    mixes the *outputs*. SITREP tokens are decoded by mixing the two
    branches' per-step log-probabilities, which is decision-level fusion for
    generation rather than a single decoder over a joint memory.
    """

    fusion_name = "late"

    def __init__(
        self,
        dim_i: int,
        dim_t: int,
        code_dim: int,
        vocab_size: int,
        d_model: int = 128,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 256,
        max_sitrep_len: int = 48,
        dropout: float = 0.1,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> None:
        super().__init__()
        common = dict(
            vocab_size=vocab_size, d_model=d_model, n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers,
            n_heads=n_heads, ffn_dim=ffn_dim, max_sitrep_len=max_sitrep_len, dropout=dropout,
            pad_id=pad_id, bos_id=bos_id, eos_id=eos_id,
        )
        self.img_branch = _UnimodalBranch(dim_i, code_dim, is_image=True, **common)
        self.txt_branch = _UnimodalBranch(dim_t, code_dim, is_image=False, **common)
        # One gate per output head, each a 2-logit softmax over (image, text).
        self.gate_damage = nn.Parameter(torch.zeros(2))
        self.gate_bag = nn.Parameter(torch.zeros(2))
        self.gate_sitrep = nn.Parameter(torch.zeros(2))
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        self.max_len = max_sitrep_len

    def _branches(self, part1_out, report_geo, report_time, report_pad_mask):
        h_k = part1_out["text_out"]["h_k"]
        geo_time = (
            torch.cat([report_geo, report_time], dim=-1) if h_k.shape[1] > 0 else None
        )
        z_i, mem_i = self.img_branch.encode(
            part1_out["h_i"], part1_out["patch_tokens"],
            torch.zeros(part1_out["h_i"].shape[0], NUM_PATCHES, dtype=torch.bool, device=part1_out["h_i"].device),
        )
        z_t, mem_t = self.txt_branch.encode(part1_out["h_t"], h_k, report_pad_mask, geo_time)
        return z_i, mem_i, z_t, mem_t

    def _mix(self, gate, a, b):
        w = torch.softmax(gate, dim=0)
        return w[0] * a + w[1] * b

    def forward(self, part1_out, report_geo, report_time, report_pad_mask, target_tokens=None) -> dict:
        z_i, mem_i, z_t, mem_t = self._branches(part1_out, report_geo, report_time, report_pad_mask)

        damage = self._mix(self.gate_damage, self.img_branch.damage_head(z_i), self.txt_branch.damage_head(z_t))
        bag = self._mix(
            self.gate_bag, self.img_branch.bag_head(z_i).squeeze(-1), self.txt_branch.bag_head(z_t).squeeze(-1)
        )

        result = {
            "bag_logit": bag,
            "damage_logits": damage,
            "mil_alpha": z_i.new_zeros((z_i.shape[0], report_pad_mask.shape[1])),
            "encoder_out": mem_i,
            "gate_damage": torch.softmax(self.gate_damage, dim=0).detach(),
            "gate_bag": torch.softmax(self.gate_bag, dim=0).detach(),
            "gate_sitrep": torch.softmax(self.gate_sitrep, dim=0).detach(),
        }
        if target_tokens is not None:
            di = self.img_branch.decoder.forward_teacher_forcing(z_i, target_tokens, mem_i)
            dt = self.txt_branch.decoder.forward_teacher_forcing(z_t, target_tokens, mem_t)
            w = torch.softmax(self.gate_sitrep, dim=0)
            mixed = torch.log(
                w[0] * F.softmax(di["logits"], dim=-1) + w[1] * F.softmax(dt["logits"], dim=-1) + 1e-9
            )
            result["decoder_out"] = {
                "logits": mixed,
                "cross_attn": di["cross_attn"],
                "image_patch_mass": di.get("image_patch_mass"),
                "context_mass": di.get("context_mass"),
                "report_mass": dt.get("report_mass"),
            }
        return result

    @torch.no_grad()
    def infer(self, part1_out, report_geo, report_time, report_pad_mask, beam_size: int = 1, max_len=None) -> dict:
        z_i, mem_i, z_t, mem_t = self._branches(part1_out, report_geo, report_time, report_pad_mask)
        damage = self._mix(self.gate_damage, self.img_branch.damage_head(z_i), self.txt_branch.damage_head(z_t))
        bag = self._mix(
            self.gate_bag, self.img_branch.bag_head(z_i).squeeze(-1), self.txt_branch.bag_head(z_t).squeeze(-1)
        )

        max_len = max_len or self.max_len
        B = z_i.shape[0]
        device = z_i.device
        w = torch.softmax(self.gate_sitrep, dim=0)
        generated = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        def step(branch, z, mem, gen):
            ctx = branch.decoder.ctx_proj(z).unsqueeze(1)
            x = torch.cat([ctx, branch.decoder.token_emb(gen)], dim=1)
            x = x + branch.decoder.pos_emb[:, : x.shape[1], :]
            x, cw = branch.decoder._run_layers(x, mem["memory"], mem["mem_pad_mask"])
            return branch.decoder.out_proj(x[:, -1, :]), cw[-1]

        last_i = last_t = None
        for _ in range(max_len - 1):
            li, last_i = step(self.img_branch, z_i, mem_i, generated)
            lt, last_t = step(self.txt_branch, z_t, mem_t, generated)
            probs = w[0] * F.softmax(li, dim=-1) + w[1] * F.softmax(lt, dim=-1)
            nxt = probs.argmax(dim=-1)
            nxt = torch.where(finished, torch.full_like(nxt, self.pad_id), nxt)
            generated = torch.cat([generated, nxt.unsqueeze(1)], dim=1)
            finished = finished | (nxt == self.eos_id)
            if finished.all():
                break

        img_mass = last_i[..., : mem_i["n_patches"]].sum(dim=-1) if last_i is not None else None
        rep_mass = last_t[..., : mem_t["n_reports"]].sum(dim=-1) if last_t is not None else None
        return {
            "bag_logit": bag,
            "damage_logits": damage,
            "mil_alpha": z_i.new_zeros((B, report_pad_mask.shape[1])),
            "encoder_out": mem_i,
            "tokens": generated,
            "cross_attn": last_i,
            "image_patch_mass": img_mass,
            "report_mass": rep_mass,
            "context_mass": None,
            "gate_damage": torch.softmax(self.gate_damage, dim=0),
            "gate_bag": torch.softmax(self.gate_bag, dim=0),
            "gate_sitrep": w,
        }


def build_fusion_model(fusion: str, cfg: dict, part1, tokenizer):
    """Construct the Part-2-equivalent model for a named fusion strategy."""
    m = cfg["model"]
    common = dict(
        d_model=m["d_model"], n_enc_layers=m["n_enc_layers"], n_dec_layers=m["n_dec_layers"],
        n_heads=m["n_heads"], ffn_dim=m["ffn_dim"], max_sitrep_len=cfg["data"]["max_sitrep_tokens"],
        pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
    )
    if fusion == "early":
        return EarlyFusionSeq2Seq(
            part1.dim_i, part1.dim_t, joint_dim=m["shared_dim"], vocab_size=tokenizer.vocab_size, **common
        )
    if fusion == "late":
        return LateFusionSeq2Seq(
            part1.dim_i, part1.dim_t, code_dim=m["shared_dim"], vocab_size=tokenizer.vocab_size, **common
        )
    raise ValueError(f"build_fusion_model handles 'early' and 'late'; got {fusion!r}")
