"""Part 2: cross-attention encoder-decoder producing the SITREP, damage
class, bag (report-relevance) label, and attention-based evidence maps.

Design choices (see README):
  * "Report tokens" in this module are the *per-report* pooled embeddings
    (``h_k`` from Part 1's text encoder), not raw word tokens -- this keeps
    the encoder sequence bounded (256 patches + <=32 reports + 3 context
    tokens) and is where the attention-based MIL pooling operates.
  * All attention ops that can see a fully-padded key/value set (k=0
    samples) are routed through ``_safe_mha``, which temporarily unmasks
    such rows to avoid a softmax-over-all -inf NaN, then zeroes the output
    for those rows explicitly -- so k=0 never produces NaNs, only zeros.
  * The decoder's start token is a learned projection of z_s (the "<ctx>"
    token from the spec); teacher forcing shifts the target sequence by
    one and predicts every position including the first real token.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH_GRID = 16  # 16x16 = 256 patches
NUM_PATCHES = PATCH_GRID * PATCH_GRID

DAMAGE_WORD_TO_CLASS = {
    "destroyed": 3,
    "gone": 3,
    "collapsed": 3,
    "rubble": 3,
    "major": 2,
    "badly": 2,
    "minor": 1,
    "lightly": 1,
    "undamaged": 0,
    "intact": 0,
}


def _safe_mha(
    mha: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
    attn_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """nn.MultiheadAttention wrapper that never produces NaNs on rows whose
    keys are entirely padded (e.g. k=0 samples): those rows are unmasked
    for the call (so softmax stays finite) and their output is zeroed
    afterwards. Also short-circuits zero-length query/key sequences, which
    PyTorch's MultiheadAttention returns as a None weights tensor for."""
    B = query.shape[0]
    if key.shape[1] == 0 or query.shape[1] == 0:
        out = query.new_zeros((B, query.shape[1], query.shape[-1]))
        weights = query.new_zeros((B, query.shape[1], key.shape[1]))
        return out, weights

    if key_padding_mask is not None:
        all_masked = key_padding_mask.all(dim=1)
        eff_mask = key_padding_mask.clone()
        if all_masked.any():
            eff_mask[all_masked] = False
    else:
        all_masked = torch.zeros(query.shape[0], dtype=torch.bool, device=query.device)
        eff_mask = None

    out, weights = mha(
        query, key, value, key_padding_mask=eff_mask, attn_mask=attn_mask, need_weights=True, average_attn_weights=True
    )
    if all_masked.any():
        out = out.masked_fill(all_masked.view(-1, 1, 1), 0.0)
        weights = weights.masked_fill(all_masked.view(-1, 1, 1), 0.0)
    return out, weights


class MILPooling(nn.Module):
    """Attention-based MIL pooling over reports (Ilse et al. 2018)."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.v = nn.Linear(dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1, bias=False)
        self.null_bag_embedding = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, h_k: torch.Tensor, report_pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, k, D = h_k.shape
        if k == 0:
            return self.null_bag_embedding.unsqueeze(0).expand(B, D).contiguous(), h_k.new_zeros((B, 0))

        scores = self.w(torch.tanh(self.v(h_k))).squeeze(-1)  # (B, k)
        all_pad = report_pad_mask.all(dim=1)
        scores = scores.masked_fill(report_pad_mask, float("-inf"))
        scores_eff = torch.where(all_pad.unsqueeze(-1), torch.zeros_like(scores), scores)
        alpha = torch.softmax(scores_eff, dim=1)
        alpha = alpha.masked_fill(report_pad_mask, 0.0)
        bag = (alpha.unsqueeze(-1) * h_k).sum(dim=1)

        bag = torch.where(all_pad.unsqueeze(-1), self.null_bag_embedding.unsqueeze(0).expand(B, D), bag)
        alpha = torch.where(all_pad.unsqueeze(-1), torch.zeros_like(alpha), alpha)
        return bag, alpha


class EncoderCoAttentionLayer(nn.Module):
    """self-attn (per modality) -> bidirectional co-attention -> FFN, with residual+LN."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.img_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.txt_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.co_i2t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)  # Q=img, K/V=txt
        self.co_t2i = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)  # Q=txt, K/V=img

        self.ln_img_self = nn.LayerNorm(d_model)
        self.ln_txt_self = nn.LayerNorm(d_model)
        self.ln_img_co = nn.LayerNorm(d_model)
        self.ln_txt_co = nn.LayerNorm(d_model)
        self.ln_img_ffn = nn.LayerNorm(d_model)
        self.ln_txt_ffn = nn.LayerNorm(d_model)

        self.ffn_img = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.ReLU(inplace=True), nn.Linear(ffn_dim, d_model))
        self.ffn_txt = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.ReLU(inplace=True), nn.Linear(ffn_dim, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, img: torch.Tensor, txt: torch.Tensor, txt_pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        img_sa, _ = self.img_self_attn(img, img, img, need_weights=False)
        img = self.ln_img_self(img + self.dropout(img_sa))

        txt_sa, _ = _safe_mha(self.txt_self_attn, txt, txt, txt, key_padding_mask=txt_pad_mask)
        txt = self.ln_txt_self(txt + self.dropout(txt_sa))

        img_co, w_i2t = _safe_mha(self.co_i2t, img, txt, txt, key_padding_mask=txt_pad_mask)
        img = self.ln_img_co(img + self.dropout(img_co))

        txt_co, w_t2i = _safe_mha(self.co_t2i, txt, img, img, key_padding_mask=None)
        txt = self.ln_txt_co(txt + self.dropout(txt_co))

        img = self.ln_img_ffn(img + self.dropout(self.ffn_img(img)))
        txt = self.ln_txt_ffn(txt + self.dropout(self.ffn_txt(txt)))

        return img, txt, {"img_to_txt": w_i2t, "txt_to_img": w_t2i}


class Part2Encoder(nn.Module):
    def __init__(
        self,
        dim_i: int,
        dim_t: int,
        shared_dim: int,
        private_dim: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 256,
        max_reports: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_proj = nn.Linear(dim_i, d_model)
        self.row_embed = nn.Parameter(torch.randn(PATCH_GRID, d_model) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(PATCH_GRID, d_model) * 0.02)

        self.report_proj = nn.Linear(dim_t, d_model)
        self.geo_time_mlp = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model))

        self.ctx_s_proj = nn.Linear(shared_dim, d_model)
        self.ctx_i_proj = nn.Linear(private_dim, d_model)
        self.ctx_t_proj = nn.Linear(private_dim, d_model)

        self.layers = nn.ModuleList(
            [EncoderCoAttentionLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )
        self.max_reports = max_reports

    def patch_pos_encoding(self, device: torch.device) -> torch.Tensor:
        pos = self.row_embed.unsqueeze(1) + self.col_embed.unsqueeze(0)  # (16, 16, d_model)
        return pos.reshape(NUM_PATCHES, self.d_model).to(device)

    def forward(
        self,
        patch_tokens: torch.Tensor,  # (B, 256, dim_i)
        h_k: torch.Tensor,  # (B, k, dim_t)
        report_geo: torch.Tensor,  # (B, k, 2)
        report_time: torch.Tensor,  # (B, k, 1)
        report_pad_mask: torch.Tensor,  # (B, k)
        z_s: torch.Tensor,
        z_i_p: torch.Tensor,
        z_t_p: torch.Tensor,
    ) -> dict:
        B = patch_tokens.shape[0]
        device = patch_tokens.device

        patches = self.patch_proj(patch_tokens) + self.patch_pos_encoding(device).unsqueeze(0)

        ctx_s = self.ctx_s_proj(z_s).unsqueeze(1)
        ctx_i = self.ctx_i_proj(z_i_p).unsqueeze(1)
        ctx_t = self.ctx_t_proj(z_t_p).unsqueeze(1)
        img_stream = torch.cat([patches, ctx_s, ctx_i, ctx_t], dim=1)  # (B, 259, d_model)

        k = h_k.shape[1]
        if k > 0:
            geo_time = torch.cat([report_geo, report_time], dim=-1)
            txt_stream = self.report_proj(h_k) + self.geo_time_mlp(geo_time)
        else:
            txt_stream = h_k.new_zeros((B, 0, self.d_model))

        attn_maps = []
        for layer in self.layers:
            img_stream, txt_stream, w = layer(img_stream, txt_stream, report_pad_mask)
            attn_maps.append(w)

        memory = torch.cat([img_stream, txt_stream], dim=1)
        n_img = img_stream.shape[1]
        mem_pad_mask = torch.cat(
            [torch.zeros(B, n_img, dtype=torch.bool, device=device), report_pad_mask], dim=1
        )

        return {
            "img_stream": img_stream,
            "txt_stream": txt_stream,
            "memory": memory,
            "mem_pad_mask": mem_pad_mask,
            "n_patches": NUM_PATCHES,
            "n_context": 3,
            "n_img_total": n_img,
            "n_reports": k,
            "attn_maps": attn_maps,
        }


def _causal_mask(L: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.ReLU(inplace=True), nn.Linear(ffn_dim, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, mem_pad_mask: torch.Tensor, causal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sa, _ = self.self_attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.ln1(x + self.dropout(sa))
        ca, w = _safe_mha(self.cross_attn, x, memory, memory, key_padding_mask=mem_pad_mask)
        x = self.ln2(x + self.dropout(ca))
        x = self.ln3(x + self.dropout(self.ffn(x)))
        return x, w


class Part2Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        shared_dim: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 256,
        max_len: int = 64,
        dropout: float = 0.1,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len + 1, d_model) * 0.02)
        self.ctx_proj = nn.Linear(shared_dim, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def _run_layers(
        self, x: torch.Tensor, memory: torch.Tensor, mem_pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        L = x.shape[1]
        causal_mask = _causal_mask(L, x.device)
        weights = []
        for layer in self.layers:
            x, w = layer(x, memory, mem_pad_mask, causal_mask)
            weights.append(w)
        return x, weights

    def forward_teacher_forcing(self, z_s: torch.Tensor, target_tokens: torch.Tensor, memory: dict) -> dict:
        """target_tokens: (B, L) with BOS at [:,0] and EOS/pad afterwards.
        Predicts target_tokens[:, t] at decoder position t (position 0 is the <ctx> token)."""
        B, L = target_tokens.shape
        dec_in_tokens = target_tokens[:, :-1]
        tok_emb = self.token_emb(dec_in_tokens)
        ctx_tok = self.ctx_proj(z_s).unsqueeze(1)
        x = torch.cat([ctx_tok, tok_emb], dim=1)  # (B, L, d_model)
        x = x + self.pos_emb[:, :L, :]

        x, cross_weights = self._run_layers(x, memory["memory"], memory["mem_pad_mask"])
        logits = self.out_proj(x)  # (B, L, V) aligned with target_tokens

        last_w = cross_weights[-1]  # (B, L, mem_len)
        masses = _split_attention_mass(last_w, memory)
        return {"logits": logits, "cross_attn": last_w, **masses}

    @torch.no_grad()
    def generate_greedy(self, z_s: torch.Tensor, memory: dict, max_len: int | None = None) -> dict:
        B = z_s.shape[0]
        device = z_s.device
        max_len = max_len or self.max_len
        ctx_tok = self.ctx_proj(z_s).unsqueeze(1)
        generated = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        all_cross_w = None

        for _ in range(max_len - 1):
            tok_emb = self.token_emb(generated)
            x = torch.cat([ctx_tok, tok_emb], dim=1)
            x = x + self.pos_emb[:, : x.shape[1], :]
            x, cross_weights = self._run_layers(x, memory["memory"], memory["mem_pad_mask"])
            logits = self.out_proj(x[:, -1, :])
            next_tok = logits.argmax(dim=-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, self.pad_id), next_tok)
            generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == self.eos_id)
            all_cross_w = cross_weights[-1]
            if finished.all():
                break
        masses = _split_attention_mass(all_cross_w, memory) if all_cross_w is not None else {}
        return {"tokens": generated, "cross_attn": all_cross_w, **masses}

    @torch.no_grad()
    def generate_beam(self, z_s: torch.Tensor, memory: dict, beam_size: int = 4, max_len: int | None = None) -> torch.Tensor:
        """Simple batched beam search, one sample at a time (tiny configs / small batches)."""
        max_len = max_len or self.max_len
        B = z_s.shape[0]
        outputs = []
        for b in range(B):
            zb = z_s[b : b + 1]
            mem_b = {
                "memory": memory["memory"][b : b + 1],
                "mem_pad_mask": memory["mem_pad_mask"][b : b + 1],
                "n_patches": memory["n_patches"],
                "n_context": memory["n_context"],
                "n_img_total": memory["n_img_total"],
            }
            beams = [(0.0, [self.bos_id])]
            for _ in range(max_len - 1):
                candidates = []
                all_done = True
                for score, seq in beams:
                    if seq[-1] == self.eos_id:
                        candidates.append((score, seq))
                        continue
                    all_done = False
                    tok = torch.tensor([seq], dtype=torch.long, device=zb.device)
                    ctx_tok = self.ctx_proj(zb).unsqueeze(1)
                    tok_emb = self.token_emb(tok)
                    x = torch.cat([ctx_tok, tok_emb], dim=1) + self.pos_emb[:, : len(seq) + 1, :]
                    x, _ = self._run_layers(x, mem_b["memory"], mem_b["mem_pad_mask"])
                    logits = self.out_proj(x[:, -1, :])
                    logp = F.log_softmax(logits, dim=-1).squeeze(0)
                    topk = torch.topk(logp, beam_size)
                    for lp, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                        candidates.append((score + lp, seq + [idx]))
                if all_done:
                    break
                candidates.sort(key=lambda c: c[0], reverse=True)
                beams = candidates[:beam_size]
            best = max(beams, key=lambda c: c[0])[1]
            outputs.append(best)
        max_out_len = max(len(o) for o in outputs)
        padded = torch.full((B, max_out_len), self.pad_id, dtype=torch.long, device=z_s.device)
        for b, o in enumerate(outputs):
            padded[b, : len(o)] = torch.tensor(o, dtype=torch.long, device=z_s.device)
        return padded


def _split_attention_mass(cross_w: torch.Tensor | None, memory: dict) -> dict:
    if cross_w is None:
        return {}
    n_patches, n_context = memory["n_patches"], memory["n_context"]
    patch_mass = cross_w[..., :n_patches].sum(dim=-1)
    context_mass = cross_w[..., n_patches : n_patches + n_context].sum(dim=-1)
    report_mass = cross_w[..., n_patches + n_context :].sum(dim=-1)
    return {"image_patch_mass": patch_mass, "context_mass": context_mass, "report_mass": report_mass}


class DamageHead(nn.Module):
    def __init__(self, shared_dim: int, bag_dim: int, hidden_dim: int, n_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shared_dim + bag_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, z_s: torch.Tensor, bag: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_s, bag], dim=-1))


class BagHead(nn.Module):
    def __init__(self, bag_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(bag_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        return self.net(bag).squeeze(-1)


class Part2Seq2Seq(nn.Module):
    def __init__(
        self,
        dim_i: int,
        dim_t: int,
        shared_dim: int,
        private_dim: int,
        vocab_size: int,
        d_model: int = 128,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 256,
        max_reports: int = 32,
        max_sitrep_len: int = 48,
        dropout: float = 0.1,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = Part2Encoder(
            dim_i, dim_t, shared_dim, private_dim, d_model, n_enc_layers, n_heads, ffn_dim, max_reports, dropout
        )
        self.mil = MILPooling(dim_t)
        self.decoder = Part2Decoder(
            vocab_size, shared_dim, d_model, n_dec_layers, n_heads, ffn_dim, max_sitrep_len, dropout, pad_id, bos_id, eos_id
        )
        self.bag_head = BagHead(dim_t, hidden_dim=max(dim_t, 16))
        self.damage_head = DamageHead(shared_dim, dim_t, hidden_dim=max(shared_dim, 16))

    def forward(
        self,
        part1_out: dict,
        report_geo: torch.Tensor,
        report_time: torch.Tensor,
        report_pad_mask: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
    ) -> dict:
        h_k = part1_out["text_out"]["h_k"]
        bag, alpha = self.mil(h_k, report_pad_mask)
        bag_logit = self.bag_head(bag)
        damage_logits = self.damage_head(part1_out["z_s"], bag)

        enc_out = self.encoder(
            part1_out["patch_tokens"], h_k, report_geo, report_time, report_pad_mask,
            part1_out["z_s"], part1_out["z_i_p"], part1_out["z_t_p"],
        )

        result = {"bag_logit": bag_logit, "mil_alpha": alpha, "damage_logits": damage_logits, "encoder_out": enc_out}
        if target_tokens is not None:
            dec_out = self.decoder.forward_teacher_forcing(part1_out["z_s"], target_tokens, enc_out)
            result["decoder_out"] = dec_out
        return result

    @torch.no_grad()
    def infer(
        self,
        part1_out: dict,
        report_geo: torch.Tensor,
        report_time: torch.Tensor,
        report_pad_mask: torch.Tensor,
        beam_size: int = 1,
        max_len: int | None = None,
    ) -> dict:
        h_k = part1_out["text_out"]["h_k"]
        bag, alpha = self.mil(h_k, report_pad_mask)
        bag_logit = self.bag_head(bag)
        damage_logits = self.damage_head(part1_out["z_s"], bag)
        enc_out = self.encoder(
            part1_out["patch_tokens"], h_k, report_geo, report_time, report_pad_mask,
            part1_out["z_s"], part1_out["z_i_p"], part1_out["z_t_p"],
        )
        if beam_size > 1:
            tokens = self.decoder.generate_beam(part1_out["z_s"], enc_out, beam_size=beam_size, max_len=max_len)
            gen = {"tokens": tokens}
        else:
            gen = self.decoder.generate_greedy(part1_out["z_s"], enc_out, max_len=max_len)
        return {"bag_logit": bag_logit, "mil_alpha": alpha, "damage_logits": damage_logits, "encoder_out": enc_out, **gen}


def grounding_loss(
    cross_attn: torch.Tensor,  # (B, L, mem_len)
    target_tokens: torch.Tensor,  # (B, L)
    id_to_word: dict[int, str],
    building_masks: torch.Tensor,  # (B, S, 256, 256)
    building_damage: torch.Tensor,  # (B, S)
    n_patches: int,
) -> torch.Tensor:
    """Push decoder attention for damage words toward patches that fall
    inside a building whose ground-truth damage matches that word."""
    B, L, _ = cross_attn.shape
    grid = PATCH_GRID
    ph = building_masks.shape[-2] // grid
    pw = building_masks.shape[-1] // grid
    # patch_building_damage[b, patch] = damage class of the majority structure covering that patch, or -1
    bm = building_masks.float()  # (B, S, H, W)
    bm_patches = bm.unfold(2, ph, ph).unfold(3, pw, pw).sum(dim=(-1, -2))  # (B, S, grid, grid)
    bm_patches = bm_patches.reshape(building_masks.shape[0], building_masks.shape[1], grid * grid)  # (B,S,256)

    losses = []
    for b in range(B):
        for t in range(L):
            tid = int(target_tokens[b, t].item())
            word = id_to_word.get(tid)
            if word not in DAMAGE_WORD_TO_CLASS:
                continue
            dmg = DAMAGE_WORD_TO_CLASS[word]
            covers = bm_patches[b] > 0  # (S, 256)
            matching_struct = building_damage[b] == dmg
            if not matching_struct.any():
                continue
            patch_hit = covers[matching_struct].any(dim=0)  # (256,)
            if not patch_hit.any():
                continue
            target_dist = patch_hit.float()
            target_dist = target_dist / target_dist.sum()
            pred = cross_attn[b, t, :n_patches]
            pred = pred / pred.sum().clamp(min=1e-8)
            loss = -(target_dist * torch.log(pred.clamp(min=1e-8))).sum()
            losses.append(loss)
    if not losses:
        return cross_attn.new_zeros(())
    return torch.stack(losses).mean()


def part2_loss(
    out: dict,
    target_tokens: torch.Tensor,
    damage: torch.Tensor,
    bag_label: torch.Tensor,
    pad_id: int,
    lambda_d: float = 1.0,
    lambda_b: float = 1.0,
    lambda_g: float = 0.0,
    id_to_word: dict[int, str] | None = None,
    building_masks: torch.Tensor | None = None,
    building_damage: torch.Tensor | None = None,
) -> dict:
    logits = out["decoder_out"]["logits"]
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_tokens.reshape(-1), ignore_index=pad_id)
    damage_ce = F.cross_entropy(out["damage_logits"], damage)
    bag_bce = F.binary_cross_entropy_with_logits(out["bag_logit"], bag_label)

    total = ce + lambda_d * damage_ce + lambda_b * bag_bce
    ground = logits.new_zeros(())
    if lambda_g > 0 and id_to_word is not None and building_masks is not None:
        ground = grounding_loss(
            out["decoder_out"]["cross_attn"], target_tokens, id_to_word, building_masks, building_damage,
            out["encoder_out"]["n_patches"],
        )
        total = total + lambda_g * ground

    return {
        "loss": total,
        "sitrep_ce": ce.detach(),
        "damage_ce": damage_ce.detach(),
        "bag_bce": bag_bce.detach(),
        "grounding": ground.detach(),
    }
