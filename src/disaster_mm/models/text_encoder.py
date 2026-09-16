"""Text encoder E_T.

Reports are concatenated into one sequence per sample, each prefixed with a
separator token, so a single transformer pass yields both token-level
outputs and, at each separator position, a per-report pooled embedding
(``h_k``) -- this is the "report-index per token" bookkeeping the spec asks
for (the index is simply which segment a token falls in).

``config='tiny'`` uses a small randomly-initialized transformer over a
word-level vocab (offline, CPU-friendly). ``config='bert'`` uses HF
``bert-base-uncased`` as the backbone (downloads weights on first use).

Samples with zero reports (k=0) are handled explicitly: the pooled text
representation falls back to a learned ``null_report_embedding`` instead of
an average over zero elements, so no NaNs are produced.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _TinyTextBackbone(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, depth: int, num_heads: int, max_len: int = 4096) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, ids: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        L = ids.shape[1]
        x = self.token_emb(ids) + self.pos_emb[:, :L, :]
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class TextEncoder(nn.Module):
    def __init__(
        self,
        variant: str,
        vocab_size: int,
        sep_id: int,
        pad_id: int,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 2,
        bert_name: str = "bert-base-uncased",
    ) -> None:
        super().__init__()
        self.variant = variant
        self.sep_id = sep_id
        self.pad_id = pad_id
        if variant == "tiny":
            self.backbone = _TinyTextBackbone(vocab_size, embed_dim, depth, num_heads)
            self.embed_dim = embed_dim
        elif variant == "bert":
            from transformers import BertModel

            self.backbone = BertModel.from_pretrained(bert_name)
            self.embed_dim = self.backbone.config.hidden_size
        else:
            raise ValueError(f"Unknown text encoder variant: {variant}")
        self.null_report_embedding = nn.Parameter(torch.randn(self.embed_dim) * 0.02)

    def _run_backbone(self, ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        if self.variant == "tiny":
            return self.backbone(ids, key_padding_mask=pad_mask)
        attention_mask = (~pad_mask).long()
        return self.backbone(input_ids=ids, attention_mask=attention_mask).last_hidden_state

    def forward(self, report_tokens: torch.Tensor, report_pad_mask: torch.Tensor) -> dict:
        """
        report_tokens: (B, k, Lr) long
        report_pad_mask: (B, k) bool, True = this report slot is padding
        """
        B, k, Lr = report_tokens.shape
        device = report_tokens.device
        D = self.embed_dim

        if k == 0:
            pooled = self.null_report_embedding.unsqueeze(0).expand(B, D).contiguous()
            return {
                "h_k": report_tokens.new_zeros((B, 0, D), dtype=torch.float32),
                "pooled": pooled,
                "token_states": report_tokens.new_zeros((B, 0, Lr, D), dtype=torch.float32),
                "report_index": report_tokens.new_zeros((B, 0, Lr), dtype=torch.long),
                "report_pad_mask": report_pad_mask,
            }

        sep_col = torch.full((B, k, 1), self.sep_id, dtype=torch.long, device=device)
        seq = torch.cat([sep_col, report_tokens], dim=-1)  # (B, k, Lr+1)

        token_is_pad = seq == self.pad_id
        pad_mask_token = report_pad_mask.unsqueeze(-1).expand(B, k, Lr + 1) | token_is_pad

        all_pad_sample = report_pad_mask.all(dim=1)  # (B,)
        eff_mask = pad_mask_token.clone()
        eff_mask[all_pad_sample] = False  # avoid fully-masked rows -> NaN softmax

        flat_ids = seq.view(B, k * (Lr + 1))
        flat_mask = eff_mask.view(B, k * (Lr + 1))

        hidden = self._run_backbone(flat_ids, flat_mask)
        hidden = hidden.view(B, k, Lr + 1, D)

        h_k = hidden[:, :, 0, :]
        token_states = hidden[:, :, 1:, :]

        valid = (~report_pad_mask).float().unsqueeze(-1)  # (B, k, 1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        pooled = (h_k * valid).sum(dim=1) / denom
        pooled = torch.where(
            all_pad_sample.unsqueeze(-1), self.null_report_embedding.unsqueeze(0).expand(B, D), pooled
        )

        report_index = torch.arange(k, device=device).view(1, k, 1).expand(B, k, Lr)

        return {
            "h_k": h_k,
            "pooled": pooled,
            "token_states": token_states,
            "report_index": report_index,
            "report_pad_mask": report_pad_mask,
        }
