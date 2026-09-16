import pytest
import torch

from disaster_mm.models.part2_seq2seq import MILPooling, part2_loss


def _p1_out(part1, batch):
    part1.eval()
    with torch.no_grad():
        return part1.encode(batch["image"], batch["report_tokens"], batch["report_pad_mask"])


def test_encoder_co_attention_weights_sum_to_one_and_respect_padding(part1, part2, batch):
    p1_out = _p1_out(part1, batch)
    part2.eval()  # disable attention dropout so weight rows sum to exactly 1
    enc_out = part2.encoder(
        p1_out["patch_tokens"], p1_out["text_out"]["h_k"], batch["report_geo"], batch["report_time"],
        batch["report_pad_mask"], p1_out["z_s"], p1_out["z_i_p"], p1_out["z_t_p"],
    )
    w_i2t = enc_out["attn_maps"][-1]["img_to_txt"]  # (B, n_img, k)
    B = batch["image"].shape[0]
    for b in range(B):
        has_reports = (~batch["report_pad_mask"][b]).any()
        row_sums = w_i2t[b].sum(dim=-1)
        if has_reports:
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
            # padded report slots should carry ~0 weight
            pad_cols = batch["report_pad_mask"][b]
            if pad_cols.any():
                assert w_i2t[b][:, pad_cols].abs().max() < 1e-3
        else:
            assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-4)


def test_mil_weights_sum_to_one_when_reports_present(part1, part2, batch):
    p1_out = _p1_out(part1, batch)
    bag, alpha = part2.mil(p1_out["text_out"]["h_k"], batch["report_pad_mask"])
    B = batch["image"].shape[0]
    assert torch.isnan(bag).sum() == 0
    for b in range(B):
        has_reports = (~batch["report_pad_mask"][b]).any()
        if has_reports:
            assert alpha[b].sum().item() == pytest.approx(1.0, abs=1e-4)
        else:
            assert alpha[b].sum().item() == pytest.approx(0.0, abs=1e-4)


def test_mil_pooling_k0_uses_null_embedding():
    mil = MILPooling(dim=8)
    h_k = torch.zeros(3, 0, 8)
    mask = torch.zeros(3, 0, dtype=torch.bool)
    bag, alpha = mil(h_k, mask)
    assert bag.shape == (3, 8)
    assert alpha.shape == (3, 0)
    assert torch.allclose(bag, mil.null_bag_embedding.unsqueeze(0).expand(3, 8))


def test_decoder_forward_shapes_and_loss(part1, part2, tokenizer, batch):
    p1_out = _p1_out(part1, batch)
    out = part2(p1_out, batch["report_geo"], batch["report_time"], batch["report_pad_mask"], target_tokens=batch["sitrep_tokens"])
    B, L = batch["sitrep_tokens"].shape
    assert out["decoder_out"]["logits"].shape == (B, L, tokenizer.vocab_size)
    assert out["damage_logits"].shape == (B, 4)
    assert out["bag_logit"].shape == (B,)

    losses = part2_loss(out, batch["sitrep_tokens"], batch["damage"], batch["bag_label"], pad_id=tokenizer.pad_id)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


def test_cross_attention_mass_partitions_sum_to_one(part1, part2, batch):
    p1_out = _p1_out(part1, batch)
    part2.eval()  # disable attention dropout so weight rows sum to exactly 1
    out = part2(p1_out, batch["report_geo"], batch["report_time"], batch["report_pad_mask"], target_tokens=batch["sitrep_tokens"])
    dec = out["decoder_out"]
    total = dec["image_patch_mass"] + dec["context_mass"] + dec["report_mass"]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-3)


def test_greedy_generation_runs_and_respects_max_len(part1, part2, batch):
    p1_out = _p1_out(part1, batch)
    part2.eval()
    enc_out = part2.encoder(
        p1_out["patch_tokens"], p1_out["text_out"]["h_k"], batch["report_geo"], batch["report_time"],
        batch["report_pad_mask"], p1_out["z_s"], p1_out["z_i_p"], p1_out["z_t_p"],
    )
    gen = part2.decoder.generate_greedy(p1_out["z_s"], enc_out, max_len=10)
    assert gen["tokens"].shape[0] == batch["image"].shape[0]
    assert gen["tokens"].shape[1] <= 10
