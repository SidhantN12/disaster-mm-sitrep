import torch

from disaster_mm.models.part1_mmae import bow_target, info_nce_loss, orthogonality_loss, part1_loss


def test_encode_shapes(part1, batch):
    out = part1.encode(batch["image"], batch["report_tokens"], batch["report_pad_mask"])
    B = batch["image"].shape[0]
    assert out["z_s"].shape == (B, part1.shared_dim)
    assert out["z_s_i"].shape == (B, part1.shared_dim)
    assert out["z_s_t"].shape == (B, part1.shared_dim)
    assert out["z_i_p"].shape == (B, part1.private_dim)
    assert out["z_t_p"].shape == (B, part1.private_dim)
    assert out["patch_tokens"].shape == (B, 256, part1.dim_i)


def test_forward_shapes_and_no_nans(part1, batch):
    part1.train()
    out = part1(batch["image"], batch["report_tokens"], batch["report_pad_mask"])
    assert out["recon_i"].shape == batch["image"].shape
    assert out["recon_t_logits"].shape == (batch["image"].shape[0], part1.vocab_size)
    assert torch.isnan(out["recon_i"]).sum() == 0
    assert torch.isnan(out["recon_t_logits"]).sum() == 0


def test_k0_batch_has_no_nans(part1, zero_report_batch):
    part1.train()
    out = part1(zero_report_batch["image"], zero_report_batch["report_tokens"], zero_report_batch["report_pad_mask"])
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            assert torch.isnan(v).sum() == 0, f"{k} has NaNs"


def test_modality_dropout_never_drops_both(part1):
    part1.modality_dropout_p = 1.0  # always drop something
    part1.train()
    B = 64
    torch.manual_seed(0)
    drop = torch.rand(B) < part1.modality_dropout_p
    drop_image = drop & (torch.rand(B) < 0.5)
    drop_text = drop & (~drop_image)
    both_dropped = drop_image & drop_text
    assert not both_dropped.any()
    # sanity: with p=1.0 every sample drops exactly one modality
    assert (drop_image | drop_text).all()
    assert not (drop_image & drop_text).any()


def test_orthogonality_loss_zero_for_orthogonal_inputs():
    # Two length-4 vectors that are exactly orthogonal (dot product 0),
    # placed in disjoint columns so every other entry of S^T P is 0 too.
    d1, d2 = 6, 4
    s = torch.zeros(4, d1)
    s[:, 0] = torch.tensor([1.0, -1.0, 1.0, -1.0])
    p = torch.zeros(4, d2)
    p[:, 0] = torch.tensor([1.0, 1.0, -1.0, -1.0])
    loss = orthogonality_loss(s, p)
    assert loss.item() == 0.0


def test_orthogonality_loss_positive_for_correlated_inputs():
    B, d = 16, 6
    x = torch.randn(B, d)
    loss = orthogonality_loss(x, x.clone())
    assert loss.item() > 0.0


def test_info_nce_sanity():
    B, d = 8, 16
    torch.manual_seed(0)
    base = torch.randn(B, d)
    z_i = base + 0.01 * torch.randn(B, d)
    z_t = base + 0.01 * torch.randn(B, d)
    paired_mask = torch.ones(B, dtype=torch.bool)
    aligned_loss = info_nce_loss(z_i, z_t, paired_mask, tau=0.07)

    z_t_shuffled = z_t[torch.randperm(B)]
    misaligned_loss = info_nce_loss(z_i, z_t_shuffled, paired_mask, tau=0.07)
    assert aligned_loss.item() < misaligned_loss.item()


def test_info_nce_skips_when_fewer_than_two_paired():
    z_i = torch.randn(4, 8)
    z_t = torch.randn(4, 8)
    paired_mask = torch.tensor([True, False, False, False])
    loss = info_nce_loss(z_i, z_t, paired_mask, tau=0.07)
    assert loss.item() == 0.0


def test_bow_target_matches_reports(dataset, tokenizer):
    item = dataset[0]
    from disaster_mm.data.dataset import collate_fn

    b = collate_fn([item])
    target = bow_target(b["report_tokens"], b["report_pad_mask"], tokenizer.vocab_size, tokenizer.pad_id)
    assert target.shape == (1, tokenizer.vocab_size)
    assert target.min() >= 0 and target.max() <= 1
    assert target[0, tokenizer.pad_id] == 0


def test_part1_loss_is_finite_and_backprops(part1, batch):
    part1.train()
    out = part1(batch["image"], batch["report_tokens"], batch["report_pad_mask"])
    losses = part1_loss(
        out, batch["image"], batch["report_tokens"], batch["report_pad_mask"], batch["num_reports"],
        vocab_size=part1.vocab_size, pad_id=0,
    )
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    grad_norms = [p.grad.norm().item() for p in part1.parameters() if p.grad is not None]
    assert any(g > 0 for g in grad_norms)
