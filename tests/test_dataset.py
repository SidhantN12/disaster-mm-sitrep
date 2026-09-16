import torch


def test_collate_shapes(batch):
    B = 8
    assert batch["image"].shape == (B, 6, 256, 256)
    assert batch["report_tokens"].ndim == 3
    assert batch["report_tokens"].shape[0] == B
    assert batch["report_pad_mask"].shape[:2] == batch["report_tokens"].shape[:2]
    assert batch["sitrep_tokens"].shape[0] == B
    assert batch["damage"].shape == (B,)
    assert batch["bag_label"].shape == (B,)
    assert batch["building_masks"].shape[0] == B
    assert batch["building_damage"].shape[0] == B


def test_pad_mask_matches_num_reports(batch):
    for i in range(batch["image"].shape[0]):
        k = batch["num_reports"][i].item()
        mask_row = batch["report_pad_mask"][i]
        assert (~mask_row).sum().item() == k


def test_all_k0_batch_has_stable_shape(zero_report_batch):
    b = zero_report_batch
    assert b["report_tokens"].shape[1] >= 1
    assert b["report_pad_mask"].all()
    assert torch.isnan(b["image"]).sum() == 0
