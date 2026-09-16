import torch

from disaster_mm.models.part1_mmae import part1_loss
from disaster_mm.models.part2_seq2seq import part2_loss


def test_part1_overfits_eight_samples(part1, dataset):
    from disaster_mm.data.dataset import collate_fn

    items = [dataset[i] for i in range(8)]
    batch = collate_fn(items)

    opt = torch.optim.Adam(part1.parameters(), lr=3e-3)
    part1.train()

    losses = []
    for _ in range(150):
        out = part1(batch["image"], batch["report_tokens"], batch["report_pad_mask"])
        loss = part1_loss(
            out, batch["image"], batch["report_tokens"], batch["report_pad_mask"], batch["num_reports"],
            vocab_size=part1.vocab_size, pad_id=0,
        )["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.7


def test_part2_overfits_eight_samples(part1, part2, dataset, tokenizer):
    from disaster_mm.data.dataset import collate_fn

    items = [dataset[i] for i in range(8)]
    batch = collate_fn(items)

    for p in part1.parameters():
        p.requires_grad_(False)
    part1.eval()

    opt = torch.optim.Adam(part2.parameters(), lr=3e-3)
    part2.train()

    losses = []
    for _ in range(60):
        with torch.no_grad():
            p1_out = part1.encode(batch["image"], batch["report_tokens"], batch["report_pad_mask"])
        out = part2(p1_out, batch["report_geo"], batch["report_time"], batch["report_pad_mask"], target_tokens=batch["sitrep_tokens"])
        loss = part2_loss(out, batch["sitrep_tokens"], batch["damage"], batch["bag_label"], pad_id=tokenizer.pad_id)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.6
