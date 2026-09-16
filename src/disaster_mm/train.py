"""Three-stage training CLI.

Stage 1: train Part 1 (multimodal autoencoder) with L1.
Stage 2: freeze Part 1 encoders, train Part 2 (seq2seq) with L2.
Stage 3: unfreeze everything, joint fine-tune at a low LR.

    python -m disaster_mm.train --config configs/tiny.yaml --stage 1
    python -m disaster_mm.train --config configs/tiny.yaml --stage 2
    python -m disaster_mm.train --config configs/tiny.yaml --stage 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .builder import build_datasets, build_models, build_tokenizer_and_splits
from .data.dataset import collate_fn
from .data.tokenizer import TinyTokenizer
from .models.part1_mmae import part1_loss
from .models.part2_seq2seq import part2_loss
from .utils import get_device, load_config, seed_everything


def _run_dir(cfg: dict) -> Path:
    d = Path(cfg.get("checkpoints", {}).get("dir", "runs/default"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_stage1(cfg: dict, device: torch.device) -> None:
    run_dir = _run_dir(cfg)
    vocab_path = str(run_dir / "vocab.json")
    tokenizer, _syn_cfg, splits = build_tokenizer_and_splits(cfg, vocab_path=vocab_path)
    if isinstance(tokenizer, TinyTokenizer):
        tokenizer.vocab.save(vocab_path)
    train_ds, _val_ds, _ = build_datasets(cfg, tokenizer, splits)
    part1, _part2 = build_models(cfg, tokenizer)
    part1.to(device)

    bs = cfg["train"]["batch_size"]
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    opt = torch.optim.Adam(part1.parameters(), lr=cfg["train"]["lr_stage1"])
    l1_cfg = cfg.get("loss1", {})

    part1.train()
    history = []
    for epoch in range(cfg["train"]["epochs_stage1"]):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = part1(batch["image"], batch["report_tokens"], batch["report_pad_mask"])
            losses = part1_loss(
                out, batch["image"], batch["report_tokens"], batch["report_pad_mask"], batch["num_reports"],
                vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id,
                lambda_r=l1_cfg.get("lambda_r", 1.0), lambda_c=l1_cfg.get("lambda_c", 0.5),
                lambda_orth=l1_cfg.get("lambda_orth", 0.1), lambda_n=l1_cfg.get("lambda_n", 0.5),
                tau=l1_cfg.get("tau", 0.07),
            )
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            epoch_loss += losses["loss"].item()
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        history.append(avg)
        print(f"[stage1][epoch {epoch + 1}/{cfg['train']['epochs_stage1']}] loss={avg:.4f}")

    torch.save(part1.state_dict(), run_dir / "part1_stage1.pt")
    print(f"Saved {run_dir / 'part1_stage1.pt'}")


def train_stage2(cfg: dict, device: torch.device) -> None:
    run_dir = _run_dir(cfg)
    vocab_path = str(run_dir / "vocab.json")
    tokenizer, _syn_cfg, splits = build_tokenizer_and_splits(cfg, vocab_path=vocab_path)
    train_ds, _val_ds, _ = build_datasets(cfg, tokenizer, splits)
    part1, part2 = build_models(cfg, tokenizer)
    part1.load_state_dict(torch.load(run_dir / "part1_stage1.pt", map_location=device))
    part1.to(device)
    part2.to(device)

    for p in part1.parameters():
        p.requires_grad_(False)
    part1.eval()

    bs = cfg["train"]["batch_size"]
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    opt = torch.optim.Adam(part2.parameters(), lr=cfg["train"]["lr_stage2"])
    l2_cfg = cfg.get("loss2", {})

    id_to_word = None
    if isinstance(tokenizer, TinyTokenizer):
        id_to_word = {i: w for i, w in enumerate(tokenizer.vocab.idx2word)}

    part2.train()
    for epoch in range(cfg["train"]["epochs_stage2"]):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                p1_out = part1.encode(batch["image"], batch["report_tokens"], batch["report_pad_mask"])
            out = part2(p1_out, batch["report_geo"], batch["report_time"], batch["report_pad_mask"], target_tokens=batch["sitrep_tokens"])
            losses = part2_loss(
                out, batch["sitrep_tokens"], batch["damage"], batch["bag_label"], pad_id=tokenizer.pad_id,
                lambda_d=l2_cfg.get("lambda_d", 1.0), lambda_b=l2_cfg.get("lambda_b", 1.0),
                lambda_g=l2_cfg.get("lambda_g", 0.0), id_to_word=id_to_word,
                building_masks=batch["building_masks"], building_damage=batch["building_damage"],
            )
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            epoch_loss += losses["loss"].item()
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        print(f"[stage2][epoch {epoch + 1}/{cfg['train']['epochs_stage2']}] loss={avg:.4f}")

    torch.save(part2.state_dict(), run_dir / "part2_stage2.pt")
    print(f"Saved {run_dir / 'part2_stage2.pt'}")


def train_stage3(cfg: dict, device: torch.device) -> None:
    run_dir = _run_dir(cfg)
    vocab_path = str(run_dir / "vocab.json")
    tokenizer, _syn_cfg, splits = build_tokenizer_and_splits(cfg, vocab_path=vocab_path)
    train_ds, _val_ds, _ = build_datasets(cfg, tokenizer, splits)
    part1, part2 = build_models(cfg, tokenizer)
    part1.load_state_dict(torch.load(run_dir / "part1_stage1.pt", map_location=device))
    part2.load_state_dict(torch.load(run_dir / "part2_stage2.pt", map_location=device))
    part1.to(device)
    part2.to(device)
    for p in part1.parameters():
        p.requires_grad_(True)

    bs = cfg["train"]["batch_size"]
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    params = list(part1.parameters()) + list(part2.parameters())
    opt = torch.optim.Adam(params, lr=cfg["train"]["lr_stage3"])
    l1_cfg, l2_cfg = cfg.get("loss1", {}), cfg.get("loss2", {})

    id_to_word = None
    if isinstance(tokenizer, TinyTokenizer):
        id_to_word = {i: w for i, w in enumerate(tokenizer.vocab.idx2word)}

    part1.train()
    part2.train()
    for epoch in range(cfg["train"]["epochs_stage3"]):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            p1_out = part1(batch["image"], batch["report_tokens"], batch["report_pad_mask"])
            l1 = part1_loss(
                p1_out, batch["image"], batch["report_tokens"], batch["report_pad_mask"], batch["num_reports"],
                vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id,
                lambda_r=l1_cfg.get("lambda_r", 1.0), lambda_c=l1_cfg.get("lambda_c", 0.5),
                lambda_orth=l1_cfg.get("lambda_orth", 0.1), lambda_n=l1_cfg.get("lambda_n", 0.5),
                tau=l1_cfg.get("tau", 0.07),
            )
            out = part2(p1_out, batch["report_geo"], batch["report_time"], batch["report_pad_mask"], target_tokens=batch["sitrep_tokens"])
            l2 = part2_loss(
                out, batch["sitrep_tokens"], batch["damage"], batch["bag_label"], pad_id=tokenizer.pad_id,
                lambda_d=l2_cfg.get("lambda_d", 1.0), lambda_b=l2_cfg.get("lambda_b", 1.0),
                lambda_g=l2_cfg.get("lambda_g", 0.0), id_to_word=id_to_word,
                building_masks=batch["building_masks"], building_damage=batch["building_damage"],
            )
            total = l1["loss"] + l2["loss"]
            opt.zero_grad()
            total.backward()
            opt.step()
            epoch_loss += total.item()
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        print(f"[stage3][epoch {epoch + 1}/{cfg['train']['epochs_stage3']}] loss={avg:.4f}")

    torch.save(part1.state_dict(), run_dir / "part1_stage3.pt")
    torch.save(part2.state_dict(), run_dir / "part2_stage3.pt")
    print(f"Saved {run_dir / 'part1_stage3.pt'} and {run_dir / 'part2_stage3.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg)

    if args.stage == 1:
        train_stage1(cfg, device)
    elif args.stage == 2:
        train_stage2(cfg, device)
    else:
        train_stage3(cfg, device)


if __name__ == "__main__":
    main()
