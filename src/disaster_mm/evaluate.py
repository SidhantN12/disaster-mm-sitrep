"""Evaluation CLI: runs the trained (or stage-1/2/3) checkpoints over the
test split and reports the metrics from metrics.py, including a
modality-ablation pass (drop image / drop text at test time).

    python -m disaster_mm.evaluate --config configs/tiny.yaml --stage 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .builder import build_datasets, build_models, build_tokenizer_and_splits
from .data.dataset import collate_fn
from .metrics import bleu, classification_f1, count_fact_accuracy
from .utils import get_device, load_config, seed_everything


def _load_stage(run_dir: Path, part1, part2, stage: int, device: torch.device) -> None:
    if stage == 1:
        part1.load_state_dict(torch.load(run_dir / "part1_stage1.pt", map_location=device))
    elif stage == 2:
        part1.load_state_dict(torch.load(run_dir / "part1_stage1.pt", map_location=device))
        part2.load_state_dict(torch.load(run_dir / "part2_stage2.pt", map_location=device))
    else:
        part1.load_state_dict(torch.load(run_dir / "part1_stage3.pt", map_location=device))
        part2.load_state_dict(torch.load(run_dir / "part2_stage3.pt", map_location=device))


@torch.no_grad()
def run_eval(cfg: dict, device: torch.device, stage: int, ablate: str | None = None) -> dict:
    run_dir = Path(cfg.get("checkpoints", {}).get("dir", "runs/default"))
    vocab_path = str(run_dir / "vocab.json")
    tokenizer, _syn_cfg, splits = build_tokenizer_and_splits(cfg, vocab_path=vocab_path)
    _train_ds, _val_ds, test_ds = build_datasets(cfg, tokenizer, splits)
    part1, part2 = build_models(cfg, tokenizer)
    _load_stage(run_dir, part1, part2, stage, device)
    part1.to(device).eval()
    part2.to(device).eval()

    loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_fn)

    all_damage_preds, all_damage_true = [], []
    all_bag_preds, all_bag_true = [], []
    hyps, refs = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        image = batch["image"]
        report_tokens = batch["report_tokens"]
        report_pad_mask = batch["report_pad_mask"]
        report_geo, report_time = batch["report_geo"], batch["report_time"]

        if ablate == "image":
            image = torch.zeros_like(image)
        elif ablate == "text":
            report_tokens = report_tokens[:, :0, :]
            report_geo = report_geo[:, :0, :]
            report_time = report_time[:, :0, :]
            report_pad_mask = report_pad_mask[:, :0]

        p1_out = part1.encode(image, report_tokens, report_pad_mask)
        out = part2.infer(p1_out, report_geo, report_time, report_pad_mask, beam_size=1)

        damage_pred = out["damage_logits"].argmax(dim=-1)
        bag_pred = (torch.sigmoid(out["bag_logit"]) > 0.5).float()

        all_damage_preds.append(damage_pred.cpu().numpy())
        all_damage_true.append(batch["damage"].cpu().numpy())
        all_bag_preds.append(bag_pred.cpu().numpy())
        all_bag_true.append(batch["bag_label"].cpu().numpy())

        for i in range(image.shape[0]):
            gen_text = tokenizer.decode(out["tokens"][i].cpu().tolist())
            ref_text = tokenizer.decode(batch["sitrep_tokens"][i].cpu().tolist())
            hyps.append(gen_text)
            refs.append(ref_text)

    damage_preds = np.concatenate(all_damage_preds)
    damage_true = np.concatenate(all_damage_true)
    bag_preds = np.concatenate(all_bag_preds)
    bag_true = np.concatenate(all_bag_true)

    damage_f1 = classification_f1(damage_preds, damage_true, n_classes=4)
    bag_f1 = classification_f1(bag_preds.astype(int), bag_true.astype(int), n_classes=2)
    bleu_score = bleu(hyps, refs)
    facts = count_fact_accuracy(hyps, refs)

    return {
        "ablate": ablate or "none",
        "damage_f1": damage_f1,
        "bag_f1": bag_f1,
        "bleu": bleu_score,
        "count_fact_accuracy": facts,
        "n_samples": len(hyps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--out", default=None, help="Optional path to dump metrics JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg)

    results = {
        "full": run_eval(cfg, device, args.stage, ablate=None),
        "drop_image": run_eval(cfg, device, args.stage, ablate="image"),
        "drop_text": run_eval(cfg, device, args.stage, ablate="text"),
    }
    print(json.dumps(results, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
