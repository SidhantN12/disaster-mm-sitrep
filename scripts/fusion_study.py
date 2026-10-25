"""Three-way fusion comparison: early vs late vs hybrid (cross-attention).

Protocol (identical for all three variants):
  Stage 1  trained ONCE -- the Part 1 multimodal autoencoder. All three
           variants then load the same stage-1 encoders and freeze them, so
           the unimodal representations are literally identical and the only
           difference between runs is the fusion mechanism.
  Stage 2  each variant trains its own fusion + decoder + heads, same data,
           same seed, same optimiser, same batch size, same epoch budget.

Reports, per variant: convergence, damage/bag weighted + macro F1, BLEU,
template fact accuracy, modality-ablation behaviour, parameter count and
wall-clock cost, plus the constant-predictor baselines every number must beat.

    python scripts/fusion_study.py --config configs/tiny.yaml --out runs/fusion
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from disaster_mm.builder import build_datasets, build_models, build_tokenizer_and_splits
from disaster_mm.data.dataset import collate_fn
from disaster_mm.data.tokenizer import TinyTokenizer
from disaster_mm.metrics import bleu, count_fact_accuracy
from disaster_mm.models.fusion_variants import build_fusion_model
from disaster_mm.models.part1_mmae import part1_loss
from disaster_mm.models.part2_seq2seq import part2_loss
from disaster_mm.utils import load_config, seed_everything

FUSIONS = ["early", "late", "hybrid"]


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def train_stage1(cfg, part1, train_ds, tokenizer, device):
    loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_fn)
    opt = torch.optim.Adam(part1.parameters(), lr=cfg["train"]["lr_stage1"])
    l1 = cfg.get("loss1", {})
    part1.train()
    history = []
    for epoch in range(cfg["train"]["epochs_stage1"]):
        tot, nb = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = part1(batch["image"], batch["report_tokens"], batch["report_pad_mask"])
            losses = part1_loss(
                out, batch["image"], batch["report_tokens"], batch["report_pad_mask"], batch["num_reports"],
                vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id,
                lambda_r=l1.get("lambda_r", 1.0), lambda_c=l1.get("lambda_c", 0.5),
                lambda_orth=l1.get("lambda_orth", 0.1), lambda_n=l1.get("lambda_n", 0.5), tau=l1.get("tau", 0.07),
            )
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            tot += float(losses["loss"].detach())
            nb += 1
        history.append(tot / max(nb, 1))
        print(f"  [stage1][epoch {epoch+1}/{cfg['train']['epochs_stage1']}] loss={history[-1]:.4f}", flush=True)
    return history


def train_stage2(cfg, part1, model, train_ds, tokenizer, device, tag):
    """Part 1 frozen; only the fusion model trains. Identical for all variants."""
    loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_fn)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr_stage2"])
    l2 = cfg.get("loss2", {})
    for p in part1.parameters():
        p.requires_grad_(False)
    part1.eval()
    model.train()
    history = []
    t0 = time.time()
    for epoch in range(cfg["train"]["epochs_stage2"]):
        tot, nb = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                p1 = part1.encode(batch["image"], batch["report_tokens"], batch["report_pad_mask"])
            out = model(
                p1, batch["report_geo"], batch["report_time"], batch["report_pad_mask"], batch["sitrep_tokens"]
            )
            losses = part2_loss(
                out, batch["sitrep_tokens"], batch["damage"], batch["bag_label"], pad_id=tokenizer.pad_id,
                lambda_d=l2.get("lambda_d", 1.0), lambda_b=l2.get("lambda_b", 1.0), lambda_g=0.0,
            )
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            tot += float(losses["loss"].detach())
            nb += 1
        history.append(tot / max(nb, 1))
        print(f"  [{tag}][epoch {epoch+1}/{cfg['train']['epochs_stage2']}] loss={history[-1]:.4f}", flush=True)
    return history, time.time() - t0


@torch.no_grad()
def evaluate(cfg, part1, model, test_ds, tokenizer, device, ablate=None):
    loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_fn)
    part1.eval()
    model.eval()
    dp, dt_, bp, bt_, hyps, refs = [], [], [], [], [], []
    img_mass, rep_mass = [], []
    t0 = time.time()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        image = batch["image"]
        rt, rg, rtime = batch["report_tokens"], batch["report_geo"], batch["report_time"]
        rpm = batch["report_pad_mask"]
        if ablate == "image":
            image = torch.zeros_like(image)
        elif ablate == "text":
            rt, rg, rtime, rpm = rt[:, :0, :], rg[:, :0, :], rtime[:, :0, :], rpm[:, :0]

        p1 = part1.encode(image, rt, rpm)
        out = model.infer(p1, rg, rtime, rpm, beam_size=1)

        dp += out["damage_logits"].argmax(-1).cpu().tolist()
        dt_ += batch["damage"].cpu().tolist()
        bp += (torch.sigmoid(out["bag_logit"]) > 0.5).long().flatten().cpu().tolist()
        bt_ += batch["bag_label"].long().cpu().tolist()
        for i in range(image.shape[0]):
            hyps.append(tokenizer.decode(out["tokens"][i].cpu().tolist()))
            refs.append(tokenizer.decode(batch["sitrep_tokens"][i].cpu().tolist()))
        if out.get("image_patch_mass") is not None:
            img_mass.append(float(out["image_patch_mass"].mean()))
        if out.get("report_mass") is not None:
            rep_mass.append(float(out["report_mass"].mean()))
    elapsed = time.time() - t0

    res = {
        "ablate": ablate or "none",
        "damage_weighted_f1": float(f1_score(dt_, dp, average="weighted", zero_division=0)),
        "damage_macro_f1": float(f1_score(dt_, dp, average="macro", zero_division=0)),
        "damage_pred_dist": dict(Counter(dp)),
        "bag_weighted_f1": float(f1_score(bt_, bp, average="weighted", zero_division=0)),
        "bag_macro_f1": float(f1_score(bt_, bp, average="macro", zero_division=0)),
        "bleu": float(bleu(hyps, refs)),
        "fact": count_fact_accuracy(hyps, refs),
        "distinct_outputs": len(set(hyps)),
        "n_samples": len(hyps),
        "eval_seconds": elapsed,
        "ms_per_tile": 1000 * elapsed / max(len(hyps), 1),
        "sample_hyp": hyps[0] if hyps else "",
        "mean_image_mass": float(np.mean(img_mass)) if img_mass else None,
        "mean_report_mass": float(np.mean(rep_mass)) if rep_mass else None,
    }
    if hasattr(model, "gate_damage"):
        res["gates"] = {
            "damage": torch.softmax(model.gate_damage, 0).cpu().tolist(),
            "bag": torch.softmax(model.gate_bag, 0).cpu().tolist(),
            "sitrep": torch.softmax(model.gate_sitrep, 0).cpu().tolist(),
        }
    return res


def baselines(train_ds, test_ds):
    def lab(ds):
        d, b = [], []
        for i in range(len(ds)):
            s = ds[i]
            d.append(int(s["damage"]))
            b.append(int(s["bag_label"]))
        return d, b
    dtr, btr = lab(train_ds)
    dte, bte = lab(test_ds)
    maj_d = Counter(dtr).most_common(1)[0][0]
    out = {
        "train_damage_dist": dict(Counter(dtr)), "test_damage_dist": dict(Counter(dte)),
        "train_bag_dist": dict(Counter(btr)), "test_bag_dist": dict(Counter(bte)),
        "damage_majority_class": maj_d,
        "damage_majority_weighted_f1": float(f1_score(dte, [maj_d] * len(dte), average="weighted", zero_division=0)),
        "damage_majority_macro_f1": float(f1_score(dte, [maj_d] * len(dte), average="macro", zero_division=0)),
    }
    for c in (0, 1):
        out[f"bag_always{c}_weighted_f1"] = float(
            f1_score(bte, [c] * len(bte), average="weighted", zero_division=0)
        )
        out[f"bag_always{c}_macro_f1"] = float(f1_score(bte, [c] * len(bte), average="macro", zero_division=0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny.yaml")
    ap.add_argument("--out", default="runs/fusion")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get("device", "cpu"))

    seed_everything(cfg["seed"])
    tokenizer, _syn, splits = build_tokenizer_and_splits(cfg, vocab_path=str(out_dir / "vocab.json"))
    if isinstance(tokenizer, TinyTokenizer):
        tokenizer.vocab.save(str(out_dir / "vocab.json"))
    train_ds, _val_ds, test_ds = build_datasets(cfg, tokenizer, splits)

    results = {"config": args.config, "seed": cfg["seed"], "baselines": baselines(train_ds, test_ds)}
    print("baselines:", json.dumps(results["baselines"], indent=1), flush=True)

    # ---- Stage 1, once, shared by every variant ----
    seed_everything(cfg["seed"])
    part1_proto, _ = build_models(cfg, tokenizer)
    part1_proto.to(device)
    print("\n=== Stage 1 (shared by all fusion variants) ===", flush=True)
    results["stage1_history"] = train_stage1(cfg, part1_proto, train_ds, tokenizer, device)
    torch.save(part1_proto.state_dict(), out_dir / "part1_stage1.pt")
    results["part1_params"] = n_params(part1_proto)

    # ---- Stage 2, once per fusion variant, from the same frozen encoders ----
    for fusion in FUSIONS:
        print(f"\n=== Stage 2: {fusion} fusion ===", flush=True)
        seed_everything(cfg["seed"])
        part1, hybrid = build_models(cfg, tokenizer)
        part1.load_state_dict(torch.load(out_dir / "part1_stage1.pt", map_location=device))
        part1.to(device)
        model = hybrid if fusion == "hybrid" else build_fusion_model(fusion, cfg, part1, tokenizer)
        model.to(device)

        hist, secs = train_stage2(cfg, part1, model, train_ds, tokenizer, device, fusion)
        torch.save(model.state_dict(), out_dir / f"model_{fusion}.pt")

        entry = {
            "fusion": fusion,
            "fusion_params": n_params(model),
            "train_seconds": secs,
            "stage2_history": hist,
            "eval": {},
        }
        for ab in (None, "image", "text"):
            entry["eval"][ab or "full"] = evaluate(cfg, part1, model, test_ds, tokenizer, device, ablate=ab)
            e = entry["eval"][ab or "full"]
            print(
                f"  [{fusion}/{ab or 'full'}] damage wF1={e['damage_weighted_f1']:.4f} "
                f"macro={e['damage_macro_f1']:.4f} bag wF1={e['bag_weighted_f1']:.4f} "
                f"BLEU={e['bleu']:.2f} distinct={e['distinct_outputs']}/{e['n_samples']}",
                flush=True,
            )
        results[fusion] = entry

    with open(out_dir / "fusion_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nwrote", out_dir / "fusion_results.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
