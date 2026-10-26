"""How much training does a showable SITREP actually need?

Trains stage 1 once, then trains a fresh cross-attention Part 2 for a range of
stage-2 epoch budgets, and records what the decoder writes at each budget.
The point is the generated text, not the score: it shows the budget at which
output stops being degenerate and starts being presentable.

    python scripts/budget_sweep.py --config configs/tiny.yaml --out runs/sweep
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from disaster_mm.builder import build_datasets, build_models, build_tokenizer_and_splits
from disaster_mm.data.dataset import collate_fn
from disaster_mm.data.tokenizer import TinyTokenizer
from disaster_mm.metrics import bleu
from disaster_mm.models.part1_mmae import part1_loss
from disaster_mm.models.part2_seq2seq import part2_loss
from disaster_mm.utils import load_config, seed_everything

BUDGETS = [1, 2, 3, 6, 12, 24, 48]

COUNT = re.compile(r"(\d+)\s+destroyed\s+(\d+)\s+major damage\s+(\d+)\s+minor damage\s+(\d+)\s+undamaged")
BRIDGE = re.compile(r"bridge is (\w+)")
LEVEL = re.compile(r"overall damage level (\w+)")


def fields(text):
    m = COUNT.search(text)
    b = BRIDGE.search(text)
    lv = LEVEL.search(text)
    out = {}
    out["counts"] = tuple(int(x) for x in m.groups()) if m else None
    out["bridge"] = b.group(1) if b else None
    out["level"] = lv.group(1) if lv else None
    return out


def repetition(text):
    """Fraction of tokens that simply repeat the previous token."""
    toks = text.split()
    if len(toks) < 2:
        return 0.0
    rep = sum(1 for i in range(1, len(toks)) if toks[i] == toks[i - 1])
    return rep / (len(toks) - 1)


def train_stage1(cfg, part1, ds, tok, device):
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_fn)
    opt = torch.optim.Adam(part1.parameters(), lr=cfg["train"]["lr_stage1"])
    l1 = cfg.get("loss1", {})
    part1.train()
    for _ in range(cfg["train"]["epochs_stage1"]):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = part1(batch["image"], batch["report_tokens"], batch["report_pad_mask"])
            losses = part1_loss(
                out, batch["image"], batch["report_tokens"], batch["report_pad_mask"], batch["num_reports"],
                vocab_size=tok.vocab_size, pad_id=tok.pad_id,
                lambda_r=l1.get("lambda_r", 1.0), lambda_c=l1.get("lambda_c", 0.5),
                lambda_orth=l1.get("lambda_orth", 0.1), lambda_n=l1.get("lambda_n", 0.5), tau=l1.get("tau", 0.07))
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()


def train_part2(cfg, part1, model, ds, tok, device, epochs):
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_fn)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr_stage2"])
    l2 = cfg.get("loss2", {})
    for p in part1.parameters():
        p.requires_grad_(False)
    part1.eval()
    model.train()
    history = []
    t0 = time.time()
    for _ in range(epochs):
        tot = 0.0
        nb = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                p1 = part1.encode(batch["image"], batch["report_tokens"], batch["report_pad_mask"])
            out = model(p1, batch["report_geo"], batch["report_time"], batch["report_pad_mask"], batch["sitrep_tokens"])
            losses = part2_loss(out, batch["sitrep_tokens"], batch["damage"], batch["bag_label"],
                                pad_id=tok.pad_id, lambda_d=l2.get("lambda_d", 1.0),
                                lambda_b=l2.get("lambda_b", 1.0), lambda_g=0.0)
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            tot += float(losses["loss"].detach())
            nb += 1
        history.append(tot / max(nb, 1))
    return history, time.time() - t0


@torch.no_grad()
def evaluate(cfg, part1, model, ds, tok, device):
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    part1.eval()
    model.eval()
    hyps = []
    refs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        p1 = part1.encode(batch["image"], batch["report_tokens"], batch["report_pad_mask"])
        o = model.infer(p1, batch["report_geo"], batch["report_time"], batch["report_pad_mask"], beam_size=1)
        for i in range(batch["image"].shape[0]):
            hyps.append(tok.decode(o["tokens"][i].tolist()))
            refs.append(tok.decode(batch["sitrep_tokens"][i].tolist()))
    got = {"counts": 0, "bridge": 0, "level": 0}
    ref_ok = {"counts": 0, "bridge": 0, "level": 0}
    for h, r in zip(hyps, refs):
        g = fields(h)
        t = fields(r)
        for k in got:
            if t[k] is not None:
                ref_ok[k] += 1
                if g[k] == t[k]:
                    got[k] += 1
    return {
        "bleu": float(bleu(hyps, refs)),
        "distinct": len(set(hyps)),
        "n": len(hyps),
        "repetition": sum(repetition(h) for h in hyps) / max(len(hyps), 1),
        "bridge_acc": got["bridge"] / max(ref_ok["bridge"], 1),
        "level_acc": got["level"] / max(ref_ok["level"], 1),
        "counts_acc": got["counts"] / max(ref_ok["counts"], 1),
        "sample": hyps[0],
        "reference": refs[0],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny.yaml")
    ap.add_argument("--out", default="runs/sweep")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    seed_everything(cfg["seed"])
    tok, _syn, splits = build_tokenizer_and_splits(cfg, vocab_path=str(out_dir / "vocab.json"))
    if isinstance(tok, TinyTokenizer):
        tok.vocab.save(str(out_dir / "vocab.json"))
    train_ds, _val, test_ds = build_datasets(cfg, tok, splits)

    seed_everything(cfg["seed"])
    part1, _h = build_models(cfg, tok)
    part1.to(device)
    print("training stage 1 once ...", flush=True)
    train_stage1(cfg, part1, train_ds, tok, device)
    torch.save(part1.state_dict(), out_dir / "part1.pt")

    rows = []
    for ep in BUDGETS:
        seed_everything(cfg["seed"])
        p1, hybrid = build_models(cfg, tok)
        p1.load_state_dict(torch.load(out_dir / "part1.pt", map_location=device))
        p1.to(device)
        hybrid.to(device)
        hist, secs = train_part2(cfg, p1, hybrid, train_ds, tok, device, ep)
        res = evaluate(cfg, p1, hybrid, test_ds, tok, device)
        res["epochs"] = ep
        res["final_loss"] = hist[-1]
        res["seconds"] = secs
        rows.append(res)
        print(f"[{ep:3d} ep] loss={hist[-1]:.3f} bleu={res['bleu']:5.2f} "
              f"rep={res['repetition']:.2f} distinct={res['distinct']}/{res['n']} "
              f"bridge={res['bridge_acc']:.2f} level={res['level_acc']:.2f} counts={res['counts_acc']:.2f}",
              flush=True)
        print(f"        -> {res['sample'][:110]}", flush=True)

    with open(out_dir / "budget_sweep.json", "w") as f:
        json.dump({"reference": rows[0]["reference"], "rows": rows}, f, indent=1)
    print("\nwrote", out_dir / "budget_sweep.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
