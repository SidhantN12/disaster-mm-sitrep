"""Single-tile inference CLI.

    python -m disaster_mm.infer --config configs/tiny.yaml --stage 3 \
        --image path/to/tile.npy --reports reports.json --out_dir out/

``--image`` may be a ``.npy`` file holding a (6, 256, 256) float array; if
omitted, a fresh synthetic tile is generated instead so the command is
runnable end-to-end with zero external inputs. ``reports.json`` is a list of
``{"text": ..., "dx": ..., "dy": ..., "dt": ...}`` objects (dx/dy in km,
dt in hours); if omitted, an empty report list (k=0) is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .builder import build_models, build_tokenizer_and_splits
from .data.dataset import DatasetConfig
from .data.synthetic import generate_tile
from .retrieval_check import embed_sentences, flag_unverified, split_sentences
from .utils import get_device, load_config, seed_everything
from .viz import save_attention_heatmap


def _load_image(path: str | None, syn_cfg) -> np.ndarray:
    if path is None:
        rng = np.random.default_rng(0)
        return generate_tile(rng, syn_cfg).image
    return np.load(path).astype(np.float32)


def _load_reports(path: str | None) -> list[dict]:
    if path is None:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--image", default=None)
    parser.add_argument("--reports", default=None)
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--beam_size", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg)

    run_dir = Path(cfg.get("checkpoints", {}).get("dir", "runs/default"))
    vocab_path = str(run_dir / "vocab.json")
    tokenizer, syn_cfg, _splits = build_tokenizer_and_splits(cfg, vocab_path=vocab_path)
    ds_cfg = DatasetConfig(
        max_report_tokens=cfg["data"].get("max_report_tokens", 16),
        max_reports=cfg["data"].get("max_reports", 8),
        max_sitrep_tokens=cfg["data"].get("max_sitrep_tokens", 40),
    )

    part1, part2 = build_models(cfg, tokenizer)
    if args.stage >= 3 and (run_dir / "part1_stage3.pt").exists():
        part1.load_state_dict(torch.load(run_dir / "part1_stage3.pt", map_location=device))
        part2.load_state_dict(torch.load(run_dir / "part2_stage3.pt", map_location=device))
    else:
        part1.load_state_dict(torch.load(run_dir / "part1_stage1.pt", map_location=device))
        part2.load_state_dict(torch.load(run_dir / "part2_stage2.pt", map_location=device))
    part1.to(device).eval()
    part2.to(device).eval()

    image = torch.from_numpy(_load_image(args.image, syn_cfg)).unsqueeze(0).to(device)
    reports = _load_reports(args.reports)

    if reports:
        report_tokens = torch.tensor(
            [tokenizer.encode(r["text"], ds_cfg.max_report_tokens) for r in reports], dtype=torch.long
        ).unsqueeze(0).to(device)
        report_geo = torch.tensor([[r["dx"], r["dy"]] for r in reports], dtype=torch.float32).unsqueeze(0).to(device)
        report_time = torch.tensor([[r["dt"]] for r in reports], dtype=torch.float32).unsqueeze(0).to(device)
        report_pad_mask = torch.zeros((1, len(reports)), dtype=torch.bool).to(device)
    else:
        report_tokens = torch.zeros((1, 0, ds_cfg.max_report_tokens), dtype=torch.long).to(device)
        report_geo = torch.zeros((1, 0, 2), dtype=torch.float32).to(device)
        report_time = torch.zeros((1, 0, 1), dtype=torch.float32).to(device)
        report_pad_mask = torch.zeros((1, 0), dtype=torch.bool).to(device)

    with torch.no_grad():
        p1_out = part1.encode(image, report_tokens, report_pad_mask)
        out = part2.infer(p1_out, report_geo, report_time, report_pad_mask, beam_size=args.beam_size)

    sitrep = tokenizer.decode(out["tokens"][0].cpu().tolist())
    damage_class = out["damage_logits"][0].argmax().item()
    damage_name = ["none", "minor", "major", "destroyed"][damage_class]

    sentences = split_sentences(sitrep)
    sent_embeds = embed_sentences(part1, tokenizer, sentences, ds_cfg.max_report_tokens, device)
    flags = flag_unverified(sent_embeds, p1_out["z_s_i"][0], out["mil_alpha"][0])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if out.get("image_patch_mass") is not None and out["cross_attn"] is not None:
        last_step = min(2, out["cross_attn"].shape[1] - 1)
        patch_attn = out["cross_attn"][0, last_step, : p1_out["patch_tokens"].shape[1]]
        heatmap_path = save_attention_heatmap(image[0, 3:6], patch_attn, str(out_dir / "attention_heatmap.png"))
    else:
        heatmap_path = None

    print("SITREP:", sitrep)
    print("Damage class:", damage_name)
    print("Unverified sentence flags:")
    for s, f in zip(sentences, flags):
        print(f"  [{'UNVERIFIED' if f else 'ok'}] {s}")
    if heatmap_path:
        print("Saved heatmap to", heatmap_path)


if __name__ == "__main__":
    main()
