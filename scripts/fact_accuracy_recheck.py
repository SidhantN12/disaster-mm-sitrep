"""Re-measure template fact accuracy with a tokenizer-aware parser.

``metrics.count_fact_accuracy`` matches case-sensitive, punctuated patterns
("Bridge is", "Overall damage level:"), but the tiny tokenizer lower-cases
text and strips punctuation, so the decoded *reference* never parses either.
``parseable_fraction`` is therefore structurally 0.0 for every model at every
budget, and the three fact accuracies are 0.0 by construction rather than by
measurement. This script re-parses the same generations with patterns that
match the decoded form, and reports what the metric should have said.

    python scripts/fact_accuracy_recheck.py --run runs/long_s42 --config configs/tiny_long.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from disaster_mm.builder import build_datasets, build_models, build_tokenizer_and_splits
from disaster_mm.data.dataset import collate_fn
from disaster_mm.models.fusion_variants import build_fusion_model
from disaster_mm.utils import load_config, seed_everything

# Decoded form: lower-cased, punctuation stripped by the word-level tokenizer.
COUNT = re.compile(r"(\d+)\s+destroyed\s+(\d+)\s+major damage\s+(\d+)\s+minor damage\s+(\d+)\s+undamaged")
BRIDGE = re.compile(r"bridge is (\w+)")
LEVEL = re.compile(r"overall damage level (\w+)")


def parse(text: str):
    m, b, lv = COUNT.search(text), BRIDGE.search(text), LEVEL.search(text)
    return {
        "counts": tuple(int(x) for x in m.groups()) if m else None,
        "bridge": b.group(1) if b else None,
        "level": lv.group(1) if lv else None,
    }


def fact_accuracy(hyps, refs):
    n = len(hyps)
    got = {"counts": 0, "bridge": 0, "level": 0}
    ref_has = {"counts": 0, "bridge": 0, "level": 0}
    for h, r in zip(hyps, refs):
        g, t = parse(h), parse(r)
        for k in got:
            if t[k] is not None:
                ref_has[k] += 1
                if g[k] == t[k]:
                    got[k] += 1
    return {
        "n": n,
        "reference_parseable": {k: ref_has[k] / max(n, 1) for k in ref_has},
        "accuracy": {k: got[k] / max(ref_has[k], 1) for k in got},
    }


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    run = Path(args.run)
    device = torch.device("cpu")

    seed_everything(cfg["seed"])
    tokenizer, _s, splits = build_tokenizer_and_splits(cfg, vocab_path=str(run / "vocab.json"))
    _tr, _va, test_ds = build_datasets(cfg, tokenizer, splits)

    out = {}
    for fusion in ["early", "late", "hybrid"]:
        seed_everything(cfg["seed"])
        part1, hybrid = build_models(cfg, tokenizer)
        part1.load_state_dict(torch.load(run / "part1_stage1.pt", map_location=device))
        model = hybrid if fusion == "hybrid" else build_fusion_model(fusion, cfg, part1, tokenizer)
        model.load_state_dict(torch.load(run / f"model_{fusion}.pt", map_location=device))
        part1.eval()
        model.eval()

        hyps, refs = [], []
        for batch in DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn):
            p1 = part1.encode(batch["image"], batch["report_tokens"], batch["report_pad_mask"])
            o = model.infer(p1, batch["report_geo"], batch["report_time"], batch["report_pad_mask"], beam_size=1)
            for i in range(batch["image"].shape[0]):
                hyps.append(tokenizer.decode(o["tokens"][i].tolist()))
                refs.append(tokenizer.decode(batch["sitrep_tokens"][i].tolist()))
        out[fusion] = fact_accuracy(hyps, refs)
        out[fusion]["example"] = hyps[0]
    out["reference_example"] = refs[0]
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
