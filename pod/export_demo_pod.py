"""Export real model outputs for the interactive presentation frontend.

Pod-ready version: takes paths on the command line, runs on the GPU if one is
present, and caps how many test tiles go into the JSON (the base config has a
2,000-tile test split and the page only shows a strip of them).

    python export_demo_pod.py \
        --config configs/base.yaml \
        --run /workspace/runs/base_fusion \
        --out /workspace/demo_data.json \
        --limit 16
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from disaster_mm.builder import build_datasets, build_models, build_tokenizer_and_splits
from disaster_mm.data.dataset import collate_fn
from disaster_mm.models.fusion_variants import build_fusion_model
from disaster_mm.utils import load_config, seed_everything

DAMAGE = ['none', 'minor', 'major', 'destroyed']
MAX_ATT_TOKENS = 14
THUMB = 192
FUSIONS = ['early', 'late', 'hybrid']


def png_uri(arr_chw: np.ndarray) -> str:
    """arr_chw: (3, H, W) float in [0,1] -> base64 PNG data URI."""
    a = np.clip(arr_chw.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    im = Image.fromarray(a).resize((THUMB, THUMB), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/base.yaml')
    ap.add_argument('--run', required=True, help='fusion_study.py --out directory')
    ap.add_argument('--out', default='demo_data.json')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=16, help='tiles to export')
    args = ap.parse_args()

    run = Path(args.run)
    for name in ['vocab.json', 'part1_stage1.pt', 'model_early.pt', 'model_late.pt', 'model_hybrid.pt']:
        if not (run / name).exists():
            print(f'missing {run / name} - did fusion_study.py finish?', file=sys.stderr)
            return 1

    cfg = load_config(args.config)
    cfg['seed'] = args.seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    seed_everything(args.seed)

    tokenizer, _syn, splits = build_tokenizer_and_splits(cfg, vocab_path=str(run / 'vocab.json'))
    _tr, _va, test_ds = build_datasets(cfg, tokenizer, splits)
    test_samples = splits[2]

    models = {}
    part1 = None
    for fusion in FUSIONS:
        seed_everything(args.seed)
        p1, hyb = build_models(cfg, tokenizer)
        p1.load_state_dict(torch.load(run / 'part1_stage1.pt', map_location='cpu'))
        m = hyb if fusion == 'hybrid' else build_fusion_model(fusion, cfg, p1, tokenizer)
        m.load_state_dict(torch.load(run / f'model_{fusion}.pt', map_location='cpu'))
        p1.to(device)
        m.to(device)
        p1.eval()
        m.eval()
        models[fusion] = m
        part1 = p1

    tiles = []
    idx = 0
    grid = 16
    for batch in DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn):
        dev_batch = {}
        for key in batch:
            dev_batch[key] = batch[key].to(device)

        p1out = part1.encode(dev_batch['image'], dev_batch['report_tokens'], dev_batch['report_pad_mask'])
        outs = {}
        for fusion in FUSIONS:
            outs[fusion] = models[fusion].infer(
                p1out,
                dev_batch['report_geo'],
                dev_batch['report_time'],
                dev_batch['report_pad_mask'],
                beam_size=1,
            )

        for i in range(batch['image'].shape[0]):
            if idx >= args.limit:
                break
            s = test_samples[idx]
            img = batch['image'][i].numpy()
            k = int(batch['num_reports'][i])

            alpha = outs['hybrid'].get('mil_alpha')
            reports = []
            for j, r in enumerate(s.reports[:k]):
                a = None
                if alpha is not None and j < alpha.shape[1]:
                    a = round(float(alpha[i, j]), 4)
                reports.append({
                    'text': r.text,
                    'relevant': bool(r.relevant),
                    'dx': round(float(r.dx), 3),
                    'dy': round(float(r.dy), 3),
                    'dt': round(float(r.dt), 2),
                    'alpha': a,
                })

            variants = {}
            for fusion in FUSIONS:
                o = outs[fusion]
                variants[fusion] = {
                    'sitrep': tokenizer.decode(o['tokens'][i].tolist()),
                    'damage': int(o['damage_logits'][i].argmax()),
                    'bag': int(torch.sigmoid(o['bag_logit'][i]) > 0.5),
                }

            # hybrid per-token attention over the image patches
            att = []
            h = outs['hybrid']
            toks = h['tokens'][i].tolist()
            cw = h.get('cross_attn')
            n_patch = h['encoder_out']['n_patches']
            n_ctx = h['encoder_out']['n_context']
            grid = int(round(n_patch ** 0.5))
            if cw is not None:
                for ti, tok in enumerate(toks):
                    if ti == 0 or tok == tokenizer.pad_id:
                        continue
                    word = tokenizer.decode([tok]).strip()
                    if not word:
                        continue
                    row = cw[i, min(ti, cw.shape[1] - 1)].float().cpu()
                    patches = row[:n_patch].numpy()
                    tot = float(patches.sum())
                    # encode each patch as a multiple of uniform attention, x100
                    uni = tot / n_patch if n_patch else 1.0
                    rel = [v / uni for v in patches] if uni else [1.0] * n_patch
                    att.append({
                        'w': word,
                        'img': round(tot, 4),
                        'rep': round(float(row[n_patch + n_ctx:].sum()), 4),
                        'map': [int(round(100 * r)) for r in rel],
                        'spread': round(float(max(rel) - min(rel)), 4),
                    })
                    if len(att) >= MAX_ATT_TOKENS:
                        break

            tiles.append({
                'id': idx,
                'pre': png_uri(img[:3]),
                'post': png_uri(img[3:]),
                'truth': {
                    'damage': int(batch['damage'][i]),
                    'bag': int(batch['bag_label'][i]),
                    'sitrep': tokenizer.decode(batch['sitrep_tokens'][i].tolist()),
                    'raw_sitrep': s.sitrep,
                },
                'n_reports': k,
                'reports': reports,
                'variants': variants,
                'attention': att,
            })
            idx += 1

        if idx >= args.limit:
            break

    meta = {
        'damage_names': DAMAGE,
        'n_tiles': len(tiles),
        'seed': args.seed,
        'budget': f"{cfg['train']['epochs_stage2']} stage-2 epochs, {cfg['data']['n_train']} tiles",
        'patch_grid': grid,
    }
    out = {'meta': meta, 'tiles': tiles}
    p = Path(args.out)
    p.write_text(json.dumps(out, separators=(',', ':')))
    print('wrote', p, round(p.stat().st_size / 1e6, 2), 'MB,', len(tiles), 'tiles')
    print('patch grid:', grid, 'x', grid)
    print('reports per tile:', [t['n_reports'] for t in tiles])
    print('example hybrid sitrep:', tiles[0]['variants']['hybrid']['sitrep'][:90])
    return 0


if __name__ == '__main__':
    sys.exit(main())
