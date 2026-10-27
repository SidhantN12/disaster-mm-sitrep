"""Inference service for the multimodal SITREP model.

Loads the checkpoints once at startup and serves single-tile inference over
HTTP. The response carries the evidence as well as the prediction - which
report each claim leaned on, and how each generated word split its attention
between imagery and text - because for this domain an unexplained SITREP is
not actionable.

Configured by environment:

    MODEL_DIR   directory holding part1_stage1.pt and model_<fusion>.pt
                (the layout fusion_study.py writes). Default /weights.
    CONFIG      model config to build from. Default configs/base.yaml.
    FUSION      early | late | hybrid. Default hybrid.
    DEVICE      cpu | cuda. Default cpu.
"""
from __future__ import annotations

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from disaster_mm.builder import build_models, build_synthetic_config
from disaster_mm.data.dataset import DatasetConfig
from disaster_mm.data.synthetic import generate_tile
from disaster_mm.data.tokenizer import build_tokenizer
from disaster_mm.data.vocab import Vocab
from disaster_mm.models.fusion_variants import build_fusion_model
from disaster_mm.utils import load_config, seed_everything

MODEL_DIR = os.environ.get('MODEL_DIR', '/weights')
CONFIG = os.environ.get('CONFIG', 'configs/base.yaml')
FUSION = os.environ.get('FUSION', 'hybrid')
DEVICE = os.environ.get('DEVICE', 'cpu')
DAMAGE = ['none', 'minor', 'major', 'destroyed']

M = {}


def load_models():
    """Build and load once. Called at startup, never per request."""
    cfg = load_config(CONFIG)
    device = torch.device(DEVICE)
    seed_everything(cfg.get('seed', 42))

    # Build the tokenizer directly rather than via build_tokenizer_and_splits,
    # which generates the whole dataset to do it. At base scale that is 24,000
    # tiles of 6x256x256 float32 - about 38 GB resident - purely to produce a
    # tokenizer the service could have had for free. A serving process needs no
    # splits at all.
    text_variant = cfg['model']['text_variant']
    if text_variant == 'tiny':
        vocab_path = os.path.join(MODEL_DIR, 'vocab.json')
        if not os.path.exists(vocab_path):
            raise RuntimeError(f'missing {vocab_path} - the tiny tokenizer needs its vocab')
        tokenizer = build_tokenizer('tiny', Vocab.load(vocab_path))
    else:
        tokenizer = build_tokenizer('bert')
    syn_cfg = build_synthetic_config(cfg)

    part1, hybrid = build_models(cfg, tokenizer)
    p1_path = os.path.join(MODEL_DIR, 'part1_stage1.pt')
    p2_path = os.path.join(MODEL_DIR, f'model_{FUSION}.pt')
    for p in (p1_path, p2_path):
        if not os.path.exists(p):
            raise RuntimeError(f'missing checkpoint {p} - is MODEL_DIR mounted?')

    part1.load_state_dict(torch.load(p1_path, map_location='cpu'))
    model = hybrid if FUSION == 'hybrid' else build_fusion_model(FUSION, cfg, part1, tokenizer)
    model.load_state_dict(torch.load(p2_path, map_location='cpu'))

    part1.to(device).eval()
    model.to(device).eval()

    M.update({
        'cfg': cfg,
        'device': device,
        'tokenizer': tokenizer,
        'syn_cfg': syn_cfg,
        'part1': part1,
        'model': model,
        'ds_cfg': DatasetConfig(
            max_report_tokens=cfg['data'].get('max_report_tokens', 64),
            max_reports=cfg['data'].get('max_reports', 7),
            max_sitrep_tokens=cfg['data'].get('max_sitrep_tokens', 96),
        ),
        'params': sum(p.numel() for p in part1.parameters()) + sum(p.numel() for p in model.parameters()),
    })


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_models()
    yield
    M.clear()


app = FastAPI(
    title='Multimodal SITREP inference',
    description='Single-tile situation-report generation from pre/post imagery and geotagged field reports.',
    version='1.0.0',
    lifespan=lifespan,
)


class Report(BaseModel):
    text: str
    dx: float = Field(0.0, description='east-west offset from tile centre, km')
    dy: float = Field(0.0, description='north-south offset from tile centre, km')
    dt: float = Field(0.0, description='time offset from the event, hours')


class SitrepRequest(BaseModel):
    reports: List[Report] = Field(default_factory=list)
    image_npy_b64: Optional[str] = Field(
        None, description='base64 of a .npy holding a (6, 256, 256) float array. Omitted: a synthetic tile is generated.'
    )
    beam_size: int = 1


def decode_image(b64: Optional[str]) -> np.ndarray:
    if b64 is None:
        rng = np.random.default_rng(0)
        return generate_tile(rng, M['syn_cfg']).image
    try:
        # allow_pickle stays at its default of False: a .npy arriving over the
        # wire is untrusted, and a pickled one would execute on load.
        arr = np.load(io.BytesIO(base64.b64decode(b64)))
    except Exception:
        raise HTTPException(400, 'image_npy_b64 is not a readable, unpickled .npy array')
    arr = arr.astype(np.float32)
    if arr.ndim != 3 or arr.shape[0] != 6:
        raise HTTPException(400, f'expected a (6, H, W) array, got {arr.shape}')
    return arr


@app.get('/health')
def health():
    ok = bool(M)
    return {
        'status': 'ok' if ok else 'loading',
        'fusion': FUSION,
        'device': DEVICE,
        'config': CONFIG,
        'parameters': M.get('params'),
    }


@app.post('/sitrep')
@torch.no_grad()
def sitrep(req: SitrepRequest):
    if not M:
        raise HTTPException(503, 'models still loading')
    t0 = time.perf_counter()
    dev = M['device']
    tok = M['tokenizer']
    ds = M['ds_cfg']

    reports = req.reports[: ds.max_reports]
    image = torch.from_numpy(decode_image(req.image_npy_b64)).unsqueeze(0).to(dev)

    k = len(reports)
    if k:
        report_tokens = torch.tensor(
            [tok.encode(r.text, ds.max_report_tokens) for r in reports], dtype=torch.long
        ).unsqueeze(0).to(dev)
        report_geo = torch.tensor([[r.dx, r.dy] for r in reports], dtype=torch.float32).unsqueeze(0).to(dev)
        report_time = torch.tensor([[r.dt] for r in reports], dtype=torch.float32).unsqueeze(0).to(dev)
    else:
        report_tokens = torch.zeros((1, 0, ds.max_report_tokens), dtype=torch.long, device=dev)
        report_geo = torch.zeros((1, 0, 2), dtype=torch.float32, device=dev)
        report_time = torch.zeros((1, 0, 1), dtype=torch.float32, device=dev)
    pad_mask = torch.zeros((1, k), dtype=torch.bool, device=dev)

    p1 = M['part1'].encode(image, report_tokens, pad_mask)
    out = M['model'].infer(p1, report_geo, report_time, pad_mask, beam_size=req.beam_size)

    text = tok.decode(out['tokens'][0].cpu().tolist())
    dmg = int(out['damage_logits'][0].argmax())
    bag_p = float(torch.sigmoid(out['bag_logit'][0]))

    weights = []
    alpha = out.get('mil_alpha')
    if alpha is not None and k:
        for j, r in enumerate(reports):
            if j < alpha.shape[1]:
                weights.append({'text': r.text, 'weight': round(float(alpha[0, j]), 4)})

    att = []
    cw = out.get('cross_attn')
    enc = out.get('encoder_out', {})
    n_patch = enc.get('n_patches', 0)
    n_ctx = enc.get('n_context', 0)
    if cw is not None and n_patch:
        toks = out['tokens'][0].cpu().tolist()
        for ti, t in enumerate(toks):
            if ti == 0 or t == tok.pad_id:
                continue
            word = tok.decode([t]).strip()
            if not word:
                continue
            row = cw[0, min(ti, cw.shape[1] - 1)].float().cpu()
            img_mass = float(row[:n_patch].sum())
            rep_mass = float(row[n_patch + n_ctx:].sum())
            tot = img_mass + rep_mass
            att.append({
                'word': word,
                'from_image': round(img_mass / tot, 4) if tot else None,
                'from_reports': round(rep_mass / tot, 4) if tot else None,
            })

    return {
        'sitrep': text,
        'damage': {'class': dmg, 'label': DAMAGE[dmg]},
        'report_relevance': {'label': int(bag_p > 0.5), 'probability': round(bag_p, 4)},
        'report_weights': weights,
        'attention': att,
        'n_reports': k,
        'latency_ms': round((time.perf_counter() - t0) * 1000, 1),
    }
