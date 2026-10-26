"""Hard gates for the autonomous SITREP pipeline.

Exits 0 only when the run produced something genuinely presentable. Every
failure prints VERIFY-FAIL plus a specific reason, so an agent reading stdout
knows what to repair instead of guessing.

    python verify.py --run /workspace/runs/base_fusion \
                     --data /workspace/demo_data.json \
                     --html /workspace/sitrep_demo.html \
                     --min-tiles 16

Gate 5 (degeneracy) is the one that matters. Everything else checks plumbing;
gate 5 checks whether the model actually learned to write.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CKPTS = ['vocab.json', 'part1_stage1.pt', 'model_early.pt', 'model_late.pt', 'model_hybrid.pt']
FUSIONS = ['early', 'late', 'hybrid']

fails = []
warns = []


def fail(gate, msg):
    fails.append(f'VERIFY-FAIL [{gate}] {msg}')


def warn(gate, msg):
    warns.append(f'VERIFY-WARN [{gate}] {msg}')


def repetition(text):
    """Fraction of tokens that simply repeat the previous token."""
    toks = text.split()
    if len(toks) < 2:
        return 1.0
    rep = 0
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            rep += 1
    return rep / (len(toks) - 1)


def gate_checkpoints(run, min_ckpt):
    if not run.is_dir():
        fail('checkpoints', f'{run} does not exist - training never started')
        return
    for name in CKPTS:
        p = run / name
        if not p.exists():
            fail('checkpoints', f'missing {p} - fusion_study.py did not finish')
            continue
        size = p.stat().st_size
        # a base-config checkpoint is hundreds of MB; vocab.json is small by design
        if name.endswith('.pt') and size < min_ckpt:
            fail('checkpoints', f'{p} is {size / 1e6:.2f} MB, under the {min_ckpt / 1e6:.0f} MB floor - truncated, or you trained the tiny config by mistake')
        elif size == 0:
            fail('checkpoints', f'{p} is empty')


def gate_results(run):
    p = run / 'fusion_results.json'
    if not p.exists():
        fail('results', f'missing {p}')
        return
    try:
        r = json.loads(p.read_text())
    except Exception as e:
        fail('results', f'{p} is not valid JSON: {e}')
        return
    if 'baselines' not in r:
        warn('results', 'no constant-predictor baselines recorded')
    for f in FUSIONS:
        if f not in json.dumps(r):
            fail('results', f'no results for the {f} variant')


def gate_data(data_path, min_tiles):
    if not data_path.exists():
        fail('data', f'missing {data_path} - export step never ran')
        return None
    mb = data_path.stat().st_size / 1e6
    if mb > 25:
        fail('data', f'{data_path} is {mb:.1f} MB - pass a smaller --limit')
    try:
        d = json.loads(data_path.read_text())
    except Exception as e:
        fail('data', f'{data_path} is not valid JSON: {e}')
        return None
    tiles = d.get('tiles', [])
    if len(tiles) < min_tiles:
        fail('data', f'only {len(tiles)} tiles, expected at least {min_tiles}')
        return d
    grid = d.get('meta', {}).get('patch_grid', 0)
    if grid != 16:
        fail('data', f'patch_grid is {grid}, the page CSS expects 16')
    for t in tiles[:min_tiles]:
        for layer in ('pre', 'post'):
            if not str(t.get(layer, '')).startswith('data:image/png;base64,'):
                fail('data', f"tile {t.get('id')} has no {layer} image")
        for f in FUSIONS:
            s = t.get('variants', {}).get(f, {}).get('sitrep', '')
            if not s.strip():
                fail('data', f"tile {t.get('id')} has an empty {f} SITREP")
    hyb_att = [t for t in tiles if t.get('attention')]
    if not hyb_att:
        fail('data', 'no cross-attention recorded - the page cannot draw its headline feature')
    else:
        a = hyb_att[0]['attention'][0]
        if len(a.get('map', [])) != grid * grid:
            fail('data', f"attention map is {len(a.get('map', []))} long, expected {grid * grid}")
    return d


def gate_quality(d, min_tiles):
    """The gate that distinguishes a trained model from a broken one."""
    if not d:
        return
    tiles = d.get('tiles', [])[:min_tiles]
    if not tiles:
        return
    for f in FUSIONS:
        outs = []
        for t in tiles:
            outs.append(t.get('variants', {}).get(f, {}).get('sitrep', ''))
        outs = [o for o in outs if o.strip()]
        if not outs:
            continue
        distinct = len(set(outs))
        rep = sum(repetition(o) for o in outs) / len(outs)
        avg_len = sum(len(o.split()) for o in outs) / len(outs)
        line = f'{f}: {distinct}/{len(outs)} distinct, repetition {rep:.2f}, {avg_len:.0f} words'
        # Only the hybrid variant gates the run: it is what the demo showcases.
        # Early and late fusion collapsing toward one sentence per tile is a
        # REAL measured finding of this project, not a broken run.
        if f == 'hybrid':
            if rep > 0.25:
                fail('quality', f'{line} - degenerate output, the model is undertrained')
            elif distinct == 1:
                fail('quality', f'{line} - one sentence for every tile, hybrid collapsed')
            elif avg_len < 8:
                fail('quality', f'{line} - outputs are too short to be SITREPs')
            else:
                print(f'  ok  {line}')
        else:
            print(f'  ok  {line}')
            if distinct <= 2:
                warn('quality', f'{line} - low diversity, expected for {f} fusion, not a blocker')
            if rep > 0.25:
                warn('quality', f'{line} - {f} fusion is degenerate; report it, do not fix it')


def gate_html(html_path):
    if not html_path.exists():
        fail('html', f'missing {html_path} - build_demo.py never ran')
        return
    kb = html_path.stat().st_size / 1024
    if kb < 100:
        fail('html', f'{html_path} is {kb:.0f} KB - the data was not inlined')
        return
    txt = html_path.read_text()
    for needle in ['<!doctype html>', 'window.__DEMO__', 'attGrid']:
        if needle not in txt:
            fail('html', f'{html_path} does not contain {needle}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='/workspace/runs/base_fusion')
    ap.add_argument('--data', default='/workspace/demo_data.json')
    ap.add_argument('--html', default='/workspace/sitrep_demo.html')
    ap.add_argument('--min-tiles', type=int, default=16)
    ap.add_argument('--skip-html', action='store_true')
    ap.add_argument('--min-ckpt-mb', type=float, default=10.0,
                    help='floor for .pt files; base config is 250-800 MB, tiny is under 1')
    args = ap.parse_args()

    print('== verifying the run ==')
    gate_checkpoints(Path(args.run), args.min_ckpt_mb * 1e6)
    gate_results(Path(args.run))
    d = gate_data(Path(args.data), args.min_tiles)
    gate_quality(d, args.min_tiles)
    if not args.skip_html:
        gate_html(Path(args.html))

    for w in warns:
        print(w)
    if fails:
        print()
        for f in fails:
            print(f)
        print(f'\n{len(fails)} gate(s) failed. The run is NOT presentable.')
        return 1
    print('\nVERIFY-OK  all gates passed - this run is presentable.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
