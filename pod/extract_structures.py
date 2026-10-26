"""Annotate demo tiles with structure positions and types.

The demo page draws an icon over each building. Rather than trusting a
separate record of the layout, this recovers it from the exported image
itself: the synthetic renderer fills each structure with one exact flat
colour, so a connected-component pass per colour gives back every building's
bounding box and kind.

This only adds an overlay for the viewer. The pre/post images in
demo_data.json are left untouched, so what the page displays underneath the
icons is exactly what the model was given.

    python extract_structures.py --in demo_data.json --out demo_data.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# Exactly the fills used by synthetic._render
KIND = {
    (180, 140, 100): 'house',
    (150, 150, 160): 'warehouse',
    (200, 170, 90): 'shop',
    (220, 220, 220): 'clinic',
    (210, 120, 90): 'market',
    (160, 130, 130): 'apartments',
    (170, 140, 90): 'bridge',
}
MIN_AREA_PX = 90


def structures(data_uri: str) -> list[dict]:
    """Bounding boxes, normalised to [0,1], for every structure in a pre image."""
    raw = base64.b64decode(data_uri.split(',')[1])
    arr = np.array(Image.open(io.BytesIO(raw)).convert('RGB'))
    h_img, w_img = arr.shape[:2]
    found = []
    for rgb, kind in KIND.items():
        mask = np.all(arr == np.array(rgb, dtype=arr.dtype), axis=-1)
        if not mask.any():
            continue
        labelled, _n = ndimage.label(mask)
        for ys, xs in ndimage.find_objects(labelled):
            h = ys.stop - ys.start
            w = xs.stop - xs.start
            if h * w < MIN_AREA_PX:
                continue
            found.append({
                'k': kind,
                'x': round(xs.start / w_img, 4),
                'y': round(ys.start / h_img, 4),
                'w': round(w / w_img, 4),
                'h': round(h / h_img, 4),
            })
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', default='demo_data.json')
    ap.add_argument('--out', dest='dst', default='demo_data.json')
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f'missing {src}', file=sys.stderr)
        return 1
    data = json.loads(src.read_text())

    total = 0
    for tile in data['tiles']:
        tile['structures'] = structures(tile['pre'])
        total += len(tile['structures'])
    n = len(data['tiles'])
    Path(args.dst).write_text(json.dumps(data, separators=(',', ':')))
    print(f'annotated {n} tiles, {total} structures ({total / max(n, 1):.1f} per tile)')
    print('a 4x4 grid plus one bridge should give exactly 17.0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
