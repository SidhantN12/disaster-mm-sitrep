"""Assemble the standalone SITREP demo page from its parts.

Produces ONE self-contained .html file with the model outputs inlined, so it
opens straight off disk with no server and no network. Unlike the published
artifact (where the host supplies the document skeleton) this writes a complete
document: doctype, charset, viewport.

    python build_demo.py --data demo_data.json --out sitrep_demo.html

Needs page_head.html, page_body.html and page_script.js beside it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SKELETON = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{head}
</head>
<body>
{body}
<script>window.__DEMO__={data};</script>
<script>
{script}
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='demo_data.json')
    ap.add_argument('--out', default='sitrep_demo.html')
    ap.add_argument('--src', default='.', help='directory holding the three page parts')
    args = ap.parse_args()

    src = Path(args.src)
    parts = {}
    for key, name in [('head', 'page_head.html'), ('body', 'page_body.html'), ('script', 'page_script.js')]:
        p = src / name
        if not p.exists():
            print(f'missing {p}', file=sys.stderr)
            return 1
        parts[key] = p.read_text()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f'missing {data_path}', file=sys.stderr)
        return 1
    data = json.loads(data_path.read_text())
    n = len(data.get('tiles', []))
    if n == 0:
        print('no tiles in the data file', file=sys.stderr)
        return 1

    # </script> anywhere inside the JSON would close the tag early
    compact = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')

    html = SKELETON.format(
        head=parts['head'],
        body=parts['body'],
        data=compact,
        script=parts['script'],
    )
    out = Path(args.out)
    out.write_text(html)
    mb = out.stat().st_size / 1e6
    print(f'wrote {out}  {mb:.2f} MB  {n} tiles')
    print(f"budget: {data.get('meta', {}).get('budget', 'unknown')}")
    print('open it by double-clicking - no server needed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
