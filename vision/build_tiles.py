"""Generate the tile set for the product-concept console.

Everything written here is ILLUSTRATIVE. The imagery is drawn by aerial.py and
the SITREPs are composed from the ground-truth damage levels, not produced by
the trained model. The concept page says so on every screen.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from aerial import make_pair

OUT = Path('/home/claude/vision')
SIZE = 448
RNG = np.random.default_rng

PLACES = [
    ('Riverside North', 'RN-04'), ('Mill Quarter', 'MQ-11'), ('Old Market', 'OM-02'),
    ('Canal Row', 'CR-07'), ('Station Flats', 'SF-19'), ('Hilltop East', 'HE-05'),
    ('Lower Ford', 'LF-13'), ('Granary Lane', 'GL-08'), ('Church Field', 'CF-16'),
]

REPORTS = {
    'rescue': ["people trapped under the {k} on {street}", "shouting heard inside collapsed {k}", "neighbours digging at {k}, need lifting gear"],
    'medical': ["two injured outside the {k}, bleeding", "clinic on {street} is overwhelmed", "need bandages and a medic at {street}"],
    'water': ["mains burst at {street}, no drinking water", "well water is brown near {k}", "water tank cracked by the {k}"],
    'shelter': ["about forty people sleeping in the open at {street}", "roofs gone on {street}, families outside", "need tarpaulins near the {k}"],
    'food': ["no food distribution reached {street} yet", "market stalls destroyed, nothing to buy", "children not fed since yesterday at {street}"],
    'access': ["bridge at {street} is down, cannot cross", "road blocked by debris near the {k}", "truck cannot reach {street}"],
}
STREETS = ['Mill Lane', 'River Road', 'Market Street', 'Canal Row', 'Station Road', 'Church Walk', 'Granary Lane']
KINDS = ['warehouse', 'clinic', 'apartment block', 'market hall', 'school', 'grain store']


def uri(im):
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=72, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def sitrep(counts, bridge, needs):
    d, mj, mn, ok = counts
    lvl = 'destroyed' if d >= 5 else ('major' if d + mj >= 6 else 'minor')
    need = ', '.join(needs) if needs else 'none reported'
    return (f"Tile assessment: {d} destroyed, {mj} major damage, {mn} minor damage, "
            f"{ok} undamaged structures. Bridge is {bridge}. "
            f"Overall damage level: {lvl}. Reported needs: {need}.")


def main():
    rng = RNG(11)
    tiles = []
    profiles = [
        (0.55, 'destroyed', ['rescue', 'medical', 'water']),
        (0.42, 'damaged', ['rescue', 'shelter']),
        (0.30, 'intact', ['medical', 'food']),
        (0.62, 'destroyed', ['rescue', 'access', 'water']),
        (0.18, 'intact', ['food']),
        (0.08, 'intact', []),
        (0.48, 'damaged', ['shelter', 'water']),
        (0.25, 'intact', ['food', 'shelter']),
        (0.12, 'intact', []),
    ]
    for i, (sev, bridge, need_kinds) in enumerate(profiles):
        levels = {}
        for b in range(17):
            r = rng.random()
            if r < sev * 0.55:
                levels[b] = 3
            elif r < sev * 0.85:
                levels[b] = 2
            elif r < sev * 1.05:
                levels[b] = 1
            else:
                levels[b] = 0
        pre, post, builds = make_pair(100 + i * 7, levels, size=SIZE)
        vals = list(levels.values())
        counts = (vals.count(3), vals.count(2), vals.count(1), vals.count(0))

        reps = []
        for k in need_kinds:
            for tmpl in rng.choice(REPORTS[k], size=int(rng.integers(1, 3)), replace=False):
                reps.append({
                    'need': k,
                    'text': tmpl.format(k=str(rng.choice(KINDS)), street=str(rng.choice(STREETS))),
                    'dist': round(float(rng.uniform(0.04, 0.38)), 2),
                    'age': round(float(rng.uniform(0.3, 7.5)), 1),
                    'w': 0.0,
                })
        if reps:
            raw = rng.random(len(reps)) + 0.6
            raw = raw / raw.sum()
            for r, w in zip(reps, raw):
                r['w'] = round(float(w), 3)

        needs = sorted(set(r['need'] for r in reps))
        name, code = PLACES[i]
        severity = counts[0] * 3 + counts[1] * 2 + counts[2]
        access = 0 if bridge == 'destroyed' else (1 if bridge == 'damaged' else 2)
        priority = round(severity * 1.6 + len(reps) * 2.4 + (2 - access) * 6, 1)
        tiles.append({
            'id': i, 'name': name, 'code': code,
            'pre': uri(pre), 'post': uri(post),
            'counts': counts, 'bridge': bridge, 'needs': needs,
            'reports': reps, 'priority': priority,
            'severity': severity, 'access': access,
            'sitrep': sitrep(counts, bridge, needs),
            'conf': {
                'counts': round(float(rng.uniform(0.74, 0.93)), 2),
                'bridge': round(float(rng.uniform(0.81, 0.97)), 2),
                'level': round(float(rng.uniform(0.88, 0.98)), 2),
                'needs': round(float(rng.uniform(0.79, 0.95)), 2),
            },
            'boxes': [{'x': round(b[0][0] / SIZE, 4), 'y': round(b[0][1] / SIZE, 4),
                       'w': round((b[0][2] - b[0][0]) / SIZE, 4),
                       'h': round((b[0][3] - b[0][1]) / SIZE, 4),
                       'd': levels[j]} for j, b in enumerate(builds)],
        })
        print(f"  {code} {name:<16} dmg={counts} bridge={bridge:<9} reports={len(reps)} prio={priority}")

    data = {'tiles': tiles, 'generated': 'illustrative'}
    p = OUT / 'vision_data.json'
    p.write_text(json.dumps(data, separators=(',', ':')))
    print(f"\nwrote {p}  {p.stat().st_size/1e6:.2f} MB  {len(tiles)} tiles")


if __name__ == '__main__':
    main()
