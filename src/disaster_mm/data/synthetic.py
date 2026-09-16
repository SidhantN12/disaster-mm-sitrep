"""Fully-offline procedural generator for disaster-response tiles.

Each generated tile is a self-contained sample: a pre/post satellite chip
with a river, road grid and a grid of buildings (plus a bridge, both treated
uniformly as "structures"), a set of geotagged text reports about some of
those structures (some visually obvious, some not -- so the model must read
text for those), and a templated SITREP that summarizes ground truth.

Nothing here downloads anything; generation is a pure function of a seeded
``numpy.random.Generator``, so it is safe to call from tests and CI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TILE_PX = 256
STREET_NAMES = ["Station Rd", "Market Ave", "River St", "Mill Ln", "Harbor Way", "Oak Blvd"]
BUILDING_KINDS = ["house", "warehouse", "shop", "clinic", "market stall", "apartment block"]
DAMAGE_NAMES = ["none", "minor", "major", "destroyed"]
NEEDS = ["water", "medical supplies", "food", "shelter", "rescue"]


@dataclass
class SyntheticConfig:
    grid_rows: int = 4
    grid_cols: int = 4
    tile_meters: float = 480.0
    min_reports: int = 0
    max_reports: int = 8
    max_reports_cap: int = 32
    prob_zero_reports: float = 0.2
    prob_distractor: float = 0.25
    max_report_chars_tokens: int = 64
    geo_noise_km: float = 0.01
    time_noise_hours: float = 2.0


@dataclass
class Structure:
    idx: int
    name: str
    kind: str
    cx: int
    cy: int
    half: int
    damage: int
    is_bridge: bool = False


@dataclass
class Report:
    text: str
    dx: float
    dy: float
    dt: float
    relevant: bool
    struct_idx: int | None = None


@dataclass
class TileSample:
    image: np.ndarray  # (6, H, W) float32 in [0, 1], pre RGB + post RGB
    reports: list[Report]
    sitrep: str
    damage: int
    bag_label: int
    building_masks: np.ndarray  # (S, H, W) bool
    building_damage: np.ndarray  # (S,) int64
    structure_names: list[str] = field(default_factory=list)


def _layout_structures(cfg: SyntheticConfig, rng: np.random.Generator) -> list[Structure]:
    structs: list[Structure] = []
    cell_w = TILE_PX // cfg.grid_cols
    cell_h = TILE_PX // cfg.grid_rows
    idx = 0
    for r in range(cfg.grid_rows):
        for c in range(cfg.grid_cols):
            cx = c * cell_w + cell_w // 2
            cy = r * cell_h + cell_h // 2
            half = int(min(cell_w, cell_h) * 0.32)
            street = STREET_NAMES[(r + c) % len(STREET_NAMES)]
            kind = BUILDING_KINDS[rng.integers(0, len(BUILDING_KINDS))]
            name = f"{kind} #{idx} on {street}"
            structs.append(Structure(idx, name, kind, cx, cy, half, damage=0))
            idx += 1
    # Bridge sits where the main horizontal road crosses the river band.
    bridge_cy = TILE_PX // 2
    bridge_cx = TILE_PX // 2
    structs.append(
        Structure(idx, "bridge on Station Rd", "bridge", bridge_cx, bridge_cy, half=14, damage=0, is_bridge=True)
    )
    return structs


def _river_mask(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    # Diagonal band from (0, 40) to (w, 180).
    y0, y1 = 40.0, 180.0
    center = y0 + (y1 - y0) * (xx / max(w - 1, 1))
    return np.abs(yy - center) < 12


def _road_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 2 - 5 : h // 2 + 5, :] = True
    mask[:, w // 2 - 5 : w // 2 + 5] = True
    return mask


def _assign_damage(structs: list[Structure], rng: np.random.Generator) -> None:
    epicenter = rng.uniform(0, TILE_PX, size=2)
    max_dist = math.hypot(TILE_PX, TILE_PX)
    for s in structs:
        dist = math.hypot(s.cx - epicenter[0], s.cy - epicenter[1]) / max_dist
        severity = max(0.0, 1.0 - dist) ** 1.5
        weights = np.array(
            [
                0.7 - 0.55 * severity,
                0.2 + 0.05 * severity,
                0.07 + 0.25 * severity,
                0.03 + 0.25 * severity,
            ]
        )
        weights = np.clip(weights, 1e-3, None)
        weights /= weights.sum()
        s.damage = int(rng.choice(4, p=weights))


def _render(structs: list[Structure], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h = w = TILE_PX
    ground = np.array([106, 153, 78], dtype=np.float32)
    pre = np.tile(ground, (h, w, 1))
    river = _river_mask(h, w)
    road = _road_mask(h, w)
    pre[road] = np.array([120, 120, 120], dtype=np.float32)
    pre[river] = np.array([64, 110, 200], dtype=np.float32)

    kind_color = {
        "house": np.array([180, 140, 100], dtype=np.float32),
        "warehouse": np.array([150, 150, 160], dtype=np.float32),
        "shop": np.array([200, 170, 90], dtype=np.float32),
        "clinic": np.array([220, 220, 220], dtype=np.float32),
        "market stall": np.array([210, 120, 90], dtype=np.float32),
        "apartment block": np.array([160, 130, 130], dtype=np.float32),
        "bridge": np.array([170, 140, 90], dtype=np.float32),
    }

    struct_masks = np.zeros((len(structs), h, w), dtype=bool)
    for s in structs:
        y0, y1 = max(s.cy - s.half, 0), min(s.cy + s.half, h)
        x0, x1 = max(s.cx - s.half, 0), min(s.cx + s.half, w)
        struct_masks[s.idx, y0:y1, x0:x1] = True
        pre[y0:y1, x0:x1] = kind_color[s.kind]

    post = pre.copy()
    rubble = np.array([90, 80, 75], dtype=np.float32)
    for s in structs:
        if s.damage == 0:
            continue
        m = struct_masks[s.idx]
        ys, xs = np.where(m)
        if len(ys) == 0:
            continue
        frac = {1: 0.15, 2: 0.5, 3: 0.9}[s.damage]
        noise = rng.random(len(ys))
        hit = noise < frac
        darken = {1: 0.85, 2: 0.55, 3: 0.25}[s.damage]
        post[ys, xs] = post[ys, xs] * darken
        post[ys[hit], xs[hit]] = rubble + rng.normal(0, 6, size=(hit.sum(), 3))

    pre = np.clip(pre, 0, 255).astype(np.float32) / 255.0
    post = np.clip(post, 0, 255).astype(np.float32) / 255.0
    image = np.concatenate([pre.transpose(2, 0, 1), post.transpose(2, 0, 1)], axis=0)
    return image.astype(np.float32), struct_masks


def _make_reports(
    structs: list[Structure], cfg: SyntheticConfig, rng: np.random.Generator
) -> list[Report]:
    if rng.random() < cfg.prob_zero_reports:
        return []

    k = int(rng.integers(cfg.min_reports, cfg.max_reports + 1))
    k = min(k, cfg.max_reports_cap)
    reports: list[Report] = []
    m_per_px = cfg.tile_meters / TILE_PX

    candidates = [s for s in structs if s.damage > 0] or structs
    for _ in range(k):
        is_distractor = rng.random() < cfg.prob_distractor
        if is_distractor:
            text = rng.choice(
                [
                    "Traffic congestion reported downtown.",
                    "Weather is clear this afternoon.",
                    "Community meeting scheduled next week.",
                    "Local market reopened for business.",
                ]
            )
            dx = float(rng.uniform(-0.35, 0.35))
            dy = float(rng.uniform(-0.35, 0.35))
            dt = float(rng.normal(0, cfg.time_noise_hours * 2))
            reports.append(Report(text, dx, dy, dt, relevant=False))
            continue

        s = candidates[rng.integers(0, len(candidates))]
        need_only = rng.random() < 0.35
        if need_only:
            need = rng.choice(NEEDS)
            street = STREET_NAMES[s.idx % len(STREET_NAMES)] if not s.is_bridge else "Station Rd"
            text = f"Shelter needs {need} near {street}."
        elif s.is_bridge:
            text = "Bridge near Station Rd gone." if s.damage >= 2 else "Bridge near Station Rd reported intact."
        else:
            dmg_word = {0: "undamaged", 1: "lightly damaged", 2: "badly damaged", 3: "destroyed"}[s.damage]
            phrasing = rng.choice(
                [
                    f"{s.kind.capitalize()} on {s.name.split(' on ')[-1]} is {dmg_word}.",
                    f"Family on roof near {s.name.split(' on ')[-1]}." if s.damage >= 2 else
                    f"{s.kind.capitalize()} near {s.name.split(' on ')[-1]} looks {dmg_word}.",
                ]
            )
            text = phrasing

        cx_m = (s.cx - TILE_PX / 2) * m_per_px
        cy_m = (s.cy - TILE_PX / 2) * m_per_px
        dx = cx_m / 1000.0 + float(rng.normal(0, cfg.geo_noise_km))
        dy = cy_m / 1000.0 + float(rng.normal(0, cfg.geo_noise_km))
        dt = float(rng.normal(0, cfg.time_noise_hours))
        reports.append(Report(text, dx, dy, dt, relevant=True, struct_idx=s.idx))

    return reports


def _make_sitrep(structs: list[Structure], reports: list[Report]) -> tuple[str, int]:
    counts = [0, 0, 0, 0]
    for s in structs:
        counts[s.damage] += 1
    tile_damage = max(s.damage for s in structs)
    bridge = next(s for s in structs if s.is_bridge)
    bridge_status = "destroyed" if bridge.damage >= 2 else ("damaged" if bridge.damage == 1 else "intact")

    needs = sorted({n for r in reports if r.relevant for n in NEEDS if n in r.text})
    needs_str = ", ".join(needs) if needs else "none reported"

    text = (
        f"Tile assessment: {counts[3]} destroyed, {counts[2]} major damage, "
        f"{counts[1]} minor damage, {counts[0]} undamaged structures. "
        f"Bridge is {bridge_status}. "
        f"Overall damage level: {DAMAGE_NAMES[tile_damage]}. "
        f"Reported needs: {needs_str}."
    )
    return text, tile_damage


def generate_tile(rng: np.random.Generator, cfg: SyntheticConfig | None = None) -> TileSample:
    cfg = cfg or SyntheticConfig()
    structs = _layout_structures(cfg, rng)
    _assign_damage(structs, rng)
    image, struct_masks = _render(structs, rng)
    reports = _make_reports(structs, cfg, rng)
    sitrep, tile_damage = _make_sitrep(structs, reports)
    bag_label = int(any(r.relevant for r in reports))
    building_damage = np.array([s.damage for s in structs], dtype=np.int64)
    return TileSample(
        image=image,
        reports=reports,
        sitrep=sitrep,
        damage=tile_damage,
        bag_label=bag_label,
        building_masks=struct_masks,
        building_damage=building_damage,
        structure_names=[s.name for s in structs],
    )


def generate_dataset(n: int, seed: int, cfg: SyntheticConfig | None = None) -> list[TileSample]:
    rng = np.random.default_rng(seed)
    return [generate_tile(rng, cfg) for _ in range(n)]
