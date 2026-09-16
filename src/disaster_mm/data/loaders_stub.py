"""Loader stubs for real datasets.

None of these download or read anything at import time -- they document the
expected on-disk layout and the conversion needed to reach the
``TileSample`` contract used throughout this repo (see
``disaster_mm.data.synthetic.TileSample``). See the README "Plugging in
real datasets" section for manual download instructions; nothing here is
exercised by CI.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any


class XBDLoader:
    """xView2 / xBD (Gupta et al. 2019): pre/post satellite image pairs with
    per-building polygons labeled with a damage class.

    Expected layout (after manual download from https://xview2.org/):
        <root>/
          images/<tile_id>_pre_disaster.png
          images/<tile_id>_post_disaster.png
          labels/<tile_id>_pre_disaster.json   # building polygons
          labels/<tile_id>_post_disaster.json  # + damage labels

    xBD ships no text reports, so ``reports`` must come from elsewhere (a
    simulated feed, or a paired social-media dataset via geo/time bagging --
    see ``bag_reports_by_geo_time`` below). ``damage`` should be derived by
    reducing per-building damage (xBD's ``subtype`` field: no-damage,
    minor-damage, major-damage, destroyed) to a tile-level label, e.g. the
    max across buildings, matching this repo's synthetic-data convention.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def __len__(self) -> int:
        raise NotImplementedError("Download xBD manually and implement tile listing; see class docstring.")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError(
            "Parse the pre/post PNG pair and building-polygon JSON at `idx` into the "
            "TileSample contract: image (6,H,W), building_masks (S,H,W), building_damage (S,)."
        )


class CrisisMMDLoader:
    """CrisisMMD (Alam et al. 2018): tweets with text + attached images,
    labeled for informativeness and humanitarian category, with some
    damage-severity annotations.

    Expected layout (after manual download, see https://crisisnlp.qcri.org/):
        <root>/
          <event_name>/<event_name>_data/*.jpg
          <event_name>/<event_name>_devset.tsv | _trainset.tsv | _testset.tsv

    CrisisMMD provides single images (not pre/post pairs) plus real geotagged
    (approximate) text -- use it primarily as a source of realistic report
    text and image/text pairing statistics; pre/post pairing must come from a
    separate source (e.g. paired with xBD tiles by location/time).
    """

    def __init__(self, root: str | Path, event_name: str, split: str = "train") -> None:
        self.root = Path(root)
        self.event_name = event_name
        self.split = split

    def __len__(self) -> int:
        raise NotImplementedError("Download CrisisMMD manually and implement TSV parsing; see class docstring.")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError("Parse the TSV row + image at `idx` into a raw (image, text, label) triple.")


class FloodNetLoader:
    """FloodNet (Rahnemoonfar et al. 2021): UAV post-flood imagery with
    semantic segmentation masks (building-flooded / building-non-flooded /
    road-flooded / road-non-flooded / water / tree / vehicle / pool / grass).

    Expected layout (after manual download, see https://github.com/BinaLab/FloodNet-Supervised_v1.0):
        <root>/
          train/train-org-img/*.jpg
          train/train-label-img/*.png   # per-pixel class id

    FloodNet has no pre-disaster imagery and no text; it is most useful as a
    source of realistic post-disaster building/flood segmentation masks to
    substitute for the synthetic ``building_masks``, paired with a
    synthetic or externally-sourced pre-disaster image and report feed.
    """

    def __init__(self, root: str | Path, split: str = "train") -> None:
        self.root = Path(root)
        self.split = split

    def __len__(self) -> int:
        raise NotImplementedError("Download FloodNet manually and implement mask parsing; see class docstring.")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError("Parse the image + segmentation mask at `idx` into a raw (image, mask) pair.")


def bag_reports_by_geo_time(
    tile_center: tuple[float, float],
    tile_time: float,
    reports: list[dict[str, Any]],
    radius_km: float = 2.0,
    time_window_hours: float = 24.0,
) -> list[dict[str, Any]]:
    """Select and geo/time-normalize reports near a tile.

    ``reports`` items must have ``lat``, ``lon``, ``timestamp_hours`` (any
    consistent epoch) and ``text``. Returns the subset within
    ``radius_km``/``time_window_hours`` of the tile, each augmented with
    ``dx``, ``dy`` (km offset, equirectangular approximation) and ``dt``
    (hours offset) -- i.e. exactly the per-report geo/time fields the
    ``TileSample``/dataset contract expects.
    """
    lat0, lon0 = tile_center
    lat0_rad = math.radians(lat0)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(lat0_rad)

    out = []
    for r in reports:
        dy = (r["lat"] - lat0) * km_per_deg_lat
        dx = (r["lon"] - lon0) * km_per_deg_lon
        dt = r["timestamp_hours"] - tile_time
        if math.hypot(dx, dy) <= radius_km and abs(dt) <= time_window_hours:
            out.append({**r, "dx": dx, "dy": dy, "dt": dt})
    return out
