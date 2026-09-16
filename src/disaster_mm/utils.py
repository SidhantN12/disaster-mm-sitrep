"""Small shared utilities: seeding and config loading."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_device(cfg: dict[str, Any]) -> torch.device:
    want = cfg.get("device", "cpu")
    if want == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(want)
