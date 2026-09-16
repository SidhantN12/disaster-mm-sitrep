"""torch Dataset/collate wrapping the synthetic tile generator.

Handles the k=0 (no reports) case throughout: samples with zero reports get
a (0, max_report_tokens) report tensor which ``collate_fn`` pads like any
other, producing an all-True padding mask row -- consumers (text encoder,
Part-2 encoder, MIL pooling) are responsible for not producing NaNs on a
fully-masked row (see ``models/text_encoder.py`` and ``part2_seq2seq.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .synthetic import SyntheticConfig, TileSample, generate_dataset
from .tokenizer import BertTokenizerWrapper, TinyTokenizer
from .vocab import PAD_ID, Vocab


@dataclass
class DatasetConfig:
    max_report_tokens: int = 16
    max_reports: int = 8
    max_sitrep_tokens: int = 48


def build_vocab(samples: list[TileSample], max_size: int = 30000) -> Vocab:
    texts = [s.sitrep for s in samples]
    for s in samples:
        texts.extend(r.text for r in s.reports)
    return Vocab(max_size=max_size).build(texts)


class DisasterTileDataset(Dataset):
    def __init__(
        self,
        samples: list[TileSample],
        tokenizer: TinyTokenizer | BertTokenizerWrapper,
        cfg: DatasetConfig | None = None,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.cfg = cfg or DatasetConfig()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        cfg = self.cfg
        reports = s.reports[: cfg.max_reports]
        k = len(reports)

        if k > 0:
            report_tokens = torch.tensor(
                [self.tokenizer.encode(r.text, cfg.max_report_tokens) for r in reports],
                dtype=torch.long,
            )
            report_geo = torch.tensor([[r.dx, r.dy] for r in reports], dtype=torch.float32)
            report_time = torch.tensor([[r.dt] for r in reports], dtype=torch.float32)
        else:
            report_tokens = torch.zeros((0, cfg.max_report_tokens), dtype=torch.long)
            report_geo = torch.zeros((0, 2), dtype=torch.float32)
            report_time = torch.zeros((0, 1), dtype=torch.float32)

        sitrep_ids = self.tokenizer.encode(s.sitrep, cfg.max_sitrep_tokens, add_bos_eos=True)
        sitrep_tokens = torch.tensor(sitrep_ids, dtype=torch.long)

        return {
            "image": torch.from_numpy(s.image),
            "report_tokens": report_tokens,
            "report_geo": report_geo,
            "report_time": report_time,
            "num_reports": k,
            "sitrep_tokens": sitrep_tokens,
            "damage": torch.tensor(s.damage, dtype=torch.long),
            "bag_label": torch.tensor(float(s.bag_label), dtype=torch.float32),
            "building_masks": torch.from_numpy(s.building_masks),
            "building_damage": torch.from_numpy(s.building_damage),
        }


def collate_fn(batch: list[dict]) -> dict:
    B = len(batch)
    max_k = max((b["report_tokens"].shape[0] for b in batch), default=0)
    max_k = max(max_k, 1)  # keep a stable, non-degenerate report axis
    Lr = batch[0]["report_tokens"].shape[1]

    report_tokens = torch.full((B, max_k, Lr), PAD_ID, dtype=torch.long)
    report_geo = torch.zeros((B, max_k, 2), dtype=torch.float32)
    report_time = torch.zeros((B, max_k, 1), dtype=torch.float32)
    report_pad_mask = torch.ones((B, max_k), dtype=torch.bool)
    num_reports = torch.zeros((B,), dtype=torch.long)

    for i, b in enumerate(batch):
        k = b["report_tokens"].shape[0]
        num_reports[i] = k
        if k > 0:
            report_tokens[i, :k] = b["report_tokens"]
            report_geo[i, :k] = b["report_geo"]
            report_time[i, :k] = b["report_time"]
            report_pad_mask[i, :k] = False

    images = torch.stack([b["image"] for b in batch])
    sitrep_tokens = torch.stack([b["sitrep_tokens"] for b in batch])
    damage = torch.stack([b["damage"] for b in batch])
    bag_label = torch.stack([b["bag_label"] for b in batch])
    building_masks = torch.stack([b["building_masks"] for b in batch])
    building_damage = torch.stack([b["building_damage"] for b in batch])

    return {
        "image": images,
        "report_tokens": report_tokens,
        "report_geo": report_geo,
        "report_time": report_time,
        "report_pad_mask": report_pad_mask,
        "num_reports": num_reports,
        "sitrep_tokens": sitrep_tokens,
        "damage": damage,
        "bag_label": bag_label,
        "building_masks": building_masks,
        "building_damage": building_damage,
    }


def make_synthetic_splits(
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    syn_cfg: SyntheticConfig | None = None,
) -> tuple[list[TileSample], list[TileSample], list[TileSample]]:
    syn_cfg = syn_cfg or SyntheticConfig()
    train = generate_dataset(n_train, seed, syn_cfg)
    val = generate_dataset(n_val, seed + 1, syn_cfg)
    test = generate_dataset(n_test, seed + 2, syn_cfg)
    return train, val, test


def structure_grid_shape(syn_cfg: SyntheticConfig) -> int:
    return syn_cfg.grid_rows * syn_cfg.grid_cols + 1  # + bridge
