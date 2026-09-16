"""Shared tiny fixtures: everything here is sized to be near-instant on CPU
and never touches the network (no pretrained-weight downloads)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from disaster_mm.data.dataset import DatasetConfig, DisasterTileDataset, build_vocab, collate_fn
from disaster_mm.data.synthetic import SyntheticConfig, generate_dataset
from disaster_mm.data.tokenizer import TinyTokenizer
from disaster_mm.models.part1_mmae import Part1MMAE
from disaster_mm.models.part2_seq2seq import Part2Seq2Seq

torch.manual_seed(0)


@pytest.fixture
def syn_cfg() -> SyntheticConfig:
    return SyntheticConfig(grid_rows=2, grid_cols=2, max_reports=4, prob_zero_reports=0.3, prob_distractor=0.25)


@pytest.fixture
def ds_cfg() -> DatasetConfig:
    return DatasetConfig(max_report_tokens=8, max_reports=4, max_sitrep_tokens=24)


@pytest.fixture
def samples(syn_cfg):
    return generate_dataset(12, seed=1, cfg=syn_cfg)


@pytest.fixture
def tokenizer(samples):
    vocab = build_vocab(samples, max_size=200)
    return TinyTokenizer(vocab)


@pytest.fixture
def dataset(samples, tokenizer, ds_cfg):
    return DisasterTileDataset(samples, tokenizer, ds_cfg)


@pytest.fixture
def batch(dataset):
    items = [dataset[i] for i in range(8)]
    return collate_fn(items)


@pytest.fixture
def zero_report_batch(dataset, tokenizer, ds_cfg):
    """A batch where every sample has k=0 reports (forces the all-padding path)."""
    items = []
    for i in range(4):
        item = dataset[i]
        item = dict(item)
        item["report_tokens"] = torch.zeros((0, ds_cfg.max_report_tokens), dtype=torch.long)
        item["report_geo"] = torch.zeros((0, 2), dtype=torch.float32)
        item["report_time"] = torch.zeros((0, 1), dtype=torch.float32)
        items.append(item)
    return collate_fn(items)


@pytest.fixture
def part1(tokenizer):
    return Part1MMAE(
        image_variant="tiny",
        text_variant="tiny",
        vocab_size=tokenizer.vocab_size,
        sep_id=tokenizer.sep_id,
        pad_id=tokenizer.pad_id,
        img_embed_dim=16,
        img_depth=1,
        img_heads=2,
        txt_embed_dim=16,
        txt_depth=1,
        txt_heads=2,
        shared_dim=12,
        private_dim=6,
        decoder_base_channels=8,
        modality_dropout_p=0.3,
    )


@pytest.fixture
def part2(part1, tokenizer):
    return Part2Seq2Seq(
        dim_i=part1.dim_i,
        dim_t=part1.dim_t,
        shared_dim=part1.shared_dim,
        private_dim=part1.private_dim,
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_enc_layers=1,
        n_dec_layers=1,
        n_heads=2,
        ffn_dim=32,
        max_reports=4,
        max_sitrep_len=24,
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
    )


def rng() -> np.random.Generator:
    return np.random.default_rng(0)
