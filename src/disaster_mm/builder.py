"""Assembles datasets, tokenizer and models from a loaded YAML config dict.
Shared by train.py, evaluate.py and infer.py so all three stay consistent."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .data.dataset import DatasetConfig, DisasterTileDataset, build_vocab, make_synthetic_splits
from .data.synthetic import SyntheticConfig
from .data.tokenizer import BertTokenizerWrapper, TinyTokenizer, build_tokenizer
from .data.vocab import Vocab
from .models.part1_mmae import Part1MMAE
from .models.part2_seq2seq import Part2Seq2Seq


def build_synthetic_config(cfg: dict[str, Any]) -> SyntheticConfig:
    d = cfg["data"]
    return SyntheticConfig(
        grid_rows=d.get("grid_rows", 4),
        grid_cols=d.get("grid_cols", 4),
        max_reports=d.get("max_reports", 8),
        prob_zero_reports=d.get("prob_zero_reports", 0.2),
        prob_distractor=d.get("prob_distractor", 0.25),
    )


def build_dataset_config(cfg: dict[str, Any]) -> DatasetConfig:
    d = cfg["data"]
    return DatasetConfig(
        max_report_tokens=d.get("max_report_tokens", 16),
        max_reports=d.get("max_reports", 8),
        max_sitrep_tokens=d.get("max_sitrep_tokens", 40),
    )


def build_tokenizer_and_splits(cfg: dict[str, Any], vocab_path: str | None = None):
    d = cfg["data"]
    syn_cfg = build_synthetic_config(cfg)
    train, val, test = make_synthetic_splits(
        d.get("n_train", 64), d.get("n_val", 16), d.get("n_test", 16), cfg.get("seed", 42), syn_cfg
    )

    text_variant = cfg["model"]["text_variant"]
    if text_variant == "tiny":
        if vocab_path and Path(vocab_path).exists():
            vocab = Vocab.load(vocab_path)
        else:
            vocab = build_vocab(train, max_size=d.get("vocab_size", 30000))
        tokenizer = build_tokenizer("tiny", vocab)
    else:
        tokenizer = build_tokenizer("bert")

    return tokenizer, syn_cfg, (train, val, test)


def build_datasets(cfg: dict[str, Any], tokenizer, splits) -> tuple[DisasterTileDataset, DisasterTileDataset, DisasterTileDataset]:
    ds_cfg = build_dataset_config(cfg)
    train, val, test = splits
    return (
        DisasterTileDataset(train, tokenizer, ds_cfg),
        DisasterTileDataset(val, tokenizer, ds_cfg),
        DisasterTileDataset(test, tokenizer, ds_cfg),
    )


def build_models(cfg: dict[str, Any], tokenizer: TinyTokenizer | BertTokenizerWrapper) -> tuple[Part1MMAE, Part2Seq2Seq]:
    m = cfg["model"]
    vocab_size = tokenizer.vocab_size
    part1 = Part1MMAE(
        image_variant=m["image_variant"],
        text_variant=m["text_variant"],
        vocab_size=vocab_size,
        sep_id=tokenizer.sep_id,
        pad_id=tokenizer.pad_id,
        img_embed_dim=m.get("img_embed_dim", 64),
        img_depth=m.get("img_depth", 2),
        img_heads=m.get("img_heads", 2),
        txt_embed_dim=m.get("txt_embed_dim", 64),
        txt_depth=m.get("txt_depth", 2),
        txt_heads=m.get("txt_heads", 2),
        shared_dim=m.get("shared_dim", 32),
        private_dim=m.get("private_dim", 16),
        decoder_base_channels=m.get("decoder_base_channels", 32),
        modality_dropout_p=m.get("modality_dropout_p", 0.3),
    )
    part2 = Part2Seq2Seq(
        dim_i=part1.dim_i,
        dim_t=part1.dim_t,
        shared_dim=part1.shared_dim,
        private_dim=part1.private_dim,
        vocab_size=vocab_size,
        d_model=m.get("d_model", 128),
        n_enc_layers=m.get("n_enc_layers", 4),
        n_dec_layers=m.get("n_dec_layers", 4),
        n_heads=m.get("n_heads", 4),
        ffn_dim=m.get("ffn_dim", 256),
        max_reports=cfg["data"].get("max_reports", 32),
        max_sitrep_len=cfg["data"].get("max_sitrep_tokens", 48),
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
    )
    return part1, part2
