"""Unified tokenizer interface used by both the ``tiny`` and ``bert`` configs.

Both wrappers expose the same minimal surface (``encode``, ``pad_id``,
``sep_id``, ``vocab_size``) so the rest of the codebase (dataset, text
encoder) does not need to know which one is in use.
"""
from __future__ import annotations

from .vocab import BOS_ID, EOS_ID, PAD_ID, Vocab


class TinyTokenizer:
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        # No dedicated [SEP] in the tiny vocab; reuse EOS as the report separator.
        self.sep_id = EOS_ID
        self.vocab_size = len(vocab)

    def encode(self, text: str, max_len: int, add_bos_eos: bool = False) -> list[int]:
        ids = self.vocab.encode(text, max_len=max_len, add_bos_eos=add_bos_eos)
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len
        return ids[:max_len]

    def decode(self, ids: list[int]) -> str:
        return self.vocab.decode(ids)


class BertTokenizerWrapper:
    def __init__(self, name: str = "bert-base-uncased") -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        self.pad_id = self.tok.pad_token_id
        self.bos_id = self.tok.cls_token_id
        self.eos_id = self.tok.sep_token_id
        self.sep_id = self.tok.sep_token_id
        self.vocab_size = self.tok.vocab_size

    def encode(self, text: str, max_len: int, add_bos_eos: bool = False) -> list[int]:
        ids = self.tok.encode(text, add_special_tokens=False)
        if add_bos_eos:
            ids = [self.bos_id] + ids + [self.eos_id]
        ids = ids[:max_len]
        ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode([i for i in ids if i != self.pad_id], skip_special_tokens=True)


def build_tokenizer(variant: str, vocab: Vocab | None = None) -> TinyTokenizer | BertTokenizerWrapper:
    if variant == "tiny":
        assert vocab is not None, "tiny tokenizer requires a pre-built Vocab"
        return TinyTokenizer(vocab)
    if variant == "bert":
        return BertTokenizerWrapper()
    raise ValueError(f"Unknown tokenizer variant: {variant}")
