"""A minimal word-level vocabulary used by the ``tiny`` text pipeline.

Used both to tokenize input reports and to tokenize/detokenize SITREP target
sequences when the ``tiny`` text-encoder config is selected. For the ``bert``
config, reports and SITREP targets are instead tokenized with the
HuggingFace ``bert-base-uncased`` tokenizer (see ``text_encoder.py``); no
download happens unless that config is explicitly requested.
"""
from __future__ import annotations

import json
import re
from collections import Counter

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Vocab:
    def __init__(self, max_size: int = 30000) -> None:
        self.max_size = max_size
        self.word2idx: dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.idx2word: list[str] = list(SPECIAL_TOKENS)

    def build(self, texts: list[str]) -> "Vocab":
        counter: Counter[str] = Counter()
        for t in texts:
            counter.update(tokenize(t))
        budget = self.max_size - len(SPECIAL_TOKENS)
        for word, _count in counter.most_common(budget):
            if word not in self.word2idx:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)
        return self

    def __len__(self) -> int:
        return len(self.idx2word)

    def encode(self, text: str, max_len: int | None = None, add_bos_eos: bool = False) -> list[int]:
        ids = [self.word2idx.get(w, UNK_ID) for w in tokenize(text)]
        if add_bos_eos:
            ids = [BOS_ID] + ids + [EOS_ID]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"max_size": self.max_size, "idx2word": self.idx2word}, f)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = cls(max_size=data["max_size"])
        v.idx2word = data["idx2word"]
        v.word2idx = {w: i for i, w in enumerate(v.idx2word)}
        return v

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        words = []
        for i in ids:
            w = self.idx2word[i] if 0 <= i < len(self.idx2word) else UNK
            if strip_special and w in SPECIAL_TOKENS:
                if w == EOS:
                    break
                continue
            words.append(w)
        return " ".join(words)
