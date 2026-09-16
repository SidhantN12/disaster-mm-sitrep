import torch

from disaster_mm.retrieval_check import (
    embed_sentences,
    flag_unverified,
    recall_at_k,
    split_sentences,
    text_to_tile_retrieval,
)


def test_split_sentences():
    text = "Bridge is destroyed. Overall damage level: major. Reported needs: water."
    sents = split_sentences(text)
    assert len(sents) == 3


def test_embed_sentences_shape(part1, tokenizer):
    sents = ["Bridge is destroyed.", "Shelter needs water."]
    embeds = embed_sentences(part1, tokenizer, sents, max_len=8, device=torch.device("cpu"))
    assert embeds.shape == (2, part1.shared_dim)


def test_embed_sentences_empty():
    class DummyModel:
        shared_dim = 4

    embeds = embed_sentences(DummyModel(), None, [], max_len=8, device=torch.device("cpu"))
    assert embeds.shape == (0, 4)


def test_flag_unverified_low_sim_low_alpha_flags_true():
    sent_embeds = torch.tensor([[1.0, 0.0]])
    tile_z = torch.tensor([0.0, 1.0])  # orthogonal -> cosine sim 0
    mil_alpha = torch.tensor([0.1, 0.1])
    flags = flag_unverified(sent_embeds, tile_z, mil_alpha, sim_threshold=0.2, alpha_threshold=0.3)
    assert flags == [True]


def test_flag_unverified_high_sim_not_flagged():
    sent_embeds = torch.tensor([[1.0, 0.0]])
    tile_z = torch.tensor([1.0, 0.0])
    mil_alpha = torch.tensor([0.1])
    flags = flag_unverified(sent_embeds, tile_z, mil_alpha, sim_threshold=0.2, alpha_threshold=0.3)
    assert flags == [False]


def test_flag_unverified_supported_by_report_not_flagged():
    sent_embeds = torch.tensor([[1.0, 0.0]])
    tile_z = torch.tensor([0.0, 1.0])
    mil_alpha = torch.tensor([0.9])
    flags = flag_unverified(sent_embeds, tile_z, mil_alpha, sim_threshold=0.2, alpha_threshold=0.3)
    assert flags == [False]


def test_recall_at_k_perfect():
    n = 5
    sim = torch.eye(n)
    gt = torch.arange(n)
    r = recall_at_k(sim, gt, ks=[1])
    assert r[1] == 1.0


def test_text_to_tile_retrieval():
    text_embeds = torch.eye(4)
    tile_embeds = torch.eye(4)
    gt = torch.arange(4)
    r = text_to_tile_retrieval(text_embeds, tile_embeds, gt, ks=[1, 2])
    assert r[1] == 1.0
    assert r[2] == 1.0
