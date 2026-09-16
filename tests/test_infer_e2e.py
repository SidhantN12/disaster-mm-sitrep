import torch

from disaster_mm.retrieval_check import embed_sentences, flag_unverified, split_sentences
from disaster_mm.viz import save_attention_heatmap


def test_end_to_end_infer_on_synthetic_tile(part1, part2, dataset, tokenizer, tmp_path):
    from disaster_mm.data.dataset import collate_fn

    item = dataset[0]
    batch = collate_fn([item])

    part1.eval()
    part2.eval()
    with torch.no_grad():
        p1_out = part1.encode(batch["image"], batch["report_tokens"], batch["report_pad_mask"])
        out = part2.infer(p1_out, batch["report_geo"], batch["report_time"], batch["report_pad_mask"], beam_size=1, max_len=12)

    sitrep = tokenizer.decode(out["tokens"][0].tolist())
    assert isinstance(sitrep, str)

    damage_class = out["damage_logits"][0].argmax().item()
    assert damage_class in (0, 1, 2, 3)

    sentences = split_sentences(sitrep) or [sitrep]
    sent_embeds = embed_sentences(part1, tokenizer, sentences, max_len=8, device=torch.device("cpu"))
    flags = flag_unverified(sent_embeds, p1_out["z_s_i"][0], out["mil_alpha"][0])
    assert len(flags) == len(sentences)

    n_patches = p1_out["patch_tokens"].shape[1]
    patch_attn = out["cross_attn"][0, 0, :n_patches]
    out_path = tmp_path / "heatmap.png"
    save_attention_heatmap(batch["image"][0, 3:6], patch_attn, str(out_path))
    assert out_path.exists()


def test_beam_search_runs(part1, part2, dataset):
    from disaster_mm.data.dataset import collate_fn

    item = dataset[0]
    batch = collate_fn([item])
    part1.eval()
    part2.eval()
    with torch.no_grad():
        p1_out = part1.encode(batch["image"], batch["report_tokens"], batch["report_pad_mask"])
        out = part2.infer(p1_out, batch["report_geo"], batch["report_time"], batch["report_pad_mask"], beam_size=2, max_len=8)
    assert out["tokens"].shape[0] == 1
