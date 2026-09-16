import numpy as np
import torch

from disaster_mm.metrics import bleu, classification_f1, count_fact_accuracy, pointing_accuracy


def test_classification_f1_perfect():
    preds = np.array([0, 1, 2, 3, 0, 1])
    targets = np.array([0, 1, 2, 3, 0, 1])
    result = classification_f1(preds, targets, n_classes=4)
    assert result["weighted_f1"] == 1.0
    assert all(f == 1.0 for f in result["per_class_f1"])


def test_classification_f1_imperfect_is_lower():
    preds = np.array([0, 0, 0, 0])
    targets = np.array([0, 1, 2, 3])
    result = classification_f1(preds, targets, n_classes=4)
    assert result["weighted_f1"] < 1.0


def test_bleu_identical_is_high():
    hyps = ["the bridge is destroyed", "no damage reported here"]
    refs = ["the bridge is destroyed", "no damage reported here"]
    score = bleu(hyps, refs)
    assert score > 90.0


def test_bleu_empty_hyps_is_zero():
    assert bleu([], []) == 0.0


def test_count_fact_accuracy_perfect_match(samples):
    refs = [s.sitrep for s in samples]
    facts = count_fact_accuracy(refs, refs)
    assert facts["parseable_fraction"] == 1.0
    assert facts["count_accuracy"] == 1.0
    assert facts["bridge_accuracy"] == 1.0
    assert facts["level_accuracy"] == 1.0


def test_count_fact_accuracy_mismatch_is_penalized(samples):
    refs = [s.sitrep for s in samples]
    wrong = ["Tile assessment: 9 destroyed, 9 major damage, 9 minor damage, 9 undamaged structures. "
             "Bridge is destroyed. Overall damage level: destroyed. Reported needs: none reported."] * len(refs)
    facts = count_fact_accuracy(wrong, refs)
    assert facts["count_accuracy"] < 1.0


def test_pointing_accuracy_hits_when_patch_inside_matching_building():
    masks = torch.zeros(2, 256, 256, dtype=torch.bool)
    masks[0, :16, :16] = True  # covers patch (0,0) -> patch index 0
    damage = torch.tensor([3, 0])
    acc = pointing_accuracy([0], [3], [masks], [damage])
    assert acc == 1.0


def test_pointing_accuracy_misses_when_damage_class_wrong():
    masks = torch.zeros(2, 256, 256, dtype=torch.bool)
    masks[0, :16, :16] = True
    damage = torch.tensor([3, 0])
    acc = pointing_accuracy([0], [1], [masks], [damage])
    assert acc == 0.0
