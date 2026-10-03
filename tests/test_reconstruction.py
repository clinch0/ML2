"""Синтетические фикстуры: реконструкция пар BTZSC → multiclass."""

import pytest

from btzsc_ru.reconstruction import (
    check_single_gold,
    infer_n_classes,
    infer_n_classes_by_hypothesis,
    reconstruct,
)

HYPS = ["про бизнес", "про науку", "про спорт"]


def make_pairs(n_texts, gold_per_text, hyps=HYPS):
    texts, hypotheses, labels = [], [], []
    for i in range(n_texts):
        for j, h in enumerate(hyps):
            texts.append(f"text-{i}")
            hypotheses.append(h)
            labels.append(1 if j == gold_per_text[i] else 0)
    return texts, hypotheses, labels


def test_infer_n_classes_from_binary_pattern():
    """Официальная эвристика (btzsc.data._get_n_classes) верна, когда gold первых
    двух примеров — один и тот же класс."""
    _, _, labels = make_pairs(3, [0, 0, 0])
    assert infer_n_classes(labels) == 3


def test_binary_pattern_heuristic_is_fragile_hypotheses_are_not():
    """Причина, по которой мы определяем число классов по гипотезам (см. docs/btzsc_code_notes.md)."""
    _, hyps, labels = make_pairs(3, [0, 1, 2])
    assert infer_n_classes(labels) == 4          # официальная эвристика ошибается
    assert infer_n_classes_by_hypothesis(hyps) == 3


def test_infer_n_classes_from_hypotheses():
    _, hyps, _ = make_pairs(4, [0, 1, 2, 0])
    assert infer_n_classes_by_hypothesis(hyps) == 3


def test_reconstruct_recovers_gold():
    gold = [0, 1, 2, 1]
    texts, hyps, labels = make_pairs(4, gold)
    task = reconstruct("synthetic", texts, hyps, labels)
    assert task.n_classes == 3
    assert task.n_samples == 4
    assert task.gold == gold
    assert task.verbalizers == HYPS
    assert task.texts == [f"text-{i}" for i in range(4)]
    check_single_gold(task)


def test_reconstruct_drops_groups_without_single_gold():
    texts, hyps, labels = make_pairs(3, [0, 1, 2])
    labels[3:6] = [0, 0, 0]          # у второго текста нет положительной гипотезы
    task = reconstruct("synthetic", texts, hyps, labels)
    assert task.n_samples == 2
    assert task.dropped == 1


def test_reconstruct_rejects_broken_hypothesis_order():
    texts, hyps, labels = make_pairs(2, [0, 1])
    hyps[3], hyps[4] = hyps[4], hyps[3]
    with pytest.raises(ValueError):
        reconstruct("synthetic", texts, hyps, labels)


def test_reconstruct_rejects_ragged_columns():
    texts, hyps, labels = make_pairs(2, [0, 1])
    with pytest.raises(ValueError):
        reconstruct("synthetic", texts[:-1], hyps, labels)


def test_20_samples_means_20_texts_not_20_rows():
    gold = [i % 3 for i in range(20)]
    texts, hyps, labels = make_pairs(20, gold)
    task = reconstruct("synthetic", texts, hyps, labels)
    assert len(texts) == 60 and task.n_samples == 20
