"""Метрики и delta EN-RU."""

import pytest

from btzsc_ru.metrics import compute_metrics, delta_en_ru, summarize


def test_perfect_prediction():
    m = compute_metrics([0, 1, 2, 1], [0, 1, 2, 1], 3)
    assert m["macro_f1"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["pred_class_coverage"] == 1.0


def test_constant_prediction_penalised_by_macro_f1():
    gold = [0] * 8 + [1, 2]
    pred = [0] * 10
    m = compute_metrics(gold, pred, 3)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["macro_f1"] < 0.4
    assert m["pred_class_coverage"] == pytest.approx(1 / 3)


def test_missing_class_counts_in_macro_average():
    m = compute_metrics([0, 1], [0, 1], 4)
    assert m["macro_f1"] == pytest.approx(0.5)
    assert m["gold_class_coverage"] == 0.5


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0], 2)


def test_delta_requires_same_label_set():
    en = {"macro_f1": 0.7}
    ru = {"macro_f1": 0.5}
    assert delta_en_ru(en, ru, ["a", "b"], ["a", "b"])["delta"] == pytest.approx(0.2)
    blocked = delta_en_ru(en, ru, ["a", "b"], ["a", "c"])
    assert blocked["status"] == "не рассчитано"


def test_summarize_ignores_failed_rows():
    rows = [
        {"model_id": "m1", "status": "OK", "macro_f1": "0.6", "accuracy": "0.7", "ms_per_example": "10"},
        {"model_id": "m1", "status": "FAILED", "macro_f1": "", "accuracy": "", "ms_per_example": ""},
        {"model_id": "m2", "status": "SKIPPED", "macro_f1": "", "accuracy": "", "ms_per_example": ""},
    ]
    s = {r["model_id"]: r for r in summarize(rows)}
    assert s["m1"]["mean_macro_f1"] == pytest.approx(0.6)
    assert s["m1"]["n_ok"] == 1 and s["m1"]["n_total"] == 2
    assert s["m2"]["mean_macro_f1"] is None


def test_parallel_pairs_share_identical_verbalizers():
    """Методология статьи: описание класса одно и то же для обеих половин пары EN/RU.

    Иначе сравнение меняет сразу две переменные — язык текста и язык описания класса.
    """
    from btzsc_ru.data import emotion_verbalizer, massive_verbalizer, scenario_verbalizer

    for fn, label in (
        (massive_verbalizer, "alarm_set"),
        (scenario_verbalizer, "weather"),
        (emotion_verbalizer, "gratitude"),
    ):
        assert fn(label, "en") == fn(label, "ru"), fn.__name__
        assert fn(label) == fn(label, "ru"), f"{fn.__name__}: язык не должен влиять на описание"


def test_verbalizer_uses_dataset_label_text():
    """Текст метки берётся из датасета (слаг), мы его не переводим и не переписываем."""
    from btzsc_ru.data import emotion_verbalizer, massive_verbalizer

    assert "alarm set" in massive_verbalizer("alarm_set")
    assert "gratitude" in emotion_verbalizer("gratitude")
