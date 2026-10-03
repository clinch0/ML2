"""Метрики. Главная — macro-F1 (как в статье, раздел 3.1)."""

from __future__ import annotations

METRIC_KEYS = ("macro_f1", "accuracy", "macro_precision", "macro_recall")


def compute_metrics(gold: list[int], pred: list[int], n_classes: int) -> dict[str, float]:
    """macro-F1 / accuracy / macro-P / macro-R + покрытие классов.

    Усреднение по всем `n_classes` классам (labels=range(n_classes)), чтобы
    отсутствующий в выборке класс не исчезал из знаменателя молча.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    if len(gold) != len(pred):
        raise ValueError("длины gold и pred не совпадают")
    if not gold:
        raise ValueError("пустая выборка")
    labels = list(range(n_classes))
    out = {
        "macro_f1": float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_precision": float(
            precision_score(gold, pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(gold, pred, labels=labels, average="macro", zero_division=0)),
        "n": len(gold),
        "n_classes": n_classes,
        "gold_class_coverage": len(set(gold)) / n_classes,
        "pred_class_coverage": len(set(pred)) / n_classes,
    }
    return out


def delta_en_ru(
    metrics_en: dict[str, float],
    metrics_ru: dict[str, float],
    labels_en: list[str],
    labels_ru: list[str],
    metric: str = "macro_f1",
) -> dict[str, object]:
    """Delta EN−RU считается только при совпадающем наборе меток (протокол эксперимента)."""
    if sorted(labels_en) != sorted(labels_ru):
        return {"status": "не рассчитано", "reason": "наборы меток EN и RU не совпадают"}
    return {
        "status": "OK",
        "metric": metric,
        "en": metrics_en[metric],
        "ru": metrics_ru[metric],
        "delta": metrics_en[metric] - metrics_ru[metric],
    }


def summarize(rows: list[dict]) -> list[dict]:
    """Сводка по моделям: среднее macro-F1 по датасетам со status=OK."""
    import statistics

    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model_id"], []).append(r)
    summary = []
    for model_id, items in sorted(by_model.items()):
        ok = [i for i in items if i.get("status") == "OK"]
        summary.append(
            {
                "model_id": model_id,
                "n_ok": len(ok),
                "n_total": len(items),
                "mean_macro_f1": statistics.fmean([float(i["macro_f1"]) for i in ok]) if ok else None,
                "mean_accuracy": statistics.fmean([float(i["accuracy"]) for i in ok]) if ok else None,
                "mean_ms_per_example": (
                    statistics.fmean([float(i["ms_per_example"]) for i in ok if i.get("ms_per_example")])
                    if any(i.get("ms_per_example") for i in ok)
                    else None
                ),
            }
        )
    return summary
