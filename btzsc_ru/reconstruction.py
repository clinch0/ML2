"""Реконструкция multiclass-примеров из парного формата BTZSC.

Источник протокола — официальный код (шаг D2):
`src/btzsc/data.py` официального проекта,
функции `_get_n_classes` и `load_btzsc_dataset`:
каждая строка HF-датасета — пара (text, hypothesis, labels∈{0,1}); подряд идущие
`n_classes` строк относятся к одному тексту; gold — индекс строки с labels==1.

Отличия от официальной функции задокументированы в docs/btzsc_code_notes.md:
1. порядок гипотез проверяется для каждой группы (официальный код берёт его только из первой);
2. группы, где число положительных гипотез != 1, не «схлопываются» в класс 0, а отбрасываются
   с подсчётом в `dropped` (gold должен быть ровно один).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MultiClassTask:
    """Восстановленный multiclass-датасет."""

    name: str
    texts: list[str]
    verbalizers: list[str]          # описания классов в каноническом порядке
    gold: list[int]                 # индекс правильного класса
    n_classes: int
    dropped: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.texts)


def infer_n_classes(labels: list[int]) -> int:
    """Число классов = период повторения бинарного паттерна (официальный `_get_n_classes`)."""
    if not labels:
        raise ValueError("пустая колонка labels")
    first = labels[0]
    for i in range(1, len(labels)):
        if labels[i] == first:
            return i
    return len(labels)


def infer_n_classes_by_hypothesis(hypotheses: list[str]) -> int:
    """Более устойчивый вариант: период повторения списка гипотез.

    Используется как перекрёстная проверка `infer_n_classes`: бинарный паттерн
    ломается, если в группе нет ни одной единицы.
    """
    if len(hypotheses) < 2:
        raise ValueError("колонка hypothesis короче двух строк")
    first = hypotheses[0]
    for i in range(1, len(hypotheses)):
        if hypotheses[i] == first:
            return i
    return len(hypotheses)


def reconstruct(
    name: str,
    texts: list[str],
    hypotheses: list[str],
    labels: list[int],
    *,
    strict: bool = True,
) -> MultiClassTask:
    """Собирает пары в multiclass-примеры.

    Args:
        name: имя датасета (agnews / imdb / ...).
        texts, hypotheses, labels: колонки HF-датасета в исходном порядке.
        strict: при True расхождение порядка гипотез внутри группы — ошибка.

    Returns:
        MultiClassTask с ровно одним gold на пример.
    """
    if not (len(texts) == len(hypotheses) == len(labels)):
        raise ValueError("колонки разной длины")
    n_classes = infer_n_classes_by_hypothesis(hypotheses)
    if n_classes < 2:
        raise ValueError(f"{name}: определено {n_classes} классов")
    if len(texts) % n_classes != 0:
        raise ValueError(f"{name}: {len(texts)} строк не делится на {n_classes} классов")

    canonical = hypotheses[:n_classes]
    out_texts: list[str] = []
    out_gold: list[int] = []
    dropped = 0

    for start in range(0, len(texts), n_classes):
        group_h = hypotheses[start : start + n_classes]
        if group_h != canonical:
            if strict:
                raise ValueError(f"{name}: порядок гипотез меняется на строке {start}")
            dropped += 1
            continue
        group_y = [int(v) for v in labels[start : start + n_classes]]
        if sum(group_y) != 1:
            dropped += 1
            continue
        group_text = texts[start : start + n_classes]
        if len(set(group_text)) != 1:
            if strict:
                raise ValueError(f"{name}: текст меняется внутри группы на строке {start}")
            dropped += 1
            continue
        out_texts.append(group_text[0])
        out_gold.append(group_y.index(1))

    if not out_texts:
        raise ValueError(f"{name}: не осталось примеров с ровно одним gold")

    return MultiClassTask(
        name=name,
        texts=out_texts,
        verbalizers=list(canonical),
        gold=out_gold,
        n_classes=n_classes,
        dropped=dropped,
        meta={"n_pairs": len(texts), "n_groups": len(texts) // n_classes},
    )


def check_single_gold(task: MultiClassTask) -> None:
    """Контроль инварианта: каждый gold — валидный индекс класса."""
    for i, g in enumerate(task.gold):
        if not (0 <= g < task.n_classes):
            raise ValueError(f"{task.name}: пример {i} имеет gold={g} вне диапазона")
    if len(task.gold) != len(task.texts):
        raise ValueError(f"{task.name}: длины texts и gold расходятся")
