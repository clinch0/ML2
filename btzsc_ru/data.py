"""Загрузка датасетов и построение задач классификации.

Все контракты (config / split / имена колонок) проверены на шаге D3 и
зафиксированы в contracts.json; здесь они только используются.
`datasets` импортируется лениво — модуль можно импортировать без сети.
"""

from __future__ import annotations

from .config import DatasetSpec, DATASETS_BY_KEY
from .reconstruction import (
    MultiClassTask,
    check_single_gold,
    infer_n_classes_by_hypothesis,
    reconstruct,
)

# Шаблоны вербализаций. Фиксируются ДО оценки (протокол эксперимента).
#
# МЕТОДОЛОГИЯ СТАТЬИ, БЕЗ ИЗМЕНЕНИЙ (§4): метка превращается в одну короткую
# англоязычную фразу-описание, текст метки берётся из самого датасета. Мы её не
# переводим и не переписываем. Для параллельных пар EN/RU это к тому же обязательное
# условие корректности: обе половины получают ОДИН И ТОТ ЖЕ набор описаний классов,
# поэтому различается ровно одна переменная — язык входного текста.
MASSIVE_INTENT_TEMPLATE = "The user request to the voice assistant is about {label}"
MASSIVE_SCENARIO_TEMPLATE = "The user request to the voice assistant belongs to the {label} scenario"
EMOTION_TEMPLATE = "The emotion expressed in this comment is {label}"


def massive_verbalizer(label_slug: str, language: str = "en") -> str:
    """Интент MASSIVE. Метка — англоязычный слаг датасета в обоих конфигах.

    Аргумент language сохранён для совместимости вызовов и НЕ влияет на результат:
    вербализация одинакова для английской и русской половины (протокол статьи).
    """
    return MASSIVE_INTENT_TEMPLATE.format(label=label_slug.replace("_", " "))


def scenario_verbalizer(label_slug: str, language: str = "en") -> str:
    """Сценарий MASSIVE (18 доменов). Описание класса одинаково для EN и RU."""
    return MASSIVE_SCENARIO_TEMPLATE.format(label=label_slug.replace("_", " "))


def emotion_verbalizer(label_name: str, language: str = "en") -> str:
    """Эмоция GoEmotions. Описание класса одинаково для EN и RU."""
    return EMOTION_TEMPLATE.format(label=label_name.replace("_", " "))


def task_from_go_emotions(spec: DatasetSpec, ds) -> "MultiClassTask":
    """GoEmotions → задача с одной меткой на пример.

    Протокол статьи одноклассовый (argmax по описаниям), а разметка GoEmotions
    многометочная, поэтому берутся строки ровно с одной меткой. Это подготовка данных,
    а не изменение протокола: сам способ оценки остаётся авторским. Число отброшенных
    строк пишется в артефакты прогона.

    Русская и английская половины — это одни и те же строки (колонки ru_text и text),
    поэтому пары строго параллельны, а фильтр по числу меток применяется к ним одинаково.
    """
    names = list(ds.features[spec.label_column].feature.names)
    verbalizers = [emotion_verbalizer(n) for n in names]

    texts, gold, dropped = [], [], 0
    for text, labels in zip(ds[spec.text_column], ds[spec.label_column]):
        if labels is None or len(labels) != 1:
            dropped += 1
            continue
        texts.append(str(text))
        gold.append(int(labels[0]))

    task = MultiClassTask(
        name=spec.key,
        texts=texts,
        verbalizers=verbalizers,
        gold=gold,
        n_classes=len(verbalizers),
        dropped=dropped,
    )
    task.meta["label_slugs"] = names
    task.meta["note"] = f"оставлены только однометочные примеры; отброшено {dropped}"
    return task


def _local_parquet(spec: DatasetSpec, limit: int | None):
    """Офлайн-источник данных для самопроверки конвейера.

    Включается переменной окружения BTZSC_LOCAL_DATA=<каталог с parquet>: нужен, чтобы прогнать
    весь путь (прогон → экспорт → импорт → отчёт) без доступа к Hugging Face.
    В настоящем прогоне не используется — данные берутся из официального датасета.
    """
    import os
    from pathlib import Path

    root = os.environ.get("BTZSC_LOCAL_DATA")
    if not root or spec.hf_id != "btzsc/btzsc":
        return None
    path = Path(root) / f"{spec.config}.parquet"
    if not path.exists():
        return None

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if limit is not None:
        table = table.slice(0, limit)
    return {name: table.column(name).to_pylist() for name in table.column_names}


def load_hf_split(spec: DatasetSpec, *, cache_dir: str | None = None, limit: int | None = None):
    """Загружает split датасета. Ленивый импорт `datasets`; сеть нужна только здесь."""
    local = _local_parquet(spec, limit)
    if local is not None:
        print(f"  [офлайн] {spec.key}: локальный parquet, строк {len(local['text'])}", flush=True)
        return local

    from datasets import load_dataset

    split = spec.split if limit is None else f"{spec.split}[:{limit}]"
    if spec.hf_id == "btzsc/btzsc":
        return load_dataset(spec.hf_id, name=spec.config, split=split, cache_dir=cache_dir)
    if spec.config and spec.config != "default":
        return load_dataset(spec.hf_id, spec.config, split=split, cache_dir=cache_dir)
    return load_dataset(spec.hf_id, split=split, cache_dir=cache_dir)


def task_from_btzsc_pairs(spec: DatasetSpec, ds, *, trim_tail: bool = False) -> MultiClassTask:
    """BTZSC: пары (text, hypothesis, labels) → multiclass (официальный протокол, D2).

    `trim_tail=True` отрезает незавершённую последнюю группу. Нужно только для пробной
    загрузки в preflight: там берутся первые N строк, и N не обязан делиться на число
    классов (например, 200 строк при 3 классах у financialphrasebank или 77 у banking77).
    В полном прогоне флаг всегда False: там неделимость — это признак битых данных.
    """
    texts = list(ds["text"])
    hypotheses = list(ds["hypothesis"])
    labels = [int(v) for v in ds["labels"]]

    if trim_tail and texts:
        n_classes = infer_n_classes_by_hypothesis(hypotheses)
        usable = len(texts) - len(texts) % n_classes
        if usable == 0:
            raise ValueError(
                f"{spec.key}: загружено {len(texts)} строк, а один пример занимает {n_classes} — "
                "для проверки нужно больше строк"
            )
        texts, hypotheses, labels = texts[:usable], hypotheses[:usable], labels[:usable]

    task = reconstruct(
        name=spec.key,
        texts=texts,
        hypotheses=hypotheses,
        labels=labels,
    )
    check_single_gold(task)
    task.meta.update({"hf_id": spec.hf_id, "config": spec.config, "split": spec.split, "language": spec.language})
    return task


def task_from_flat(spec: DatasetSpec, ds) -> MultiClassTask:
    """Плоский датасет (одна строка = один пример) → MultiClassTask."""
    texts = [str(t) for t in ds[spec.text_column]]
    raw_labels = list(ds[spec.label_column])

    if spec.key == "ru_sentiment":
        verbalizers = list(spec.verbalizers)
        gold = [int(v) for v in raw_labels]
        if gold and (min(gold) < 0 or max(gold) >= len(verbalizers)):
            raise ValueError(
                f"{spec.key}: метки вне диапазона 0..{len(verbalizers) - 1}; "
                "mapping карточки MonoHime не подтверждён — остановиться и перепроверить"
            )
    elif spec.key.startswith("massive_scenario"):
        slugs = sorted({str(v) for v in raw_labels})
        verbalizers = [scenario_verbalizer(s, spec.language) for s in slugs]
        index = {s: i for i, s in enumerate(slugs)}
        gold = [index[str(v)] for v in raw_labels]
    elif spec.key.startswith("massive"):
        slugs = sorted({str(v) for v in raw_labels})
        verbalizers = [massive_verbalizer(s, spec.language) for s in slugs]
        index = {s: i for i, s in enumerate(slugs)}
        gold = [index[str(v)] for v in raw_labels]
    else:
        slugs = sorted({str(v) for v in raw_labels})
        verbalizers = list(slugs)
        index = {s: i for i, s in enumerate(slugs)}
        gold = [index[str(v)] for v in raw_labels]

    task = MultiClassTask(
        name=spec.key,
        texts=texts,
        verbalizers=verbalizers,
        gold=gold,
        n_classes=len(verbalizers),
        dropped=0,
        meta={
            "hf_id": spec.hf_id,
            "config": spec.config,
            "split": spec.split,
            "language": spec.language,
            "label_slugs": sorted({str(v) for v in raw_labels}),
        },
    )
    check_single_gold(task)
    return task


def load_task(dataset_key: str, *, cache_dir: str | None = None, limit: int | None = None,
              trim_tail: bool = False) -> MultiClassTask:
    """Единая точка входа: ключ датасета → MultiClassTask."""
    spec = DATASETS_BY_KEY[dataset_key]
    ds = load_hf_split(spec, cache_dir=cache_dir, limit=limit)
    if spec.kind == "btzsc_pairs":
        return task_from_btzsc_pairs(spec, ds, trim_tail=trim_tail)
    if spec.kind == "go_emotions":
        return task_from_go_emotions(spec, ds)
    return task_from_flat(spec, ds)


def label_sets_match(task_a: MultiClassTask, task_b: MultiClassTask) -> bool:
    """Проверка перед расчётом delta EN−RU: совпадают ли наборы меток (по слагам)."""
    a = task_a.meta.get("label_slugs")
    b = task_b.meta.get("label_slugs")
    if a is None or b is None:
        return False
    return sorted(a) == sorted(b)
