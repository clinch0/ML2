"""Отчёт и слайды обязаны совпадать с артефактами прогона.

Эти проверки ловят самый дорогой класс ошибок в такой работе: число в тексте живёт
своей жизнью, потому что его один раз вписали руками, а прогон с тех пор переделали.
Здесь числа из `Эксперимент_BTZSC.md` сверяются со строками `results/run5/results.csv`
и с вердиктами `results/run5/replication_check.csv`.

Второй блок проверок — про шкалу: средние считаются только внутри одного теста.
Смешивать английские датасеты статьи с русским расширением нельзя, и об этом должен
заботиться код фигур и презентации, а не внимательность автора.
"""

from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results" / "run5"
REPORT = ROOT / "Эксперимент_BTZSC.md"

pytestmark = pytest.mark.skipif(not (RUN / "results.csv").exists(),
                                reason="нет импортированного прогона run5")

# Короткое имя из таблицы отчёта → model_id прогона.
SHORT_TO_ID = {
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "e5-large-v2": "intfloat/e5-large-v2",
    "e5-base-v2": "intfloat/e5-base-v2",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "USER-bge-m3": "deepvk/USER-bge-m3",
    "FRIDA": "ai-forever/FRIDA",
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    "rubert-mini-frida (32M)": "sergeyzh/rubert-mini-frida",
    "Vikhr-Qwen-2.5-1.5B": "Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct",
}


def rows(test: str) -> list[dict]:
    with (RUN / "results.csv").open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["status"] == "OK" and r["test"] == test]


def mean_by_model(test: str) -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for r in rows(test):
        per.setdefault(r["model_id"], []).append(float(r["macro_f1"]))
    return {m: statistics.mean(v) for m, v in per.items()}


def test_main_table_of_report_equals_results_csv():
    """Таблица «обе группы на одной шкале» — ровно средние по block_a и block_b."""
    text = REPORT.read_text(encoding="utf-8")
    table = text[text.index("| Блок | Модель | ср. по 7"):]
    table = table[: table.index("\n\n")]

    means = mean_by_model("block_a") | mean_by_model("block_b")
    checked = 0
    for line in table.splitlines()[2:]:
        cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
        name, claimed = cells[1], float(cells[2])
        model_id = SHORT_TO_ID[name]
        assert model_id in means, f"{name}: модели нет в прогоне"
        assert claimed == pytest.approx(means[model_id], abs=0.001), \
            f"{name}: в отчёте {claimed}, в прогоне {means[model_id]:.3f}"
        checked += 1
    assert checked == 10, "в таблице должны быть все десять моделей обоих блоков"


def test_replication_claim_matches_verdicts():
    """Фраза «11 пар из 15 совпали» должна следовать из replication_check.csv."""
    with (RUN / "replication_check.csv").open(encoding="utf-8") as fh:
        verdicts = [r["verdict"] for r in csv.DictReader(fh)]
    matched = verdicts.count("совпало")
    text = REPORT.read_text(encoding="utf-8")
    claim = re.search(r"\*\*(\d+) пар из (\d+) совпали\*\*", text)
    assert claim, "в отчёте должна быть явная формулировка про число совпавших пар"
    assert (int(claim.group(1)), int(claim.group(2))) == (matched, len(verdicts)), \
        f"в отчёте {claim.group(0)}, в прогоне совпало {matched} из {len(verdicts)}"


def test_blocks_share_one_scale():
    """Блоки А и Б сравнимы: одни и те же датасеты и один и тот же размер выборки."""
    a, b = rows("block_a"), rows("block_b")
    assert {r["dataset_key"] for r in a} == {r["dataset_key"] for r in b}
    assert {r["n_samples"] for r in a} == {r["n_samples"] for r in b}
    assert {r["dataset_key"] for r in a}.isdisjoint({r["dataset_key"] for r in rows("ru_extension")}), \
        "русское расширение — другой набор задач, его строки не должны попадать в блоки"


def test_figures_and_slides_average_inside_one_test():
    """Код фигур и презентации обязан фильтровать строки по тесту."""
    figures = (ROOT / "scripts/make_figures.py").read_text(encoding="utf-8")
    assert 'r.get("test") in ("block_a", "block_b")' in figures, \
        "столбики качества строятся только по блокам на одной шкале"
    assert 'r.get("test") == "ru_extension"' in figures, \
        "график качество/задержка берёт один набор задач"

    pptx = (ROOT / "scripts/build_btzsc_pptx.py").read_text(encoding="utf-8")
    assert 'SLIDE_TEST = "ru_extension"' in pptx and 'r.get("test") == SLIDE_TEST' in pptx, \
        "средние на слайдах считаются внутри одного теста"
