"""Отчёт и слайды обязаны совпадать с артефактами прогона.

Эти проверки ловят самый дорогой класс ошибок в такой работе: число в тексте живёт
своей жизнью, потому что его один раз вписали руками, а прогон с тех пор переделали.
Здесь числа из README сверяются со строками выбранного прогона и его
вердиктами `replication_check.csv`.

Второй блок проверок — про шкалу: средние считаются только внутри одного теста.
Смешивать английские датасеты статьи с русским расширением нельзя, и об этом должен
заботиться код фигур и презентации, а не внимательность автора.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "README.md"
RUN_MATCH = re.search(r"<!-- REPORT:RUN_ID=([A-Za-z0-9._-]+) -->", REPORT.read_text(encoding="utf-8"))
assert RUN_MATCH, "в README нет идентификатора прогона"
RUN = ROOT / "results" / RUN_MATCH.group(1)

pytestmark = pytest.mark.skipif(not (RUN / "results.csv").exists(),
                                reason="нет импортированного прогона отчёта")

def rows(test: str) -> list[dict]:
    with (RUN / "results.csv").open(encoding="utf-8") as fh:
        found = [r for r in csv.DictReader(fh) if r["status"] == "OK" and r["test"] == test]
    latest = {(r["model_id"], r["dataset_key"]): r for r in found}
    return list(latest.values())


def mean_by_model(test: str) -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for r in rows(test):
        per.setdefault(r["model_id"], []).append(float(r["macro_f1"]))
    return {m: statistics.mean(v) for m, v in per.items()}


@pytest.mark.skipif(not REPORT.exists(), reason="отчёт не входит в эту сборку")
def test_main_table_of_report_equals_results_csv():
    """Таблица «обе группы на одной шкале» — ровно средние по block_a и block_b."""
    text = REPORT.read_text(encoding="utf-8")
    if "| Блок | Модель | ср. macro-F1" not in text:
        pytest.skip("в текущем прогоне нет таблицы блоков А и Б")
    table = text[text.index("| Блок | Модель | ср. macro-F1"):]
    table = table[: table.index("\n\n")]

    means = mean_by_model("block_a") | mean_by_model("block_b")
    checked = 0
    for line in table.splitlines()[2:]:
        cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
        model_id, claimed = cells[1].strip("`"), float(cells[2])
        assert model_id in means, f"{model_id}: модели нет в прогоне"
        assert claimed == pytest.approx(means[model_id], abs=0.001), \
            f"{model_id}: в отчёте {claimed}, в прогоне {means[model_id]:.3f}"
        checked += 1
    assert checked == len(means), "в таблице должны быть все модели обоих блоков"


@pytest.mark.skipif(not REPORT.exists(), reason="отчёт не входит в эту сборку")
@pytest.mark.skipif(not (RUN / "replication_check.csv").exists(),
                    reason="для этого прогона нет сверки с публикацией")
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
    if not a or not b:
        pytest.skip("текущий прогон не содержит оба блока")
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


def test_all_generated_figures_are_in_readme():
    manifest_path = ROOT / "figures" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == RUN.name
    report = REPORT.read_text(encoding="utf-8")
    for filename in manifest["files"]:
        assert (ROOT / "figures" / filename).exists()
        assert f"(figures/{filename})" in report
