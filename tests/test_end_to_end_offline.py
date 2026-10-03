"""Сквозная самопроверка конвейера БЕЗ сети и без скачивания моделей.

Запускает настоящий `experiment.py --mode all` в подпроцессе на локальных parquet
(`data/*.parquet`) с псевдо-моделями и проверяет весь путь:
прогон → results.csv с колонкой test → export → import → пересчёт метрик.

Именно этого теста не хватало: ошибки вида «конструктор не принимает аргумент» или
«экспорт не знает про подкаталог» видны только при реальном проходе всей цепочки.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOCAL_DATASETS = "btzsc_rottentomatoes,btzsc_financialphrasebank"

pytestmark = pytest.mark.skipif(
    not (DATA / "rottentomatoes.parquet").exists(),
    reason="нет локальных parquet в data/",
)


def run_pipeline(tmp_path: Path) -> dict:
    env = os.environ | {
        "BTZSC_LOCAL_DATA": str(DATA),
        "BTZSC_FAKE_MODELS": "1",
        "BTZSC_BLOCK_DATASETS": LOCAL_DATASETS,
        "BTZSC_BLOCK_A_MODELS": "intfloat/e5-base-v2,BAAI/bge-base-en-v1.5",
        "BTZSC_BLOCK_B_MODELS": "deepvk/USER-bge-m3,sergeyzh/BERTA",
        # Якорь по умолчанию включает btzsc_imdb, а его parquet в data/ нет (большой).
        # Берём локальные наборы + agnews: якорь проверяется и на длинном датасете.
        "BTZSC_ANCHOR_DATASETS": LOCAL_DATASETS + ",btzsc_agnews",
    }
    proc = subprocess.run(
        [sys.executable, "experiment.py", "--mode", "all", "--run-id", "e2e",
         "--out", str(tmp_path / "runs"), "--n-samples", "6",
         # дообучение и русские датасеты требуют сети и настоящих весов — офлайн их не зовём
         "--no-ru-extension", "--no-finetune"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    return {"stdout": proc.stdout, "run_dir": tmp_path / "runs" / "e2e"}


def test_full_pipeline_offline(tmp_path):
    out = run_pipeline(tmp_path)
    run_dir = out["run_dir"]

    # 1. Все этапы прошли
    report = json.loads((run_dir / "run_all_report.json").read_text(encoding="utf-8"))
    assert [v["status"] for v in report.values()] == ["OK"] * len(report), report
    assert set(report) == {"preflight", "anchor", "block_a", "block_b", "export"}

    # 2. Результаты обоих блоков в одном файле и помечены колонкой test
    rows = list(csv.DictReader((run_dir / "results.csv").open(encoding="utf-8")))
    assert {r["test"] for r in rows} == {"baseline", "block_a", "block_b"}, "якорь пишется как baseline"
    assert all(r["status"] == "OK" for r in rows), [r["reason"] for r in rows if r["status"] != "OK"]
    block_rows = [r for r in rows if r["test"].startswith("block_")]
    assert len(block_rows) == 8, "2 блока × 2 модели × 2 датасета"

    # Признак сравнимости: блоки (подвыборка 6) — несравнимы, якорь на полном сплите — сравним
    assert all(r["comparable_to_paper"] == "False" for r in block_rows), "подвыборка не сравнима с публикацией"
    anchor_rows = [r for r in rows if r["test"] == "baseline"]
    assert anchor_rows and all(r["comparable_to_paper"] == "True" for r in anchor_rows), \
        [r["comparability_reason"] for r in anchor_rows]

    # 3. Прогресс печатался в структурированном виде
    assert "=== этап block_a" in out["stdout"] and "=== этап block_b" in out["stdout"]
    assert "пара 4/4" in out["stdout"] and "█" in out["stdout"]

    # 4. Архив собран и содержит обязательные файлы
    archive = run_dir.parent / "colab_outputs_e2e.zip"
    assert archive.exists()

    # 5. Импорт пересчитывает метрики из предсказаний и не находит расхождений
    sys.path.insert(0, str(ROOT))
    from scripts.import_colab_outputs import import_zip

    info = import_zip(archive, tmp_path / "results", tmp_path / "report.md")
    assert info["mismatches"] == [], info["mismatches"]
    expected_pairs = len({(r["model_id"], r["dataset_key"]) for r in rows})
    assert info["n_pairs"] == expected_pairs, "импорт сводит строки по парам модель×датасет"

    imported = list(csv.DictReader((tmp_path / "results" / "e2e" / "results.csv").open(encoding="utf-8")))
    assert {"block_a", "block_b", "baseline"} <= {r["test"] for r in imported}
    recomputed = list(csv.DictReader((tmp_path / "results" / "e2e" / "results_recomputed.csv").open(encoding="utf-8")))
    assert recomputed and "test" in recomputed[0]


def test_resume_does_not_recompute(tmp_path):
    """Повторный запуск не пересчитывает уже готовые пары."""
    run_pipeline(tmp_path)
    second = run_pipeline(tmp_path)
    assert "уже посчитано, пропуск" in second["stdout"]

    rows = list(csv.DictReader((second["run_dir"] / "results.csv").open(encoding="utf-8")))
    keys = [(r["test"], r["model_id"], r["dataset_key"]) for r in rows]
    assert len(keys) == len(set(keys)), "дубликатов после повторного прогона быть не должно"
