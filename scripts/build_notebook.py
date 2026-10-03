"""Генерация BTZSC_Colab.ipynb с загрузкой кода из репозитория.

    python scripts/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "BTZSC_Colab.ipynb"


def build() -> Path:
    cells: list[dict] = []

    def md(source: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})

    def code(source: str) -> None:
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "outputs": [], "source": source.splitlines(keepends=True)})

    md("""# BTZSC-RU — прогон в Google Colab

Ноутбук **вызывает** `experiment.py` из клонированного репозитория.
Код проекта загружается из <https://github.com/clinch0/ML2> (ячейка 2).
Откройте .ipynb в Colab и выполняйте ячейки сверху вниз.

**Токен Hugging Face не нужен**: все модели и датасеты публичные.

Runtime → Change runtime type → **T4 GPU**. Без GPU энкодеры считаются на CPU, LLM получают `SKIPPED_NO_GPU`.

**Ничего выбирать не нужно.** `MODE = "all"` в ячейке 1 прогоняет всё по порядку:
preflight → smoke (20 примеров) → evaluate (300) → **replicate** → finetune → export.
Достаточно нажать Runtime → Run all. Отдельные режимы остались на случай, когда нужен один этап.

## Что делает прогон: два блока по 5 моделей

| Блок | Модели | Данные | Зачем |
|---|---|---|---|
| **А** | 5 моделей из статьи: all-MiniLM-L6-v2, e5-base-v2, e5-large-v2, bge-base-en-v1.5, bge-large-en-v1.5 | 7 датасетов статьи по 300 примеров | воспроизвести её результат нашим кодом |
| **Б** | 5 наших: USER-bge-m3, BERTA, rubert-mini-frida, ru-en-RoSBERTa, multilingual-e5-base | те же данные, та же выборка | сравнить с блоком А на одной шкале |

10 500 предсказаний на блок, примерно по 10-12 минут каждый на T4. Расширения (дообучение,
остальные модели, русские датасеты, увеличенные выборки) включаются флагами в ячейке 1 и по
умолчанию выключены.

Порядок этапов подобран так, чтобы ценное считалось первым: сначала основная оценка и экспорт,
потом тяжёлые этапы на больших выборках. Если сессия Colab умрёт на тяжёлом этапе, архив с
основными результатами уже сохранён, а повторный запуск с тем же `RUN_ID` досчитает остальное
(resume). Скачайте архив результатов до завершения сессии Colab.

Ориентировочное время на T4: основная часть ~45 минут, тяжёлые этапы ещё ~60 минут при
`FULL_CAP = 3000`. Общий дедлайн `DEADLINE_MIN` не даёт прогону уйти в бесконечность.""")

    code('''# 1. Параметры прогона
MODE = "all"                # ВСЁ САМО: preflight → БЛОК А → БЛОК Б → export.
                            # Менять не нужно. Этапы по отдельности:
                            # preflight | smoke | evaluate | replicate | finetune | export
N_SAMPLES = None            # None = 300 примеров на датасет в режиме all; число = вручную
RUN_ID = "run1"             # тот же RUN_ID = продолжение прогона (resume)
# Код проекта клонируется из репозитория:
GIT_REPO_URL = "https://github.com/clinch0/ML2"
GIT_BRANCH = "main"
PROJECT_DIR = "/content/btzsc_project"
HF_CACHE = "/content/hf_cache"
RUNS_ROOT = "/content/runs"
# Ничего отключать не нужно: по умолчанию выполняется весь конвейер.
# Флаги ниже — аварийные, если времени в сессии совсем нет.
# По умолчанию считается всё, на чём держатся выводы отчёта:
# якорь → блок А → блок Б → дообучение → русское расширение.
SKIP_FINETUNE = False       # выключить дообучение rubert-tiny2 (иначе +1-3 мин, даёт слайд «разметка против размера»)
SKIP_RU_EXTENSION = False   # выключить русские датасеты и разрыв EN−RU (иначе +20 мин)
WITH_EXTRA_MODELS = True    # остальные модели сверх блока Б: tiny-энкодеры и две LLM (+30 мин)
WITH_FULL_TESTS = False     # увеличенные выборки для блоков А и Б (+60 мин)
# Якорь к публикации (модели статьи на ПОЛНОМ сплите трёх дешёвых датасетов) включён всегда:
# это единственные числа прогона, сравнимые с Таблицей 2 статьи. Отключать не рекомендуется.
SKIP_ANCHOR = False
DEADLINE_MIN = 150          # общий дедлайн: тяжёлые этапы за ним помечаются NOT_RUN,
                            # повторный запуск с тем же RUN_ID досчитает их через resume
FULL_CAP = 3000             # потолок примеров на датасет для «полных» этапов (0 = весь сплит)
print("MODE:", MODE, "| RUN_ID:", RUN_ID, "| N_SAMPLES:", N_SAMPLES, "| репозиторий:", GIT_REPO_URL)''')

    code('''# 2. Загрузка кода проекта
import os, shutil, subprocess

shutil.rmtree(PROJECT_DIR, ignore_errors=True)
cmd = ["git", "clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_REPO_URL, PROJECT_DIR]
subprocess.run(cmd, check=True)
if not os.path.isfile(os.path.join(PROJECT_DIR, "experiment.py")):
    raise FileNotFoundError("В клонированном репозитории нет experiment.py")
print("код проекта загружен:", PROJECT_DIR)
print(sorted(os.listdir(PROJECT_DIR))[:20])''')

    code('''# 3. Установка зависимостей
!pip -q install -r {PROJECT_DIR}/requirements-colab.txt''')

    code('''# 4. Каталоги прогона и переменные окружения HF (до импорта huggingface_hub)
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"   # иначе лог тонет в «Loading weights»
os.environ["HF_HUB_DISABLE_XET"] = "1"            # и в «downloading bytes …» от xet-клиента
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE, "datasets")
os.makedirs(HF_CACHE, exist_ok=True)

os.makedirs(RUNS_ROOT, exist_ok=True)
RUN_DIR = os.path.join(RUNS_ROOT, RUN_ID)
print("результаты прогона (локально):", RUN_DIR)''')

    code('''# 5. Публичные модели и датасеты загружаются анонимно.
print("Hugging Face: анонимная загрузка публичных моделей и датасетов")''')

    code('''# 6. Импорт experiment.py и план прогона (ноутбук вызывает модуль, а не копирует логику)
import sys

sys.path.insert(0, PROJECT_DIR)
import experiment

print("experiment:", experiment.__doc__.splitlines()[0])
print("режимы:", experiment.MODES)
%cd {PROJECT_DIR}
!python experiment.py --mode {MODE} --run-id {RUN_ID} --dry-run''')

    code('''# 7. Preflight: контракты датасетов и среда. Модели не загружаются.
%cd {PROJECT_DIR}
!python experiment.py --mode preflight --run-id {RUN_ID} --out {RUNS_ROOT} --cache-dir {HF_CACHE}
import os
print(open(os.path.join(RUN_DIR, "preflight.json"), encoding="utf-8").read()[:1500])''')

    code('''# 8. Запуск. Основной запуск эксперимента. Этапы идут сами,
# Прогресс печатается структурированными строками: возврат каретки в Colab через поток не виден,
# поэтому бар рисуется текстом и печатается отдельными строками:
#   [block_b 2/5 · BERTA] btzsc_imdb 3/7 [######....] 60% · 180/300 · 14 мс/пример · фон: frida
#   где: этап и номер модели · текущий датасет · бар пары · сколько примеров · скорость ·
#        какая модель в это время качается в фоне
# В конце каждой пары — строка с macro-F1 и общим прогрессом этапа с оценкой остатка.
# с resume и записью после каждого батча:
#   preflight → anchor → block_a → block_b → export → finetune → extra_models → ru_extension → export_final
# Три теста прогона: anchor (модели статьи на полном сплите — проверка кода),
# block_a + block_b (обе группы моделей на одной выборке), ru_extension (русские задачи).
import shlex, subprocess

cmd = f"python experiment.py --mode {MODE} --run-id {RUN_ID} --out {RUNS_ROOT} --cache-dir {HF_CACHE}"
cmd += f" --deadline-min {DEADLINE_MIN} --full-cap {FULL_CAP}"
if N_SAMPLES is not None:
    cmd += f" --n-samples {N_SAMPLES}"
for flag, option in (
    (SKIP_ANCHOR, "--no-anchor"),
    (SKIP_FINETUNE, "--no-finetune"),
    (SKIP_RU_EXTENSION, "--no-ru-extension"),
    (WITH_EXTRA_MODELS, "--with-extra-models"),
    (WITH_FULL_TESTS, "--with-full-baseline --with-full-ours"),
):
    if flag:
        cmd += " " + option
print(cmd)

proc = subprocess.Popen(
    shlex.split(cmd), cwd=PROJECT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1,
)
for line in proc.stdout:          # лог идёт в реальном времени, а не одной простынёй в конце
    print(line, end="")
proc.wait()
print("код возврата:", proc.returncode)

import json
report_path = os.path.join(RUN_DIR, "run_all_report.json")
if os.path.exists(report_path):
    report = json.load(open(report_path, encoding="utf-8"))
    print()
    print("итог по этапам:")
    for stage, info in report.items():
        mark = "OK " if info["status"] == "OK" else "СБОЙ"
        print(f"  {mark} {stage:11} {info['seconds']:>7.1f} с")
        if info["status"] != "OK":
            print(f"       причина: {str(info['detail'])[:300]}")
''')

    code('''# 9. Быстрый просмотр результатов
import os, pandas as pd

res = os.path.join(RUN_DIR, "results.csv")
if os.path.exists(res):
    df = pd.read_csv(res)
    display(df[["model_id", "dataset_key", "macro_f1", "accuracy", "status", "reason"]])
    print(df["status"].value_counts().to_dict())
else:
    print("results.csv ещё нет — это нормально после preflight; для чисел нужен MODE='smoke' или 'evaluate'")

rep = os.path.join(RUN_DIR, "replication", "results.csv")
if os.path.exists(rep):
    ref = {
        ("sentence-transformers/all-MiniLM-L6-v2", "btzsc_agnews"): 0.495,
        ("sentence-transformers/all-MiniLM-L6-v2", "btzsc_imdb"): 0.340,
        ("intfloat/e5-base-v2", "btzsc_agnews"): 0.761,
        ("intfloat/e5-base-v2", "btzsc_imdb"): 0.898,
        ("intfloat/e5-large-v2", "btzsc_agnews"): 0.787,
        ("intfloat/e5-large-v2", "btzsc_imdb"): 0.928,
        ("BAAI/bge-base-en-v1.5", "btzsc_agnews"): 0.635,
        ("BAAI/bge-base-en-v1.5", "btzsc_imdb"): 0.896,
        ("BAAI/bge-large-en-v1.5", "btzsc_agnews"): 0.766,
        ("BAAI/bge-large-en-v1.5", "btzsc_imdb"): 0.935,
    }
    rdf = pd.read_csv(rep)
    rdf = rdf[rdf["status"] == "OK"].copy()
    rdf["статья"] = [ref.get((m, d)) for m, d in zip(rdf["model_id"], rdf["dataset_key"])]
    rdf["Δ"] = (rdf["macro_f1"] - rdf["статья"]).round(3)
    display(rdf[["model_id", "dataset_key", "macro_f1", "статья", "Δ", "n_samples"]])
    print("сверка с числами статьи: |Δ| max =", rdf["Δ"].abs().max())
else:
    print("сверки ещё нет — она появится после этапа replicate")''')

    code('''# 10. Скачать архив. В режиме all экспорт уже выполнен ячейкой 8.
import os, subprocess, zipfile

zip_path = os.path.join(RUNS_ROOT, f"colab_outputs_{RUN_ID}.zip")
if not os.path.exists(zip_path):
    res = os.path.join(RUN_DIR, "results.csv")
    if not os.path.exists(res):
        print(f"В {RUN_DIR} нет results.csv — экспортировать нечего.")
        raise SystemExit("Сначала выполните ячейку 8.")
    proc = subprocess.run(
        ["python", "experiment.py", "--mode", "export", "--run-id", RUN_ID, "--out", RUNS_ROOT],
        capture_output=True, text=True, cwd=PROJECT_DIR,
    )
    print(proc.stdout[-2000:])

names = zipfile.ZipFile(zip_path).namelist() if os.path.exists(zip_path) else []
print(zip_path, os.path.getsize(zip_path) if names else 0, "байт | файлов:", len(names))
print(names)
assert names, "архив пуст — не скачивайте его, сначала выполните ячейку 11"

from google.colab import files
files.download(zip_path)''')

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return OUT


if __name__ == "__main__":
    path = build()
    print("saved", path)
