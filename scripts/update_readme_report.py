"""Обновляет числовой отчёт в README из сохранённого прогона и построенных графиков."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START = "<!-- REPORT:START -->"
END = "<!-- REPORT:END -->"

FIGURES = (
    ("fig6_families.png", "Семейства моделей", "Схема способов классификации и их вычислительной цены."),
    ("fig7_who.png", "Какие модели вошли в сравнение", "Схема происхождения и отбора моделей."),
    ("fig0_core_vs_paper.png", "Обе группы на общей шкале", "Блоки А и Б измерены на общей выборке; значения статьи показаны как внешний ориентир."),
    ("fig8_quality_bars.png", "Качество и пиковая память", "Средний macro-F1 по общим задачам; подписи показывают пиковую память."),
    ("fig1_paper_vs_ours.png", "Числа статьи и нашего кода", "Датасеты совпадают, но размер выборки в публикации и нашем основном прогоне различается."),
    ("fig4_replication.png", "Сверка с опубликованными числами", "Разница macro-F1 для моделей статьи на полном тестовом сплите."),
    ("fig2_delta_en_ru.png", "Английский и русский MASSIVE", "Парное сравнение качества на двух языках."),
    ("fig5_delta_by_task.png", "Разница по типам задач", "Языковой разрыв зависит от набора меток и типа задачи."),
    ("fig3_quality_vs_latency.png", "Качество и задержка", "Сопоставление macro-F1 со временем обработки примера."),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def current_rows(run_dir: Path) -> list[dict[str, str]]:
    rows = read_csv(run_dir / "results.csv")
    if not rows:
        raise ValueError(f"Нет результатов в {run_dir / 'results.csv'}")
    # После возобновления прогона одна пара может встретиться несколько раз.
    latest: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("test", ""), row["model_id"], row["dataset_key"])
        previous = latest.get(key)
        if previous is None or int(row.get("n_samples") or 0) >= int(previous.get("n_samples") or 0):
            latest[key] = row
    return list(latest.values())


def score_rows(rows: list[dict[str, str]], tests: set[str]) -> list[tuple[str, str, float, float, int, int]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("test") in tests and row.get("status") == "OK" and row.get("macro_f1"):
            grouped[(row["test"], row["model_id"])].append(row)
    scores = []
    for (test, model), values in grouped.items():
        f1 = statistics.fmean(float(row["macro_f1"]) for row in values)
        accuracy = statistics.fmean(float(row["accuracy"]) for row in values)
        scores.append((test, model, f1, accuracy, len(values),
                       max(int(row.get("n_samples") or 0) for row in values)))
    return sorted(scores, key=lambda item: (item[0], -item[2], item[1]))


def table(scores: list[tuple[str, str, float, float, int, int]], *, block: bool) -> list[str]:
    lines = [
        "| Блок | Модель | ср. macro-F1 | ср. accuracy | задач | примеров на задачу |"
        if block else
        "| Модель | ср. macro-F1 | ср. accuracy | задач | примеров на задачу |",
        "|---|---|---:|---:|---:|---:|" if block else "|---|---:|---:|---:|---:|",
    ]
    for test, model, f1, accuracy, count, n in scores:
        label = f"`{model}`"
        cells = [test, label] if block else [label]
        cells += [f"{f1:.3f}", f"{accuracy:.3f}", str(count), str(n)]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def blocks_comparable(rows: list[dict[str, str]]) -> bool:
    """Каждая модель обоих блоков должна иметь тот же набор задач и размер выборки."""
    coverage: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        if row.get("test") in {"block_a", "block_b"} and row.get("status") == "OK":
            coverage[(row["test"], row["model_id"])].add(
                (row["dataset_key"], row.get("n_samples", "")))
    return (bool(coverage)
            and {"block_a", "block_b"} <= {test for test, _ in coverage}
            and len({frozenset(keys) for keys in coverage.values()}) == 1)


def render(run_id: str) -> str:
    run_dir = ROOT / "results" / run_id
    rows = current_rows(run_dir)
    statuses = Counter(row.get("status", "") for row in rows)
    tests = Counter(row.get("test", "") for row in rows)
    common = score_rows(rows, {"block_a", "block_b"})
    comparable = blocks_comparable(rows)
    russian = score_rows(rows, {"ru_extension"})
    replication = read_csv(run_dir / "replication_check.csv")
    verdicts = Counter(row["verdict"] for row in replication)
    manifest_path = ROOT / "figures" / "manifest.json"
    figures = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_id") == run_id:
            figures = list(manifest.get("files", []))

    lines = [
        START,
        f"<!-- REPORT:RUN_ID={run_id} -->",
        "## Результаты эксперимента",
        "",
        f"Источник: [`results/{run_id}/results.csv`](results/{run_id}/results.csv). "
        "Значения ниже пересчитаны из сохранённых результатов; графики строятся скриптом "
        "[`make_figures.py`](scripts/make_figures.py).",
        "",
        "### Коротко",
        "",
        f"- Прогон `{run_id}`: {len(rows)} пар модель × датасет; "
        f"успешных {statuses.get('OK', 0)}. "
        + ", ".join(f"`{test}` — {count}" for test, count in sorted(tests.items())) + ".",
    ]
    if common:
        a = max((row for row in common if row[0] == "block_a"), key=lambda row: row[2], default=None)
        b = max((row for row in common if row[0] == "block_b"), key=lambda row: row[2], default=None)
        if a and b and comparable:
            lines.append(
                f"- На общем наборе задач лучший результат блока А — `{a[1]}` "
                f"({a[2]:.3f} macro-F1), блока Б — `{b[1]}` ({b[2]:.3f}). "
                "Средние посчитаны отдельно по одинаковым задачам."
            )
    if russian:
        best = max(russian, key=lambda row: row[2])
        lines.append(
            f"- В русском расширении лучший средний macro-F1 — {best[2]:.3f} "
            f"у `{best[1]}` на {best[4]} задачах."
        )
    if replication:
        lines.append(
            f"- Сверка с опубликованными числами: **{verdicts.get('совпало', 0)} пар "
            f"из {len(replication)} совпали**; на границе — {verdicts.get('на границе', 0)}, "
            f"расхождений — {verdicts.get('расхождение', 0)}."
        )
    lines += ["", "### Методика и сопоставимость", ""]
    if common:
        lines.append("Блок А содержит модели статьи, блок Б — наши модели. Сравнение блоков "
                     "допустимо на общих датасетах при одинаковом размере выборки. "
                     "В таблицах ниже приведено невзвешенное среднее macro-F1 по задачам для каждой модели.")
    if tests.get("baseline"):
        lines.append("`baseline` использует полные тестовые сплиты для сверки с публикацией.")
    if russian:
        lines.append("`ru_extension` — отдельный набор русских и парных языковых задач; "
                     "его средние не смешиваются со средними блоков А и Б.")
    lines.append("")
    if common:
        heading = "### Блоки А и Б: одна шкала" if comparable else "### Результаты блоков А и Б"
        lines += [heading, ""]
        if not comparable:
            lines += ["В этом прогоне наборы задач или размеры выборки различаются; "
                      "средние между моделями напрямую не сравниваются.", ""]
        lines += table(common, block=True)
        lines += ["", "Источник: строки `block_a` и `block_b` в CSV прогона.", ""]
    if russian:
        lines += ["### Русское расширение", ""]
        lines += table(russian, block=False)
        lines += ["", "Источник: строки `ru_extension` в CSV прогона.", ""]
    if replication:
        lines += [
            "### Сверка на моделях статьи",
            "",
            f"Источник: [`results/{run_id}/replication_check.csv`](results/{run_id}/replication_check.csv) "
            "и [опубликованные JSON](paper_scores/). Δ = наш macro-F1 минус значение статьи.",
            "",
            "| Модель | Датасет | Статья | Наш код | Δ | Вердикт |",
            "|---|---|---:|---:|---:|---|",
        ]
        for row in replication:
            lines.append(
                f"| `{row['model_id']}` | `{row['dataset_key']}` | "
                f"{float(row['paper_macro_f1']):.3f} | {float(row['our_macro_f1']):.3f} | "
                f"{float(row['diff']):+.3f} | {row['verdict']} |"
            )
        lines.append("")
    lines += ["### Графики", ""]
    if figures:
        for filename, title, caption in FIGURES:
            if filename not in figures:
                continue
            if not (ROOT / "figures" / filename).exists():
                raise FileNotFoundError(ROOT / "figures" / filename)
            if filename in {"fig6_families.png", "fig7_who.png"}:
                source = "Схема по коду и списку моделей."
            elif filename in {"fig0_core_vs_paper.png", "fig1_paper_vs_ours.png",
                              "fig4_replication.png"}:
                source = f"Источник — прогон `{run_id}` и опубликованные результаты статьи."
            else:
                source = f"Источник — прогон `{run_id}`."
            lines += [f"#### {title}", "", f"![{title}](figures/{filename})", "",
                      f"{caption} {source}", ""]
    else:
        lines += ["Графики для этого прогона ещё не построены.", ""]
    lines += [
        "### Источники и воспроизведение",
        "",
        "- Исходная статья: [BTZSC, arXiv:2603.11991](https://arxiv.org/abs/2603.11991).",
        "- Публикация авторских метрик: [`paper_scores/`](paper_scores/).",
        f"- Артефакты нашего прогона: [`results/{run_id}/`](results/{run_id}/).",
        "- Метод загрузки и запуска: [инструкция Colab](docs/colab.md).",
        "",
        END,
    ]
    return "\n".join(lines)


def update_readme(run_id: str, readme_path: Path = ROOT / "README.md") -> bool:
    source = readme_path.read_text(encoding="utf-8")
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError("README должен содержать ровно одну пару маркеров REPORT:START/END")
    block = render(run_id)
    before = source.split(START, 1)[0]
    after = source.split(END, 1)[1]
    updated = before + block + after
    if updated == source:
        return False
    readme_path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    changed = update_readme(args.run_id)
    print("README обновлён" if changed else "README уже актуален")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
