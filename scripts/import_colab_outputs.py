"""Импорт результатов Colab: incoming/colab_outputs_<run_id>.zip → results/<run_id>/.

Что делает:
1. проверяет структуру zip (обязательные файлы, наличие run_id);
2. пересчитывает метрики ИЗ predictions.jsonl (не доверяя results.csv);
3. пишет results/<run_id>/ (results.csv, results_recomputed.csv, summary.csv, остальные артефакты);
4. обновляет раздел «Результаты» в Эксперимент_BTZSC.md таблицей с числами из артефактов.

Использование:
    python scripts/import_colab_outputs.py incoming/colab_outputs_run1.zip
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED = ("results.csv", "predictions.jsonl", "sample_manifest.csv", "run_config.json")
OPTIONAL = ("summary.csv", "environment.json", "errors.log", "manifest_info.json",
            "preflight.json", "ft_metrics.csv", "ft_config.json")
REPORT_PATH = ROOT / "Эксперимент_BTZSC.md"
RESULTS_MARK_START = "<!-- RESULTS:START -->"
RESULTS_MARK_END = "<!-- RESULTS:END -->"


class ImportError_(RuntimeError):
    """Структура архива не соответствует контракту."""


def run_id_from_name(name: str) -> str:
    m = re.match(r"colab_outputs_(.+)\.zip$", Path(name).name)
    if not m:
        raise ImportError_(f"имя архива должно быть colab_outputs_<run_id>.zip, получено {Path(name).name}")
    return m.group(1)


def validate(zf: zipfile.ZipFile) -> None:
    names = set(zf.namelist())
    if not names:
        raise ImportError_(
            "архив пуст (0 файлов). Значит, в каталоге прогона на Colab нечего было экспортировать: "
            "скорее всего MODE так и остался 'preflight' и ни smoke, ни evaluate не запускались. "
            "Порядок: MODE='smoke' → выполнить ячейку запуска → убедиться, что в results.csv есть строки "
            "→ только потом MODE='export'."
        )
    missing = [r for r in REQUIRED if r not in names]
    if missing:
        raise ImportError_(
            f"в архиве нет обязательных файлов: {', '.join(missing)}. "
            f"Что есть в архиве: {', '.join(sorted(names)) or '(ничего)'}"
        )


def read_predictions(zf: zipfile.ZipFile) -> list[dict]:
    raw = zf.read("predictions.jsonl").decode("utf-8")
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def rows_by_run_key(zf: zipfile.ZipFile) -> dict[str, dict]:
    """Строки results.csv по run_key: у каждого прогона свой манифест и свой размер выборки."""
    text = zf.read("results.csv").decode("utf-8")
    return {row["run_key"]: row for row in csv.DictReader(io.StringIO(text)) if row.get("run_key")}


def recompute(records: list[dict], reported: dict[str, dict] | None = None) -> list[dict]:
    """Пересчёт метрик из предсказаний, отдельно по каждому run_key.

    run_key включает манифест, поэтому прогоны на 20 и на 300 примеров не смешиваются;
    число классов берётся из results.csv (в выборку могли попасть не все классы).
    """
    from btzsc_ru.io_utils import deduplicate_predictions
    from btzsc_ru.metrics import compute_metrics

    reported = reported or {}
    records = deduplicate_predictions(records)
    groups: dict[tuple[str, str, str], dict[str, dict]] = {}
    for rec in records:
        key = (rec["model_id"], rec["dataset_key"], rec.get("run_key", ""))
        groups.setdefault(key, {})[rec["sample_id"]] = rec

    rows = []
    for (model_id, dataset_key, run_key), items in sorted(groups.items()):
        values = list(items.values())
        gold = [int(i["gold_index"]) for i in values]
        pred = [int(i["pred_index"]) for i in values]
        info = reported.get(run_key, {})
        n_classes = int(info["n_classes"]) if info.get("n_classes") else max(max(gold), max(pred)) + 1
        m = compute_metrics(gold, pred, n_classes)
        rows.append({
            "test": info.get("test", ""),
            "model_id": model_id,
            "dataset_key": dataset_key,
            "run_key": run_key,
            "n_samples_reported": info.get("n_samples", ""),
            "ms_per_example": info.get("ms_per_example", ""),
            "peak_vram_mb": info.get("peak_vram_mb", ""),
            "timestamp": info.get("timestamp", ""),
            **m,
        })
    return rows


def primary_rows(recomputed: list[dict]) -> list[dict]:
    """Для отчёта берём по одной строке на пару модель×датасет — с наибольшей выборкой."""
    best: dict[tuple[str, str], dict] = {}
    for row in recomputed:
        key = (row["model_id"], row["dataset_key"])
        current = best.get(key)
        if current is None or row["n"] > current["n"]:
            best[key] = row
    return sorted(best.values(), key=lambda r: (-r["macro_f1"], r["model_id"]))


def write_outputs(zf: zipfile.ZipFile, run_id: str, recomputed: list[dict], out_root: Path) -> Path:
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED + OPTIONAL:
        if name in zf.namelist():
            (out_dir / name).write_bytes(zf.read(name))
    # Подкаталоги прогона (сверка на моделях статьи, диагностика ST, графики):
    # раньше терялись при импорте.
    subdirs = ("replication/", "replication_st/", "baseline/", "ours_full/", "plots/")
    for name in zf.namelist():
        if name.startswith(subdirs) and not name.endswith("/"):
            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))
    fields = ["test", "model_id", "dataset_key", "run_key", "macro_f1", "accuracy", "macro_precision", "macro_recall",
              "n", "n_classes", "gold_class_coverage", "pred_class_coverage",
              "n_samples_reported", "ms_per_example", "peak_vram_mb", "timestamp"]
    with (out_dir / "results_recomputed.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in recomputed:
            writer.writerow({k: row.get(k, "") for k in fields})
    return out_dir


def compare_with_reported(zf: zipfile.ZipFile, recomputed: list[dict], tol: float = 1e-6) -> list[str]:
    """Сверка пересчитанных метрик с results.csv по run_key — однозначное соответствие."""
    reported = rows_by_run_key(zf)
    problems = []
    for row in recomputed:
        info = reported.get(row["run_key"])
        if not info or info.get("status") != "OK" or not info.get("macro_f1"):
            continue
        if abs(float(info["macro_f1"]) - row["macro_f1"]) > tol:
            problems.append(
                f"{row['model_id']} × {row['dataset_key']} (run_key {row['run_key'][:8]}): "
                f"results.csv={float(info['macro_f1']):.6f} пересчёт={row['macro_f1']:.6f}"
            )
    return problems


def render_table(run_id: str, recomputed: list[dict], mismatches: list[str]) -> str:
    recomputed = primary_rows(recomputed)
    lines = [
        RESULTS_MARK_START,
        f"### Результаты прогона `{run_id}`",
        "",
        "Числа пересчитаны из `predictions.jsonl` скриптом `scripts/import_colab_outputs.py`.",
        "",
        "| Модель | Датасет | macro-F1 | accuracy | macro-P | macro-R | N | Классов | Покрытие предсказаний |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(recomputed, key=lambda r: (-r["macro_f1"], r["model_id"])):
        lines.append(
            "| {model_id} | {dataset_key} | {macro_f1:.4f} | {accuracy:.4f} | {macro_precision:.4f} | "
            "{macro_recall:.4f} | {n} | {n_classes} | {pred_class_coverage:.2f} |".format(**row)
        )
    lines.append("")
    if mismatches:
        lines.append("Расхождения с `results.csv` из Colab:")
        lines.extend(f"- {m}" for m in mismatches)
    else:
        lines.append("Расхождений между `results.csv` из Colab и пересчётом нет.")
    lines.append(RESULTS_MARK_END)
    return "\n".join(lines)


def update_report(block: str, report_path: Path = REPORT_PATH) -> bool:
    if not report_path.exists():
        return False
    text = report_path.read_text(encoding="utf-8")
    if RESULTS_MARK_START in text and RESULTS_MARK_END in text:
        new = re.sub(
            re.escape(RESULTS_MARK_START) + r".*?" + re.escape(RESULTS_MARK_END),
            block,
            text,
            flags=re.S,
        )
    else:
        new = text.rstrip() + "\n\n" + block + "\n"
    report_path.write_text(new, encoding="utf-8")
    return True


MIN_OK_SHARE = 0.70


def check_before_overwrite(run_id: str, out_root: Path, force: bool) -> None:
    """Не затирать существующий прогон молча.

    Реальный случай: архив сломанного прогона импортировали под тем же run_id, он перезаписал
    хорошие результаты, а автосборка успела пересобрать по ним отчёт, графики и презентацию.
    """
    target = out_root / run_id
    if target.exists() and not force:
        raise ImportError_(
            f"каталог {target} уже существует. Импорт под тем же run_id перезапишет результаты.\n"
            f"Варианты: переименовать архив в colab_outputs_<новый_run_id>.zip "
            f"или запустить с --force, если перезапись осознанная."
        )


def check_run_quality(zf: zipfile.ZipFile, allow_failed: bool) -> float:
    """Отказ собирать отчёт по заведомо сломанному прогону."""
    text = zf.read("results.csv").decode("utf-8")
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r.get("status")]
    if not rows:
        raise ImportError_("в results.csv нет ни одной строки со статусом")
    ok_share = sum(1 for r in rows if r["status"] == "OK") / len(rows)
    if ok_share < MIN_OK_SHARE and not allow_failed:
        reasons = {r.get("reason", "")[:90] for r in rows if r["status"] == "FAILED"}
        raise ImportError_(
            f"успешных строк всего {ok_share:.0%} (нужно ≥ {MIN_OK_SHARE:.0%}) — прогон сломан, "
            f"отчёт по нему собирать нельзя.\nПричины отказов: {'; '.join(sorted(reasons)[:3])}\n"
            f"Если импорт всё же нужен (например, для разбора), добавьте --allow-failed."
        )
    return ok_share


def import_zip(zip_path: Path, out_root: Path | None = None, report_path: Path | None = None,
               *, force: bool = False, allow_failed: bool = False) -> dict:
    zip_path = Path(zip_path)
    out_root = Path(out_root) if out_root else ROOT / "results"
    report_path = Path(report_path) if report_path else REPORT_PATH
    run_id = run_id_from_name(zip_path.name)
    check_before_overwrite(run_id, out_root, force)
    with zipfile.ZipFile(zip_path) as zf:
        validate(zf)
        ok_share = check_run_quality(zf, allow_failed)
        records = read_predictions(zf)
        if not records:
            raise ImportError_("predictions.jsonl пуст — импортировать нечего")
        recomputed = recompute(records, rows_by_run_key(zf))
        mismatches = compare_with_reported(zf, recomputed)
        out_dir = write_outputs(zf, run_id, recomputed, out_root)
    block = render_table(run_id, recomputed, mismatches)
    updated = update_report(block, report_path)
    return {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "n_pairs": len(primary_rows(recomputed)),
        "n_run_keys": len(recomputed),
        "n_predictions": len(records),
        "ok_share": round(ok_share, 3),
        "mismatches": mismatches,
        "report_updated": updated,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--force", action="store_true",
                        help="перезаписать результаты существующего run_id (по умолчанию запрещено)")
    parser.add_argument("--allow-failed", action="store_true",
                        help="импортировать прогон, где успешных строк меньше 70 % (только для разбора)")
    args = parser.parse_args()
    info = import_zip(Path(args.zip_path), args.out_root, force=args.force, allow_failed=args.allow_failed)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
