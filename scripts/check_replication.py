"""Сверка наших чисел с опубликованными числами статьи на её же моделях.

Читает `results/<run_id>/replication/results.csv` (наш прогон моделей статьи на agnews/imdb)
и сравнивает с `btzsc_ru.config.PAPER_REFERENCE` — значениями из `results.by_dataset`
лидерборда официального репозитория.

Вывод: `results/<run_id>/replication_check.csv` + markdown-раздел в `Эксперимент_BTZSC.md`
между маркерами REPLICATION.

    python scripts/check_replication.py --run-id run1
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btzsc_ru.config import PAPER_REFERENCE  # noqa: E402

REPORT = ROOT / "Эксперимент_BTZSC.md"
START = "<!-- REPLICATION:START -->"
END = "<!-- REPLICATION:END -->"
# Порог «совпало»: различия объёма выборки, версии библиотек и fp32/bf16 дают шум,
# но структурная ошибка в коде даёт отклонение куда больше.
TOL_OK = 0.03
TOL_WARN = 0.06
DATASET_TITLE = {
    "btzsc_agnews": "AG News",
    "btzsc_imdb": "IMDb",
    "btzsc_rottentomatoes": "Rotten Tomatoes",
    "btzsc_financialphrasebank": "Financial PhraseBank",
    "btzsc_emotiondair": "Emotion (DAIR)",
    "btzsc_massive": "MASSIVE intent",
    "btzsc_banking77": "Banking77",
}


def load_rows(run_dir: Path, sub: str = "replication") -> list[dict]:
    """Строки для сверки: из подкаталога, а если его нет — из основного results.csv.

    Этап `anchor` пишет модели статьи прямо в корневой results.csv с `test = baseline`,
    поэтому отдельного подкаталога у него нет.
    """
    path = run_dir / sub / "results.csv"
    if not path.exists() and sub == "replication":
        path = run_dir / "results.csv"
    if not path.exists():
        print(f"нет {path}: этап сверки не запускался — раздел отчёта остаётся заглушкой")
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") == "OK" and r.get("macro_f1")]
    # Сверка касается только моделей статьи: строки нашего блока сюда попадать не должны.
    allowed_tests = {"replication", "replication_st", "baseline", "block_a", ""}
    rows = [r for r in rows if r.get("test", "") in allowed_tests]
    rows = [r for r in rows if r["model_id"] in PAPER_REFERENCE]

    # Вердикт «совпало/разошлось» имеет смысл только для строк, сравнимых с публикацией:
    # у статьи (A.4) метрика считается на полном сплите, у нас — не всегда.
    comparable, skipped = [], []
    for row in rows:
        flag = str(row.get("comparable_to_paper", "")).strip().lower()
        if flag in {"true", "1", "yes"}:
            comparable.append(row)
        elif flag in {"false", "0", "no"}:
            skipped.append(row)
        else:                      # старые файлы без колонки — решаем по размеру выборки
            from btzsc_ru.paper_compat import comparable_to_paper as _cmp

            n = int(float(row.get("n_samples") or 0)) or None
            (comparable if _cmp(row["dataset_key"], n) else skipped).append(row)
    if skipped:
        print(f"  вне сверки (несравнимо с публикацией): {len(skipped)} строк, "
              f"например {skipped[0]['model_id'].split('/')[-1]} × {skipped[0]['dataset_key']} "
              f"— {skipped[0].get('comparability_reason') or 'подвыборка вместо полного сплита'}")
    rows = comparable
    # В каталоге копятся строки нескольких запусков: берём по одной на пару модель×датасет
    # (самую большую выборку, при равенстве — последнюю по времени).
    best: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["model_id"], row["dataset_key"])
        cur = best.get(key)
        if cur is None or (int(row.get("n_samples") or 0), row.get("timestamp", "")) >= (
            int(cur.get("n_samples") or 0), cur.get("timestamp", "")
        ):
            best[key] = row
    return list(best.values())


def compare(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        ref = PAPER_REFERENCE.get(row["model_id"], {}).get(row["dataset_key"])
        if ref is None:
            continue
        ours = float(row["macro_f1"])
        diff = ours - ref
        verdict = "совпало" if abs(diff) <= TOL_OK else ("на границе" if abs(diff) <= TOL_WARN else "расхождение")
        out.append({
            "model_id": row["model_id"],
            "dataset_key": row["dataset_key"],
            "paper_macro_f1": ref,
            "our_macro_f1": round(ours, 4),
            "diff": round(diff, 4),
            "abs_diff": round(abs(diff), 4),
            "n": row.get("n_samples", ""),
            "verdict": verdict,
        })
    return sorted(out, key=lambda r: -r["abs_diff"])


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return None if den == 0 else num / den


def render(run_id: str, rows: list[dict], skipped: int = 0) -> str:
    if not rows:
        return (
            f"{START}\n**Статус: ожидается прогон.** Сверка запускается режимом `replicate` "
            "(в режиме `all` — автоматически): те же пять чекпоинтов, что в статье "
            "(all-MiniLM-L6-v2, e5-base-v2, e5-large-v2, bge-base-en-v1.5, bge-large-en-v1.5), "
            "прогоняются нашим кодом на agnews и imdb, и числа сравниваются с опубликованными.\n"
            f"Результаты появятся здесь после импорта архива с каталогом `replication/`.\n{END}"
        )
    ours = [r["our_macro_f1"] for r in rows]
    paper = [r["paper_macro_f1"] for r in rows]
    corr = pearson(paper, ours)
    mae = statistics.mean(r["abs_diff"] for r in rows)
    ok = sum(1 for r in rows if r["verdict"] == "совпало")

    lines = [
        START,
        f"### Сверка кода на моделях статьи (прогон `{run_id}`)",
        "",
        "Те же чекпоинты, что в статье, прогнаны **нашим** кодом на её же датасетах. "
        "Эталон — `results.by_dataset` лидерборда официального репозитория.",
        "",
        "| Модель статьи | Датасет | статья | мы | Δ | N у нас | вердикт |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["dataset_key"], -x["our_macro_f1"])):
        lines.append(
            f"| {r['model_id']} | {DATASET_TITLE.get(r['dataset_key'], r['dataset_key'])} | "
            f"{r['paper_macro_f1']:.3f} | {r['our_macro_f1']:.3f} | {r['diff']:+.3f} | {r['n']} | {r['verdict']} |"
        )
    lines += [
        "",
        f"**Итог сверки:** пар {len(rows)}, совпало в пределах ±{TOL_OK:.2f} — {ok}; "
        f"средняя абсолютная разница {mae:.3f}"
        + (f"; корреляция Пирсона с числами статьи {corr:.3f}." if corr is not None else "."),
        "",
        "Что означает расхождение: наш код воспроизводит протокол статьи (та же вербализация, "
        "cosine, argmax, macro-F1), но отличается объёмом выборки, версиями библиотек и точностью "
        "вычислений, поэтому идеального совпадения до третьего знака не ожидается. Значимым "
        f"считается отклонение больше ±{TOL_WARN:.2f} — оно указывало бы на ошибку в реализации.",
        END,
    ]
    return "\n".join(lines)


def update_report(block: str, report: Path = REPORT) -> bool:
    if not report.exists():
        return False
    text = report.read_text(encoding="utf-8")
    if START in text and END in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), block, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n## 12. Сверка кода на моделях статьи\n\n" + block + "\n"
    report.write_text(text, encoding="utf-8")
    return True


def render_st_block(run_dir: Path, hf_rows: list[dict]) -> str:
    """Диагностика: те же модели, но код-путь статьи (SentenceTransformer)."""
    st_rows = load_rows(run_dir, "replication_st")
    if not st_rows:
        return ""
    hf_by_key = {(r["model_id"], r["dataset_key"]): float(r["macro_f1"]) for r in hf_rows}
    lines = [
        "",
        "### Диагностика: наш код против кода статьи на тех же чекпоинтах",
        "",
        "`SentenceTransformer.encode(normalize_embeddings=True)` — это буквально код-путь авторов.",
        "",
        "| Модель | Датасет | наш код | код статьи (ST) | разница | опубликовано |",
        "|---|---|---:|---:|---:|---:|",
    ]
    max_gap = 0.0
    for row in sorted(st_rows, key=lambda r: (r["model_id"], r["dataset_key"])):
        key = (row["model_id"], row["dataset_key"])
        st_value = float(row["macro_f1"])
        hf_value = hf_by_key.get(key)
        ref = PAPER_REFERENCE.get(row["model_id"], {}).get(row["dataset_key"])
        gap = abs(st_value - hf_value) if hf_value is not None else None
        if gap is not None:
            max_gap = max(max_gap, gap)
        lines.append(
            f"| {row['model_id']} | {DATASET_TITLE.get(row['dataset_key'], row['dataset_key'])} | "
            f"{'—' if hf_value is None else f'{hf_value:.4f}'} | {st_value:.4f} | "
            f"{'—' if gap is None else f'{gap:.4f}'} | {'—' if ref is None else f'{ref:.3f}'} |"
        )
    lines += [
        "",
        f"Максимальное расхождение между нашей реализацией и кодом статьи: **{max_gap:.4f}**.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="run1")
    parser.add_argument("--results-root", default=str(ROOT / "results"))
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()

    run_dir = Path(args.results_root) / args.run_id
    hf_rows = load_rows(run_dir)
    rows = compare(hf_rows)
    block = render(args.run_id, rows)
    st_block = render_st_block(run_dir, hf_rows)
    if st_block:
        block = block.replace(END, st_block + "\n" + END)
    update_report(block, Path(args.report))
    if not rows:
        print("сверка пока не выполнена: в отчёт записана заглушка")
        return 0
    out_csv = run_dir / "replication_check.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), )
        writer.writeheader()
        writer.writerows(rows)
    print(f"пар сверки: {len(rows)} → {out_csv}")
    for r in rows:
        print(f"  {r['model_id']:42} {r['dataset_key']:13} статья {r['paper_macro_f1']:.3f} "
              f"мы {r['our_macro_f1']:.3f} Δ {r['diff']:+.3f} {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
