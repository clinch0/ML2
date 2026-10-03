"""Единый сборщик отчётов: zip из Colab → числа в отчёте, графики, сверка, презентация.

Вызывается вручную и автоматически (workflow .github/workflows/import-results.yml,
триггер — появление colab_outputs_*.zip в репозитории).

Что делает по порядку:
  1. находит архивы прогонов (incoming/*.zip и colab_outputs_*.zip в корне);
  2. импортирует каждый: проверка структуры, пересчёт метрик из predictions.jsonl → results/<run_id>/;
  3. строит графики figures/*.png;
  4. сверяет наши числа с числами статьи на её моделях (если был режим replicate);
  5. обновляет числовой отчёт и графики в README.md;
  6. пересобирает презентацию, если нужны слайды.

    python scripts/build_reports.py                 # обработать всё, что лежит
    python scripts/build_reports.py --delete-zip    # с удалением архивов после импорта
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON = sys.executable


def find_archives() -> list[Path]:
    found = sorted(ROOT.glob("incoming/colab_outputs_*.zip")) + sorted(ROOT.glob("colab_outputs_*.zip"))
    # «colab_outputs_run1 (2).zip» — тоже валидное имя после скачивания браузером
    found += sorted(p for p in ROOT.glob("colab_outputs_*(*).zip") if p not in found)
    found += sorted(p for p in ROOT.glob("incoming/colab_outputs_*(*).zip") if p not in found)
    seen, out = set(), []
    for p in found:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def normalise_name(path: Path) -> Path:
    """«colab_outputs_run1 (3).zip» → «incoming/colab_outputs_run1.zip» (run_id без мусора)."""
    stem = path.stem
    if "(" in stem:
        stem = stem.split("(")[0].strip()
    target = ROOT / "incoming" / f"{stem}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve() != target.resolve():
        target.write_bytes(path.read_bytes())
    return target


def run(cmd: list[str], *, required: bool = True) -> bool:
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if out:
        print(out[-3000:])
    if proc.returncode != 0:
        print(f"[неуспех, код {proc.returncode}]")
        if err:
            print(err[-1500:])
        if required:
            raise SystemExit(proc.returncode)
        return False
    return True


def run_id_of(path: Path) -> str:
    return path.stem.removeprefix("colab_outputs_")


def figure_sample_size(run_id: str) -> str:
    """Берём наибольшую успешную выборку основного прогона, исключая baseline."""
    path = ROOT / "results" / run_id / "results.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts: dict[str, int] = {}
    for row in rows:
        size = row.get("n_samples", "")
        if row.get("status") == "OK" and size and row.get("test") != "baseline":
            counts[size] = counts.get(size, 0) + 1
    if not counts:
        raise ValueError(f"В {path} нет успешных строк основного прогона")
    return max(counts, key=int)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete-zip", action="store_true",
                        help="удалить архив после успешного импорта (по умолчанию выключено — режим отладки)")
    parser.add_argument("--skip-pptx", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="разрешить перезапись результатов уже импортированного run_id")
    args = parser.parse_args()

    archives = find_archives()
    if not archives:
        print("архивов colab_outputs_*.zip не найдено — нечего собирать")
        return 0

    processed: list[str] = []
    failed: list[str] = []
    for raw in archives:
        target = normalise_name(raw)
        run_id = run_id_of(target)
        print(f"\n=== прогон {run_id}: {raw.name} ===")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        run_dir = ROOT / "results" / run_id
        digest_path = run_dir / "source_archive.sha256"
        if run_dir.exists() and digest_path.exists() and digest_path.read_text().strip() == digest and not args.force:
            print("этот архив уже импортирован — пропуск")
            continue
        import_cmd = [PYTHON, "scripts/import_colab_outputs.py", str(target)]
        if args.force or run_dir.exists():
            import_cmd.append("--force")
        if not run(import_cmd, required=False):
            print("импорт отклонён (см. причину выше) — отчёт не обновляется")
            failed.append(raw.name)
            continue
        processed.append(run_id)

        run([PYTHON, "scripts/make_figures.py", "--run-id", run_id,
             "--n", figure_sample_size(run_id)])
        run([PYTHON, "scripts/check_replication.py", "--run-id", run_id], required=False)
        run([PYTHON, "scripts/update_readme_report.py", "--run-id", run_id])
        digest_path.write_text(digest + "\n", encoding="utf-8")

        # --- удаление обработанного архива ---
        # Пока отключено намеренно: на отладке архивы нужны, чтобы перезапускать импорт.
        # Включается флагом --delete-zip (в workflow он закомментирован).
        if args.delete_zip:
            for candidate in {raw, target}:
                if candidate.exists() and candidate.name.startswith("colab_outputs_"):
                    candidate.unlink()
                    print(f"удалён архив: {candidate.relative_to(ROOT)}")
        else:
            print("архив оставлен на месте (удаление включается флагом --delete-zip)")

    if not args.skip_pptx:
        run([PYTHON, "scripts/build_btzsc_pptx.py"], required=False)

    print(f"\nготово. Обработано прогонов: {', '.join(processed)}")
    if failed:
        print("не импортированы:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
