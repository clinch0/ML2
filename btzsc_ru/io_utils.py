"""Запись артефактов прогона, run_key, resume, бюджет времени, очистка памяти."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

RESULTS_COLUMNS = [
    "run_id",
    "test",          # какой именно тест: block_a | block_b | smoke | replication | ...
    "run_key",
    "model_id",
    "revision",
    "role",
    "dataset_key",
    "hf_id",
    "config",
    "split",
    "language",
    "n_samples",
    "n_classes",
    "macro_f1",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "gold_class_coverage",
    "pred_class_coverage",
    "ms_per_example",
    "load_seconds",
    "peak_vram_mb",
    "truncated_prompts",
    "comparable_to_paper",   # можно ли класть это число рядом с публикацией (A.4: полный сплит)
    "comparability_reason",
    "device",
    "dtype",
    "batch_size",
    "status",
    "reason",
    "timestamp",
]

STATUSES = ("OK", "PARTIAL", "FAILED", "SKIPPED", "NOT_RUN")

# Значения колонки test: по ней три разных теста больше не смешиваются в одном файле.
TESTS = ("block_a", "block_b", "smoke", "replication", "replication_st", "finetune",
         "extra_models", "ru_extension", "baseline", "ours_full", "evaluate")


def dedup_results(rows: list[dict]) -> list[dict]:
    """Одна строка на (test, model_id, dataset_key, n_samples): выигрывает последняя.

    Прогон может повторяться (resume, другой размер выборки, перезапуск этапа), и без
    дедупликации в отчёт попадает случайная из дублей — числа становятся невоспроизводимыми.
    """
    best: dict[tuple, dict] = {}
    for row in rows:
        key = (row.get("test", ""), row.get("model_id", ""), row.get("dataset_key", ""),
               str(row.get("n_samples", "")))
        best[key] = row
    return list(best.values())


def run_key(
    model_id: str,
    revision: str,
    dataset_key: str,
    config: str | None,
    split: str,
    manifest_hash: str,
    config_hash: str,
) -> str:
    """Ключ resume: модель+revision+датасет+config+split+manifest+config_hash."""
    payload = "|".join(
        [model_id, revision or "", dataset_key, config or "", split, manifest_hash, config_hash]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ResultsWriter:
    """Инкрементальная запись results.csv: дозапись строки после каждой пары модель×датасет."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8", newline="") as fh:
                csv.DictWriter(fh, fieldnames=RESULTS_COLUMNS).writeheader()

    def existing_keys(self) -> set[str]:
        if not self.path.exists():
            return set()
        with self.path.open("r", encoding="utf-8", newline="") as fh:
            return {row["run_key"] for row in csv.DictReader(fh) if row.get("run_key")}

    def append(self, row: dict) -> None:
        if row.get("status") not in STATUSES:
            raise ValueError(f"недопустимый status: {row.get('status')}")
        if row.get("test") not in TESTS:
            raise ValueError(f"недопустимый test: {row.get('test')!r}; ожидается одно из {TESTS}")
        clean = {k: row.get(k, "") for k in RESULTS_COLUMNS}
        clean.setdefault("timestamp", "")
        if not clean["timestamp"]:
            clean["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self.path.open("a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=RESULTS_COLUMNS).writerow(clean)

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))


class PredictionsWriter:
    """predictions.jsonl: дозапись после каждого батча (устойчивость к обрыву Colab)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def done_sample_ids(self, key: str) -> set[str]:
        done: set[str] = set()
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("run_key") == key:
                    done.add(rec["sample_id"])
        return done

    def append_batch(self, records: list[dict]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def read(self, key: str | None = None) -> list[dict]:
        out = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if key is None or rec.get("run_key") == key:
                    out.append(rec)
        return out


def deduplicate_predictions(records: list[dict]) -> list[dict]:
    """Последняя запись по (run_key, sample_id) побеждает; порядок сохраняется."""
    seen: dict[tuple[str, str], int] = {}
    out: list[dict] = []
    for rec in records:
        k = (rec.get("run_key", ""), rec["sample_id"])
        if k in seen:
            out[seen[k]] = rec
        else:
            seen[k] = len(out)
            out.append(rec)
    return out


@dataclass
class TimeBudget:
    """Мягкий бюджет Colab: проверка между батчами, запас на экспорт."""

    total_s: int = 7200
    reserve_s: int = 600
    started: float = 0.0

    def __post_init__(self) -> None:
        if not self.started:
            self.started = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def remaining(self) -> float:
        return self.total_s - self.reserve_s - self.elapsed()

    def exhausted(self) -> bool:
        return self.remaining() <= 0


def code_revision() -> dict:
    """Хэш исходного кода для воспроизводимости без истории Git."""
    import hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for path in sorted((root / "btzsc_ru").glob("*.py")):
        digest.update(path.read_bytes())
    return {"code_sha256": digest.hexdigest()[:16]}


def environment_info() -> dict:
    """Снимок среды. Без падения, если torch не установлен."""
    info = {
        **code_revision(),      # чем именно порождён артефакт
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:  # pragma: no cover — зависит от среды Colab
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2, 1
            )
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"not available: {exc}"
        info["cuda_available"] = False
    try:  # pragma: no cover
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        info["transformers"] = "not available"
    return info


def write_json(path: Path | str, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def log_error(path: Path | str, message: str) -> None:
    """Дописывает строку в errors.log и НИКОГДА не бросает исключение.

    Логирование не должно ронять прогон: в Colab внешнее хранилище может быть недоступно
    (`OSError: Transport endpoint is not connected`), и падать из-за записи лога нельзя —
    иначе теряется весь прогон вместе с уже посчитанными результатами.
    """
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:  # noqa: BLE001
        print(f"[лог недоступен: {exc}] {line}")


def free_memory(*objects: object) -> None:
    """del + gc.collect() + empty_cache() после каждой модели (протокол эксперимента)."""
    for obj in objects:
        del obj
    gc.collect()
    try:  # pragma: no cover
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def peak_vram_mb() -> float | None:  # pragma: no cover — требует GPU
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    except Exception:  # noqa: BLE001
        return None
    return None
