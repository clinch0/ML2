"""Сэмплирование и sample_manifest.csv.

Один манифест (seed=42) используется всеми моделями: он фиксирует, какие именно
примеры оцениваются, и входит в ключ resume (см. io_utils.run_key).
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

MANIFEST_COLUMNS = ["dataset_key", "sample_id", "source_index", "gold_index", "gold_label", "n_classes"]


@dataclass(frozen=True)
class ManifestRow:
    dataset_key: str
    sample_id: str
    source_index: int
    gold_index: int
    gold_label: str
    n_classes: int

    def as_dict(self) -> dict:
        return {
            "dataset_key": self.dataset_key,
            "sample_id": self.sample_id,
            "source_index": self.source_index,
            "gold_index": self.gold_index,
            "gold_label": self.gold_label,
            "n_classes": self.n_classes,
        }


def stratified_indices(gold: list[int], n: int | None, seed: int = 42) -> list[int]:
    """Стратифицированная выборка индексов.

    При n=None возвращает все индексы. Классы обходятся по кругу, поэтому при
    маленьком n покрытие классов максимально; порядок внутри класса задан seed.
    """
    import numpy as np

    total = len(gold)
    if n is None or n >= total:
        return list(range(total))
    if n <= 0:
        return []

    rng = np.random.default_rng(seed)
    buckets: dict[int, list[int]] = {}
    for i, g in enumerate(gold):
        buckets.setdefault(int(g), []).append(i)
    for c in buckets:
        order = rng.permutation(len(buckets[c]))
        buckets[c] = [buckets[c][k] for k in order]

    classes = sorted(buckets)
    chosen: list[int] = []
    pos = {c: 0 for c in classes}
    while len(chosen) < n:
        progressed = False
        for c in classes:
            if len(chosen) >= n:
                break
            if pos[c] < len(buckets[c]):
                chosen.append(buckets[c][pos[c]])
                pos[c] += 1
                progressed = True
        if not progressed:
            break
    chosen.sort()
    return chosen


def build_manifest_rows(
    dataset_key: str,
    gold: list[int],
    verbalizers: list[str],
    indices: list[int],
) -> list[ManifestRow]:
    return [
        ManifestRow(
            dataset_key=dataset_key,
            sample_id=f"{dataset_key}:{idx}",
            source_index=int(idx),
            gold_index=int(gold[idx]),
            gold_label=verbalizers[int(gold[idx])],
            n_classes=len(verbalizers),
        )
        for idx in indices
    ]


def write_manifest(rows: list[ManifestRow], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return path


def read_manifest(path: Path | str) -> list[ManifestRow]:
    path = Path(path)
    out: list[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            out.append(
                ManifestRow(
                    dataset_key=rec["dataset_key"],
                    sample_id=rec["sample_id"],
                    source_index=int(rec["source_index"]),
                    gold_index=int(rec["gold_index"]),
                    gold_label=rec["gold_label"],
                    n_classes=int(rec["n_classes"]),
                )
            )
    return out


def manifest_hash(rows: list[ManifestRow]) -> str:
    payload = "|".join(f"{r.dataset_key}:{r.source_index}:{r.gold_index}" for r in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def class_coverage(rows: list[ManifestRow]) -> dict[str, float]:
    """Доля классов датасета, представленных в выборке."""
    per_ds: dict[str, set[int]] = {}
    n_cls: dict[str, int] = {}
    for r in rows:
        per_ds.setdefault(r.dataset_key, set()).add(r.gold_index)
        n_cls[r.dataset_key] = r.n_classes
    return {k: len(v) / n_cls[k] for k, v in per_ds.items()}
