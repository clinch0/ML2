"""Режим finetune (пункт 8+ задания): дообучение rubert-tiny2 с головой классификации.

Код готов к запуску В COLAB. Локально и в CI он только импортируется —
обучение здесь не запускается (протокол эксперимента, запрет №2).

Конфигурация — из протокол эксперимента → «Обучение (8+)»:
    датасет MonoHime/ru_sentiment_dataset, train ≤ 2000, dev выделяется из train,
    held-out = опубликованный validation, seed 42, max_len 256, batch 16, lr 2e-5, 2 эпохи.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

FT_METRICS_COLUMNS = [
    "run_id",
    "stage",           # dev | heldout
    "model_id",
    "revision",
    "dataset_key",
    "split",
    "epoch",
    "n_train",
    "n_eval",
    "macro_f1",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "train_seconds",
    "device",
    "status",
    "reason",
]


@dataclass(frozen=True)
class FineTuneConfig:
    model_id: str = "cointegrated/rubert-tiny2"
    revision: str = "e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae"
    dataset_key: str = "ru_sentiment"
    train_split: str = "train"
    heldout_split: str = "validation"
    max_train: int = 2000
    dev_fraction: float = 0.1
    seed: int = 42
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 2
    num_labels: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_FT_CONFIG = FineTuneConfig()


class FTMetricsWriter:
    """Отдельный writer: результаты обучения не смешиваются с zero-shot results.csv."""

    def __init__(self, path: Path | str) -> None:
        import csv

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8", newline="") as fh:
                csv.DictWriter(fh, fieldnames=FT_METRICS_COLUMNS).writeheader()

    def append(self, row: dict) -> None:
        import csv

        clean = {k: row.get(k, "") for k in FT_METRICS_COLUMNS}
        with self.path.open("a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FT_METRICS_COLUMNS).writerow(clean)


def split_train_dev(n: int, dev_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Детерминированное разбиение train → (train, dev). Held-out не трогается."""
    import numpy as np

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_dev = max(1, int(round(n * dev_fraction))) if n > 1 else 0
    dev = sorted(int(i) for i in order[:n_dev])
    train = sorted(int(i) for i in order[n_dev:])
    return train, dev


def run_finetune(cfg: FineTuneConfig, out_dir: Path | str, *, cache_dir: str | None = None) -> dict:  # pragma: no cover
    """Обучение и оценка. Вызывается ТОЛЬКО из Colab (MODE=finetune)."""
    import time

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from .config import DATASETS_BY_KEY
    from .data import load_hf_split
    from .metrics import compute_metrics

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = FTMetricsWriter(out_dir / "ft_metrics.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    spec = DATASETS_BY_KEY[cfg.dataset_key]
    train_spec = type(spec)(**{**spec.__dict__, "split": cfg.train_split})
    heldout_spec = type(spec)(**{**spec.__dict__, "split": cfg.heldout_split})
    train_ds = load_hf_split(train_spec, cache_dir=cache_dir, limit=cfg.max_train)
    heldout_ds = load_hf_split(heldout_spec, cache_dir=cache_dir)

    texts = [str(t) for t in train_ds[spec.text_column]]
    labels = [int(v) for v in train_ds[spec.label_column]]
    tr_idx, dev_idx = split_train_dev(len(texts), cfg.dev_fraction, cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id, revision=cfg.revision, num_labels=cfg.num_labels
    ).to(device)

    class _DS(Dataset):
        def __init__(self, idx):
            self.idx = idx

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            return texts[j], labels[j]

    def collate(batch):
        bt, by = zip(*batch)
        enc = tokenizer(list(bt), padding=True, truncation=True, max_length=cfg.max_length, return_tensors="pt")
        enc["labels"] = torch.tensor(by, dtype=torch.long)
        return enc

    loader = DataLoader(_DS(tr_idx), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    def evaluate(eval_texts, eval_labels):
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(eval_texts), cfg.batch_size):
                enc = tokenizer(
                    eval_texts[start : start + cfg.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=cfg.max_length,
                    return_tensors="pt",
                ).to(device)
                preds.extend(model(**enc).logits.argmax(dim=-1).cpu().tolist())
        return compute_metrics(list(eval_labels), preds, cfg.num_labels)

    t0 = time.perf_counter()
    summary = {}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optim.step()
            optim.zero_grad()
        dev_metrics = evaluate([texts[i] for i in dev_idx], [labels[i] for i in dev_idx])
        writer.append(
            {
                "stage": "dev",
                "model_id": cfg.model_id,
                "revision": cfg.revision,
                "dataset_key": cfg.dataset_key,
                "split": f"{cfg.train_split}[dev]",
                "epoch": epoch,
                "n_train": len(tr_idx),
                "n_eval": len(dev_idx),
                "train_seconds": round(time.perf_counter() - t0, 1),
                "device": device,
                "status": "OK",
                **{k: dev_metrics[k] for k in ("macro_f1", "accuracy", "macro_precision", "macro_recall")},
            }
        )
        summary[f"dev_epoch_{epoch}"] = dev_metrics

    ho_texts = [str(t) for t in heldout_ds[spec.text_column]]
    ho_labels = [int(v) for v in heldout_ds[spec.label_column]]
    ho_metrics = evaluate(ho_texts, ho_labels)
    writer.append(
        {
            "stage": "heldout",
            "model_id": cfg.model_id,
            "revision": cfg.revision,
            "dataset_key": cfg.dataset_key,
            "split": cfg.heldout_split,
            "epoch": cfg.epochs,
            "n_train": len(tr_idx),
            "n_eval": len(ho_labels),
            "train_seconds": round(time.perf_counter() - t0, 1),
            "device": device,
            "status": "OK",
            **{k: ho_metrics[k] for k in ("macro_f1", "accuracy", "macro_precision", "macro_recall")},
        }
    )
    summary["heldout"] = ho_metrics
    (out_dir / "ft_config.json").write_text(
        __import__("json").dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
