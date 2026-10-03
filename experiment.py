"""Точка входа эксперимента BTZSC-RU.

Импорт модуля не имеет побочных эффектов: ни сети, ни загрузки моделей,
ни чтения датасетов. Тяжёлые зависимости (torch, transformers, datasets)
подтягиваются лениво внутри функций пакета `btzsc_ru`.

Режимы эксперимента:
    preflight | smoke | evaluate | finetune | export

Запуск — только в Google Colab (см. docs/colab.md). Локально допустимы
`--mode preflight --dry-run` и импорт для тестов.

Сохранённые функции MVP (реэкспорт, старый код продолжает работать):
    class_period       → btzsc_ru.reconstruction.infer_n_classes_by_hypothesis
    stratified_indices → btzsc_ru.sampling.stratified_indices
    prf_counts         → btzsc_ru.metrics.compute_metrics
    mean_pool          → btzsc_ru.adapters.EncoderAdapter._mean_pool
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from btzsc_ru import config as _config
from btzsc_ru import io_utils, metrics, reconstruction, runner, sampling
from btzsc_ru.config import DATASETS, MODELS, MODES, RunConfig

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "runs"

# ── совместимость с MVP ──────────────────────────────────────────────────────

def class_period(hypotheses: list[str]) -> int:
    """MVP-совместимая обёртка: период повторения блока гипотез."""
    return reconstruction.infer_n_classes_by_hypothesis(hypotheses)


def stratified_indices(refs, n: int, seed: int = _config.SEED):
    """MVP-совместимая обёртка над стратифицированным сэмплированием."""
    return sampling.stratified_indices([int(x) for x in refs], n, seed)


def prf_counts(y_true, y_pred, n_cls: int) -> dict:
    """MVP-совместимая обёртка: macro-P/R/F1 + accuracy."""
    return metrics.compute_metrics([int(x) for x in y_true], [int(x) for x in y_pred], n_cls)


def mean_pool(hidden, mask):
    """MVP-совместимая обёртка над mean-pooling энкодера."""
    from btzsc_ru.adapters import EncoderAdapter

    return EncoderAdapter._mean_pool(hidden, mask)


def quiet_logs() -> None:
    """Глушит служебный вывод HF: прогресс-бары, LOAD REPORT, предупреждения.

    Без этого полезный лог прогона тонет в сотнях строк «Loading weights» и таблицах
    UNEXPECTED/MISSING, которые к результату отношения не имеют.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DATASETS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Xet-клиент печатает свои «downloading bytes …» мимо tqdm и ломает структуру лога.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_config(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig(mode=args.mode, n_samples=args.n_samples)
    if getattr(args, "encoder_backend", None):
        cfg.encoder_backend = args.encoder_backend
    if args.datasets:
        cfg.dataset_keys = tuple(args.datasets)
    if args.models:
        cfg.model_ids = tuple(args.models)
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BTZSC-RU experiment runner")
    parser.add_argument("--mode", choices=MODES, default=_config.MODE)
    parser.add_argument("--run-id", default="local")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--no-anchor", action="store_true",
                        help="пропустить якорь к публикации (модели статьи на полном сплите) — не рекомендуется")
    parser.add_argument("--with-smoke", action="store_true", help="добавить быструю проверку на 20 примерах")
    parser.add_argument("--encoder-backend", choices=("hf", "st"), default=None,
                        help="st = считать энкодеры кодом статьи (SentenceTransformer) — диагностика эквивалентности")
    parser.add_argument("--with-full-baseline", action="store_true",
                        help="добавить модели статьи на увеличенной выборке (долго)")
    parser.add_argument("--with-full-ours", action="store_true",
                        help="добавить наши энкодеры на увеличенной выборке (долго)")
    parser.add_argument("--with-extra-models", action="store_true",
                        help="добавить остальные модели (tiny-энкодеры и LLM) сверх блока Б")
    parser.add_argument("--no-ru-extension", action="store_true",
                        help="пропустить русские датасеты (по умолчанию считаются: это наш вклад)")
    parser.add_argument("--no-replication", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-finetune", action="store_true",
                        help="пропустить дообучение rubert-tiny2 (по умолчанию считается: "
                             "это контрольная точка «разметка против размера модели», ~1 минута на T4)")
    # Старый флаг оставлен, чтобы не ломать уже записанные команды: дообучение и так включено.
    parser.add_argument("--with-finetune", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mirror-dir", default=None,
                        help="куда копировать результаты после каждого этапа (например, резервная папка)")
    parser.add_argument("--deadline-min", type=int, default=None,
                        help="общий дедлайн прогона в минутах; тяжёлые этапы за ним помечаются NOT_RUN")
    parser.add_argument("--full-cap", type=int, default=None,
                        help="потолок примеров на датасет для полных этапов (по умолчанию 3000; 0 = весь сплит)")
    parser.add_argument("--verbose-libs", action="store_true",
                        help="не глушить служебный вывод transformers/datasets (по умолчанию глушится)")
    parser.add_argument("--dry-run", action="store_true", help="печатает план и выходит, ничего не грузит")
    args = parser.parse_args(argv)
    if not args.verbose_libs:
        quiet_logs()

    if args.full_cap is not None:
        _config.FULL_TEST_CAP = None if args.full_cap == 0 else args.full_cap

    cfg = build_config(args)
    if cfg.mode == "replicate" and not args.models:
        from btzsc_ru.config import as_replication

        cfg = as_replication(cfg)
    out_dir = Path(args.out) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"mode={cfg.mode} run_id={args.run_id} config_hash={cfg.config_hash()}")
        if cfg.mode == "all":
            stages = ["preflight"] + (["smoke"] if args.with_smoke else []) \
                + ([] if args.no_anchor else ["anchor"]) + ["block_a", "block_b", "export"]
            extra = ([] if args.no_finetune else ["finetune"]) \
                + (["extra_models"] if args.with_extra_models else []) \
                + ([] if args.no_ru_extension else ["ru_extension"]) \
                + (["baseline"] if args.with_full_baseline else []) \
                + (["ours_full"] if args.with_full_ours else [])
            print("этапы: " + " → ".join(stages + extra + (["export_final"] if extra else [])))
            print(f"блок А: {len(_config.BLOCK_A_MODEL_IDS)} моделей статьи, "
                  f"блок Б: {len(_config.BLOCK_B_MODEL_IDS)} наших; "
                  f"{len(_config.BLOCK_DATASET_KEYS)} датасетов × {_config.BLOCK_N} примеров = "
                  f"{len(_config.BLOCK_A_MODEL_IDS) * len(_config.BLOCK_DATASET_KEYS) * _config.BLOCK_N:,} "
                  f"предсказаний на блок")
        print(f"n_samples={cfg.resolved_n()}")
        print("datasets: " + ", ".join(f"{d.key}[{d.hf_id}:{d.config}/{d.split}]" for d in DATASETS if d.key in cfg.dataset_keys))
        print("models: " + ", ".join(f"{m.model_id}@{m.revision[:8]}" for m in MODELS if m.model_id in cfg.model_ids))
        return 0

    if cfg.mode == "all":
        report = runner.run_all(cfg, out_dir, cache_dir=args.cache_dir, run_id=args.run_id,
                                mirror_dir=args.mirror_dir,
                                deadline_s=args.deadline_min * 60 if args.deadline_min else None,
                                smoke_first=args.with_smoke,
                                with_anchor=not args.no_anchor,
                                with_finetune=not args.no_finetune,
                                with_extra_models=args.with_extra_models,
                                with_ru_extension=not args.no_ru_extension,
                                with_baseline=args.with_full_baseline,
                                with_ours_full=args.with_full_ours)
        return 0 if all(v["status"] == "OK" for v in report.values()) else 1

    if cfg.mode == "preflight":
        report = runner.preflight(cfg, out_dir, cache_dir=args.cache_dir)
        ok = sum(1 for v in report["datasets"].values() if v["status"] == "OK")
        print(f"preflight: датасетов OK {ok}/{len(report['datasets'])}, отчёт {out_dir/'preflight.json'}")
        return 0

    if cfg.mode in {"smoke", "evaluate"}:
        path = runner.evaluate(cfg, out_dir, cache_dir=args.cache_dir, run_id=args.run_id)
        print(f"{cfg.mode}: записано {path}")
        return 0

    if cfg.mode == "block_a":
        path = runner.block_a(cfg, out_dir, cache_dir=args.cache_dir, run_id=args.run_id)
        print(f"блок А: записано {path}")
        return 0

    if cfg.mode == "block_b":
        path = runner.block_b(cfg, out_dir, cache_dir=args.cache_dir, run_id=args.run_id)
        print(f"блок Б: записано {path}")
        return 0

    if cfg.mode == "baseline":
        path = runner.baseline(cfg, out_dir, cache_dir=args.cache_dir, run_id=args.run_id)
        print(f"baseline: записано {path}")
        return 0

    if cfg.mode == "ours_full":
        path = runner.ours_full(cfg, out_dir, cache_dir=args.cache_dir, run_id=args.run_id)
        print(f"ours_full: записано {path}")
        return 0

    if cfg.mode == "reference":
        path = runner.reference(cfg, out_dir, cache_dir=args.cache_dir, run_id=args.run_id)
        print(f"reference: записано {path}")
        return 0

    if cfg.mode == "replicate":
        path = runner.replicate(cfg, out_dir, cache_dir=args.cache_dir, run_id=args.run_id)
        print(f"replicate: записано {path}")
        return 0

    if cfg.mode == "st_check":
        path = runner.st_check(cfg, out_dir, cache_dir=args.cache_dir, run_id=args.run_id)
        print(f"st_check: записано {path}")
        return 0

    if cfg.mode == "finetune":
        from btzsc_ru.finetune import DEFAULT_FT_CONFIG, run_finetune

        summary = run_finetune(DEFAULT_FT_CONFIG, out_dir, cache_dir=args.cache_dir)
        print(f"finetune: held-out macro-F1 {summary['heldout']['macro_f1']:.4f}")
        return 0

    if cfg.mode == "export":
        target = runner.export_zip(out_dir, args.run_id)
        print(f"export: {target}")
        return 0

    raise SystemExit(f"неизвестный режим: {cfg.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
