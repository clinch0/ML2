"""Оркестрация прогона: preflight / smoke / evaluate / finetune / export.

Запускается только в Colab. Локально модуль импортируется (без побочных
эффектов) и проверяется тестами на синтетике.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import io_utils
from .config import DATASETS_BY_KEY, EXTRA_MODEL_IDS, MODELS_BY_ID, RU_EXTENSION_KEYS, RunConfig
from .paper_compat import comparable_to_paper
from .sampling import (
    build_manifest_rows,
    class_coverage,
    manifest_hash,
    read_manifest,
    stratified_indices,
    write_manifest,
)


def build_manifest(cfg: RunConfig, out_dir: Path, *, cache_dir: str | None = None) -> Path:
    """Создаёт единый sample_manifest.csv для всех моделей."""
    from .data import load_task

    rows = []
    for key in cfg.dataset_keys:
        task = load_task(key, cache_dir=cache_dir)
        idx = stratified_indices(task.gold, cfg.resolved_n(), cfg.seed)
        rows.extend(build_manifest_rows(key, task.gold, task.verbalizers, idx))
    path = write_manifest(rows, out_dir / "sample_manifest.csv")
    io_utils.write_json(
        out_dir / "manifest_info.json",
        {
            "manifest_hash": manifest_hash(rows),
            "n_rows": len(rows),
            "requested_n": cfg.resolved_n(),
            "mode": cfg.mode,
            "dataset_keys": list(cfg.dataset_keys),
            "class_coverage": class_coverage(rows),
        },
    )
    return path


def manifest_is_stale(cfg: RunConfig, out_dir: Path) -> str:
    """Причина, по которой существующий манифест не годится для текущего конфига, или ''.

    Без этой проверки прогон с тем же RUN_ID молча переиспользовал манифест предыдущего режима:
    evaluate с n=300 считался по 20 примерам smoke-манифеста (реальный случай, прогон run1).
    """
    info_path = Path(out_dir) / "manifest_info.json"
    if not (Path(out_dir) / "sample_manifest.csv").exists():
        return "манифеста нет"
    if not info_path.exists():
        return "нет manifest_info.json — размер прежней выборки неизвестен"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if "requested_n" not in info:
        return "manifest_info.json старого формата, без requested_n"
    # None = «весь сплит» (режим replicate). Раньше int(None) ронял этап сверки при resume.
    was, want = info["requested_n"], cfg.resolved_n()
    human = lambda v: "весь сплит" if v is None else f"{v} примеров"  # noqa: E731
    if (was is None) != (want is None) or (was is not None and int(was) != int(want)):
        return f"выборка была на {human(was)}, запрошено {human(want)}"
    have = set(info.get("dataset_keys", []))
    want = set(cfg.dataset_keys)
    if not want <= have:
        missing = ", ".join(sorted(want - have))
        return f"в манифесте нет датасетов: {missing}"
    return ""


def preflight(cfg: RunConfig, out_dir: Path, *, cache_dir: str | None = None) -> dict:
    """Проверка контрактов без запуска моделей: колонки, splits, число классов."""
    from .data import load_task

    report = {"mode": "preflight", "config_hash": cfg.config_hash(), "datasets": {}, "models": {}}
    for key in cfg.dataset_keys:
        spec = DATASETS_BY_KEY[key]
        try:
            # 400 строк хватает минимум на 5 примеров даже при 77 классах; хвост отрезается
            task = load_task(
                key,
                cache_dir=cache_dir,
                limit=400 if spec.kind == "btzsc_pairs" else 50,
                trim_tail=spec.kind == "btzsc_pairs",
            )
            report["datasets"][key] = {
                "status": "OK",
                "hf_id": spec.hf_id,
                "config": spec.config,
                "split": spec.split,
                "n_classes": task.n_classes,
                "n_samples_probe": task.n_samples,
                "verbalizers_head": task.verbalizers[:3],
            }
        except Exception as exc:  # noqa: BLE001
            report["datasets"][key] = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
            io_utils.log_error(out_dir / "errors.log", f"preflight {key}: {exc}")
    for model_id in cfg.model_ids:
        spec = MODELS_BY_ID[model_id]
        report["models"][model_id] = {
            "role": spec.role,
            "revision": spec.revision,
            "pooling": spec.pooling,
            "query_prefix": spec.query_prefix,
            "status": "NOT_RUN",
        }
    report["environment"] = io_utils.environment_info()
    io_utils.write_json(out_dir / "preflight.json", report)
    return report


def run_pair(
    adapter,
    task,
    manifest_rows,
    *,
    run_id: str,
    key: str,
    model_spec,
    dataset_spec,
    batch_size: int,
    predictions: io_utils.PredictionsWriter,
    budget: io_utils.TimeBudget,
    progress=None,
) -> dict:
    """Одна пара модель×датасет с сохранением после каждого батча и resume."""
    from .metrics import compute_metrics

    done = predictions.done_sample_ids(key)
    pending = [r for r in manifest_rows if r.sample_id not in done]
    t0 = time.perf_counter()
    processed = 0
    interrupted = False

    starts = list(range(0, len(pending), batch_size))
    for start in starts:
        if budget.exhausted():
            interrupted = True
            break
        chunk = pending[start : start + batch_size]
        texts = [task.texts[r.source_index] for r in chunk]
        preds = adapter.predict(texts, task.verbalizers, batch_size=batch_size)
        predictions.append_batch(
            [
                {
                    "run_id": run_id,
                    "run_key": key,
                    "model_id": model_spec.model_id,
                    "dataset_key": dataset_spec.key,
                    "sample_id": r.sample_id,
                    "pred_index": int(p),
                    "gold_index": int(r.gold_index),
                }
                for r, p in zip(chunk, preds)
            ]
        )
        processed += len(chunk)
        if progress is not None:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            progress.update(
                dataset_spec.key,
                done=len(done) + processed,
                total=len(manifest_rows),
                ms_per_example=elapsed_ms / processed if processed else None,
            )

    records = io_utils.deduplicate_predictions(predictions.read(key))
    if not records:
        return {"status": "FAILED", "reason": "нет предсказаний", "processed": 0}
    gold = [r["gold_index"] for r in records]
    pred = [r["pred_index"] for r in records]
    metrics = compute_metrics(gold, pred, task.n_classes)
    elapsed = time.perf_counter() - t0
    metrics.update(
        {
            "status": "PARTIAL" if interrupted or len(records) < len(manifest_rows) else "OK",
            "reason": "бюджет времени исчерпан" if interrupted else "",
            "ms_per_example": round(elapsed * 1000 / processed, 2) if processed else "",
            "processed": processed,
        }
    )
    return metrics


def evaluate(cfg: RunConfig, out_dir: Path, *, cache_dir: str | None = None, run_id: str = "local",
             test: str | None = None) -> Path:
    """Основной цикл: одна модель в памяти за раз, resume, бюджет времени."""
    from .adapters import build_adapter
    from .data import load_task

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "sample_manifest.csv"
    stale = manifest_is_stale(cfg, out_dir)
    if stale:
        if manifest_path.exists():
            message = f"манифест пересоздан: {stale}. Прежние строки results.csv остаются в истории."
            print(message)
            io_utils.log_error(out_dir / "errors.log", message)
        build_manifest(cfg, out_dir, cache_dir=cache_dir)
    rows = read_manifest(manifest_path)
    m_hash = manifest_hash(rows)
    c_hash = cfg.config_hash()

    results = io_utils.ResultsWriter(out_dir / "results.csv")
    predictions = io_utils.PredictionsWriter(out_dir / "predictions.jsonl")
    budget = io_utils.TimeBudget(cfg.time_budget_s, cfg.time_reserve_s)
    done_keys = results.existing_keys()
    io_utils.write_json(out_dir / "environment.json", io_utils.environment_info())
    io_utils.write_json(out_dir / "run_config.json", cfg.to_dict() | {"config_hash": c_hash, "run_id": run_id})

    tasks = {key: load_task(key, cache_dir=cache_dir) for key in cfg.dataset_keys}

    from .prefetch import ModelPrefetcher
    from .progress import StageProgress

    bar = StageProgress(
        stage=test or cfg.mode,
        models=list(cfg.model_ids),
        datasets=list(cfg.dataset_keys),
        n_samples=cfg.resolved_n() or "весь сплит",
    )
    bar.start_stage()

    prefetcher = ModelPrefetcher(cache_dir=cache_dir, enabled=cfg.prefetch_next_model)
    prefetcher.__enter__()
    pair_no = 0

    for model_index, model_id in enumerate(cfg.model_ids):
        model_spec = MODELS_BY_ID[model_id]
        bar.start_model(model_index)
        # Пока эта модель считается на GPU, фоном тянем веса следующей: сеть и видеокарта —
        # разные ресурсы, держать их по очереди значит простаивать.
        if model_index + 1 < len(cfg.model_ids):
            nxt = MODELS_BY_ID[cfg.model_ids[model_index + 1]]
            prefetcher.request(nxt.model_id, nxt.revision)
        bar.background = prefetcher.status
        adapter = None
        load_seconds = ""
        try:
            bar.loading()
            # Сбрасываем счётчик пиковой памяти ДО загрузки модели: иначе в peak_vram_mb
            # попадает всё, что оставили предыдущие этапы процесса (в прогоне run1 у
            # rubert-tiny2 так получилось 1159 МБ вместо ~140 МБ собственного размера).
            io_utils.free_memory()
            t0 = time.perf_counter()
            adapter = build_adapter(
                model_spec,
                backend=cfg.encoder_backend,
                prefix_policy=cfg.prefix_policy,
                use_chat_template=cfg.use_chat_template,
            )
            load_seconds = round(time.perf_counter() - t0, 1)
            bar.loaded(load_seconds)
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            status = "SKIPPED" if "SKIPPED_NO_GPU" in reason else "FAILED"
            io_utils.log_error(out_dir / "errors.log", f"load {model_id}: {reason}")
            for key in cfg.dataset_keys:
                ds = DATASETS_BY_KEY[key]
                load_verdict = comparable_to_paper(
                    key, cfg.resolved_n(), encoder_backend=cfg.encoder_backend,
                    prefix_policy=cfg.prefix_policy, use_chat_template=cfg.use_chat_template,
                )
                results.append(
                    {
                        "run_id": run_id,
                        "test": test or cfg.mode,
                        "comparable_to_paper": load_verdict.comparable,
                        "comparability_reason": load_verdict.reason,
                        "run_key": io_utils.run_key(model_id, model_spec.revision, key, ds.config, ds.split, m_hash, c_hash),
                        "model_id": model_id,
                        "revision": model_spec.revision,
                        "role": model_spec.role,
                        "dataset_key": key,
                        "hf_id": ds.hf_id,
                        "config": ds.config,
                        "split": ds.split,
                        "language": ds.language,
                        "status": status,
                        "reason": reason,
                    }
                )
            continue

        batch_size = model_spec.batch_size if model_spec.role == "llm" else cfg.encoder_batch_sizes[0]
        for key in cfg.dataset_keys:
            pair_no += 1
            bar.background = prefetcher.status
            ds = DATASETS_BY_KEY[key]
            rkey = io_utils.run_key(model_id, model_spec.revision, key, ds.config, ds.split, m_hash, c_hash)
            if rkey in done_keys:
                bar.say(f"{key}: уже посчитано, пропуск")
                bar.pairs_done += 1
                continue
            ds_rows_preview = [r for r in rows if r.dataset_key == key]
            bar.start_pair(cfg.dataset_keys.index(key), key, len(ds_rows_preview))
            ds_rows = [r for r in rows if r.dataset_key == key]
            verdict = comparable_to_paper(
                key,
                cfg.resolved_n(),
                encoder_backend=cfg.encoder_backend,
                prefix_policy=cfg.prefix_policy,
                use_chat_template=cfg.use_chat_template,
            )
            base = {
                "run_id": run_id,
                "test": test or cfg.mode,
                "comparable_to_paper": verdict.comparable,
                "comparability_reason": verdict.reason,
                "run_key": rkey,
                "model_id": model_id,
                "revision": model_spec.revision,
                "role": model_spec.role,
                "dataset_key": key,
                "hf_id": ds.hf_id,
                "config": ds.config,
                "split": ds.split,
                "language": ds.language,
                "n_samples": len(ds_rows),
                "load_seconds": load_seconds,
                "device": getattr(adapter, "device", ""),
                "batch_size": batch_size,
            }
            try:
                res = run_pair(
                    adapter,
                    tasks[key],
                    ds_rows,
                    run_id=run_id,
                    key=rkey,
                    model_spec=model_spec,
                    dataset_spec=ds,
                    batch_size=batch_size,
                    predictions=predictions,
                    budget=budget,
                    progress=bar,
                )
                base.update({k: v for k, v in res.items() if k != "processed"})
                base["peak_vram_mb"] = io_utils.peak_vram_mb() or ""
                base["truncated_prompts"] = getattr(adapter, "truncated_prompts", 0)
                base["n_classes"] = tasks[key].n_classes
            except Exception as exc:  # noqa: BLE001
                base.update({"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"})
                io_utils.log_error(out_dir / "errors.log", f"{model_id} × {key}: {exc}")
            results.append(base)
            done_keys.add(rkey)
            bar.finish_pair(
                key,
                float(base["macro_f1"]) if base.get("macro_f1") not in (None, "") else None,
                f"{base.get('status', '')}: {str(base.get('reason', ''))[:80]}",
            )

        io_utils.free_memory(adapter)
        if budget.exhausted():
            io_utils.log_error(out_dir / "errors.log", "бюджет времени исчерпан, остальные модели NOT_RUN")
            break

    prefetcher.__exit__(None, None, None)
    bar.finish_stage()
    write_summary(out_dir)
    return out_dir / "results.csv"


def write_summary(out_dir: Path) -> Path:
    """summary.csv: среднее macro-F1 по моделям (только строки status=OK)."""
    import csv

    from .metrics import summarize

    out_dir = Path(out_dir)
    rows = io_utils.ResultsWriter(out_dir / "results.csv").rows()
    summary = summarize(rows)
    path = out_dir / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["model_id", "n_ok", "n_total", "mean_macro_f1", "mean_accuracy", "mean_ms_per_example"]
        )
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    return path


def replicate(cfg: RunConfig, out_dir: Path, *, cache_dir: str | None = None, run_id: str = "local") -> Path:
    """Сверка: прогон моделей САМОЙ СТАТЬИ на её датасетах нашим кодом.

    Смысл — проверить эквивалентность реализации: у этих чекпоинтов есть опубликованные
    macro-F1 на agnews и imdb, и наши числа должны совпасть с ними в пределах разумного.
    Живёт в отдельном подкаталоге, чтобы не конфликтовать с манифестом основного прогона.
    """
    from .config import as_replication

    sub_dir = Path(out_dir) / "replication"
    sub_dir.mkdir(parents=True, exist_ok=True)
    return evaluate(as_replication(cfg), sub_dir, cache_dir=cache_dir, run_id=run_id, test="replication")


def st_check(cfg: RunConfig, out_dir: Path, *, cache_dir: str | None = None, run_id: str = "local") -> Path:
    """Диагностика: те же модели статьи, но кодом статьи (SentenceTransformer).

    Нужна, чтобы отделить «наша реализация считает иначе» от «в артефакте авторов сбой».
    Считаем только на коротких датасетах — это быстро и достаточно для ответа.
    """
    from dataclasses import replace

    from .config import ST_CHECK_DATASET_KEYS, as_replication

    sub_dir = Path(out_dir) / "replication_st"
    sub_dir.mkdir(parents=True, exist_ok=True)
    cfg_st = replace(
        as_replication(cfg),
        encoder_backend="st",
        dataset_keys=tuple(ST_CHECK_DATASET_KEYS),
    )
    return evaluate(cfg_st, sub_dir, cache_dir=cache_dir, run_id=run_id, test="replication_st")


def reference(cfg: RunConfig, out_dir: Path, *, cache_dir: str | None = None, run_id: str = "local") -> Path:
    """Эталонная группа: модели САМОЙ СТАТЬИ на наших датасетах и НАШЕЙ выборке.

    Зачем отдельно от `replicate`: сверка (`replicate`) считает модели статьи на полном тесте и
    сравнивает с её опубликованными числами — это проверка кода. А здесь те же модели проходят
    ровно тот же манифест (те же 300 примеров), что и российские модели, поэтому обе группы
    оказываются на одной шкале и их числа можно сравнивать напрямую.
    """
    from dataclasses import replace

    from .config import BTZSC_CORE_KEYS, PAPER_MODELS

    cfg_ref = replace(
        cfg,
        mode="reference",
        dataset_keys=tuple(BTZSC_CORE_KEYS),
        model_ids=tuple(m.model_id for m in PAPER_MODELS),
    )
    return evaluate(cfg_ref, out_dir, cache_dir=cache_dir, run_id=run_id, test="block_a")


def baseline(cfg: RunConfig, out_dir: Path, *, cache_dir: str | None = None, run_id: str = "local") -> Path:
    """ТЕСТ 1 «базовый»: модели статьи, её датасеты, ПОЛНЫЙ тест.

    Воспроизведение: единственное, что здесь проверяется, — совпадают ли наши числа
    с опубликованными. Результаты пишутся в подкаталог `baseline/`.
    """
    from dataclasses import replace

    from .config import BTZSC_CORE_KEYS, PAPER_MODELS

    sub_dir = Path(out_dir) / "baseline"
    sub_dir.mkdir(parents=True, exist_ok=True)
    from .config import FULL_TEST_CAP

    cfg_base = replace(
        cfg,
        mode="baseline",
        n_samples=FULL_TEST_CAP,
        dataset_keys=tuple(BTZSC_CORE_KEYS),
        model_ids=tuple(m.model_id for m in PAPER_MODELS),
    )
    return evaluate(cfg_base, sub_dir, cache_dir=cache_dir, run_id=run_id, test="baseline")


def ours_full(cfg: RunConfig, out_dir: Path, *, cache_dir: str | None = None, run_id: str = "local") -> Path:
    """ТЕСТ 3 «полный наш»: наши энкодеры на датасетах статьи, ПОЛНЫЙ тест.

    Снимает последнюю оговорку о сопоставимости: объём совпадает с авторским.
    LLM сюда не входят — полный тест для них на T4 занял бы часы.
    """
    from dataclasses import replace

    from .config import BTZSC_CORE_KEYS, ENCODERS

    sub_dir = Path(out_dir) / "ours_full"
    sub_dir.mkdir(parents=True, exist_ok=True)
    from .config import FULL_TEST_CAP

    cfg_full = replace(
        cfg,
        mode="ours_full",
        n_samples=FULL_TEST_CAP,
        dataset_keys=tuple(BTZSC_CORE_KEYS),
        model_ids=tuple(m.model_id for m in ENCODERS),
    )
    return evaluate(cfg_full, sub_dir, cache_dir=cache_dir, run_id=run_id, test="ours_full")


def mirror_results(out_dir: Path, mirror_dir: str | Path | None, run_id: str) -> None:
    """Копирует каталог прогона в зеркало (например, внешний каталог). Ошибки не критичны.

    Нужна, потому что сессия Colab может умереть в середине длинного этапа: без зеркала
    теряется всё, с зеркалом — максимум последний этап.
    """
    if not mirror_dir:
        return
    import shutil

    try:
        target = Path(mirror_dir) / run_id
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(out_dir, target, dirs_exist_ok=True)
        print(f"  результаты скопированы в {target}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  зеркало недоступно ({exc}) — данные остались в {out_dir}", flush=True)


def block_a(cfg: RunConfig, out_dir: Path, *, cache_dir: str | None = None, run_id: str = "local") -> Path:
    """БЛОК А: 5 моделей самой статьи на её датасетах, наша выборка.

    Это воспроизведение: числа сравниваются с опубликованными, а заодно задают точку отсчёта
    для блока Б. Пишет в подкаталог `block_a/`.
    """
    from dataclasses import replace

    from .config import BLOCK_A_MODEL_IDS, BLOCK_DATASET_KEYS

    cfg_a = replace(cfg, mode="block_a", n_samples=cfg.n_samples,
                    dataset_keys=tuple(BLOCK_DATASET_KEYS), model_ids=tuple(BLOCK_A_MODEL_IDS))
    # Пишем в КОРЕНЬ прогона: отличие от блока Б — в колонке test, а не в каталоге.
    # Так результаты доезжают до экспорта, импорта и графиков без отдельных путей.
    return evaluate(cfg_a, Path(out_dir), cache_dir=cache_dir, run_id=run_id, test="block_a")


def block_b(cfg: RunConfig, out_dir: Path, *, cache_dir: str | None = None, run_id: str = "local") -> Path:
    """БЛОК Б: 5 наших моделей на ТЕХ ЖЕ датасетах и ТОЙ ЖЕ выборке, что блок А.

    Одна шкала с блоком А: отличается только список моделей. Пишет в `block_b/`.
    """
    from dataclasses import replace

    from .config import BLOCK_B_MODEL_IDS, BLOCK_DATASET_KEYS

    cfg_b = replace(cfg, mode="block_b", n_samples=cfg.n_samples,
                    dataset_keys=tuple(BLOCK_DATASET_KEYS), model_ids=tuple(BLOCK_B_MODEL_IDS))
    return evaluate(cfg_b, Path(out_dir), cache_dir=cache_dir, run_id=run_id, test="block_b")


def anchor(cfg: RunConfig, out_dir: Path, *, cache_dir: str | None = None, run_id: str = "local") -> Path:
    """ЯКОРЬ: модели статьи на ПОЛНОМ сплите дешёвых датасетов статьи.

    Единственная часть прогона, числа которой можно класть рядом с опубликованными
    (comparable_to_paper=True). Стоит недорого: 3 датасета по ≤ 2000 примеров.
    """
    from dataclasses import replace

    from .config import ANCHOR_DATASET_KEYS, ANCHOR_N, BLOCK_A_MODEL_IDS

    cfg_anchor = replace(
        cfg,
        mode="baseline",
        n_samples=ANCHOR_N,
        dataset_keys=tuple(ANCHOR_DATASET_KEYS),
        model_ids=tuple(BLOCK_A_MODEL_IDS),
    )
    return evaluate(cfg_anchor, Path(out_dir), cache_dir=cache_dir, run_id=run_id, test="baseline")


def run_all(
    cfg: RunConfig,
    out_dir: Path,
    *,
    cache_dir: str | None = None,
    run_id: str = "local",
    mirror_dir: str | Path | None = None,
    deadline_s: int | None = None,
    smoke_first: bool = False,
    with_anchor: bool = True,
    with_reference: bool = False,
    with_baseline: bool = False,
    with_ours_full: bool = False,
    with_replication: bool = False,
    with_st_check: bool = False,
    with_finetune: bool = True,     # контрольная точка «1800 примеров против размера модели»
    with_extra_models: bool = False,
    with_ru_extension: bool = True,   # наш собственный вклад — считается всегда
    with_export: bool = True,
) -> dict:
    """Весь прогон: preflight → smoke → evaluate → reference → replicate → st_check → finetune → export.

    Ничего выбирать вручную не нужно: этапы идут по порядку, каждый пишет свои артефакты,
    resume пропускает уже посчитанные пары. Падение одного этапа не отменяет остальные —
    его причина попадает в errors.log и в возвращаемый отчёт.
    """
    from dataclasses import replace

    from .config import GLOBAL_DEADLINE_S

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Манифест строится ОДИН раз на все датасеты прогона. Если его строит каждый этап под свой
    # список, manifest_hash меняется, run_key перестаёт совпадать и resume обнуляется —
    # в прошлом прогоне из-за этого манифест пересоздавался между блоками и русской частью.
    if manifest_is_stale(cfg, out_dir):
        try:
            build_manifest(cfg, out_dir, cache_dir=cache_dir)
            print(f"  манифест прогона: {len(cfg.dataset_keys)} датасетов × "
                  f"{cfg.resolved_n() or 'весь сплит'} примеров", flush=True)
        except Exception as exc:  # noqa: BLE001
            # Не фатально: манифест построит первый же этап оценки под свои датасеты.
            io_utils.log_error(out_dir / "errors.log", f"общий манифест не построен: {exc}")
    report: dict[str, dict] = {}
    run_started = time.perf_counter()
    deadline = GLOBAL_DEADLINE_S if deadline_s is None else deadline_s

    def _left() -> float:
        return deadline - (time.perf_counter() - run_started)

    def _stage(name: str, fn, *, heavy: bool = False):
        if heavy and _left() <= 0:
            report[name] = {"status": "NOT_RUN", "seconds": 0.0,
                            "detail": f"пропущен: дедлайн прогона {deadline / 60:.0f} мин исчерпан"}
            print(f"[{name}] NOT_RUN — дедлайн исчерпан, запустите прогон повторно (resume досчитает)",
                  flush=True)
            return False
        started = time.perf_counter()
        try:
            value = fn()
            report[name] = {"status": "OK", "seconds": round(time.perf_counter() - started, 1), "detail": value}
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            io_utils.log_error(out_dir / "errors.log", f"этап {name}: {reason}")
            report[name] = {"status": "FAILED", "seconds": round(time.perf_counter() - started, 1), "detail": reason}
        line = f"[{name}] {report[name]['status']} за {report[name]['seconds']} с"
        if report[name]["status"] == "FAILED":
            # Молчащий провал — худший вид провала: печатаем причину сразу, а не только в errors.log.
            line += f"\n    причина: {str(report[name]['detail'])[:400]}"
        print(line, flush=True)
        mirror_results(out_dir, mirror_dir, run_id)
        print(f"  осталось времени по дедлайну: {max(0.0, _left()) / 60:.0f} мин", flush=True)
        return report[name]["status"] == "OK"

    _stage("preflight", lambda: preflight(replace(cfg, mode="preflight"), out_dir, cache_dir=cache_dir) and "ok")

    # Якорь к публикации идёт первым: если наш код врёт, это видно сразу, до блоков.
    if with_anchor:
        _stage("anchor", lambda: str(anchor(cfg, out_dir, cache_dir=cache_dir, run_id=run_id)))

    # Основная схема работы: два блока по 5 моделей на одних и тех же данных и выборке.
    _stage("block_a", lambda: str(block_a(cfg, out_dir, cache_dir=cache_dir, run_id=run_id)))
    _stage("block_b", lambda: str(block_b(cfg, out_dir, cache_dir=cache_dir, run_id=run_id)))

    if with_export:
        _stage("export", lambda: str(export_zip(out_dir, run_id)))

    # Дальше — необязательные расширения, выключены по умолчанию (см. experiment.py).
    extras_requested = any(
        (with_finetune, with_extra_models, with_ru_extension, with_baseline, with_ours_full)
    )
    if with_finetune:
        def _ft():
            from .finetune import DEFAULT_FT_CONFIG, run_finetune

            summary = run_finetune(DEFAULT_FT_CONFIG, out_dir, cache_dir=cache_dir)
            return f"held-out macro-F1 {summary['heldout']['macro_f1']:.4f}"

        _stage("finetune", _ft, heavy=True)

    if with_extra_models:
        _stage(
            "extra_models",
            lambda: str(evaluate(replace(cfg, mode="evaluate", model_ids=tuple(EXTRA_MODEL_IDS)),
                                 out_dir, cache_dir=cache_dir, run_id=run_id, test="extra_models")),
            heavy=True,
        )

    if with_ru_extension:
        _stage(
            "ru_extension",
            lambda: str(evaluate(replace(cfg, mode="evaluate", dataset_keys=tuple(RU_EXTENSION_KEYS)),
                                 out_dir, cache_dir=cache_dir, run_id=run_id, test="ru_extension")),
            heavy=True,
        )

    if with_baseline:
        _stage("baseline", lambda: str(baseline(cfg, out_dir, cache_dir=cache_dir, run_id=run_id)), heavy=True)

    if with_ours_full:
        _stage("ours_full", lambda: str(ours_full(cfg, out_dir, cache_dir=cache_dir, run_id=run_id)), heavy=True)

    if with_export and extras_requested:
        # Повторный экспорт нужен только если после первого что-то досчитывалось.
        _stage("export_final", lambda: str(export_zip(out_dir, run_id)))

    io_utils.write_json(out_dir / "run_all_report.json", report)
    ok = sum(1 for v in report.values() if v["status"] == "OK")
    print(f"итог: этапов OK {ok}/{len(report)} → {out_dir / 'run_all_report.json'}", flush=True)
    failed = {k: v["detail"] for k, v in report.items() if v["status"] == "FAILED"}
    if failed:
        print("упавшие этапы и причины:")
        for name, detail in failed.items():
            print(f"  {name}: {str(detail)[:300]}")
    return report


def export_zip(out_dir: Path, run_id: str) -> Path:
    """Собирает colab_outputs_<run_id>.zip со всеми артефактами прогона."""
    import zipfile

    out_dir = Path(out_dir)
    target = out_dir.parent / f"colab_outputs_{run_id}.zip"
    names = [
        "results.csv",
        "summary.csv",
        "predictions.jsonl",
        "sample_manifest.csv",
        "manifest_info.json",
        "run_config.json",
        "environment.json",
        "preflight.json",
        "errors.log",
        "ft_metrics.csv",
        "ft_config.json",
    ]
    if not out_dir.exists():
        raise RuntimeError(
            f"каталог прогона {out_dir} не существует — экспортировать нечего. "
            "Сначала выполните preflight и smoke/evaluate с тем же RUN_ID."
        )
    required = ("results.csv", "predictions.jsonl")
    missing = [name for name in required if not (out_dir / name).exists()]
    if missing:
        present = sorted(p.name for p in out_dir.iterdir()) or ["(каталог пуст)"]
        raise RuntimeError(
            "экспорт остановлен: в каталоге прогона нет " + ", ".join(missing) + ". "
            "Похоже, ни smoke, ни evaluate не запускались (preflight результатов не создаёт). "
            "Что лежит в каталоге: " + ", ".join(present) + ". "
            "Порядок: MODE='smoke' → запустить → проверить results.csv → затем MODE='export'."
        )

    written: list[str] = []
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            p = out_dir / name
            if p.exists():
                zf.write(p, arcname=name)
                written.append(name)
        for extra in sorted(out_dir.glob("plots/*")):
            zf.write(extra, arcname=f"plots/{extra.name}")
            written.append(f"plots/{extra.name}")
        for sub in ("baseline", "ours_full", "replication", "replication_st"):
            for extra in sorted((out_dir / sub).glob("*")):
                if extra.is_file():
                    zf.write(extra, arcname=f"{sub}/{extra.name}")
                    written.append(f"{sub}/{extra.name}")
    print(f"экспортировано файлов: {len(written)} → {target}")
    for name in written:
        print("  " + name)
    return target
