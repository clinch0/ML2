"""Сквозной режим all: пользователю не нужно выбирать этап вручную."""

from __future__ import annotations

import json

import pytest

from btzsc_ru import runner
from btzsc_ru.config import MODES, RunConfig


def test_all_is_a_mode_and_is_default():
    from btzsc_ru import config

    assert "all" in MODES
    assert config.MODE == "all", "по умолчанию запускается сквозной прогон"
    assert RunConfig(mode="all").resolved_n() == config.EVALUATE_N


def make_cfg():
    return RunConfig(mode="all", dataset_keys=["btzsc_agnews"], model_ids=["cointegrated/rubert-tiny2"])


def test_run_all_is_two_blocks_by_default(tmp_path, monkeypatch):
    """Схема по умолчанию: preflight → блок А → блок Б → export (+ русское расширение)."""
    calls: list[str] = []

    monkeypatch.setattr(runner, "evaluate", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "preflight", lambda cfg, out, **kw: calls.append("preflight") or {})
    monkeypatch.setattr(runner, "anchor", lambda cfg, out, **kw: calls.append("anchor") or out)
    monkeypatch.setattr(runner, "block_a", lambda cfg, out, **kw: calls.append("block_a") or out)
    monkeypatch.setattr(runner, "block_b", lambda cfg, out, **kw: calls.append("block_b") or out)
    monkeypatch.setattr(runner, "export_zip", lambda out, run_id: calls.append("export") or out / "z.zip")
    monkeypatch.setattr("btzsc_ru.finetune.run_finetune",
                        lambda cfg, out, **kw: {"heldout": {"macro_f1": 0.6}})

    report = runner.run_all(make_cfg(), tmp_path, run_id="t")

    assert calls == ["preflight", "anchor", "block_a", "block_b", "export", "export"], \
        "якорь к публикации идёт первым, второй export — после дообучения и русского расширения"
    saved = list(json.loads((tmp_path / "run_all_report.json").read_text(encoding="utf-8")))
    # дообучение и русское расширение считаются по умолчанию: на них держатся выводы отчёта
    assert saved == ["preflight", "anchor", "block_a", "block_b", "export",
                     "finetune", "ru_extension", "export_final"]
    assert all(v["status"] == "OK" for v in report.values())


def test_blocks_share_datasets_and_sample_but_differ_in_models(tmp_path, monkeypatch):
    """Блоки А и Б отличаются только списком моделей — иначе сравнивать нельзя."""
    seen = {}

    def fake_evaluate(cfg, out, **kw):
        seen[cfg.mode] = (tuple(cfg.model_ids), tuple(cfg.dataset_keys), cfg.resolved_n(),
                          kw.get("test"))
        return out / "results.csv"

    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    runner.block_a(make_cfg(), tmp_path, run_id="t")
    runner.block_b(make_cfg(), tmp_path, run_id="t")

    from btzsc_ru.config import BLOCK_A_MODEL_IDS, BLOCK_B_MODEL_IDS, BLOCK_DATASET_KEYS

    a, b = seen["block_a"], seen["block_b"]
    assert a[0] == tuple(BLOCK_A_MODEL_IDS) and b[0] == tuple(BLOCK_B_MODEL_IDS)
    assert len(a[0]) == 5 and len(b[0]) == 5, "в каждом блоке ровно 5 моделей"
    assert a[1] == b[1] == tuple(BLOCK_DATASET_KEYS), "датасеты одинаковые"
    assert a[2] == b[2], "выборка одинаковая"
    assert a[3] == "block_a" and b[3] == "block_b", "строки помечаются колонкой test"


def test_default_stages_cover_the_report(tmp_path, monkeypatch):
    """По умолчанию считается всё, на чём держатся выводы отчёта.

    Якорь к публикации, оба блока, дообучение и русское расширение — без флагов.
    По явному флагу остаются только дорогие необязательные этапы: extra_models и полные тесты.
    """
    calls: list[str] = []
    monkeypatch.setattr(runner, "preflight", lambda cfg, out, **kw: {})
    monkeypatch.setattr(runner, "block_a", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "block_b", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "evaluate", lambda cfg, out, **kw: calls.append(kw.get("test") or cfg.mode) or out)
    monkeypatch.setattr(runner, "anchor", lambda cfg, out, **kw: calls.append("anchor") or out)
    monkeypatch.setattr(runner, "baseline", lambda cfg, out, **kw: calls.append("baseline") or out)
    monkeypatch.setattr(runner, "ours_full", lambda cfg, out, **kw: calls.append("ours_full") or out)
    monkeypatch.setattr(runner, "export_zip", lambda out, run_id: out / "z.zip")

    monkeypatch.setattr("btzsc_ru.finetune.run_finetune",
                        lambda cfg, out, **kw: calls.append("finetune") or {"heldout": {"macro_f1": 0.6}})

    report = runner.run_all(make_cfg(), tmp_path, run_id="t")

    assert calls == ["anchor", "finetune", "ru_extension"], \
        "по умолчанию считаются якорь, дообучение и русская часть"
    assert set(report) == {"preflight", "anchor", "block_a", "block_b", "export",
                           "finetune", "ru_extension", "export_final"}
    assert "extra_models" not in report and "ours_full" not in report, \
        "дорогие необязательные этапы остаются за флагом"


def test_replicate_uses_paper_models_in_subdirectory(tmp_path, monkeypatch):
    seen = {}

    def fake_evaluate(cfg, out, **kw):
        seen["cfg"], seen["out"] = cfg, out
        return out / "results.csv"

    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    runner.replicate(make_cfg(), tmp_path, run_id="t")

    from btzsc_ru.config import PAPER_MODELS, REPLICATION_DATASET_KEYS

    assert seen["out"].name == "replication", "сверка не должна трогать манифест основного прогона"
    assert set(seen["cfg"].model_ids) == {m.model_id for m in PAPER_MODELS}
    assert seen["cfg"].dataset_keys == REPLICATION_DATASET_KEYS


def test_st_check_uses_paper_code_path_on_short_datasets(tmp_path, monkeypatch):
    seen = {}

    def fake_evaluate(cfg, out, **kw):
        seen["cfg"], seen["out"] = cfg, out
        return out / "results.csv"

    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    runner.st_check(make_cfg(), tmp_path, run_id="t")

    assert seen["out"].name == "replication_st"
    assert seen["cfg"].encoder_backend == "st", "диагностика должна идти кодом статьи"
    assert seen["cfg"].dataset_keys == ("btzsc_rottentomatoes", "btzsc_financialphrasebank")


def test_reference_runs_paper_models_on_our_manifest(tmp_path, monkeypatch):
    """Эталонная группа обязана идти по тому же манифесту, что и наши модели — иначе шкалы разные."""
    seen = {}

    def fake_evaluate(cfg, out, **kw):
        seen["cfg"], seen["out"] = cfg, out
        return out / "results.csv"

    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    runner.reference(make_cfg(), tmp_path, run_id="t")

    from btzsc_ru.config import BTZSC_CORE_KEYS, PAPER_MODELS

    assert seen["out"] == tmp_path, "эталон пишется в основной каталог, а не в подкаталог"
    assert set(seen["cfg"].model_ids) == {m.model_id for m in PAPER_MODELS}
    assert seen["cfg"].dataset_keys == tuple(BTZSC_CORE_KEYS)
    assert seen["cfg"].resolved_n() == 300, "та же выборка, что у наших моделей"


def test_manifest_allows_dataset_subset(tmp_path):
    """Прогон по части датасетов не должен пересоздавать общий манифест."""
    import json as _json

    from btzsc_ru.config import RunConfig
    from btzsc_ru.runner import manifest_is_stale

    out = tmp_path / "run"
    out.mkdir()
    (out / "sample_manifest.csv").write_text("dataset_key,sample_id\n", encoding="utf-8")
    (out / "manifest_info.json").write_text(
        _json.dumps({"requested_n": 300, "dataset_keys": ["btzsc_agnews", "btzsc_imdb", "ru_sentiment"]}),
        encoding="utf-8",
    )
    subset = RunConfig(mode="evaluate", dataset_keys=["btzsc_agnews"], model_ids=[])
    assert manifest_is_stale(subset, out) == ""
    missing = RunConfig(mode="evaluate", dataset_keys=["btzsc_banking77"], model_ids=[])
    assert "btzsc_banking77" in manifest_is_stale(missing, out)


def test_run_all_continues_after_failed_stage(tmp_path, monkeypatch):
    def boom(cfg, out, **kw):
        raise RuntimeError("нет сети")

    monkeypatch.setattr(runner, "block_a", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "block_b", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "anchor", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "preflight", boom)
    monkeypatch.setattr(runner, "evaluate", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "export_zip", lambda out, run_id: out / "z.zip")
    monkeypatch.setattr(runner, "replicate", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "st_check", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "reference", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "baseline", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "ours_full", lambda cfg, out, **kw: out)

    report = runner.run_all(make_cfg(), tmp_path, run_id="t", with_finetune=False)

    assert report["preflight"]["status"] == "FAILED"
    assert "нет сети" in report["preflight"]["detail"]
    assert report["block_a"]["status"] == "OK", "падение одного этапа не отменяет остальные"
    assert "этап preflight" in (tmp_path / "errors.log").read_text(encoding="utf-8")


def test_run_all_can_skip_smoke_and_finetune(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(runner, "preflight", lambda cfg, out, **kw: calls.append("preflight") or {})
    monkeypatch.setattr(runner, "evaluate", lambda cfg, out, **kw: calls.append(kw.get("test") or cfg.mode) or out)
    monkeypatch.setattr(runner, "anchor", lambda cfg, out, **kw: calls.append("anchor") or out)
    monkeypatch.setattr(runner, "export_zip", lambda out, run_id: calls.append("export") or out / "z.zip")

    monkeypatch.setattr(runner, "block_a", lambda cfg, out, **kw: calls.append("block_a") or out)
    monkeypatch.setattr(runner, "block_b", lambda cfg, out, **kw: calls.append("block_b") or out)
    runner.run_all(make_cfg(), tmp_path, run_id="t", with_ru_extension=False, with_finetune=False)

    assert calls == ["preflight", "anchor", "block_a", "block_b", "export"]


def test_cli_accepts_all_mode_without_arguments():
    import experiment

    assert experiment.main(["--dry-run"]) == 0, "без единого аргумента CLI должен работать"


def test_three_tests_are_separated(tmp_path, monkeypatch):
    """baseline / ours / ours_full — разные каталоги, модели и объёмы выборки."""
    seen = []

    def fake_evaluate(cfg, out, **kw):
        seen.append((cfg.mode, out.name, tuple(sorted(cfg.model_ids))[:1], cfg.resolved_n()))
        return out / "results.csv"

    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    from btzsc_ru.config import ENCODERS, PAPER_MODELS

    runner.baseline(make_cfg(), tmp_path, run_id="t")
    runner.ours_full(make_cfg(), tmp_path, run_id="t")
    runner.reference(make_cfg(), tmp_path, run_id="t")

    modes = {row[0]: row for row in seen}
    from btzsc_ru.config import FULL_TEST_CAP

    assert modes["baseline"][1] == "baseline" and modes["baseline"][3] == FULL_TEST_CAP
    assert modes["ours_full"][1] == "ours_full" and modes["ours_full"][3] == FULL_TEST_CAP
    assert modes["reference"][3] == 300, "эталон идёт по нашей выборке"

    paper_ids = {m.model_id for m in PAPER_MODELS}
    enc_ids = {m.model_id for m in ENCODERS}
    assert modes["baseline"][2][0] in paper_ids
    assert modes["ours_full"][2][0] in enc_ids
    assert modes["reference"][2][0] in paper_ids


def test_llm_matches_paper_protocol():
    """LLM-адаптер должен совпадать с btzsc/models/llm.py по алфавиту, промпту и сортировке."""
    import inspect

    from btzsc_ru.adapters import LLMAdapter

    assert len(LLMAdapter._SYMBOLS) == 100, "латиница + греческий, как в коде статьи"
    assert LLMAdapter._INSTRUCTION.startswith("You are a text classifier.")
    assert LLMAdapter._INSTRUCTION.rstrip().endswith("The correct option is letter")
    src = inspect.getsource(LLMAdapter.predict_scores)
    assert "sorted(set(labels))" in src, "канонический порядок опций как в статье"
    assert "order" in src, "столбцы возвращаются в исходный порядок меток"


def test_encoder_default_backend_is_paper_code_path():
    import inspect

    from btzsc_ru.adapters import EncoderAdapter
    from btzsc_ru.config import RunConfig

    assert RunConfig().encoder_backend == "st"
    assert inspect.signature(EncoderAdapter.__init__).parameters["backend"].default == "st"
    assert RunConfig().encoder_batch_sizes[0] == 32, "как в артефактах статьи"


def test_failed_stage_prints_reason(tmp_path, monkeypatch, capsys):
    """Молчащий провал недопустим: причина должна попадать в консольный лог прогона."""
    def boom(cfg, out, **kw):
        raise RuntimeError("датасет не загрузился")

    monkeypatch.setattr(runner, "preflight", boom)
    monkeypatch.setattr(runner, "evaluate", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "reference", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "baseline", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "ours_full", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "replicate", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "st_check", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "export_zip", lambda out, run_id: out / "z.zip")

    runner.run_all(make_cfg(), tmp_path, run_id="t", with_finetune=False)
    out = capsys.readouterr().out

    assert "причина: RuntimeError: датасет не загрузился" in out
    assert "упавшие этапы и причины:" in out


def test_quiet_logs_sets_env(monkeypatch):
    """Служебный вывод библиотек глушится, иначе лог прогона нечитаем."""
    import experiment

    for key in ("HF_HUB_DISABLE_PROGRESS_BARS", "TRANSFORMERS_VERBOSITY", "DATASETS_VERBOSITY"):
        monkeypatch.delenv(key, raising=False)
    experiment.quiet_logs()
    import os

    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"
    assert os.environ["DATASETS_VERBOSITY"] == "error"


def test_heavy_stages_skipped_after_deadline(tmp_path, monkeypatch):
    """Дедлайн: тяжёлые этапы не стартуют, если времени нет, но экспорт выполняется."""
    calls = []
    monkeypatch.setattr(runner, "preflight", lambda cfg, out, **kw: calls.append("preflight") or {})
    monkeypatch.setattr(runner, "evaluate", lambda cfg, out, **kw: calls.append("evaluate") or out)
    monkeypatch.setattr(runner, "reference", lambda cfg, out, **kw: calls.append("reference") or out)
    monkeypatch.setattr(runner, "baseline", lambda cfg, out, **kw: calls.append("baseline") or out)
    monkeypatch.setattr(runner, "ours_full", lambda cfg, out, **kw: calls.append("ours_full") or out)
    monkeypatch.setattr(runner, "replicate", lambda cfg, out, **kw: calls.append("replicate") or out)
    monkeypatch.setattr(runner, "st_check", lambda cfg, out, **kw: calls.append("st_check") or out)
    monkeypatch.setattr(runner, "export_zip", lambda out, run_id: calls.append("export") or out / "z.zip")

    monkeypatch.setattr(runner, "block_a", lambda cfg, out, **kw: calls.append("block_a") or out)
    monkeypatch.setattr(runner, "block_b", lambda cfg, out, **kw: calls.append("block_b") or out)

    report = runner.run_all(make_cfg(), tmp_path, run_id="t", deadline_s=0,
                            with_baseline=True, with_ours_full=True)

    assert "baseline" not in calls and "ours_full" not in calls, "тяжёлые этапы пропущены по дедлайну"
    assert report["baseline"]["status"] == "NOT_RUN"
    assert "дедлайн" in report["baseline"]["detail"]
    assert report["export"]["status"] == "OK", "экспорт основной части выполняется всегда"


def test_results_mirrored_after_each_stage(tmp_path, monkeypatch):
    """Каталог прогона копируется в зеркало после каждого этапа — страховка от смерти сессии."""
    out = tmp_path / "run"
    mirror = tmp_path / "drive"
    monkeypatch.setattr(runner, "preflight", lambda cfg, o, **kw: (o / "preflight.json").write_text("{}") or {})
    monkeypatch.setattr(runner, "evaluate", lambda cfg, o, **kw: o)
    monkeypatch.setattr(runner, "reference", lambda cfg, o, **kw: o)
    monkeypatch.setattr(runner, "export_zip", lambda o, run_id: o / "z.zip")

    runner.run_all(make_cfg(), out, run_id="t", smoke_first=False, with_finetune=False,
                   with_baseline=False, with_ours_full=False, with_replication=False,
                   with_st_check=False, mirror_dir=mirror)

    assert (mirror / "t" / "preflight.json").exists(), "результаты должны уехать в зеркало"


def test_manifest_built_once_for_all_stages(tmp_path, monkeypatch):
    """Регрессия: каждый этап строил манифест под свой список датасетов и обнулял resume."""
    builds: list[int] = []

    def fake_build(cfg, out, **kw):
        builds.append(len(cfg.dataset_keys))
        (out / "sample_manifest.csv").write_text("dataset_key,sample_id\n", encoding="utf-8")
        import json as _json
        (out / "manifest_info.json").write_text(
            _json.dumps({"requested_n": cfg.resolved_n(), "dataset_keys": list(cfg.dataset_keys)}),
            encoding="utf-8",
        )
        return out / "sample_manifest.csv"

    monkeypatch.setattr(runner, "build_manifest", fake_build)
    monkeypatch.setattr(runner, "preflight", lambda cfg, out, **kw: {})
    monkeypatch.setattr(runner, "evaluate", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "block_a", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "block_b", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "anchor", lambda cfg, out, **kw: out)
    monkeypatch.setattr(runner, "export_zip", lambda out, run_id: out / "z.zip")

    from btzsc_ru.config import RunConfig

    runner.run_all(RunConfig(mode="all"), tmp_path, run_id="t")

    assert len(builds) == 1, f"манифест должен строиться один раз, построен {len(builds)}"
    assert builds[0] == len(RunConfig().dataset_keys), "манифест охватывает все датасеты прогона"
