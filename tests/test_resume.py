"""Resume, run_key, отсутствие дублей, бюджет времени."""

from btzsc_ru import io_utils
from btzsc_ru.reconstruction import MultiClassTask
from btzsc_ru.runner import run_pair
from btzsc_ru.config import DATASETS_BY_KEY, MODELS_BY_ID
from btzsc_ru.sampling import build_manifest_rows

MODEL = MODELS_BY_ID["cointegrated/rubert-tiny2"]
DATASET = DATASETS_BY_KEY["ru_sentiment"]


class CountingAdapter:
    """Фейковый адаптер: gold не получает, считает число обработанных примеров."""

    def __init__(self, answer=0):
        self.answer = answer
        self.seen = 0

    def predict(self, texts, labels, batch_size=8):
        self.seen += len(texts)
        return [self.answer] * len(texts)


def make_task(n=6):
    return MultiClassTask(
        name="ru_sentiment",
        texts=[f"текст {i}" for i in range(n)],
        verbalizers=["нейтральная", "положительная", "отрицательная"],
        gold=[i % 3 for i in range(n)],
        n_classes=3,
    )


def test_run_key_is_stable_and_sensitive():
    a = io_utils.run_key("m", "rev", "ds", "cfg", "test", "mh", "ch")
    assert a == io_utils.run_key("m", "rev", "ds", "cfg", "test", "mh", "ch")
    assert a != io_utils.run_key("m", "rev2", "ds", "cfg", "test", "mh", "ch")
    assert a != io_utils.run_key("m", "rev", "ds", "cfg", "test", "mh2", "ch")


def test_second_run_adds_no_duplicates(tmp_path):
    task = make_task()
    rows = build_manifest_rows("ru_sentiment", task.gold, task.verbalizers, list(range(6)))
    preds = io_utils.PredictionsWriter(tmp_path / "predictions.jsonl")
    budget = io_utils.TimeBudget(3600, 60)
    key = io_utils.run_key(MODEL.model_id, MODEL.revision, "ru_sentiment", "default", "validation", "mh", "ch")

    first = CountingAdapter()
    res1 = run_pair(first, task, rows, run_id="r1", key=key, model_spec=MODEL, dataset_spec=DATASET,
                    batch_size=2, predictions=preds, budget=budget)
    assert first.seen == 6 and res1["status"] == "OK"

    second = CountingAdapter()
    res2 = run_pair(second, task, rows, run_id="r1", key=key, model_spec=MODEL, dataset_spec=DATASET,
                    batch_size=2, predictions=preds, budget=budget)
    assert second.seen == 0, "resume должен пропустить уже посчитанные примеры"
    assert len(io_utils.deduplicate_predictions(preds.read(key))) == 6
    assert res2["macro_f1"] == res1["macro_f1"]


def test_exhausted_budget_marks_partial(tmp_path):
    task = make_task()
    rows = build_manifest_rows("ru_sentiment", task.gold, task.verbalizers, list(range(6)))
    preds = io_utils.PredictionsWriter(tmp_path / "predictions.jsonl")
    key = "k-partial"
    budget = io_utils.TimeBudget(3600, 60)
    run_pair(CountingAdapter(), task, rows[:2], run_id="r", key=key, model_spec=MODEL, dataset_spec=DATASET,
             batch_size=2, predictions=preds, budget=budget)
    spent = io_utils.TimeBudget(1, 1)
    res = run_pair(CountingAdapter(), task, rows, run_id="r", key=key, model_spec=MODEL, dataset_spec=DATASET,
                   batch_size=2, predictions=preds, budget=spent)
    assert res["status"] == "PARTIAL"
    assert res["reason"] == "бюджет времени исчерпан"


def test_results_writer_requires_test_label(tmp_path):
    """Колонка test обязательна: иначе три разных теста смешиваются в одном файле."""
    import pytest

    from btzsc_ru.io_utils import ResultsWriter

    writer = ResultsWriter(tmp_path / "results.csv")
    with pytest.raises(ValueError, match="test"):
        writer.append({"run_id": "r", "run_key": "k", "model_id": "m", "dataset_key": "d",
                       "status": "OK"})
    writer.append({"run_id": "r", "test": "block_a", "run_key": "k", "model_id": "m",
                   "dataset_key": "d", "status": "OK"})
    assert "test" in (tmp_path / "results.csv").read_text(encoding="utf-8").splitlines()[0]


def test_results_dedup_keeps_last_per_test(tmp_path):
    """Дубли (resume, перезапуск этапа) схлопываются: побеждает последняя строка."""
    from btzsc_ru.io_utils import dedup_results

    rows = [
        {"test": "block_a", "model_id": "m", "dataset_key": "d", "n_samples": "300", "macro_f1": "0.1"},
        {"test": "block_a", "model_id": "m", "dataset_key": "d", "n_samples": "300", "macro_f1": "0.2"},
        {"test": "block_b", "model_id": "m", "dataset_key": "d", "n_samples": "300", "macro_f1": "0.3"},
        {"test": "block_a", "model_id": "m", "dataset_key": "d", "n_samples": "20", "macro_f1": "0.4"},
    ]
    out = dedup_results(rows)
    assert len(out) == 3, "разные test и разные n_samples — разные строки"
    assert [r["macro_f1"] for r in out if r["test"] == "block_a" and r["n_samples"] == "300"] == ["0.2"]


def test_results_writer_keys_and_status(tmp_path):
    w = io_utils.ResultsWriter(tmp_path / "results.csv")
    w.append({"run_key": "abc", "test": "block_a", "model_id": "m", "dataset_key": "d", "status": "OK"})
    w.append({"run_key": "def", "test": "block_a", "model_id": "m", "dataset_key": "d2",
              "status": "SKIPPED", "reason": "нет GPU"})
    assert w.existing_keys() == {"abc", "def"}
    assert len(w.rows()) == 2
    try:
        w.append({"run_key": "x", "test": "block_a", "status": "СТРАННЫЙ"})
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("недопустимый status должен отклоняться")


def test_adapter_signature_has_no_gold():
    import inspect

    from btzsc_ru.adapters import EncoderAdapter, LLMAdapter

    for cls in (EncoderAdapter, LLMAdapter):
        for name in ("predict", "predict_scores"):
            params = set(inspect.signature(getattr(cls, name)).parameters)
            assert params == {"self", "texts", "labels", "batch_size"}, (cls, name, params)


def test_export_refuses_empty_run_dir(tmp_path):
    """Пустой каталог прогона не должен превращаться в пустой zip (случай из прогона run_20260927)."""
    import pytest

    from btzsc_ru.runner import export_zip

    run_dir = tmp_path / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "preflight.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError) as err:
        export_zip(run_dir, "run1")
    assert "results.csv" in str(err.value)
    assert "smoke" in str(err.value)
    assert not list((tmp_path / "runs").glob("*.zip"))


def test_manifest_rebuilt_when_sample_size_changes(tmp_path):
    """Реальный баг прогона run1: evaluate с n=300 переиспользовал smoke-манифест на 20 примеров."""
    import json

    from btzsc_ru.config import RunConfig
    from btzsc_ru.runner import manifest_is_stale

    out = tmp_path / "run1"
    out.mkdir()
    (out / "sample_manifest.csv").write_text("dataset_key,sample_id\n", encoding="utf-8")
    smoke = RunConfig(mode="smoke", dataset_keys=["btzsc_agnews"], model_ids=[])
    evaluate = RunConfig(mode="evaluate", dataset_keys=["btzsc_agnews"], model_ids=[])
    other_datasets = RunConfig(mode="smoke", dataset_keys=["btzsc_imdb"], model_ids=[])

    (out / "manifest_info.json").write_text(
        json.dumps({"requested_n": smoke.resolved_n(), "dataset_keys": ["btzsc_agnews"]}), encoding="utf-8"
    )
    assert manifest_is_stale(smoke, out) == "", "тот же режим и та же выборка — пересоздавать нечего"
    assert "20" in manifest_is_stale(evaluate, out) and "300" in manifest_is_stale(evaluate, out)
    assert "btzsc_imdb" in manifest_is_stale(other_datasets, out), "датасета нет в манифесте — пересоздаём"

    (out / "manifest_info.json").unlink()
    assert "manifest_info.json" in manifest_is_stale(smoke, out)


def test_peak_vram_is_reset_before_each_model():
    """Пик памяти должен обнуляться до загрузки модели, иначе он наследуется от прошлых этапов."""
    import inspect

    from btzsc_ru import runner

    src = inspect.getsource(runner.evaluate)
    before_build = src.split("build_adapter(model_spec)")[0]
    assert "free_memory()" in before_build, "нет сброса счётчика памяти перед build_adapter"


def test_manifest_check_survives_full_split(tmp_path):
    """replicate идёт на полном сплите (n=None) — проверка манифеста не должна падать на int(None)."""
    import json

    from btzsc_ru.config import RunConfig
    from btzsc_ru.runner import manifest_is_stale

    out = tmp_path / "replication"
    out.mkdir()
    (out / "sample_manifest.csv").write_text("dataset_key,sample_id\n", encoding="utf-8")
    cfg = RunConfig(mode="replicate", dataset_keys=["btzsc_agnews"], model_ids=[])
    assert cfg.resolved_n() is None

    (out / "manifest_info.json").write_text(
        json.dumps({"requested_n": None, "dataset_keys": ["btzsc_agnews"]}), encoding="utf-8"
    )
    assert manifest_is_stale(cfg, out) == "", "тот же полный сплит — пересоздавать нечего"

    (out / "manifest_info.json").write_text(
        json.dumps({"requested_n": 300, "dataset_keys": ["btzsc_agnews"]}), encoding="utf-8"
    )
    reason = manifest_is_stale(cfg, out)
    assert "300 примеров" in reason and "весь сплит" in reason


def test_preflight_probe_tolerates_partial_group():
    """Пробная загрузка берёт N первых строк; N не обязан делиться на число классов."""
    import pytest

    from btzsc_ru.config import DATASETS_BY_KEY
    from btzsc_ru.data import task_from_btzsc_pairs

    spec = DATASETS_BY_KEY["btzsc_financialphrasebank"]
    hyps = ["A", "B", "C"]
    rows = 200   # 200 % 3 != 0 — ровно случай, который ронял preflight
    ds = {
        "text": [f"t{i // 3}" for i in range(rows)],
        "hypothesis": [hyps[i % 3] for i in range(rows)],
        "labels": [1 if i % 3 == 0 else 0 for i in range(rows)],
    }

    task = task_from_btzsc_pairs(spec, ds, trim_tail=True)
    assert task.n_classes == 3
    assert len(task.texts) == 66

    with pytest.raises(ValueError):
        task_from_btzsc_pairs(spec, ds, trim_tail=False)


def test_log_error_never_raises(tmp_path):
    """Отвалившийся Google Drive не должен ронять прогон на записи лога."""
    from btzsc_ru.io_utils import log_error

    log_error(tmp_path / "errors.log", "обычная запись")
    assert (tmp_path / "errors.log").read_text(encoding="utf-8").strip().endswith("обычная запись")
    log_error("/proc/definitely-not-writable/errors.log", "диск отвалился")
