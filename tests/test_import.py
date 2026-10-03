"""Импорт игрушечного colab_outputs_*.zip."""

import csv
import json
import zipfile

import pytest

from scripts.import_colab_outputs import ImportError_, import_zip, run_id_from_name

PREDICTIONS = [
    {"run_id": "toy", "run_key": "k1", "model_id": "enc/a", "dataset_key": "ru_sentiment",
     "sample_id": "ru_sentiment:0", "pred_index": 0, "gold_index": 0},
    {"run_id": "toy", "run_key": "k1", "model_id": "enc/a", "dataset_key": "ru_sentiment",
     "sample_id": "ru_sentiment:1", "pred_index": 1, "gold_index": 1},
    {"run_id": "toy", "run_key": "k1", "model_id": "enc/a", "dataset_key": "ru_sentiment",
     "sample_id": "ru_sentiment:1", "pred_index": 1, "gold_index": 1},   # дубль после resume
    {"run_id": "toy", "run_key": "k2", "model_id": "enc/b", "dataset_key": "ru_sentiment",
     "sample_id": "ru_sentiment:0", "pred_index": 1, "gold_index": 0},
    {"run_id": "toy", "run_key": "k2", "model_id": "enc/b", "dataset_key": "ru_sentiment",
     "sample_id": "ru_sentiment:1", "pred_index": 1, "gold_index": 1},
]


def make_zip(path, *, results_macro_f1="1.0", drop=()):
    with zipfile.ZipFile(path, "w") as zf:
        if "predictions.jsonl" not in drop:
            zf.writestr("predictions.jsonl", "\n".join(json.dumps(r) for r in PREDICTIONS))
        if "results.csv" not in drop:
            zf.writestr(
                "results.csv",
                "run_id,run_key,model_id,dataset_key,macro_f1,status\n"
                f"toy,k1,enc/a,ru_sentiment,{results_macro_f1},OK\n",
            )
        if "sample_manifest.csv" not in drop:
            zf.writestr("sample_manifest.csv", "dataset_key,sample_id\nru_sentiment,ru_sentiment:0\n")
        if "run_config.json" not in drop:
            zf.writestr("run_config.json", json.dumps({"mode": "smoke", "seed": 42}))
    return path


def test_run_id_parsing():
    assert run_id_from_name("colab_outputs_run-7.zip") == "run-7"
    with pytest.raises(ImportError_):
        run_id_from_name("results.zip")


def test_empty_zip_rejected_with_hint(tmp_path):
    z = tmp_path / "colab_outputs_toy.zip"
    with zipfile.ZipFile(z, "w"):
        pass
    with pytest.raises(ImportError_) as err:
        import_zip(z, tmp_path / "results", tmp_path / "report.md")
    assert "архив пуст" in str(err.value)
    assert "smoke" in str(err.value)


def test_missing_file_rejected(tmp_path):
    z = make_zip(tmp_path / "colab_outputs_toy.zip", drop=("run_config.json",))
    with pytest.raises(ImportError_):
        import_zip(z, tmp_path / "results", tmp_path / "report.md")


def test_import_recomputes_and_writes(tmp_path):
    z = make_zip(tmp_path / "colab_outputs_toy.zip")
    report = tmp_path / "Эксперимент_BTZSC.md"
    report.write_text("# Отчёт\n\n<!-- RESULTS:START -->\nрезультаты ожидаются\n<!-- RESULTS:END -->\n", encoding="utf-8")

    info = import_zip(z, tmp_path / "results", report)

    assert info["run_id"] == "toy"
    assert info["n_pairs"] == 2
    assert info["n_predictions"] == 5
    out = tmp_path / "results" / "toy"
    rows = {r["model_id"]: r for r in csv.DictReader((out / "results_recomputed.csv").open(encoding="utf-8"))}
    assert float(rows["enc/a"]["macro_f1"]) == pytest.approx(1.0)
    assert float(rows["enc/a"]["n"]) == 2, "дубль sample_id должен схлопнуться"
    assert float(rows["enc/b"]["accuracy"]) == pytest.approx(0.5)
    text = report.read_text(encoding="utf-8")
    assert "enc/a" in text and "Результаты прогона `toy`" in text
    assert "результаты ожидаются" not in text


def test_mismatch_with_colab_results_is_reported(tmp_path):
    z = make_zip(tmp_path / "colab_outputs_toy.zip", results_macro_f1="0.42")
    info = import_zip(z, tmp_path / "results", tmp_path / "report.md")
    assert info["mismatches"], "расхождение results.csv и пересчёта должно фиксироваться"


def test_replication_subdir_is_imported(tmp_path):
    """Каталог сверки на моделях статьи должен переезжать в results/<run_id>/replication/."""
    z = make_zip(tmp_path / "colab_outputs_toy.zip")
    with zipfile.ZipFile(z, "a") as zf:
        zf.writestr("replication/results.csv", "run_key,model_id,dataset_key,macro_f1,status\nk,m,d,0.5,OK\n")
        zf.writestr("replication/predictions.jsonl", "")
        zf.writestr("replication/run_config.json", json.dumps({"mode": "replicate"}))

    import_zip(z, tmp_path / "results", tmp_path / "report.md")

    rep = tmp_path / "results" / "toy" / "replication"
    assert (rep / "results.csv").exists()
    assert (rep / "run_config.json").exists()


def test_pipeline_blocks_reach_the_report(tmp_path, monkeypatch):
    """Сквозная проверка P0: run_all → export_zip → import → в results.csv есть оба блока.

    Именно этот путь был оборван: блоки писали в подкаталоги, которых не знали ни экспорт,
    ни импорт, ни графики.
    """
    import csv as _csv

    from btzsc_ru import io_utils, runner

    run_dir = tmp_path / "runs" / "run_t"
    run_dir.mkdir(parents=True)

    def fake_evaluate(cfg, out, *, cache_dir=None, run_id="local", test=None):
        results = io_utils.ResultsWriter(out / "results.csv")
        preds = io_utils.PredictionsWriter(out / "predictions.jsonl")
        for model_id in cfg.model_ids[:1]:
            results.append({"run_id": run_id, "test": test, "run_key": f"{test}-{model_id}",
                            "model_id": model_id, "dataset_key": "btzsc_agnews",
                            "n_samples": "300", "n_classes": "4", "macro_f1": "0.5",
                            "accuracy": "0.5", "status": "OK"})
            preds.append_batch([{ "run_id": run_id, "run_key": f"{test}-{model_id}",
                                  "model_id": model_id, "dataset_key": "btzsc_agnews",
                                  "sample_id": "s1", "pred_index": 0, "gold_index": 0}])
        (out / "sample_manifest.csv").write_text("dataset_key,sample_id\nbtzsc_agnews,s1\n", encoding="utf-8")
        (out / "run_config.json").write_text(json.dumps({"mode": cfg.mode}), encoding="utf-8")
        return out / "results.csv"

    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    monkeypatch.setattr(runner, "preflight", lambda cfg, out, **kw: {})

    report = runner.run_all(runner.RunConfig(mode="all"), run_dir, run_id="run_t")
    assert report["export"]["status"] == "OK", report["export"]["detail"]

    archive = run_dir.parent / "colab_outputs_run_t.zip"
    assert archive.exists(), "экспорт должен собрать архив"

    info = import_zip(archive, tmp_path / "results", tmp_path / "report.md")
    imported = list(_csv.DictReader((tmp_path / "results" / "run_t" / "results.csv").open(encoding="utf-8")))
    tests_in_file = {r["test"] for r in imported}

    assert {"block_a", "block_b"} <= tests_in_file, f"в отчёт доехали не все блоки: {tests_in_file}"
    assert info["n_predictions"] >= 2


def test_import_refuses_to_overwrite_existing_run(tmp_path):
    """Нельзя молча затереть уже импортированный прогон — так был потерян хороший run1."""
    from scripts.import_colab_outputs import ImportError_

    z = make_zip(tmp_path / "colab_outputs_toy.zip")
    results = tmp_path / "results"
    import_zip(z, results, tmp_path / "report.md")          # первый импорт

    with pytest.raises(ImportError_, match="уже существует"):
        import_zip(z, results, tmp_path / "report.md")      # второй — отказ

    info = import_zip(z, results, tmp_path / "report.md", force=True)   # осознанная перезапись
    assert info["run_id"] == "toy"


def test_import_refuses_broken_run(tmp_path):
    """Прогон, где почти всё упало, не должен попадать в отчёт."""
    import csv as _csv
    import io as _io

    from scripts.import_colab_outputs import ImportError_

    z = tmp_path / "colab_outputs_broken.zip"
    with zipfile.ZipFile(z, "w") as zf:
        rows = [{"run_key": f"k{i}", "model_id": "m", "dataset_key": "d", "macro_f1": "",
                 "status": "FAILED", "reason": "TypeError: unexpected keyword argument"} for i in range(9)]
        rows.append({"run_key": "k9", "model_id": "m", "dataset_key": "d", "macro_f1": "0.5",
                     "status": "OK", "reason": ""})
        buf = _io.StringIO()
        writer = _csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        zf.writestr("results.csv", buf.getvalue())
        zf.writestr("predictions.jsonl", json.dumps(PREDICTIONS[0]))
        zf.writestr("sample_manifest.csv", "dataset_key,sample_id\nd,s\n")
        zf.writestr("run_config.json", json.dumps({"mode": "all"}))

    with pytest.raises(ImportError_, match="сломан"):
        import_zip(z, tmp_path / "results", tmp_path / "report.md")

    info = import_zip(z, tmp_path / "results", tmp_path / "report.md", allow_failed=True)
    assert info["ok_share"] == 0.1
