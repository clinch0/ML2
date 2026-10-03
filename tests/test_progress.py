"""Прогресс прогона: структура строк и фоновая загрузка следующей модели."""

from __future__ import annotations

import time

from btzsc_ru.prefetch import ModelPrefetcher
from btzsc_ru.progress import StageProgress, bar, human_time


def make_bar() -> StageProgress:
    return StageProgress(stage="block_b", models=["deepvk/USER-bge-m3", "sergeyzh/BERTA"],
                         datasets=["btzsc_agnews", "btzsc_imdb"], n_samples=300)


def test_bar_is_text_and_fills_proportionally():
    assert bar(0.0) == "[░░░░░░░░░░]"
    assert bar(1.0) == "[██████████]"
    assert bar(0.5).count("█") == 5
    assert bar(5.0) == "[██████████]", "доля больше единицы не ломает бар"


def test_line_shows_stage_model_dataset_and_percent(capsys):
    """Строка должна отвечать на вопрос «где я сейчас»: этап, модель, датасет, доля."""
    p = make_bar()
    p.start_model(1)
    p.update("btzsc_imdb", done=150, total=300, ms_per_example=14.0)
    line = capsys.readouterr().out.strip()

    assert line.startswith("[block_b 2/2 · BERTA]"), line
    assert "btzsc_imdb" in line and "50%" in line and "150/300" in line and "14 мс/пример" in line
    assert "█" in line and "░" in line


def test_update_is_throttled(capsys):
    """Печать не чаще раза в пару секунд — иначе лог превращается в простыню."""
    p = make_bar()
    p.start_pair(0, "btzsc_agnews", 300)
    capsys.readouterr()
    for done in range(10, 60, 10):          # мелкие шаги подряд
        p.update("btzsc_agnews", done, 300, 10.0)
    assert capsys.readouterr().out == "", "частые мелкие обновления не печатаются"

    p.update("btzsc_agnews", 300, 300, 10.0)   # скачок на 100 % печатается
    assert "100%" in capsys.readouterr().out


def test_pair_finish_reports_stage_progress_and_eta(capsys):
    p = make_bar()
    p.start_model(0)
    time.sleep(0.01)
    p.finish_pair("btzsc_agnews", 0.897, "OK")
    line = capsys.readouterr().out.strip()

    assert "macro-F1 0.897" in line
    assert "пара 1/4" in line and "этап" in line
    assert "осталось" in line, "нужна оценка оставшегося времени"


def test_background_label_is_shown(capsys):
    p = make_bar()
    p.background = "rubert-mini-frida"
    p.say("загрузка весов…")
    assert "фон: rubert-mini-frida" in capsys.readouterr().out


def test_human_time():
    assert human_time(42).endswith("с")
    assert human_time(600) == "10.0 мин"


def test_prefetcher_downloads_next_model_in_background(monkeypatch):
    """Пока GPU считает текущую модель, фоновый поток тянет следующую."""
    started = []

    def fake_prefetch(model_id, revision=None, cache_dir=None):
        started.append(model_id)
        time.sleep(0.05)
        return "ok"

    monkeypatch.setattr("btzsc_ru.prefetch.prefetch_model", fake_prefetch)

    with ModelPrefetcher(cache_dir=None) as pre:
        pre.request("sergeyzh/BERTA", "rev")
        assert pre.status == "BERTA", "модель помечена как качающаяся"
        pre.request("deepvk/USER-bge-m3", "rev")   # второй запрос игнорируется, пока идёт первый
        for _ in range(100):
            if not pre.status:
                break
            time.sleep(0.01)

    assert started == ["sergeyzh/BERTA"], "одновременно качается не больше одной модели"


def test_prefetch_failure_is_not_fatal(monkeypatch):
    """Ошибка фоновой загрузки не должна ронять прогон: модель скачается в свою очередь."""
    def boom(model_id, revision=None, cache_dir=None):
        raise RuntimeError("нет сети")

    monkeypatch.setattr("btzsc_ru.prefetch.prefetch_model", boom)

    with ModelPrefetcher(cache_dir=None) as pre:
        pre.request("sergeyzh/BERTA", "rev")
        for _ in range(100):
            if not pre.status:
                break
            time.sleep(0.01)
    assert pre.status == ""


def test_prefetch_can_be_disabled():
    with ModelPrefetcher(cache_dir=None, enabled=False) as pre:
        pre.request("sergeyzh/BERTA", "rev")
        assert pre.status == ""
