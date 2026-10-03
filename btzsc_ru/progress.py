"""Структурированный прогресс прогона.

Зачем отдельный модуль: в Colab прогон запускается подпроцессом, его вывод читается
построчно, поэтому «каретка» (`\\r`) от tqdm не отображается — бар не виден. Здесь бар
рисуется текстом и печатается ОТДЕЛЬНЫМИ строками с ограничением частоты, поэтому он
одинаково читается и в терминале, и в ноутбуке, и в логе CI.

Формат строки всегда один и тот же, чтобы было видно, где именно идёт прогон:

    [block_a 2/5 · BERTA] btzsc_imdb 3/7 [██████░░░░] 60% · 180/300 · 14 мс/пример · фон: frida
     ^этап  ^модель        ^датасет        ^бар        ^доля   ^примеры  ^скорость    ^что качается
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

BAR_WIDTH = 10
MIN_INTERVAL_S = 2.0      # не чаще раза в 2 секунды
MIN_FRACTION_STEP = 0.2   # или каждые 20 % прогресса пары


def safe_print(line: str) -> None:
    """Печать, которая никогда не роняет прогон.

    Вывод может быть закрыт (перенаправление в `head`, обрыв ячейки Colab) — тогда `print`
    бросает BrokenPipeError. Терять из-за этого посчитанную пару недопустимо.
    """
    try:
        print(line, flush=True)
    except (BrokenPipeError, OSError, ValueError):
        pass


def bar(fraction: float, width: int = BAR_WIDTH) -> str:
    """Текстовый индикатор: [██████░░░░]."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} с"
    return f"{seconds / 60:.1f} мин"


@dataclass
class StageProgress:
    """Прогресс одного этапа: модели × датасеты."""

    stage: str
    models: list[str]
    datasets: list[str]
    n_samples: int | str

    started: float = field(default_factory=time.perf_counter)
    model_index: int = 0
    dataset_index: int = 0
    pairs_done: int = 0
    background: str = ""
    _last_print: float = 0.0
    _last_fraction: float = -1.0

    @property
    def total_pairs(self) -> int:
        return len(self.models) * len(self.datasets)

    def _prefix(self) -> str:
        model = self.models[self.model_index].split("/")[-1] if self.models else "—"
        return f"[{self.stage} {self.model_index + 1}/{len(self.models)} · {model}]"

    def _tail(self) -> str:
        return f" · фон: {self.background}" if self.background else ""

    def say(self, message: str) -> None:
        """Безусловная строка (начало этапа, загрузка весов, итог пары)."""
        safe_print(f"{self._prefix()} {message}{self._tail()}")
        self._last_print = time.perf_counter()

    def start_stage(self) -> None:
        safe_print(
            f"\n=== этап {self.stage}: {len(self.models)} моделей × {len(self.datasets)} датасетов "
            f"× {self.n_samples} примеров = {self.total_pairs} пар ==="
        )

    def start_model(self, index: int) -> None:
        self.model_index = index
        self._last_fraction = -1.0

    def loading(self) -> None:
        self.say("загрузка весов…")

    def loaded(self, seconds: float) -> None:
        self.say(f"модель готова за {human_time(seconds)}")

    def start_pair(self, dataset_index: int, dataset_key: str, total_examples: int) -> None:
        self.dataset_index = dataset_index
        self._last_fraction = 0.0   # отсчёт доли ведём от начала пары
        self.say(f"{dataset_key} {dataset_index + 1}/{len(self.datasets)} · {total_examples} примеров")

    def update(self, dataset_key: str, done: int, total: int, ms_per_example: float | None) -> None:
        """Строка прогресса внутри пары; печатается не чаще, чем нужно глазу."""
        fraction = done / total if total else 1.0
        now = time.perf_counter()
        if (now - self._last_print < MIN_INTERVAL_S) and (fraction - self._last_fraction < MIN_FRACTION_STEP):
            return
        speed = f" · {ms_per_example:.0f} мс/пример" if ms_per_example else ""
        self.say(
            f"{dataset_key} {self.dataset_index + 1}/{len(self.datasets)} "
            f"{bar(fraction)} {fraction * 100:3.0f}% · {done}/{total}{speed}"
        )
        self._last_fraction = fraction

    def finish_pair(self, dataset_key: str, macro_f1: float | None, status: str) -> None:
        self.pairs_done += 1
        done_fraction = self.pairs_done / self.total_pairs if self.total_pairs else 1.0
        elapsed = time.perf_counter() - self.started
        eta = elapsed / done_fraction - elapsed if done_fraction else 0.0
        score = f"macro-F1 {macro_f1:.3f}" if macro_f1 is not None else status
        safe_print(
            f"{self._prefix()} {dataset_key}: {score} │ этап {bar(done_fraction)} "
            f"{done_fraction * 100:3.0f}% · пара {self.pairs_done}/{self.total_pairs} · "
            f"прошло {human_time(elapsed)}, осталось ~{human_time(eta)}{self._tail()}"
        )
        self._last_print = time.perf_counter()

    def finish_stage(self) -> None:
        safe_print(
            f"=== этап {self.stage} завершён: {self.pairs_done}/{self.total_pairs} пар "
            f"за {human_time(time.perf_counter() - self.started)} ===\n"
        )
