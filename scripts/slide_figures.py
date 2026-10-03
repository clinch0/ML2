"""Схемы-объяснялки для презентации: рисуются кодом, а не руками.

Схема метода и график результатов для слайдов:

* `fig6_families.png` — чем отличаются эмбеддинг, реранкер, NLI и LLM и сколько каждый стоит
  в проходах модели (ответ на «методы решения» из задания);
* `fig8_quality_bars.png` — наши результаты столбиками вместо таблицы на десять строк.

Фон непрозрачный светлый: прозрачный PNG с тёмным текстом пропадает в тёмной теме.
Вызываются из `scripts/make_figures.py`, отдельно запускать не нужно.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"

NAVY = "#0b2e59"
INK = "#111827"
MUTED = "#6b7280"
CARD = "#eef2f7"
BLUE = "#1f6feb"
ORANGE = "#c2410c"
GREEN = "#15803d"
RED = "#b91c1c"
LINE = "#cbd5e1"

plt.rcParams["font.family"] = ["DejaVu Sans"]


def save(fig, name: str) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def box(ax, x, y, w, h, text, *, fill=CARD, edge=LINE, size=10, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
                                linewidth=1.1, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", color=color, zorder=3, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.2, color=color, zorder=2))


def fig_families() -> Path:
    """Четыре семейства: что подаётся на вход, что получается, сколько стоит."""
    fig, ax = plt.subplots(figsize=(13.6, 7.0))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(0, 97, "Четыре способа сравнить текст с описанием класса", fontsize=19,
            fontweight="bold", color=NAVY, va="top")
    ax.text(0, 92.0, "Пример один и тот же: текст «разбуди меня в пять утра», класс «запрос про будильник».",
            fontsize=12.5, color=MUTED, va="top")

    rows = [
        ("Эмбеддинг", BLUE, "текст → вектор", "описание → вектор", "близость\ncos = 0.81",
         "Текст и класс считаются ПОРОЗНЬ: векторы классов\nсчитаются один раз и идут ко всем текстам.",
         "377 проходов"),
        ("Реранкер", ORANGE, "текст = запрос", "описание = документ", "оценка пары\n7.4",
         "Текст и класс идут в модель ВМЕСТЕ: точнее,\nно каждую пару надо считать заново.",
         "23 100 проходов"),
        ("NLI", GREEN, "текст = посылка", "описание = гипотеза", "«следует»\n0.62",
         "Тот же принцип пары, но вопрос другой:\nследует ли утверждение из текста.",
         "23 100 проходов"),
        ("LLM", RED, "текст + варианты\nA, B, C…", "один длинный промпт", "вероятность буквы\nP(A) = 0.55",
         "Классы перечисляются прямо в промпте:\n77 вариантов часто не влезают в контекст.",
         "300 длинных промптов"),
    ]

    top, row_h = 84.0, 19.5
    for name, color, left1, left2, mid, note, cost in rows:
        y = top - row_h
        ax.add_patch(FancyBboxPatch((0, y), 100, row_h - 2.0, boxstyle="round,pad=0.01,rounding_size=0.01",
                                    linewidth=0, facecolor="#f8fafc", zorder=1))
        ax.text(0.8, y + (row_h - 2.0) / 2, name, fontsize=12.5, fontweight="bold",
                color=color, va="center", ha="left")
        box(ax, 12.4, y + 9.0, 17.8, 6.2, left1, size=10)
        box(ax, 12.4, y + 1.8, 17.8, 6.2, left2, size=10)
        arrow(ax, 30.3, y + 12.1, 35.6, y + 10.0)
        arrow(ax, 30.3, y + 4.9, 35.6, y + 7.0)
        box(ax, 35.8, y + 5.6, 9.4, 6.4, "модель", size=10, bold=True, fill="white", edge=color)
        arrow(ax, 45.4, y + 8.8, 49.4, y + 8.8)
        box(ax, 49.6, y + 5.2, 12.6, 7.2, mid, size=10, fill="white", edge=color, color=color, bold=True)
        ax.text(63.6, y + 12.6, note, fontsize=10.0, color=INK, va="top", ha="left", linespacing=1.45)
        ax.text(63.6, y + 2.8, f"цена на 77 классов и 300 текстов:  {cost}", fontsize=10.0,
                color=color, va="center", ha="left", fontweight="bold")
        top -= row_h

    ax.text(0, 2.4, "Отсюда весь компромисс: эмбеддинг дешевле на порядок, реранкер точнее, "
                    "LLM упирается в длину промпта.",
            fontsize=12.5, color=NAVY, fontweight="bold", va="center")
    return save(fig, "fig6_families.png")


def fig_quality_bars(rows: list[dict], short: dict[str, str], role: dict[str, str],
                     *, same_scale: bool = True, n_samples: str = "300") -> Path | None:
    """Результаты столбиками: длина — качество, подпись — пиковая память.

    Усреднять можно только по ОДНОМУ набору задач. Поэтому сюда передаются строки
    тестов `block_a` и `block_b` — это одни и те же 7 датасетов статьи по 300 примеров
    у обеих групп моделей. Строки русского расширения (другие датасеты) в это среднее
    не попадают: иначе столбик английской модели сравнивался бы с русской задачей.
    """
    per: dict[str, dict] = {}
    for r in rows:
        if not r.get("macro_f1"):
            continue
        item = per.setdefault(r["model_id"], {"f1": [], "vram": 0.0, "test": r.get("test", "")})
        item["f1"].append(float(r["macro_f1"]))
        if r.get("peak_vram_mb"):
            item["vram"] = max(item["vram"], float(r["peak_vram_mb"]))
    items = sorted(((m, statistics.mean(v["f1"]), v["vram"], v["test"]) for m, v in per.items()),
                   key=lambda t: t[1])
    if not items:
        return None

    def colour(mid: str, test: str) -> str:
        if role.get(mid) == "LLM 1.5B":
            return ORANGE
        return BLUE if test == "block_b" else NAVY

    fig, ax = plt.subplots(figsize=(11.6, 0.62 * len(items) + 2.8))
    fig.patch.set_facecolor("white")
    ys = list(range(len(items)))
    ax.barh(ys, [v for _, v, _, _ in items], height=0.62, zorder=2,
            color=[colour(m, tst) for m, _, _, tst in items])
    for y, (mid, value, vram, _tst) in zip(ys, items):
        label = f"{value:.3f}" + (f"   ·   {vram:.0f} МБ" if vram else "")
        ax.text(value + 0.008, y, label, va="center", fontsize=10.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([short.get(m, m.split("/")[-1]) for m, _, _, _ in items], fontsize=10.5)
    ax.set_xlim(0, max(v for _, v, _, _ in items) * 1.3)
    if same_scale:
        ax.set_xlabel(f"macro-F1, среднее по 7 датасетам статьи (по {n_samples} примеров у всех моделей)")
        subtitle = ("Обе группы считались одним кодом на одних и тех же задачах. "
                    "Подпись у столбика — пиковая память GPU в прогоне, а не размер весов.")
    else:
        ax.set_xlabel("macro-F1, среднее по датасетам прогона")
        subtitle = "Подпись у столбика — пиковая память GPU в прогоне, а не размер весов."
    ax.set_title("Качество и цена: длина столбика — качество, подпись — пиковая память",
                 fontsize=13, fontweight="bold", color=INK, pad=24)
    ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=10, color=MUTED, va="bottom")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.scatter([], [], s=90, color=NAVY, label="блок А — модели статьи")
    ax.scatter([], [], s=90, color=BLUE, label="блок Б — российские энкодеры")
    ax.scatter([], [], s=90, color=ORANGE, label="LLM 1.5B в 4-bit")
    ax.legend(loc="lower right", fontsize=9.5, frameon=False)
    return save(fig, "fig8_quality_bars.png")
