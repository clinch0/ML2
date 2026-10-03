"""Графики по результатам прогона: PNG с прозрачным фоном в figures/.

Источники — только проверенные артефакты:
  results/<run_id>/results.csv          — наши числа (строки status=OK);
  .work/btzsc_repo/hf/results_repo/...  — числа статьи (by_dataset), если клон доступен;
  paper_scores/*.json                   — та же копия чисел статьи в репозитории (fallback).

Использование:
    python scripts/make_figures.py [--run-id run1] [--n 300]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))   # соседний slide_figures.py

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"

SHORT = {
    "sergeyzh/rubert-mini-frida": "rubert-mini-frida",
    "sergeyzh/BERTA": "BERTA",
    "intfloat/multilingual-e5-small": "multilingual-e5-small",
    "cointegrated/rubert-tiny2": "rubert-tiny2",
    "sergeyzh/rubert-tiny-turbo": "rubert-tiny-turbo",
    "ai-forever/ru-en-RoSBERTa": "ru-en-RoSBERTa",
    "deepvk/USER-bge-m3": "USER-bge-m3",
    "intfloat/multilingual-e5-base": "multilingual-e5-base",
    "Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct": "Vikhr-Qwen-2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct",
}
ROLE = {
    "sergeyzh/rubert-mini-frida": "энкодер",
    "sergeyzh/BERTA": "энкодер",
    "intfloat/multilingual-e5-small": "энкодер",
    "cointegrated/rubert-tiny2": "энкодер",
    "sergeyzh/rubert-tiny-turbo": "энкодер",
    "ai-forever/ru-en-RoSBERTa": "энкодер",
    "deepvk/USER-bge-m3": "энкодер",
    "intfloat/multilingual-e5-base": "энкодер",
    "Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct": "LLM 1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct": "LLM 1.5B",
}
COLOR = {"энкодер": "#1f6feb", "LLM 1.5B": "#c2410c"}
INK = "#111827"
MUTED = "#6b7280"


# Фон прозрачный, поэтому картинку могут открыть и на белом, и на тёмном:
# тёмный текст с белой обводкой читается в обоих случаях.
HALO = [pe.withStroke(linewidth=2.6, foreground="white")]


def spread(values: list[float], gap: float) -> list[float]:
    """Сдвигает подписи по вертикали, чтобы они не налезали друг на друга."""
    out: list[float] = []
    for v in sorted(values):
        out.append(v if not out or v - out[-1] >= gap else out[-1] + gap)
    return out


def style(ax):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)


def save(fig, name: str) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def load_ours(run_id: str, n_samples: str, test: str | None = None) -> list[dict]:
    """Строки прогона. Дедупликация обязательна: один и тот же тест мог считаться несколько раз."""
    import sys

    sys.path.insert(0, str(ROOT))
    from btzsc_ru.io_utils import dedup_results

    path = ROOT / "results" / run_id / "results.csv"
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r["status"] == "OK"]
    rows = [r for r in rows if r.get("n_samples") == n_samples]
    if test:
        rows = [r for r in rows if r.get("test") == test]
    return dedup_results(rows)


def load_paper() -> list[dict]:
    """Числа статьи по датасетам: сначала клон репозитория, иначе paper_scores/."""
    files = sorted(glob.glob(str(ROOT / ".work/btzsc_repo/hf/results_repo/results/*/*.json")))
    if not files:
        files = sorted(glob.glob(str(ROOT / "paper_scores/*.json")))
    out = []
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        by = d["results"].get("by_dataset") or d["results"].get("per_dataset") or {}
        if "agnews" not in by or "imdb" not in by:
            continue
        out.append({
            "name": d["model"]["name"],
            "type": d["model"]["model_type"],
            "agnews": by["agnews"]["macro_f1"],
            "imdb": by["imdb"]["macro_f1"],
        })
    return out


def fig_paper_vs_ours(ours: list[dict], paper: list[dict], n_samples: str) -> Path:
    """Распределение моделей статьи против наших чисел на её же двух датасетах.

    В правой колонке — обе группы прогона: модели статьи, посчитанные нашим кодом (блок А),
    и российские модели (блок Б). Это не смешение шкал: датасет один и тот же, выборка одна
    и та же, разные только чекпоинты — именно это и сравнивается.
    """
    per: dict[str, dict] = {}
    block: dict[str, str] = {}
    for r in ours:
        if r.get("test") not in (None, "", "block_a", "block_b"):
            continue
        per.setdefault(r["model_id"], {})[r["dataset_key"]] = float(r["macro_f1"])
        block.setdefault(r["model_id"], r.get("test") or "")

    def colour(mid: str) -> str:
        if ROLE.get(mid) == "LLM 1.5B":
            return COLOR["LLM 1.5B"]
        return "#1f6feb" if block.get(mid) == "block_b" else "#0b2e59"

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.4))
    fig.patch.set_facecolor("white")
    top_limit = 1.02
    for ax, key, label in zip(axes, ("btzsc_agnews", "btzsc_imdb"), ("AG News", "IMDb")):
        pkey = "agnews" if "agnews" in key else "imdb"
        values = sorted(p[pkey] for p in paper)
        jitter = [0.16 * ((i % 5) - 2) / 2 for i in range(len(values))]
        ax.scatter([0.0 + j for j in jitter], values, s=46, color="#9ca3af", alpha=0.6,
                   label=f"модели статьи, n={len(values)}", zorder=2)
        median = statistics.median(values)
        ax.axhline(median, color="#6b7280", lw=1.0, ls="--", zorder=1, xmax=0.42)
        ax.text(-0.42, median + 0.014, f"медиана статьи {median:.3f}", fontsize=8.5,
                color=INK, path_effects=HALO)

        items = sorted(((m, v[key]) for m, v in per.items() if key in v), key=lambda kv: kv[1])
        label_y = spread([v for _, v in items], gap=0.052)
        # Подписи не должны уезжать выше рамки: если верхняя вышла за край, сдвигаем всю пачку.
        overflow = max(label_y) - (top_limit - 0.03) if label_y else 0.0
        if overflow > 0:
            label_y = [y - overflow for y in label_y]
        for (mid, value), ly in zip(items, label_y):
            ax.scatter([1.0], [value], s=110, color=colour(mid), zorder=4,
                       edgecolor="white", linewidth=1.0)
            if abs(ly - value) > 0.004:
                ax.plot([1.06, 1.17], [value, ly], color="#9ca3af", lw=0.8, zorder=3)
            name = SHORT.get(mid, mid.split("/")[-1])
            ax.text(1.20, ly - 0.012, f"{name}  {value:.3f}", fontsize=9.5,
                    color=INK, path_effects=HALO, zorder=5)

        ax.set_xlim(-0.55, 3.1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["статья\n(полные тесты)", f"наш прогон\n({n_samples} примеров)"], fontsize=9.5)
        ax.set_ylim(0.10, top_limit)
        ax.set_ylabel("macro-F1")
        ax.set_title(label, fontsize=13, weight="bold", pad=10)
        style(ax)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_path_effects(HALO)
        ax.yaxis.label.set_path_effects(HALO)
        ax.title.set_path_effects(HALO)
    axes[0].scatter([], [], s=110, color="#0b2e59", label="блок А — модели статьи, наш код")
    axes[0].scatter([], [], s=110, color="#1f6feb", label="блок Б — российские модели")
    axes[0].scatter([], [], s=110, color=COLOR["LLM 1.5B"], label="LLM 1.5B в 4-bit")
    leg = axes[0].legend(loc="lower right", fontsize=8.5, frameon=False)
    for txt in leg.get_texts():
        txt.set_color(INK)
        txt.set_path_effects(HALO)
    sup = fig.suptitle("Одни и те же два датасета статьи: её модели, её чекпоинты в нашем коде "
                       "и российские модели",
                       fontsize=14, color=INK, weight="bold", y=0.985)
    sup.set_path_effects(HALO)
    fig.subplots_adjust(top=0.88, wspace=0.28)
    return save(fig, "fig1_paper_vs_ours.png")


def fig_delta_en_ru(ours: list[dict], n_samples: str) -> Path:
    per = {}
    for r in ours:
        if r["dataset_key"] in ("massive_en", "massive_ru"):
            per.setdefault(r["model_id"], {})[r["dataset_key"]] = float(r["macro_f1"])
    items = sorted(((m, v["massive_en"], v["massive_ru"]) for m, v in per.items() if len(v) == 2),
                   key=lambda t: t[1] - t[2])

    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.patch.set_facecolor("white")
    ys = range(len(items))
    for y, (mid, en, ru) in zip(ys, items):
        ax.plot([ru, en], [y, y], color="#d1d5db", lw=3, solid_capstyle="round", zorder=1)
        ax.scatter([ru], [y], s=90, color="#c2410c", zorder=3, edgecolor="white", linewidth=0.8)
        ax.scatter([en], [y], s=90, color="#1f6feb", zorder=3, edgecolor="white", linewidth=0.8)
        ax.annotate(f"Δ {en - ru:+.3f}", (max(en, ru), y), xytext=(12, -3), textcoords="offset points",
                    fontsize=9, color=INK, path_effects=HALO)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([SHORT.get(m, m.split("/")[-1]) for m, _, _ in items], fontsize=9.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_path_effects(HALO)
    ax.set_xlabel(f"macro-F1 на MASSIVE (59 классов, {n_samples} примеров)")
    # Границы — по данным: при фиксированном окне модели с околонулевым качеством
    # (LLM на 59 классах) уезжали за левый край и строка выглядела пустой.
    lo = min(min(en, ru) for _, en, ru in items)
    hi = max(max(en, ru) for _, en, ru in items)
    ax.set_xlim(max(0.0, lo - 0.05), hi + 0.22)
    ax.scatter([], [], s=90, color="#1f6feb", label="английская половина")
    ax.scatter([], [], s=90, color="#c2410c", label="русская половина")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_title("Разрыв EN−RU на одной задаче с одинаковым набором меток",
                 fontsize=12, weight="bold")
    style(ax)
    ax.title.set_path_effects(HALO)
    ax.xaxis.label.set_path_effects(HALO)
    for t in ax.get_legend().get_texts():
        t.set_color(INK)
        t.set_path_effects(HALO)
    return save(fig, "fig2_delta_en_ru.png")


PAIRS = (
    ("massive_en", "massive_ru", "намерения\n59 классов"),
    ("massive_scenario_en", "massive_scenario_ru", "сценарии\n18 классов"),
    ("go_emotions_en", "go_emotions_ru", "эмоции\n28 классов"),
)


CORE_KEYS = ("btzsc_agnews", "btzsc_imdb", "btzsc_rottentomatoes", "btzsc_financialphrasebank",
             "btzsc_emotiondair", "btzsc_massive", "btzsc_banking77")


def fig_core_vs_paper(ours: list[dict], n_samples: str) -> Path | None:
    """Главный график: обе группы прогона против распределения моделей статьи.

    Правая колонка — только тесты block_a и block_b: одни и те же семь датасетов статьи
    и одна и та же выборка. Левая — опубликованные числа авторов на полных тестах.
    """
    import json as _json

    files = sorted(glob.glob(str(ROOT / ".work/btzsc_repo/hf/results_repo/results/*/*.json")))
    if not files:
        files = sorted(glob.glob(str(ROOT / "paper_scores/*.json")))
    paper = []
    for f in files:
        d = _json.loads(Path(f).read_text(encoding="utf-8"))
        by = d["results"].get("by_dataset", {})
        vals = [by[k.replace("btzsc_", "")]["macro_f1"] for k in CORE_KEYS if k.replace("btzsc_", "") in by]
        if len(vals) == len(CORE_KEYS):
            paper.append((d["model"]["name"], statistics.mean(vals)))
    if not paper:
        return None

    per: dict[str, dict[str, float]] = {}
    block: dict[str, str] = {}
    for r in ours:
        if r["dataset_key"] in CORE_KEYS and r.get("test") in (None, "", "block_a", "block_b"):
            per.setdefault(r["model_id"], {})[r["dataset_key"]] = float(r["macro_f1"])
            block.setdefault(r["model_id"], r.get("test") or "")
    mine = []
    for mid, v in per.items():
        full = len(v) == len(CORE_KEYS)
        mine.append((mid, statistics.mean(v.values()), full))
    if not mine:
        return None

    fig, ax = plt.subplots(figsize=(12.4, 6.6))
    fig.patch.set_facecolor("white")
    xs = [0.10 * ((i % 7) - 3) for i in range(len(paper))]
    ax.scatter(xs, [v for _, v in paper], s=52, color="#9ca3af", alpha=0.65, zorder=2,
               label=f"модели статьи, n={len(paper)}")
    median = statistics.median(v for _, v in paper)
    ax.axhline(median, color="#6b7280", lw=1.0, ls="--", xmax=0.40, zorder=1)
    ax.text(-0.45, median + 0.012, f"медиана статьи {median:.3f}", fontsize=9, color=INK, path_effects=HALO)
    top_name, top_value = max(paper, key=lambda t: t[1])
    ax.scatter([0], [top_value], s=80, color="#111827", zorder=3)
    ax.text(-0.45, top_value + 0.012, f"лучшая у авторов: {top_name} {top_value:.3f}",
            fontsize=9, color=INK, path_effects=HALO)

    mine.sort(key=lambda t: t[1])
    label_y = spread([v for _, v, _ in mine], gap=0.030)
    for (mid, value, full), ly in zip(mine, label_y):
        color = (COLOR["LLM 1.5B"] if ROLE.get(mid) == "LLM 1.5B"
                 else ("#1f6feb" if block.get(mid) == "block_b" else "#0b2e59"))
        ax.scatter([1.0], [value], s=120, color=color, zorder=4, edgecolor="white", linewidth=1.0)
        ax.plot([1.05, 1.18], [value, ly], color="#9ca3af", lw=0.7, zorder=3)
        note = "" if full else "  (5 из 7 наборов)"
        name = SHORT.get(mid, mid.split("/")[-1])
        ax.text(1.21, ly - 0.008, f"{name}  {value:.3f}{note}", fontsize=10, color=INK,
                path_effects=HALO, zorder=5)

    ax.set_xlim(-0.6, 2.75)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{len(paper)} моделей статьи\n(её числа, полные тесты)",
                        f"наш прогон\n(блоки А и Б, {n_samples} примеров)"], fontsize=10.5)
    ax.scatter([], [], s=120, color="#0b2e59", label="блок А — модели статьи, наш код")
    ax.scatter([], [], s=120, color="#1f6feb", label="блок Б — российские модели")
    ax.scatter([], [], s=120, color=COLOR["LLM 1.5B"], label="LLM 1.5B в 4-bit")
    ax.set_ylabel("macro-F1, среднее по 7 датасетам статьи")
    ax.set_title("Обе группы на семи датасетах статьи, одна выборка и один код",
                 fontsize=14, weight="bold", pad=14)
    leg = ax.legend(loc="lower left", fontsize=9, frameon=False)
    style(ax)
    for t in leg.get_texts():
        t.set_color(INK)
        t.set_path_effects(HALO)
    ax.title.set_path_effects(HALO)
    ax.yaxis.label.set_path_effects(HALO)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_path_effects(HALO)
    return save(fig, "fig0_core_vs_paper.png")


def fig_delta_by_task(ours: list[dict]) -> Path | None:
    """Проверка гипотез H1-H3: зависит ли разрыв EN−RU от типа задачи."""
    per: dict[str, dict[str, float]] = {}
    for r in ours:
        per.setdefault(r["model_id"], {})[r["dataset_key"]] = float(r["macro_f1"])
    groups = []
    for en, ru, title in PAIRS:
        values = [(m, v[en] - v[ru]) for m, v in per.items() if en in v and ru in v]
        if values:
            groups.append((title, values))
    if len(groups) < 2:
        return None

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    fig.patch.set_facecolor("white")
    ax.axhline(0, color="#9ca3af", lw=1.0, ls="--", zorder=1)
    for i, (title, values) in enumerate(groups):
        xs = [i + 0.10 * ((j % 5) - 2) for j in range(len(values))]
        ys = [d for _, d in values]
        colors = [COLOR[ROLE.get(m, "энкодер")] for m, _ in values]
        ax.scatter(xs, ys, s=95, c=colors, zorder=3, edgecolor="white", linewidth=0.8)
        mean = statistics.mean(ys)
        ax.plot([i - 0.32, i + 0.32], [mean, mean], color=INK, lw=2.2, zorder=4)
        ax.text(i, mean + 0.006, f"среднее {mean:+.3f}", ha="center", fontsize=10.5,
                color=INK, path_effects=HALO, zorder=5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([t for t, _ in groups], fontsize=10.5)
    ax.set_ylabel("Δ macro-F1 (английская половина − русская)")
    ax.set_ylim(-0.05, 0.165)
    ax.set_title("Разрыв EN−RU зависит от типа задачи, а не постоянен",
                 fontsize=13, weight="bold", pad=14)
    ax.scatter([], [], s=95, color=COLOR["энкодер"], label="энкодер")
    ax.scatter([], [], s=95, color=COLOR["LLM 1.5B"], label="LLM 1.5B в 4-bit")
    leg = ax.legend(loc="upper right", fontsize=9, frameon=False)
    style(ax)
    for t in leg.get_texts():
        t.set_color(INK)
        t.set_path_effects(HALO)
    ax.title.set_path_effects(HALO)
    ax.yaxis.label.set_path_effects(HALO)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_path_effects(HALO)
    return save(fig, "fig5_delta_by_task.png")


def _ru_extension_keys() -> set[str]:
    """Ключи русского расширения берём из конфига прогона, а не дублируем списком."""
    import sys

    sys.path.insert(0, str(ROOT))
    from btzsc_ru.config import RU_EXTENSION_KEYS

    return set(RU_EXTENSION_KEYS)


def fig_quality_vs_latency(ours: list[dict]) -> Path:
    """Качество против задержки на ОДНОМ наборе задач.

    Среднее берётся по семи задачам русского расширения — тому единственному набору,
    который посчитан у всех одиннадцати моделей. Подмешивать сюда английские датасеты
    статьи нельзя: у разных моделей получились бы разные наборы задач.
    """
    ru_rows = [r for r in ours if r.get("test") == "ru_extension"]
    if ru_rows:
        ours = ru_rows
        common = _ru_extension_keys()
        scale = "7 задачам русского расширения"
    else:   # старые прогоны без колонки test
        common = {"btzsc_agnews", "btzsc_imdb", "ru_sentiment", "massive_scenario_en",
                  "massive_scenario_ru", "go_emotions_en", "go_emotions_ru"}
        scale = "7 датасетам, посчитанным у всех моделей"
    agg = {}
    for r in ours:
        if r["dataset_key"] in common:
            item = agg.setdefault(r["model_id"], {"f1": [], "ms": []})
            item["f1"].append(float(r["macro_f1"]))
            item["ms"].append(float(r["ms_per_example"]))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    fig.patch.set_facecolor("white")
    import math

    points = sorted(((mid, statistics.mean(v["ms"]), statistics.mean(v["f1"])) for mid, v in agg.items()),
                    key=lambda t: t[2])
    placed: list[tuple[float, float]] = []
    for mid, x, y in points:
        role = ROLE.get(mid, "энкодер")
        ax.scatter([x], [y], s=155, color=COLOR[role], alpha=0.92, edgecolor="white", linewidth=1.0, zorder=3)
        # если рядом уже есть подпись — ставим текущую сверху, иначе снизу
        close = any(abs(py - y) < 0.025 and abs(math.log10(px) - math.log10(x)) < 0.25 for px, py in placed)
        offset = (11, 12) if close else (11, -20)
        name = SHORT.get(mid, mid.split("/")[-1])
        ax.annotate(f"{name}\n{y:.3f} / {x:.0f} мс", (x, y), xytext=offset,
                    textcoords="offset points", fontsize=9, color=INK, path_effects=HALO)
        placed.append((x, y))
    ax.set_xscale("log")
    ax.set_xlabel("мс на пример (лог. шкала, Tesla T4)")
    ax.set_ylabel(f"macro-F1, среднее по {scale}")
    ax.set_xlim(2, 700)
    # Нижняя граница — по данным: при фиксированной 0.20 модель с качеством 0.09
    # (LLM на русских задачах) просто исчезала с графика.
    lo = min(y for _, _, y in points)
    hi = max(y for _, _, y in points)
    ax.set_ylim(min(0.20, lo - 0.05), max(0.62, hi + 0.08))
    ax.scatter([], [], s=120, color=COLOR["энкодер"], label="энкодер")
    ax.scatter([], [], s=120, color=COLOR["LLM 1.5B"], label="LLM 1.5B в 4-bit")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_title("Качество против задержки: энкодеры выгоднее LLM", fontsize=12, weight="bold")
    style(ax)
    ax.title.set_path_effects(HALO)
    ax.xaxis.label.set_path_effects(HALO)
    ax.yaxis.label.set_path_effects(HALO)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_path_effects(HALO)
    for t in ax.get_legend().get_texts():
        t.set_color(INK)
        t.set_path_effects(HALO)
    return save(fig, "fig3_quality_vs_latency.png")


def fig_replication(run_id: str) -> Path | None:
    """Наши числа против опубликованных на моделях статьи (диагональ = идеальное совпадение)."""
    import sys

    sys.path.insert(0, str(ROOT))
    from btzsc_ru.config import PAPER_REFERENCE

    points = []
    # Новый формат прогона: готовая сверка, посчитанная scripts/check_replication.py.
    check = ROOT / "results" / run_id / "replication_check.csv"
    if check.exists():
        for r in csv.DictReader(check.open(encoding="utf-8")):
            points.append((r["model_id"].split("/")[-1], r["dataset_key"],
                           float(r["paper_macro_f1"]), float(r["our_macro_f1"])))
    else:
        # Старый формат: отдельный подкаталог replication/ со своими строками прогона.
        path = ROOT / "results" / run_id / "replication" / "results.csv"
        if not path.exists():
            print("сверки нет — график пропущен")
            return None
        rows_all = [r for r in csv.DictReader(path.open(encoding="utf-8"))
                    if r.get("status") == "OK" and r.get("macro_f1")]
        dedup: dict[tuple[str, str], dict] = {}
        for r in rows_all:                  # в каталоге копятся строки нескольких запусков
            dedup[(r["model_id"], r["dataset_key"])] = r
        for r in dedup.values():
            ref = PAPER_REFERENCE.get(r["model_id"], {}).get(r["dataset_key"])
            if ref is not None:
                points.append((r["model_id"].split("/")[-1], r["dataset_key"],
                               ref, float(r["macro_f1"])))
    if not points:
        return None

    fig, ax = plt.subplots(figsize=(12.0, 7.6))
    fig.patch.set_facecolor("white")
    ax.plot([0.2, 1.0], [0.2, 1.0], color="#9ca3af", lw=1.2, ls="--", zorder=1)
    ax.text(0.80, 0.815, "идеальное совпадение", fontsize=8.5, color=MUTED, rotation=26, path_effects=HALO)
    for band, alpha in ((0.03, 0.16), (0.06, 0.09)):
        ax.fill_between([0.2, 1.0], [0.2 - band, 1.0 - band], [0.2 + band, 1.0 + band],
                        color="#1f6feb", alpha=alpha, lw=0, zorder=0)

    marker = {"btzsc_agnews": "o", "btzsc_imdb": "s", "btzsc_rottentomatoes": "^",
              "btzsc_financialphrasebank": "D", "btzsc_emotiondair": "P"}
    ds_title = {"btzsc_agnews": "AG News", "btzsc_imdb": "IMDb",
                "btzsc_rottentomatoes": "Rotten Tomatoes", "btzsc_financialphrasebank": "PhraseBank",
                "btzsc_emotiondair": "Emotion"}
    points = sorted(points, key=lambda t: t[3])
    label_y = spread([p[3] for p in points], gap=0.042)
    for (name, ds, ref, our), ly in zip(points, label_y):
        gap = abs(our - ref)
        color = "#1f6feb" if gap <= 0.05 else ("#b45309" if gap <= 0.06 else "#c2410c")
        ax.scatter([ref], [our], s=130, marker=marker.get(ds, "o"), zorder=3, edgecolor="white",
                   linewidth=1.0, color=color)
        ax.plot([ref + 0.01, 1.03], [our, ly], color="#9ca3af", lw=0.7, zorder=2)
        ds_name = ds_title.get(ds, ds.replace("btzsc_", ""))
        ax.text(1.05, ly - 0.012, f"{name} · {ds_name}   Δ {our - ref:+.3f}", fontsize=9,
                color=INK, path_effects=HALO, zorder=4)
    ax.set_xlim(0.25, 1.62)
    ax.set_ylim(0.25, 1.09)
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("macro-F1, опубликовано авторами")
    ax.set_ylabel("macro-F1, наш код на тех же чекпоинтах")
    ax.set_title("Сверка реализации на моделях статьи\n"
                 "(треугольник — Rotten Tomatoes, ромб — PhraseBank, крест — Emotion, "
                 "круг — AG News, квадрат — IMDb; полосы ±0.03 и ±0.06)",
                 fontsize=12, weight="bold", pad=16)
    style(ax)
    ax.title.set_path_effects(HALO)
    ax.xaxis.label.set_path_effects(HALO)
    ax.yaxis.label.set_path_effects(HALO)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_path_effects(HALO)
    return save(fig, "fig4_replication.png")


def build_slide_figures(ours: list[dict], *, n_samples: str) -> list[Path]:
    """Схема метода и график результатов для слайдов.

    Для столбиков берём только тесты `block_a` и `block_b`: это одни и те же 7 датасетов
    статьи у обеих групп. Смешивать их со строками русского расширения нельзя — там другие
    задачи, и среднее по разным наборам задач сравнивать бессмысленно. Старые прогоны без
    колонки `test` рисуются по-прежнему, но с честной подписью «по датасетам прогона».
    """
    from slide_figures import fig_families, fig_quality_bars

    made = [fig_families()]
    comparable = [r for r in ours if r.get("test") in ("block_a", "block_b")]
    coverage: dict[tuple[str, str], set[str]] = {}
    for row in comparable:
        coverage.setdefault((row["test"], row["model_id"]), set()).add(row["dataset_key"])
    same_scale = ({"block_a", "block_b"} <= {test for test, _ in coverage}
                  and all(keys == set(CORE_KEYS) for keys in coverage.values()))
    if same_scale:
        bars = fig_quality_bars(comparable, SHORT, ROLE, same_scale=True,
                                n_samples=n_samples)
        if bars is not None:
            made.append(bars)
    return made


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="run1")
    parser.add_argument("--n", default="300", help="какой размер выборки брать из results.csv")
    args = parser.parse_args()

    ours = load_ours(args.run_id, args.n)
    if not ours:
        raise SystemExit(f"в results/{args.run_id}/results.csv нет строк OK с n_samples={args.n}")
    paper = load_paper()
    made = []
    core_coverage: dict[tuple[str, str], set[str]] = {}
    for row in ours:
        if row.get("test") in ("block_a", "block_b"):
            core_coverage.setdefault((row["test"], row["model_id"]), set()).add(row["dataset_key"])
    if ({"block_a", "block_b"} <= {test for test, _ in core_coverage}
            and all(keys == set(CORE_KEYS) for keys in core_coverage.values())):
        core = fig_core_vs_paper(ours, args.n)
        if core is not None:
            made.append(core)
    if paper and all(any(r["dataset_key"] == key and r.get("test") == test
                         for r in ours)
                     for test in ("block_a", "block_b")
                     for key in ("btzsc_agnews", "btzsc_imdb")):
        made.append(fig_paper_vs_ours(ours, paper, args.n))
    by_model: dict[str, set[str]] = {}
    for row in ours:
        by_model.setdefault(row["model_id"], set()).add(row["dataset_key"])
    if any({"massive_en", "massive_ru"} <= keys for keys in by_model.values()):
        made.append(fig_delta_en_ru(ours, args.n))
    ru_keys = _ru_extension_keys()
    latency_candidates = [r for r in ours if r.get("test") == "ru_extension"
                          and r["dataset_key"] in ru_keys
                          and float(r.get("ms_per_example") or 0) > 0]
    latency_coverage: dict[str, set[str]] = {}
    for row in latency_candidates:
        latency_coverage.setdefault(row["model_id"], set()).add(row["dataset_key"])
    complete_models = {model for model, keys in latency_coverage.items() if keys == ru_keys}
    if complete_models:
        made.append(fig_quality_vs_latency(
            [r for r in latency_candidates if r["model_id"] in complete_models]))
    by_task = fig_delta_by_task(ours)
    if by_task is not None:
        made.append(by_task)
    made += build_slide_figures(ours, n_samples=args.n)
    rep = fig_replication(args.run_id)
    if rep is not None:
        made.append(rep)
    manifest = {"run_id": args.run_id, "n_samples": args.n,
                "files": [path.name for path in made]}
    (FIGDIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for path in made:
        print(f"{path} ({path.stat().st_size} байт)")
    print(f"моделей статьи в сравнении: {len(paper)}; наших строк: {len(ours)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
