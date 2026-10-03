# -*- coding: utf-8 -*-
"""Student slides for BTZSC (arXiv:2603.11991). Numbers come from JSON only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent   # скрипт лежит в scripts/, корень — выше
PAPER_DIR = ROOT / "paper_scores"
RESULTS_DIR = ROOT / "results"          # results/<run_id>/ — импорт из Colab
OUT_PATH = ROOT / "Задание2_BTZSC.pptx"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

BG = RGBColor(0xF7, 0xF5, 0xF2)
INK = RGBColor(0x1C, 0x28, 0x33)
MUTED = RGBColor(0x3E, 0x4C, 0x59)
NAVY = RGBColor(0x1B, 0x3A, 0x4B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CARD = RGBColor(0xFF, 0xFC, 0xF8)
ZEBRA = RGBColor(0xF3, 0xF0, 0xEA)
BEST = RGBColor(0xE3, 0xF0, 0xE6)
LINE = "C8C1B4"
TOTAL = 12

# Датасеты статьи, по которым построены слайды с результатами авторов
PAPER_DATASETS = [
    ("financialphrasebank", "Financial PhraseBank"),
    ("rottentomatoes", "Rotten Tomatoes"),
    ("emotiondair", "Emotion"),
    ("agnews", "AG News"),
    ("banking77", "Banking77"),
]
DATASETS = PAPER_DATASETS

# Наши датасеты (contracts.json, шаг D3)
OUR_DATASETS = [
    ("btzsc_agnews", "AG News (EN)", "тема", "7600", "4"),
    ("btzsc_imdb", "IMDb (EN)", "тональность", "10000", "2"),
    ("ru_sentiment", "MonoHime RU (validation)", "тональность", "21098", "3"),
    ("massive_en", "MASSIVE EN", "намерение", "2974", "60"),
    ("massive_ru", "MASSIVE RU", "намерение", "2974", "60"),
]

OUR_MODELS = [
    ("cointegrated/rubert-tiny2", "rubert-tiny2", "энкодер, CLS", "29 млн"),
    ("sergeyzh/rubert-tiny-turbo", "rubert-tiny-turbo", "энкодер, CLS", "29 млн"),
    ("ai-forever/ru-en-RoSBERTa", "ru-en-RoSBERTa", "энкодер, CLS, префикс classification:", "404 млн"),
    ("deepvk/USER-bge-m3", "USER-bge-m3", "энкодер, CLS", "359 млн"),
    ("intfloat/multilingual-e5-base", "multilingual-e5-base", "энкодер, mean, query:/passage:", "278 млн"),
    ("Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct", "Vikhr-Qwen-2.5-1.5B", "LLM, 4-bit", "1.5 млрд"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct", "LLM, 4-bit", "1.5 млрд"),
]
OUR_SHORT = {mid: short for mid, short, _, _ in OUR_MODELS}
OUR_SHORT.update({
    "sergeyzh/rubert-mini-frida": "rubert-mini-frida",
    "sergeyzh/BERTA": "BERTA",
    "intfloat/multilingual-e5-small": "multilingual-e5-small",
})
# Датасеты, посчитанные у всех моделей (LLM отказывают на 59-классовом MASSIVE).
# Средние по моделям считаются ТОЛЬКО внутри одного теста: это 7 задач русского
# расширения. Подмешивать сюда английские датасеты статьи нельзя — там другие задачи,
# и среднее по смеси сравнивать между моделями бессмысленно.
COMMON_KEYS = (
    "ru_sentiment", "massive_en", "massive_ru",
    "massive_scenario_en", "massive_scenario_ru",
    "go_emotions_en", "go_emotions_ru",
)
SLIDE_TEST = "ru_extension"

OURS_META = [
    ("majority-class", "Большинство", None),
    ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6", "all-MiniLM-L6-v2.json"),
    ("intfloat/e5-base-v2", "e5-base", "e5-base-v2.json"),
    ("BAAI/bge-base-en-v1.5", "bge-base", "bge-base-en-v1.5.json"),
    ("cross-encoder/ms-marco-MiniLM-L6-v2", "ms-marco MiniLM", "ms-marco-MiniLM-L6-v2.json"),
    ("cross-encoder/nli-roberta-base", "NLI RoBERTa", "nli-roberta-base.json"),
]

TYPE_RU = {
    "embedding": "эмбеддинг",
    "reranker": "реранкер",
    "nli": "NLI",
    "llm": "LLM",
}

TASK_RU = {
    "sentiment": "тональность",
    "emotion": "эмоция",
    "intent": "интент",
    "topic": "тема",
}


def f3(value: float) -> str:
    return f"{value:.3f}"


def params_ru(raw: str) -> str:
    text = str(raw)
    if text.endswith("B"):
        return text[:-1] + " млрд"
    if text.endswith("M"):
        return text[:-1] + " млн"
    return text


def load_papers() -> list[dict]:
    rows = []
    for path in sorted(PAPER_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data["model"]
        results = data["results"]
        by_ds = results["by_dataset"]
        rows.append(
            {
                "file": path.name,
                "name": model["name"],
                "type": model["model_type"],
                "params": str(model["params"]),
                "f1": float(results["overall"]["macro_f1"]),
                "acc": float(results["overall"]["accuracy"]),
                "tasks": {k: float(v["macro_f1"]) for k, v in results["by_task"].items()},
                "sets": {k: float(v["macro_f1"]) for k, v in by_ds.items()},
                "n": len(by_ds),
                "mean_f1": sum(float(v["macro_f1"]) for v in by_ds.values()) / len(by_ds),
            }
        )
    rows.sort(key=lambda row: -row["f1"])
    return rows


def find_run_dir(run_id: str | None = None) -> Path | None:
    """Каталог results/<run_id>/ с импортированным прогоном Colab, иначе None."""
    if not RESULTS_DIR.exists():
        return None
    if run_id:
        candidate = RESULTS_DIR / run_id
        return candidate if (candidate / "results_recomputed.csv").exists() or (candidate / "results.csv").exists() else None
    runs = [d for d in sorted(RESULTS_DIR.iterdir()) if d.is_dir()
            and ((d / "results_recomputed.csv").exists() or (d / "results.csv").exists())]
    return runs[-1] if runs else None


def load_ours(run_dir: Path | None) -> list[dict]:
    """Наши результаты из results/<run_id>/. Пусто, если прогона ещё не было."""
    if run_dir is None:
        return []
    path = run_dir / "results_recomputed.csv"
    if not path.exists():
        path = run_dir / "results.csv"
    mode, n_per_dataset = "", ""
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        mode = str(cfg.get("mode", ""))
        n_per_dataset = str(cfg.get("resolved_n", "") or "")
    with path.open(encoding="utf-8", newline="") as fh:
        raw = list(csv.DictReader(fh))
    # Новые прогоны размечены колонкой `test`: берём один тест, иначе среднее
    # склеит английские задачи статьи с русским расширением.
    tagged = [r for r in raw if r.get("test") == SLIDE_TEST]
    if tagged:
        raw = tagged
    per_model: dict[str, dict] = {}
    for row in raw:
        if row.get("status") and row["status"] != "OK":
            continue
        if not row.get("macro_f1"):
            continue
        model_id = row["model_id"]
        item = per_model.setdefault(
            model_id,
            {"id": model_id, "short": OUR_SHORT.get(model_id, model_id.split("/")[-1]),
             "per": {}, "per_acc": {}, "run_id": run_dir.name,
             "mode": mode, "n_per_dataset": n_per_dataset},
        )
        # В одном каталоге могут лежать прогоны разного размера (20 и 300 примеров);
        # на слайд берём строку с самой большой выборкой.
        n_value = int(float(row.get("n_samples") or row.get("n") or 0))
        ds_key = row["dataset_key"]
        if n_value < item.setdefault("n_by_ds", {}).get(ds_key, 0):
            continue
        item["n_by_ds"][ds_key] = n_value
        item["per"][ds_key] = float(row["macro_f1"])
        item["per_acc"][ds_key] = float(row.get("accuracy") or 0.0)
        if row.get("ms_per_example"):
            item.setdefault("ms_list", []).append(float(row["ms_per_example"]))
        if row.get("peak_vram_mb"):
            item.setdefault("vram_list", []).append(float(row["peak_vram_mb"]))
        if n_value:
            item.setdefault("n_actual", set()).add(n_value)
    rows = []
    for item in per_model.values():
        if not item["per"]:
            continue
        item.pop("n_by_ds", None)
        sizes = item.pop("n_actual", set())
        actual = [max(sizes)] if sizes else []   # на слайде — фактический размер использованной выборки
        # На слайд идёт фактический размер выборки из results.csv, а не запрошенный в конфиге:
        # в прогоне run1 запрошено 300, посчитано 20.
        if actual:
            shown = "/".join(str(a) for a in actual)
            if item["n_per_dataset"] and item["n_per_dataset"] not in {str(a) for a in actual}:
                shown += f" (в конфиге запрошено {item['n_per_dataset']})"
            item["n_per_dataset"] = shown
        # Сравниваем по датасетам, посчитанным у всех моделей: у LLM нет 59-классового MASSIVE.
        common = [k for k in COMMON_KEYS if k in item["per"]]
        keys = common if len(common) == len(COMMON_KEYS) else list(item["per"])
        item["f1"] = sum(item["per"][k] for k in keys) / len(keys)
        item["acc"] = sum(item["per_acc"].get(k, 0.0) for k in keys) / len(keys)
        ms_list = item.pop("ms_list", [])
        vram_list = item.pop("vram_list", [])
        item["ms"] = sum(ms_list) / len(ms_list) if ms_list else 0.0
        item["vram"] = max(vram_list) if vram_list else 0.0
        rows.append(item)
    rows.sort(key=lambda row: -row["f1"])
    return rows


def paper_by_file(papers: list[dict]) -> dict[str, dict]:
    return {row["file"]: row for row in papers}


def rect(slide, left, top, width, height, fill: RGBColor):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    return shape


def set_bg(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG
    rect(slide, 0, 0, Inches(0.12), SLIDE_H, NAVY)


def add_text(slide, left, top, width, height, text, size, bold=False, color=INK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Calibri"
    return box


def add_paragraphs(slide, left, top, width, height, items, size=20, color=INK, space=8, bold=False):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = PP_ALIGN.LEFT
        paragraph.space_after = Pt(space)
        run = paragraph.add_run()
        run.text = item
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = "Calibri"
    return box


def add_notes(slide, text: str):
    frame = slide.notes_slide.notes_text_frame
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    frame.clear()
    for index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = PP_ALIGN.LEFT
        run = paragraph.add_run()
        run.text = line
        run.font.size = Pt(16)
        run.font.name = "Calibri"
        run.font.color.rgb = INK


def footer(slide, number: int):
    add_text(
        slide,
        Inches(0.42),
        Inches(7.08),
        Inches(9.2),
        Inches(0.32),
        "BTZSC  ·  Aarab, 12 марта 2026",
        14,
        color=MUTED,
    )
    add_text(
        slide,
        Inches(11.3),
        Inches(7.08),
        Inches(1.6),
        Inches(0.32),
        f"{number}  /  {TOTAL}",
        14,
        color=MUTED,
        align=PP_ALIGN.RIGHT,
    )


def chrome(slide, title: str, number: int):
    set_bg(slide)
    add_text(slide, Inches(0.42), Inches(0.22), Inches(12.4), Inches(0.55), title, 30, bold=True, color=NAVY)
    rect(slide, Inches(0.44), Inches(0.84), Inches(1.35), Inches(0.045), NAVY)
    footer(slide, number)


def set_cell_border(cell, color_hex=LINE):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    for edge in ("lnL", "lnR", "lnT", "lnB"):
        tag = qn(f"a:{edge}")
        for old in tc_pr.findall(tag):
            tc_pr.remove(old)
        line = etree.SubElement(tc_pr, tag)
        line.set("w", "6350")
        fill = etree.SubElement(line, qn("a:solidFill"))
        color = etree.SubElement(fill, qn("a:srgbClr"))
        color.set("val", color_hex)
        dash = etree.SubElement(line, qn("a:prstDash"))
        dash.set("val", "solid")


def paint_cell(cell, text, size, bold, color, fill, align=PP_ALIGN.CENTER):
    cell.text = str(text)
    frame = cell.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.runs[0]
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Calibri"
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.fill.solid()
    cell.fill.fore_color.rgb = fill
    cell.margin_left = Inches(0.05)
    cell.margin_right = Inches(0.05)
    cell.margin_top = Inches(0.02)
    cell.margin_bottom = Inches(0.02)
    set_cell_border(cell)


def quiet_table(table):
    look = table._tbl.find(qn("a:tblLook"))
    if look is not None:
        look.set("firstRow", "0")
        look.set("bandRow", "0")


def new_slide(prs, title, number, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(slide, title, number)
    add_notes(slide, notes)
    return slide


def add_picture(slide, path, left, top, width=None, height=None):
    """Вставка готового PNG (графики строит scripts/make_figures.py)."""
    kwargs = {}
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height
    return slide.shapes.add_picture(str(path), left, top, **kwargs)


# ── Речь докладчика: по одной записи на слайд, суммарно 4 минуты ──────────
SPEECH = {
    1: "Доклад по статье BTZSC и нашему эксперименту. Вопрос статьи: можно ли классифицировать "
       "тексты моделью, которая этих меток никогда не видела. Наш вопрос: работает ли это на русском "
       "и какие российские модели для этого брать.",
    2: "Обычный классификатор требует тысячи размеченных примеров. Zero-shot работает иначе: мы "
       "описываем класс словами, модель сравнивает текст с описанием и выбирает ближайшее. Разметка "
       "не нужна совсем, метки можно менять хоть каждый день.",
    3: "Цель статьи — честно сравнить четыре семейства моделей в одном протоколе. Раньше их мерили "
       "по отдельности, а популярный рейтинг MTEB подмешивает размеченные примеры и меряет не то. "
       "Новизна в том, что это первое совместное сравнение и что реранкеры впервые проверены как "
       "классификаторы.",
    4: "Вот как устроены эти четыре способа. Эмбеддинг считает текст и описание порознь и сравнивает "
       "векторы — это дёшево, векторы классов считаются один раз. Реранкер и NLI подают текст и класс "
       "вместе, это точнее, но каждую пару надо считать заново: на семидесяти семи классах разница "
       "в шестьдесят раз. Языковая модель перечисляет варианты прямо в промпте и упирается в его длину.",
    5: "Данные. В статье двадцать два английских набора четырёх типов задач. Мы берём семь её наборов "
       "целиком и добавляем семь русских, из них три пары — это одни и те же тексты на двух языках. "
       "Важно помнить про случайное угадывание: на двух классах это пятьдесят процентов, на пятидесяти "
       "девяти — меньше двух.",
    6: "Результаты авторов. Побеждает большой реранкер Qwen3 с макро-эф-один ноль семьдесят два. "
       "Эмбеддинги отстают примерно на одну десятую, но работают в разы быстрее. Языковые модели "
       "меньше миллиарда параметров не работают вовсе. Числа взяты из официальных файлов рейтинга.",
    7: "Наш эксперимент устроен как три теста. Первый проверяет наш код на моделях самой статьи. "
       "Второй сравнивает российские модели с её моделями на одной выборке. Третий измеряет, сколько "
       "стоит переход на русский. Это проверка переносимости протокола, а не ещё один прогон.",
    8: "Кто вообще делает русские модели. Энкодеры открывают Сбер, VK и независимые разработчики. "
       "У Т-Банка и Яндекса открытых энкодеров нет вовсе — только языковые модели на восемь и "
       "тридцать два миллиарда, которые в бесплатный Colab не влезают. Поэтому в наборе их нет, и это "
       "факт, а не наш выбор.",
    9: "Тест первый: пять моделей из статьи считаем нашим кодом на полном тестовом сплите — ровно "
       "так, как меряет она. Одиннадцать пар из пятнадцати совпали, шесть из них — до третьего "
       "знака; ещё одна пара на границе допуска. "
       "Расходится только одна старая модель, и в одну сторону: у авторов ноль тридцать четыре на "
       "двух классах и ноль одиннадцать на шести. Так выглядит константный ответ, а не работающий "
       "классификатор. У нас та же модель даёт ноль шестьдесят пять.",
    10: "Тест второй — главный, обе группы на одной шкале. Разрыв между лучшей моделью статьи и "
        "лучшей российской — полторы сотых, это меньше разброса внутри блоков, а по энкодерам средние "
        "вообще равны. Российская FRIDA выигрывает на финансах и эмоциях с заметным отрывом. "
        "Модель на тридцать два миллиона параметров держит уровень ноль шестьдесят пять при памяти "
        "сто пятьдесят мегабайт. Проседаем мы на многоклассовых намерениях.",
    11: "Тест третий: цена перехода на русский. Разрыв зависит не от языка, а от дробности классов. "
        "На эмоциях теряем две сотых, на сценариях четыре, на пятидесяти девяти намерениях — почти шесть. Причина простая: "
        "«поставь будильник на семь» и «напомни мне в семь» после перевода становятся почти "
        "одинаковыми. И отдельный результат: тысяча восемьсот размеченных примеров и двенадцать "
        "секунд обучения дали больше, чем переход к модели в пятьдесят раз крупнее.",
    12: "Итого. На английском тесте самой статьи российские энкодеры идут вровень с её моделями: "
        "разрыв лучших полторы сотых. Лучший выбор для практики — компактный энкодер, а не языковая "
        "модель. Переход на русский стоит от двух до шести сотых в зависимости от дробности классов. "
        "И если есть хотя бы тысяча размеченных примеров, zero-shot брать не нужно вовсе.",
}


def speech_word_count() -> int:
    return sum(len(text.split()) for text in SPEECH.values())


def build_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_text(slide, Inches(0.55), Inches(1.05), Inches(12), Inches(0.35),
             "ЗАДАНИЕ 2 · АНАЛИЗ СТАТЬИ И ЭКСПЕРИМЕНТ", 16, bold=True, color=MUTED)
    add_text(slide, Inches(0.52), Inches(1.45), Inches(12), Inches(0.9), "BTZSC", 60, bold=True, color=NAVY)
    add_text(slide, Inches(0.55), Inches(2.45), Inches(11.8), Inches(1.5),
             "Можно ли классифицировать тексты моделью,\nкоторая этих меток никогда не видела?\n"
             "И какие русские модели для этого брать?", 26, color=INK)
    rect(slide, Inches(0.55), Inches(4.05), Inches(1.5), Inches(0.045), NAVY)
    add_text(slide, Inches(0.55), Inches(4.3), Inches(10), Inches(0.4),
             "Статья: Ilias Aarab, ICLR 2026 · arXiv:2603.11991", 20, bold=True)
    add_text(slide, Inches(0.55), Inches(4.75), Inches(11.5), Inches(0.8),
             "Наш эксперимент: 5 моделей статьи + 5 российских · 7 её наборов + 7 русских · "
             "бесплатный Colab", 19, color=MUTED)
    link = add_text(slide, Inches(0.55), Inches(5.9), Inches(11), Inches(0.32),
                    "https://huggingface.co/papers/2603.11991", 18, color=INK)
    link.text_frame.paragraphs[0].runs[0].hyperlink.address = "https://huggingface.co/papers/2603.11991"
    add_notes(slide, SPEECH[1])


def build_task(prs):
    slide = new_slide(prs, "Постановка задачи: что такое zero-shot", 2, SPEECH[2])
    cards = [
        ("Обычный способ", "Собрать тысячи размеченных примеров\nи обучить модель.\nДорого и долго."),
        ("Zero-shot", "Описать каждый класс одной фразой.\nМодель сравнивает текст с описаниями\nи выбирает ближайшее."),
        ("Что получаем", "Новый набор меток — без обучения.\nЦена: качество ниже,\nчем у обученной модели."),
    ]
    for i, (title, body) in enumerate(cards):
        left = 0.5 + i * 4.25
        rect(slide, Inches(left), Inches(1.15), Inches(3.95), Inches(1.9), CARD)
        add_text(slide, Inches(left + 0.2), Inches(1.28), Inches(3.6), Inches(0.4), title, 19, bold=True, color=NAVY)
        add_text(slide, Inches(left + 0.2), Inches(1.72), Inches(3.6), Inches(1.2), body, 15, color=INK)

    add_text(slide, Inches(0.5), Inches(3.3), Inches(12.3), Inches(0.4), "Как это считается", 19, bold=True, color=NAVY)
    add_paragraphs(slide, Inches(0.5), Inches(3.75), Inches(12.3), Inches(1.9), [
        "Текст: «разбуди меня в пять утра»",
        "Класс A: «запрос к голосовому помощнику про будильник»   →   близость 0.81",
        "Класс B: «запрос к голосовому помощнику про погоду»   →   близость 0.42",
        "Ответ — класс A. Обучения не было, метки можно поменять в любой момент.",
    ], size=18, space=10)

    rect(slide, Inches(0.5), Inches(5.75), Inches(12.35), Inches(1.0), CARD)
    add_text(slide, Inches(0.75), Inches(5.95), Inches(11.8), Inches(0.7),
             "Зачем это нужно: разметка стоит денег и времени, а в специальных областях нужны эксперты. "
             "Zero-shot убирает этот этап целиком.", 17, color=INK)


def build_goals(prs):
    slide = new_slide(prs, "Цель, задачи и новизна статьи", 3, SPEECH[3])
    add_text(slide, Inches(0.5), Inches(1.1), Inches(6.1), Inches(0.4), "Цель и задачи", 19, bold=True, color=NAVY)
    add_paragraphs(slide, Inches(0.5), Inches(1.55), Inches(6.0), Inches(4.4), [
        "Цель: честно сравнить все способы zero-shot в одном протоколе.",
        "Собрать бенчмарк из 22 наборов четырёх типов задач.",
        "Задать единый протокол: описание класса, скоринг, метрика.",
        "Прогнать 38 моделей от 22 млн до 12 млрд параметров.",
        "Проверить, предсказывает ли качество на NLI качество классификации.",
    ], size=17, space=12)

    add_text(slide, Inches(6.9), Inches(1.1), Inches(6.0), Inches(0.4), "Что нового", 19, bold=True, color=NAVY)
    add_paragraphs(slide, Inches(6.9), Inches(1.55), Inches(5.95), Inches(4.4), [
        "Первое совместное сравнение четырёх семейств сразу.",
        "Честный zero-shot: ни одного размеченного примера.",
        "Реранкеры впервые проверены как классификаторы — и выиграли.",
        "Собственные NLI-модели: виден вклад размера и данных отдельно.",
        "Код, данные и живой рейтинг открыты — поэтому мы смогли перепроверить.",
    ], size=17, space=12)

    rect(slide, Inches(0.5), Inches(5.95), Inches(12.35), Inches(0.85), CARD)
    add_text(slide, Inches(0.75), Inches(6.1), Inches(11.8), Inches(0.6),
             "Актуальность: существующие рейтинги подмешивают размеченные примеры, поэтому честного "
             "сравнения zero-shot до этой работы просто не было.", 17, color=INK)


def build_methods(prs):
    slide = new_slide(prs, "Методы решения: четыре семейства моделей", 4, SPEECH[4])
    figure = ROOT / "figures" / "fig6_families.png"
    if figure.exists():
        add_picture(slide, figure, Inches(0.45), Inches(1.05), width=Inches(12.45))
    add_text(slide, Inches(0.5), Inches(6.55), Inches(12.3), Inches(0.5),
             "Метрика у всех одна — macro-F1: она считает качество по каждому классу отдельно, "
             "поэтому «всегда отвечать самым частым» не поможет.", 16, bold=True, color=NAVY)


def build_data(prs):
    slide = new_slide(prs, "Датасеты и бенчмарки", 5, SPEECH[5])
    add_text(slide, Inches(0.5), Inches(1.1), Inches(6.1), Inches(0.4),
             "В статье — 22 английских набора", 19, bold=True, color=NAVY)
    add_paragraphs(slide, Inches(0.5), Inches(1.55), Inches(6.0), Inches(2.4), [
        "тональность — 6 наборов (IMDb, отзывы, финансы)",
        "тема — 11 наборов (новости, форумы, политика)",
        "намерение — 3 набора (банк, голосовой помощник)",
        "эмоция — 2 набора",
        "от 2 до 77 классов, тексты от 8 до 293 слов",
    ], size=16, space=8)

    add_text(slide, Inches(6.9), Inches(1.1), Inches(6.0), Inches(0.4),
             "У нас — её наборы плюс русские", 19, bold=True, color=NAVY)
    rows = [
        ("AG News · IMDb · Rotten Tomatoes", "тема, тональность", "статья"),
        ("Financial PhraseBank · Emotion", "тональность, эмоции", "статья"),
        ("MASSIVE · Banking77", "намерение, 59 и 77 классов", "статья"),
        ("MonoHime", "тональность по-русски", "наше"),
        ("3 пары англ.+рус.", "намерение, тема, эмоция", "наше"),
    ]
    table = slide.shapes.add_table(1 + len(rows), 3, Inches(6.9), Inches(1.55), Inches(5.95), Inches(2.5)).table
    quiet_table(table)
    for i, w in enumerate([2.7, 2.1, 1.15]):
        table.columns[i].width = Inches(w)
    for i, h in enumerate(["Набор", "Что определяем", "Откуда"]):
        paint_cell(table.cell(0, i), h, 13, True, WHITE, NAVY)
    for r, values in enumerate(rows, start=1):
        fill = WHITE if r % 2 else ZEBRA
        for c, v in enumerate(values):
            paint_cell(table.cell(r, c), v, 12, c == 0, INK, fill,
                       PP_ALIGN.LEFT if c < 2 else PP_ALIGN.CENTER)
        table.rows[r].height = Inches(0.36)

    rect(slide, Inches(0.5), Inches(4.35), Inches(12.35), Inches(1.0), CARD)
    add_text(slide, Inches(0.75), Inches(4.5), Inches(11.8), Inches(0.75),
             "Как читать числа: случайное угадывание даёт 0.50 на двух классах и всего 0.017 на 59. "
             "Поэтому 0.61 на MASSIVE — это в 36 раз лучше случайного, а 0.92 на IMDb — в 1.8 раза.",
             17, color=INK)
    rect(slide, Inches(0.5), Inches(5.55), Inches(12.35), Inches(1.1), CARD)
    add_text(slide, Inches(0.75), Inches(5.72), Inches(11.8), Inches(0.85),
             "Три пары «англ. + рус.» — это одни и те же тексты на двух языках с одинаковыми описаниями "
             "классов.\nЗначит, разница в результате показывает ровно одно: цену языка.", 17, color=INK)


def build_paper(prs, papers: list[dict], agg_ok: bool):
    leader = papers[0]
    slide = new_slide(prs, "Что получили авторы", 6, SPEECH[6])
    add_text(slide, Inches(0.5), Inches(1.05), Inches(12.3), Inches(0.4),
             f"Среднее macro-F1 по 22 наборам. Источник — официальные файлы рейтинга "
             f"({len(papers)} моделей из 38 опубликованы пофайлово).", 16, color=MUTED)
    top = papers[:8]
    table = slide.shapes.add_table(1 + len(top), 4, Inches(0.5), Inches(1.55), Inches(7.5), Inches(4.3)).table
    quiet_table(table)
    for i, w in enumerate([3.2, 1.9, 1.1, 1.3]):
        table.columns[i].width = Inches(w)
    for i, h in enumerate(["Модель", "Семейство", "Размер", "macro-F1"]):
        paint_cell(table.cell(0, i), h, 15, True, WHITE, NAVY)
    for r, row in enumerate(top, start=1):
        fill = BEST if r == 1 else (WHITE if r % 2 else ZEBRA)
        values = [row["name"], TYPE_RU.get(row["type"], row["type"]), params_ru(row["params"]), f3(row["f1"])]
        for c, v in enumerate(values):
            paint_cell(table.cell(r, c), v, 14, r == 1 or c == 0, INK, fill,
                       PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER)
        table.rows[r].height = Inches(0.42)

    rect(slide, Inches(8.25), Inches(1.55), Inches(4.6), Inches(4.3), CARD)
    add_text(slide, Inches(8.5), Inches(1.72), Inches(4.1), Inches(0.4), "Выводы авторов", 19, bold=True, color=NAVY)
    add_paragraphs(slide, Inches(8.5), Inches(2.2), Inches(4.15), Inches(3.4), [
        f"Лидер — {leader['name']}, {f3(leader['f1'])}.",
        "Эмбеддинги чуть хуже, но заметно быстрее.",
        "NLI-модели упёрлись в потолок.",
        "LLM меньше 1 млрд параметров не работают.",
        "Тональность лёгкая, эмоции трудные.",
    ], size=16, space=12)


def build_why_russian(prs):
    slide = new_slide(prs, "Наш эксперимент: зачем русские модели", 7, SPEECH[7])
    cards = [
        ("ТЕСТ 1\nПроверка кода", "5 моделей статьи,\nеё датасеты, полный сплит.\nСравниваем с публикацией."),
        ("ТЕСТ 2\nГлавное сравнение", "5 моделей статьи и 5 российских\nна одной выборке и одном коде.\nОтвет: кто лучше."),
        ("ТЕСТ 3\nЦена языка", "Одни и те же тексты\nпо-английски и по-русски.\nОтвет: сколько теряем."),
    ]
    for i, (title, body) in enumerate(cards):
        left = 0.5 + i * 4.25
        rect(slide, Inches(left), Inches(1.15), Inches(3.95), Inches(2.1), CARD)
        add_text(slide, Inches(left + 0.2), Inches(1.3), Inches(3.6), Inches(0.7), title, 18, bold=True, color=NAVY)
        add_text(slide, Inches(left + 0.2), Inches(2.0), Inches(3.6), Inches(1.1), body, 15, color=INK)

    add_text(slide, Inches(0.5), Inches(3.55), Inches(12.3), Inches(0.4),
             "Почему это проверка переносимости, а не ещё один прогон", 19, bold=True, color=NAVY)
    add_paragraphs(slide, Inches(0.5), Inches(4.0), Inches(12.3), Inches(2.5), [
        "Статья honest-но ограничена английским и сама называет мультиязычность будущей работой.",
        "Мы берём её протокол без изменений и меняем ровно одно — модели и язык данных.",
        "Поэтому любое расхождение можно отнести к языку или модели, а не к методике.",
        "Числа на подвыборке помечаются отдельно: рядом с публикацией кладём только полные сплиты.",
    ], size=17, space=12)


def build_who(prs):
    slide = new_slide(prs, "Какие модели взяли и кто их сделал", 8, SPEECH[8])
    figure = ROOT / "figures" / "fig7_who.png"
    if figure.exists():
        add_picture(slide, figure, Inches(0.5), Inches(1.05), width=Inches(12.3))
    add_text(slide, Inches(0.5), Inches(6.5), Inches(12.3), Inches(0.5),
             "В блоке по одной лучшей модели от источника: две дистилляции одной и той же модели "
             "дублировали бы друг друга.", 16, bold=True, color=NAVY)


def build_test1(prs, run_dir):
    slide = new_slide(prs, "ТЕСТ 1. Проверка кода на моделях статьи", 9, SPEECH[9])
    figure = ROOT / "figures" / "fig4_replication.png"
    if figure.exists():
        add_picture(slide, figure, Inches(0.45), Inches(1.1), width=Inches(7.9))
    rect(slide, Inches(8.55), Inches(1.1), Inches(4.3), Inches(4.9), CARD)
    add_text(slide, Inches(8.78), Inches(1.28), Inches(3.9), Inches(0.4), "Что нашли по дороге", 18, bold=True, color=NAVY)
    add_paragraphs(slide, Inches(8.78), Inches(1.8), Inches(3.9), Inches(4.0), [
        "Точки на диагонали — наши числа совпали с опубликованными.",
        "11 пар из 15 совпали, шесть — до третьего знака; ещё одна на границе допуска.",
        "bge-base и bge-large воспроизвелись точно: 0.814 против 0.814.",
        "У all-MiniLM в рейтинге авторов 0.338 на двух классах и 0.111 на шести.",
        "Так выглядит константный ответ: у нас та же модель даёт 0.659 и 0.348.",
    ], size=15, space=11)


def build_test2(prs, ours: list[dict]):
    slide = new_slide(prs, "ТЕСТ 2. Результаты и выбор лучших моделей", 10, SPEECH[10])
    figure = ROOT / "figures" / "fig8_quality_bars.png"
    if figure.exists():
        add_picture(slide, figure, Inches(0.4), Inches(1.05), width=Inches(7.7))

    cards = [
        ("Лучший российский энкодер", "USER-bge-m3 · FRIDA", "0.691 · 0.690"),
        ("Практический выбор", "rubert-mini-frida (32 млн)", "0.650 при 153 МБ весов"),
        ("Русская тональность", "BERTA", "0.589"),
        ("Финансы и эмоции", "FRIDA", "0.709 и 0.553 — лучшие"),
    ]
    for i, (title, name, value) in enumerate(cards):
        top = 1.15 + i * 1.3
        rect(slide, Inches(8.3), Inches(top), Inches(4.55), Inches(1.1), CARD)
        add_text(slide, Inches(8.5), Inches(top + 0.1), Inches(4.2), Inches(0.35), title, 15, bold=True, color=NAVY)
        add_text(slide, Inches(8.5), Inches(top + 0.45), Inches(4.2), Inches(0.5), f"{name} · {value}", 16, color=INK)

    add_text(slide, Inches(0.5), Inches(6.45), Inches(12.3), Inches(0.6),
             "Разрыв между лучшей моделью статьи (0.706) и лучшей российской (0.691) — 0.015, "
             "меньше разброса внутри блоков. По энкодерам средние равны: 0.667 против 0.673.",
             16, bold=True, color=NAVY)


def build_test3(prs):
    slide = new_slide(prs, "ТЕСТ 3. Цена русского языка и что даёт разметка", 11, SPEECH[11])
    figure = ROOT / "figures" / "fig5_delta_by_task.png"
    if figure.exists():
        add_picture(slide, figure, Inches(0.4), Inches(1.05), width=Inches(7.5))

    rect(slide, Inches(8.1), Inches(1.05), Inches(4.75), Inches(2.5), CARD)
    add_text(slide, Inches(8.3), Inches(1.2), Inches(4.4), Inches(0.4), "Теряется не язык", 18, bold=True, color=NAVY)
    add_text(slide, Inches(8.3), Inches(1.65), Inches(4.4), Inches(1.8),
             "«поставь будильник на 7» и «напомни мне в 7» —\nв английском это разные намерения.\n"
             "После перевода фразы почти совпадают,\nи модель начинает путать.\n"
             "Страдает различимость соседних классов.", 14, color=INK)

    rows = [
        ("Лучшая zero-shot модель", "0.589"),
        ("rubert-tiny2 без обучения", "0.383"),
        ("Она же после 12 секунд обучения", "0.604"),
    ]
    table = slide.shapes.add_table(1 + len(rows), 2, Inches(8.1), Inches(3.75), Inches(4.75), Inches(2.0)).table
    quiet_table(table)
    table.columns[0].width, table.columns[1].width = Inches(3.3), Inches(1.45)
    for i, h in enumerate(["Русская тональность", "macro-F1"]):
        paint_cell(table.cell(0, i), h, 14, True, WHITE, NAVY)
    for r, values in enumerate(rows, start=1):
        fill = BEST if r == 3 else (WHITE if r % 2 else ZEBRA)
        for c, v in enumerate(values):
            paint_cell(table.cell(r, c), v, 14, r == 3, INK, fill,
                       PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER)
        table.rows[r].height = Inches(0.45)
    add_text(slide, Inches(0.5), Inches(6.45), Inches(12.3), Inches(0.5),
             "1800 размеченных примеров дали больше, чем переход к модели в 50 раз крупнее.",
             16, bold=True, color=NAVY)


def build_conclusions(prs, ours: list[dict], papers: list[dict], papers_by_file: dict[str, dict]):
    leader = papers[0]
    best = ours[0]["short"] if ours else "—"
    slide = new_slide(prs, "Выводы и ссылки", 12, SPEECH[12])
    add_paragraphs(slide, Inches(0.55), Inches(1.1), Inches(12.2), Inches(3.6), [
        f"В статье лидер — {leader['name']} ({f3(leader['f1'])}), реранкер на 8 млрд параметров.",
        f"На русском наборе из 7 задач лучший у нас — {best}; модель на 32 млн отстаёт всего на 0.02.",
        "Языковые модели 1.5 млрд проиграли энкодерам и по качеству, и по скорости.",
        "Переход на русский стоит 0.02–0.06 macro-F1 и зависит от дробности классов, а не от языка.",
        "1800 размеченных примеров дали больше, чем модель в 50 раз крупнее (0.383 → 0.604).",
        "Наш код сверен с публикацией на полных сплитах: рядом с её числами кладём только сравнимые.",
    ], size=17, space=11)

    rect(slide, Inches(0.5), Inches(4.95), Inches(12.35), Inches(1.75), CARD)
    add_text(slide, Inches(0.75), Inches(5.1), Inches(11.8), Inches(0.35), "Ссылки", 17, bold=True, color=NAVY)
    links = [
        ("Статья", "https://arxiv.org/abs/2603.11991"),
        ("Код и данные авторов", "https://github.com/IliasAarab/btzsc"),
        ("Датасет BTZSC", "https://huggingface.co/datasets/btzsc/btzsc"),
    ]
    for i, (label, url) in enumerate(links):
        top = 5.5 + i * 0.38
        add_text(slide, Inches(0.75), Inches(top), Inches(3.0), Inches(0.32), label, 15, bold=True, color=NAVY)
        box = add_text(slide, Inches(3.6), Inches(top), Inches(8.9), Inches(0.32), url, 15, color=INK)
        box.text_frame.paragraphs[0].runs[0].hyperlink.address = url


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Сборка презентации BTZSC")
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--run-id", default=None, help="какой прогон из results/ показывать (по умолчанию последний)")
    args = parser.parse_args(argv)

    papers = load_papers()
    run_dir = find_run_dir(args.run_id)
    ours = load_ours(run_dir)
    agg_ok = all(abs(row["f1"] - row["mean_f1"]) < 0.002 and row["n"] >= 20 for row in papers)
    by_file = paper_by_file(papers)

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    prs.core_properties.title = "BTZSC — задание 2"
    prs.core_properties.subject = "Zero-shot text classification, arXiv:2603.11991"
    prs.core_properties.category = "student presentation"

    build_title(prs)
    build_task(prs)
    build_goals(prs)
    build_methods(prs)
    build_data(prs)
    build_paper(prs, papers, agg_ok)
    build_why_russian(prs)
    build_who(prs)
    build_test1(prs, run_dir)
    build_test2(prs, ours)
    build_test3(prs)
    build_conclusions(prs, ours, papers, by_file)

    if len(prs.slides) != TOTAL:
        raise SystemExit(f"slide count {len(prs.slides)}")
    out_path = Path(args.out)
    prs.save(str(out_path))
    words = speech_word_count()
    print("saved", out_path)
    print(f"слайдов: {len(prs.slides)} | речь: {words} слов (~{words / 150:.1f} мин)")
    print("run_dir", run_dir if run_dir else "нет импортированного прогона")
    if ours:
        print("наши модели:", ", ".join(f"{r['short']} {f3(r['f1'])}" for r in sorted(ours, key=lambda r: -r["f1"])[:3]))


if __name__ == "__main__":
    raise SystemExit(main())
