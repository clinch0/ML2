"""Конфигурация прогона. Все значения — из ТЗ эксперимента и contracts.json (шаг D3).

Модуль не делает сетевых вызовов и не импортирует torch/datasets.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_PATH = ROOT / "contracts.json"

SEED = 42
MAX_LENGTH = 256

# Три теста, которые нельзя путать (подробности — в README и отчёте):
#   baseline — модели СТАТЬИ на её датасетах, ПОЛНЫЙ тест: воспроизведение её чисел;
#   ours     — наши модели, выборка 300, плюс модели статьи на той же выборке (reference);
#   ours_full— наши энкодеры на ПОЛНОМ тесте датасетов статьи: прямое сравнение с публикацией.
MODES = (
    "all",            # preflight → блок А → блок Б → отчёт → export
    "preflight",
    "block_a",        # 5 моделей статьи
    "block_b",        # 5 наших моделей
    "smoke",
    "baseline", "evaluate", "reference", "ours_full",   # расширенные режимы
    "replicate", "st_check", "finetune", "export",
)
# "all" = preflight → smoke → evaluate → reference → replicate → st_check → finetune → export,
# с resume: уже посчитанные пары пропускаются.
MODE = os.environ.get("BTZSC_MODE", "all")   # по умолчанию — сквозной прогон без ручного выбора этапа

# smoke = 20 примеров, evaluate = 300 (N_SAMPLES=None → весь split)
SMOKE_N = 20
EVALUATE_N = 300

# Мягкий бюджет времени Colab (ТЗ эксперимента → раздел Colab)
TIME_BUDGET_S = 7200
TIME_RESERVE_S = 600


@dataclass(frozen=True)
class ModelSpec:
    """Описание одной модели. Поля pooling/prefix/chat_template — из карточки (D3)."""

    model_id: str
    role: str                  # "encoder" | "llm"
    revision: str              # SHA из huggingface_hub.model_info
    params: int
    license: str
    pooling: str | None = None          # "cls" | "mean" | None (для LLM)
    query_prefix: str = ""
    label_prefix: str = ""
    prefix_source: str = ""
    chat_template: bool = False
    load_in_4bit: bool = False
    batch_size: int = 8
    notes: str = ""

    @property
    def short(self) -> str:
        return self.model_id.split("/")[-1]


@dataclass(frozen=True)
class DatasetSpec:
    """Описание одного датасета: ровно те config/split/колонки, что проверены в D3."""

    key: str
    hf_id: str
    config: str | None
    split: str
    language: str
    text_column: str
    label_column: str
    kind: str                      # "btzsc_pairs" | "flat"
    label_names: tuple[str, ...] = ()
    verbalizers: tuple[str, ...] = ()
    n_rows: int | None = None
    notes: str = ""


# ── Модели (ТЗ эксперимента, фиксированный список; revision из contracts.json) ──────

ENCODERS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="cointegrated/rubert-tiny2",
        role="encoder",
        revision="e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae",
        params=29_378_550,
        license="mit",
        pooling="cls",
        prefix_source="1_Pooling/config.json: pooling_mode_cls_token=true; prompts отсутствуют",
    ),
    ModelSpec(
        model_id="sergeyzh/rubert-tiny-turbo",
        role="encoder",
        revision="93769a3baad2b037e5c2e4312fccf6bcfe082bf1",
        params=29_378_550,
        license="mit",
        pooling="cls",
        prefix_source="1_Pooling/config.json: pooling_mode_cls_token=true; prompts отсутствуют",
        notes="параметры = архитектура rubert-tiny2 (base_model в карточке)",
    ),
    ModelSpec(
        model_id="ai-forever/ru-en-RoSBERTa",
        role="encoder",
        revision="89fb1651989adbb1cfcfdedafd7d102951ad0555",
        params=403_707_904,
        license="mit",
        pooling="cls",
        query_prefix="classification: ",
        label_prefix="classification: ",
        prefix_source="config_sentence_transformers.json prompts: classification/search_query/search_document/clustering",
    ),
    ModelSpec(
        model_id="deepvk/USER-bge-m3",
        role="encoder",
        revision="0cc6cfe48e260fb0474c753087a69369e88709ae",
        params=359_026_688,
        license="apache-2.0",
        pooling="cls",
        prefix_source="1_Pooling/config.json: cls; config_sentence_transformers.json prompts={} → префиксов нет",
    ),
    ModelSpec(
        model_id="intfloat/multilingual-e5-base",
        role="encoder",
        revision="d128750597153bb5987e10b1c3493a34e5a4502a",
        params=278_044_162,
        license="mit",
        pooling="mean",
        query_prefix="query: ",
        label_prefix="passage: ",
        prefix_source="1_Pooling/config.json: mean; карточка E5 требует query:/passage:",
    ),
    ModelSpec(
        model_id="ai-forever/FRIDA",
        role="encoder",
        revision="850455b605544a944739b25f81ddf812b6e3d0d5",
        params=823_401_216,
        license="mit",
        pooling="cls",
        query_prefix="categorize: ",
        label_prefix="categorize: ",
        prefix_source="config_sentence_transformers.json: prompts.categorize; 1_Pooling: cls",
        notes="Сбер, база FRED-T5-1.7B, 823M; топ-1 ruMTEB среди моделей до 3B (avg 0.707)",
    ),
    ModelSpec(
        model_id="sergeyzh/rubert-mini-frida",
        role="encoder",
        revision="32a82388a9f99c99eca74494731f7ee6e9a9fe3a",
        params=32_262_504,
        license="mit",
        pooling="mean",
        query_prefix="categorize: ",
        label_prefix="categorize: ",
        prefix_source="config_sentence_transformers.json: default_prompt_name=Classification → 'categorize: '",
        notes="малая русская модель 32M, добавлена для проверки гипотезы о задачах на русском",
    ),
    ModelSpec(
        model_id="sergeyzh/BERTA",
        role="encoder",
        revision="914c8c8aed14042ed890fc2c662d5e9e66b2faa7",
        params=128_345_088,
        license="mit",
        pooling="mean",
        query_prefix="categorize_entailment: ",
        label_prefix="categorize_entailment: ",
        prefix_source="config_sentence_transformers.json: default_prompt_name=Classification → 'categorize_entailment: '",
        notes="русская модель 128M (FRIDA-семейство)",
    ),
    ModelSpec(
        model_id="intfloat/multilingual-e5-small",
        role="encoder",
        revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
        params=117_654_272,
        license="mit",
        pooling="mean",
        query_prefix="query: ",
        label_prefix="passage: ",
        prefix_source="1_Pooling/config.json: mean; префиксы query:/passage: из карточки E5",
        notes="иностранный мультиязычный baseline малого размера",
    ),
)

LLMS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct",
        role="llm",
        revision="c9f763e5097587066e602ebabed4d93ea29bb60c",
        params=1_543_298_048,
        license="apache-2.0",
        chat_template=True,
        load_in_4bit=True,
        batch_size=1,
        prefix_source="tokenizer_config.chat_template (ChatML <|im_start|>/<|im_end|>)",
    ),
    ModelSpec(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        role="llm",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        params=1_543_714_304,
        license="apache-2.0",
        chat_template=True,
        load_in_4bit=True,
        batch_size=1,
        prefix_source="tokenizer_config.chat_template (ChatML), карточка Qwen2.5",
    ),
)

# ── Кросс-энкодеры: третье и четвёртое семейства статьи (reranker, NLI) ─────
# Статья сравнивает ПЯТЬ семейств; до сих пор у нас были только эмбеддеры и LLM.
# Главный тезис статьи — «реранкер обходит всех», и именно он без этих моделей не проверен.
# Оба чекпоинта идут в блок Б: тот же манифест, тот же n, та же шкала, что у остальных.
# Адаптеры — буква в букву по официальным btzsc/models/reranker.py и nli.py (см. adapters.py).
CROSS_ENCODERS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="BAAI/bge-reranker-v2-m3",
        role="reranker",
        revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        params=567_755_777,
        license="apache-2.0",
        batch_size=8,      # пар (текст × метка) за проход — дефолт btzsc/models/reranker.py
        prefix_source="config.json: num_labels=1 → скор = logits[:, 0] "
                      "(ветка `logits.shape[-1] == 1` в btzsc/models/reranker.py::_score_from_logits)",
        notes="МУЛЬТИЯЗЫЧНЫЙ реранкер (не русский!): вывод формулируется на уровне семейства — "
              "«мультиязычный реранкер против мультиязычных эмбеддеров». 568M, cross-encoder, "
              "один проход на каждую пару текст × класс",
    ),
    ModelSpec(
        model_id="cointegrated/rubert-base-cased-nli-threeway",
        role="nli",
        revision="920cbb52ef830e94461bf141ec2119979b6049e2",
        params=177_855_747,
        license="unspecified (в карточке модели лицензия не указана; база DeepPavlov/rubert-base-cased)",
        batch_size=16,     # дефолт btzsc/models/nli.py
        prefix_source="config.json: label2id={'entailment': 0, ...} → скор = logits[:, 0] "
                      "(btzsc/models/nli.py::_find_entailment_idx, ключ 'entailment')",
        notes="русское NLI-семейство, 178M, обучено на переведённых NLI-датасетах "
              "(cointegrated/nli-rus-translated-v2021); база — mBERT-наследник, английский понимает. "
              "NLI в таблице авторов — слабое семейство (bart-large-mnli 0.51)",
    ),
)

# ── Модели статьи: блок сверки (replication) ────────────────────────────────
# Нужны, чтобы проверить эквивалентность НАШЕГО кода коду статьи: у этих моделей
# есть опубликованные числа на agnews и imdb (hf/results_repo/results/**, by_dataset).
# Pooling/prefix заданы ровно так, как в официальном адаптере (models/embedding.py):
# префиксы применяются ТОЛЬКО к e5-*; bge и MiniLM подаются без префикса,
# даже если карточка модели рекомендует инструкцию для retrieval.
PAPER_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        role="encoder",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        params=22_713_728,
        license="apache-2.0",
        pooling="mean",
        prefix_source="1_Pooling/config.json: pooling_mode_mean_tokens=true; префикс не применяется (btzsc/models/embedding.py)",
        notes="модель статьи; опубликовано agnews 0.495, imdb 0.340",
    ),
    ModelSpec(
        model_id="intfloat/e5-base-v2",
        role="encoder",
        revision="f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        params=109_482_752,
        license="mit",
        pooling="mean",
        query_prefix="query: ",
        label_prefix="passage: ",
        prefix_source="1_Pooling/config.json: mean; префиксы query:/passage: как в btzsc/models/embedding.py",
        notes="модель статьи; опубликовано agnews 0.761, imdb 0.898",
    ),
    ModelSpec(
        model_id="intfloat/e5-large-v2",
        role="encoder",
        revision="f169b11e22de13617baa190a028a32f3493550b6",
        params=335_142_400,
        license="mit",
        pooling="mean",
        query_prefix="query: ",
        label_prefix="passage: ",
        prefix_source="карточка E5 + btzsc/models/embedding.py",
        notes="модель статьи; опубликовано agnews 0.787, imdb 0.928",
    ),
    ModelSpec(
        model_id="BAAI/bge-base-en-v1.5",
        role="encoder",
        revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        params=109_482_752,
        license="mit",
        pooling="cls",
        prefix_source="1_Pooling/config.json: pooling_mode_cls_token=true; префикс не применяется (btzsc/models/embedding.py)",
        notes="модель статьи; опубликовано agnews 0.635, imdb 0.896",
    ),
    ModelSpec(
        model_id="BAAI/bge-large-en-v1.5",
        role="encoder",
        revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        params=335_142_400,
        license="mit",
        pooling="cls",
        prefix_source="карточка BGE + btzsc/models/embedding.py",
        notes="модель статьи; опубликовано agnews 0.766, imdb 0.935",
    ),
)

# На каких датасетах имеет смысл сверка: англоязычные датасеты статьи.
# rottentomatoes и financialphrasebank добавлены как диагностика: тексты короткие
# (26 и 29 токенов), поэтому объясняют ли расхождение обрезка длины и размер контекста.
REPLICATION_DATASET_KEYS = (
    "btzsc_agnews",
    "btzsc_imdb",
    "btzsc_rottentomatoes",
    "btzsc_financialphrasebank",
)
# Диагностика ST идёт на коротких датасетах: там обрезка длины заведомо ни при чём.
ST_CHECK_DATASET_KEYS = ("btzsc_rottentomatoes", "btzsc_financialphrasebank")
# Размер выборки сверки: None = весь test-сплит, как у авторов.
REPLICATION_N: int | None = None

# Потолок на увеличенные этапы (baseline, ours_full, replicate). Это НЕ «полный тест»:
# для Rotten Tomatoes (1066) и PhraseBank (690) 3000 покрывает весь сплит, а для AG News (7600)
# и IMDb (10000) — только часть. В текстах писать «до 3000 примеров тестового сплита». Полное ядро — 27 210 примеров,
# это 1.5-2 часа на T4 только для энкодеров, и сессия Colab не доживает. 3000 примеров на
# датасет дают ту же картину с доверительным интервалом порядка 0.01 и укладываются в сессию.
# None = считать весь сплит (для прогона на своём железе).
FULL_TEST_CAP: int | None = 3000

# Общий дедлайн прогона в секундах: этапы, которые уже не успевают начаться, помечаются
# NOT_RUN, а экспорт выполняется в любом случае — чтобы смерть сессии не уносила результаты.
GLOBAL_DEADLINE_S = 150 * 60

# Эталон берём НЕ руками, а из опубликованных артефактов лидерборда, которые лежат
# в репозитории (paper_scores/<имя>.json, блок results.by_dataset). Так числа статьи
# невозможно «подкрутить» и всегда видно их источник.
PAPER_LEADERBOARD_FILE = {
    "sentence-transformers/all-MiniLM-L6-v2": "all-MiniLM-L6-v2.json",
    "intfloat/e5-base-v2": "e5-base-v2.json",
    "intfloat/e5-large-v2": "e5-large-v2.json",
    "BAAI/bge-base-en-v1.5": "bge-base-en-v1.5.json",
    "BAAI/bge-large-en-v1.5": "bge-large-en-v1.5.json",
}
PAPER_SCORES_DIR = Path(__file__).resolve().parent.parent / "paper_scores"
# Датасеты статьи, по которым нам нужен эталон (совпадает с ядром BTZSC_CORE_KEYS ниже).
PAPER_REFERENCE_DATASETS = (
    "btzsc_agnews", "btzsc_imdb", "btzsc_rottentomatoes", "btzsc_financialphrasebank",
    "btzsc_emotiondair", "btzsc_massive", "btzsc_banking77",
)


def load_paper_reference() -> dict[str, dict[str, float]]:
    """{model_id: {dataset_key: macro_f1}} из paper_scores/*.json (results.by_dataset)."""
    import json as _json

    out: dict[str, dict[str, float]] = {}
    for model_id, filename in PAPER_LEADERBOARD_FILE.items():
        path = PAPER_SCORES_DIR / filename
        if not path.exists():
            continue
        by_dataset = _json.loads(path.read_text(encoding="utf-8"))["results"]["by_dataset"]
        values = {}
        for key in PAPER_REFERENCE_DATASETS:   # эталон по всем датасетам статьи, которые мы считаем
            name = key.replace("btzsc_", "")
            if name in by_dataset:
                values[key] = round(float(by_dataset[name]["macro_f1"]), 4)
        if values:
            out[model_id] = values
    return out


PAPER_REFERENCE: dict[str, dict[str, float]] = load_paper_reference()


MODELS: tuple[ModelSpec, ...] = ENCODERS + CROSS_ENCODERS + LLMS + PAPER_MODELS

RESERVE_MODEL_IDS = (
    "RefalMachine/RuadaptQwen2.5-1.5B-instruct",
    "cointegrated/LaBSE-en-ru",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Qwen/Qwen2.5-0.5B-Instruct",
)


# ── Датасеты (проверено в D3, см. contracts.json) ────────────────────────────

RU_SENTIMENT_VERBALIZERS = (
    "Тональность этого русского текста нейтральная",
    "Тональность этого русского текста положительная",
    "Тональность этого русского текста отрицательная",
)
RU_SENTIMENT_LABELS = ("NEUTRAL", "POSITIVE", "NEGATIVE")  # 0/1/2 — карточка MonoHime

DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="btzsc_agnews",
        hf_id="btzsc/btzsc",
        config="agnews",
        split="test",
        language="en",
        text_column="text",
        label_column="labels",
        kind="btzsc_pairs",
        n_rows=30_400,
        notes="30400 пар / 4 класса = 7600 multiclass-примеров; гипотезы в колонке hypothesis",
    ),
    DatasetSpec(
        key="btzsc_imdb",
        hf_id="btzsc/btzsc",
        config="imdb",
        split="test",
        language="en",
        text_column="text",
        label_column="labels",
        kind="btzsc_pairs",
        n_rows=20_000,
        notes="20000 пар / 2 класса = 10000 multiclass-примеров",
    ),
    DatasetSpec(
        key="btzsc_rottentomatoes",
        hf_id="btzsc/btzsc",
        config="rottentomatoes",
        split="test",
        language="en",
        text_column="text",
        label_column="labels",
        kind="btzsc_pairs",
        n_rows=2_132,
        notes="датасет статьи: 2132 пары / 2 класса = 1066 примеров, тексты короткие (26 токенов) — диагностика сверки",
    ),
    DatasetSpec(
        key="btzsc_financialphrasebank",
        hf_id="btzsc/btzsc",
        config="financialphrasebank",
        split="test",
        language="en",
        text_column="text",
        label_column="labels",
        kind="btzsc_pairs",
        n_rows=2_070,
        notes="датасет статьи: 2070 пар / 3 класса = 690 примеров, тексты короткие (29 токенов) — диагностика сверки",
    ),
    DatasetSpec(
        key="btzsc_emotiondair",
        hf_id="btzsc/btzsc",
        config="emotiondair",
        split="test",
        language="en",
        text_column="text",
        label_column="labels",
        kind="btzsc_pairs",
        n_rows=12_000,
        notes="датасет статьи, эмоции, 6 классов = 2000 примеров",
    ),
    DatasetSpec(
        key="btzsc_banking77",
        hf_id="btzsc/btzsc",
        config="banking77",
        split="test",
        language="en",
        text_column="text",
        label_column="labels",
        kind="btzsc_pairs",
        n_rows=221_760,
        notes="датасет статьи, намерения, 77 классов (у LLM > 52 вариантов → SKIPPED)",
    ),
    DatasetSpec(
        key="btzsc_massive",
        hf_id="btzsc/btzsc",
        config="massive",
        split="test",
        language="en",
        text_column="text",
        label_column="labels",
        kind="btzsc_pairs",
        notes="датасет статьи, намерения голосового помощника (англ.), версия из релиза BTZSC",
    ),
    DatasetSpec(
        key="massive_scenario_en",
        hf_id="mteb/amazon_massive_scenario",
        config="en",
        split="test",
        language="en",
        text_column="text",
        label_column="label",
        kind="flat",
        n_rows=2_974,
        notes="18 сценариев (домены запроса) — параллельная EN/RU задача, помещается в 52 варианта LLM",
    ),
    DatasetSpec(
        key="massive_scenario_ru",
        hf_id="mteb/amazon_massive_scenario",
        config="ru",
        split="test",
        language="ru",
        text_column="text",
        label_column="label",
        kind="flat",
        n_rows=2_974,
        notes="русская половина сценариев MASSIVE; те же 18 меток",
    ),
    DatasetSpec(
        key="go_emotions_en",
        hf_id="seara/ru_go_emotions",
        config="simplified",
        split="test",
        language="en",
        text_column="text",
        label_column="labels",
        kind="go_emotions",
        notes="GoEmotions: 28 эмоций; строки параллельны русским (те же id) — оригинальный английский текст",
    ),
    DatasetSpec(
        key="go_emotions_ru",
        hf_id="seara/ru_go_emotions",
        config="simplified",
        split="test",
        language="ru",
        text_column="ru_text",
        label_column="labels",
        kind="go_emotions",
        notes="те же строки на русском (колонка ru_text) — прямая проверка гипотезы про эмоции",
    ),
    DatasetSpec(
        key="ru_sentiment",
        hf_id="MonoHime/ru_sentiment_dataset",
        config="default",
        split="validation",
        language="ru",
        text_column="text",
        label_column="sentiment",
        kind="flat",
        label_names=RU_SENTIMENT_LABELS,
        verbalizers=RU_SENTIMENT_VERBALIZERS,
        n_rows=21_098,
        notes="реальные splits: train (190k) и validation (21098). Тестового split нет — "
              "оценка идёт на validation и называется validation, не test",
    ),
    DatasetSpec(
        key="massive_en",
        hf_id="mteb/amazon_massive_intent",
        config="en",
        split="test",
        language="en",
        text_column="text",
        label_column="label_text",
        kind="flat",
        n_rows=2_974,
        notes="замена AmazonScience/massive (loading script, не грузится современным datasets); "
              "см. docs/model_audit.md → раздел «Замена датасета»",
    ),
    DatasetSpec(
        key="massive_ru",
        hf_id="mteb/amazon_massive_intent",
        config="ru",
        split="test",
        language="ru",
        text_column="text",
        label_column="label_text",
        kind="flat",
        n_rows=2_974,
        notes="параллельный ru-срез того же датасета; набор меток обязан совпасть с massive_en, иначе delta не считается",
    ),
)

def as_replication(cfg: "RunConfig") -> "RunConfig":
    """Конфиг сверки: модели статьи на её датасетах, объём — как у авторов (весь test)."""
    from dataclasses import replace

    return replace(
        cfg,
        mode="replicate",
        n_samples=cfg.n_samples if cfg.n_samples is not None else REPLICATION_N,
        dataset_keys=tuple(REPLICATION_DATASET_KEYS),
        model_ids=tuple(m.model_id for m in PAPER_MODELS),
    )


# ЯДРО: те же данные, что у авторов статьи. Русскоязычные модели проходят ТОТ ЖЕ тест,
# что и 38 моделей статьи, поэтому их результат сравним с опубликованным напрямую —
# по каждому датасету есть число из results.by_dataset лидерборда.
# Покрыты все четыре типа задач статьи: тема, тональность, эмоции, намерения.
BTZSC_CORE_KEYS = (
    "btzsc_agnews",               # тема, 4 класса
    "btzsc_imdb",                 # тональность, 2
    "btzsc_rottentomatoes",       # тональность, 2, короткие тексты
    "btzsc_financialphrasebank",  # тональность, 3, финансы
    "btzsc_emotiondair",          # эмоции, 6
    "btzsc_massive",              # намерения, 59
    "btzsc_banking77",            # намерения, 77
)

# РАСШИРЕНИЕ: русские данные, которых в статье нет. Отвечает на вопрос, что происходит
# с тем же протоколом при переходе на русский (три параллельные пары EN/RU).
RU_EXTENSION_KEYS = (
    "ru_sentiment",
    "massive_en",
    "massive_ru",
    "massive_scenario_en",
    "massive_scenario_ru",
    "go_emotions_en",
    "go_emotions_ru",
)

OUR_DATASET_KEYS = BTZSC_CORE_KEYS + RU_EXTENSION_KEYS

# ── Два блока прогона (основная схема работы) ─────────────────────────────
# Блок А: 5 моделей САМОЙ СТАТЬИ — те, что влезают в бесплатный Colab.
# Блок Б: 5 наших моделей — обученные в том числе на русском (плюс один
#          мультиязычный baseline для контроля).
# Оба блока идут по одним и тем же датасетам и по одной и той же выборке,
# поэтому их числа сравниваются напрямую.
BLOCK_A_MODEL_IDS = tuple(m.model_id for m in PAPER_MODELS)
# Принцип отбора: по одной лучшей модели от каждого источника, без дублей по происхождению.
# BERTA и rubert-mini-frida — две дистилляции ОДНОЙ модели (ai-forever/FRIDA), поэтому в блоке
# остаётся сама FRIDA и один дистиллят как нижняя точка по памяти.
BLOCK_B_MODEL_IDS = (
    "ai-forever/FRIDA",              # Сбер, 823M, топ-1 ruMTEB до 3B
    "deepvk/USER-bge-m3",            # VK, 359M
    "sergeyzh/rubert-mini-frida",    # независимый автор, 32M — нижняя точка по памяти
    "intfloat/multilingual-e5-base",  # Microsoft, мультиязычный контроль
    "Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct",  # сообщество Vikhr, русская LLM 1.5B
    # Третье и четвёртое семейства статьи. Считаются ТОЛЬКО на block_b (7 датасетов статьи):
    # утверждение «порядок семейств на русском другой, чем на английском» живёт ровно там,
    # где блоки А и Б стоят на одной шкале. В ru_extension сравнивать не с чем (нет
    # английского эталона по семействам), а 88% стоимости кросс-энкодера дают banking77 (77
    # классов) и massive (59). Цена block_b: 300 × 148 классов = 44 400 пар на модель —
    # реранкер ~9–12 мин, NLI ~3–4 мин на T4.
    "BAAI/bge-reranker-v2-m3",       # BAAI, 568M, мультиязычный реранкер (cross-encoder)
    "cointegrated/rubert-base-cased-nli-threeway",  # cointegrated, 178M, русский NLI
)

# Почему не попали (ответ на вопрос «почему именно эти»):
#   sergeyzh/BERTA            — дистиллят FRIDA, дублирует её по происхождению (ruMTEB 0.693 против 0.707)
#   ai-forever/ru-en-RoSBERTa — старая модель Сбера, в наших замерах 0.432 против 0.691 у лидера
#   rubert-tiny2/-turbo       — ниже 0.36 в zero-shot; tiny2 используется как база дообучения
#   Qwen2.5-1.5B-Instruct     — иностранная база того же семейства, что Vikhr; остаётся в расширении
#   t-tech/T-lite-it-1.0      — 7.6B, в 4-bit влезает в T4, но это снова дообученный Qwen-2.5;
#                               включается как расширение, если нужен вопрос «тянет ли русская LLM 8B»
#   ai-sage/Giga-Embeddings   — 3.4B, второе место ruMTEB; кандидат на шестой слот
#   YandexGPT-5-Lite-8B       — LLM, не энкодер; лицензия не Apache (коммерческое — до 10 млн токенов/мес)
# У Т-Банка и Яндекса открытых эмбеддеров нет вовсе, поэтому в блоке энкодеров их быть не может.
BLOCK_DATASET_KEYS = BTZSC_CORE_KEYS      # 7 датасетов статьи, все четыре типа задач
BLOCK_N = 300                              # примеров на датасет в каждом блоке

# ЯКОРЬ К ПУБЛИКАЦИИ. Без него оба блока могут одинаково врать, и заметить это не по чему.
# Берём датасеты статьи и гоняем на них модели статьи ЦЕЛИКОМ — такие строки получают
# comparable_to_paper=True и сверяются с публикацией.
#
# Длинные датасеты (agnews, imdb) добавлены не для объёма, а ради единственного
# необъяснённого места ТЕСТА 1: в якоре run5 совпали ровно модели БЕЗ префиксов (bge — ноль
# в ноль), а разошлись ровно модели С префиксами (e5-base −0.060). Обе наши реализации (hf и st)
# дают по e5 одно и то же число, значит дело не в коде вокруг. Живая гипотеза — обрезка длины:
# run1 шёл бэкендом hf с max_length=256, а у e5/bge собственный предел 512. Проверить её можно
# ТОЛЬКО на длинных текстах; три коротких датасета ниже этот вопрос не задают в принципе.
# Якорь по умолчанию (бэкенд st, полный сплит) отвечает на него напрямую.
# Цена: 5 моделей × (10 000 + 7 600) длинных примеров ≈ +18–22 минуты на T4.
ANCHOR_DATASET_KEYS = (
    "btzsc_rottentomatoes",       # 1066 примеров, короткие тексты (26 токенов)
    "btzsc_financialphrasebank",  # 690, короткие (29 токенов)
    "btzsc_emotiondair",          # 2000
    "btzsc_agnews",               # 7600 — тексты средней длины, есть опубликованные числа всех 5 моделей
    "btzsc_imdb",                 # 10000 — длинные тексты, ключ к гипотезе об обрезке
)
ANCHOR_N: int | None = None       # None = полный сплит, как в A.4 статьи
# Модели якоря. По умолчанию — все 5 моделей статьи. Минимальный вариант, если времени нет:
#   BTZSC_ANCHOR_MODELS="intfloat/e5-base-v2,BAAI/bge-base-en-v1.5" \
#   BTZSC_ANCHOR_DATASETS="btzsc_agnews,btzsc_imdb"
# — 2 модели × 17 600 текстов ≈ 7 минут. Этого хватает для вывода: разойдётся e5 и на длинных,
# и на коротких одинаково — обрезка ни при чём; разойдётся только на длинных — причина найдена.
ANCHOR_MODEL_IDS = tuple(m.model_id for m in PAPER_MODELS)

# Переопределения для самопроверки конвейера (офлайн, без скачивания моделей).
# В реальном прогоне переменные не заданы и состав блоков берётся из констант выше.
# BTZSC_ANCHOR_* дополнительно дают «минимальный якорь» (см. комментарий у ANCHOR_MODEL_IDS).
if os.environ.get("BTZSC_BLOCK_DATASETS"):
    BLOCK_DATASET_KEYS = tuple(os.environ["BTZSC_BLOCK_DATASETS"].split(","))
if os.environ.get("BTZSC_BLOCK_A_MODELS"):
    BLOCK_A_MODEL_IDS = tuple(os.environ["BTZSC_BLOCK_A_MODELS"].split(","))
if os.environ.get("BTZSC_BLOCK_B_MODELS"):
    BLOCK_B_MODEL_IDS = tuple(os.environ["BTZSC_BLOCK_B_MODELS"].split(","))
if os.environ.get("BTZSC_ANCHOR_DATASETS"):
    ANCHOR_DATASET_KEYS = tuple(os.environ["BTZSC_ANCHOR_DATASETS"].split(","))
if os.environ.get("BTZSC_ANCHOR_MODELS"):
    ANCHOR_MODEL_IDS = tuple(os.environ["BTZSC_ANCHOR_MODELS"].split(","))

# Расширенный набор (вне двух блоков, по умолчанию выключен): остальные модели и русские данные.
EXTRA_MODEL_IDS = tuple(
    m.model_id for m in ENCODERS + LLMS if m.model_id not in BLOCK_B_MODEL_IDS
)

DATASETS_BY_KEY = {d.key: d for d in DATASETS}
MODELS_BY_ID = {m.model_id: m for m in MODELS}


@dataclass
class RunConfig:
    """Полная конфигурация прогона; сериализуется в run_config.json и хэшируется."""

    mode: str = MODE
    seed: int = SEED
    max_length: int = MAX_LENGTH
    n_samples: int | None = None
    # по умолчанию — датасеты нашего эксперимента; диагностические датасеты статьи
    # (rottentomatoes, financialphrasebank) подключает только режим replicate
    dataset_keys: tuple[str, ...] = OUR_DATASET_KEYS
    # по умолчанию — только наш блок моделей; модели статьи подключает режим replicate
    model_ids: tuple[str, ...] = tuple(m.model_id for m in ENCODERS + LLMS)
    time_budget_s: int = TIME_BUDGET_S
    time_reserve_s: int = TIME_RESERVE_S
    encoder_batch_sizes: tuple[int, ...] = (32, 16, 8, 4, 1)   # 32 — как в артефактах статьи
    encoder_backend: str = "st"   # st = код-путь статьи (SentenceTransformer), hf = наша реализация
    # Откуда брать префиксы энкодера:
    #   "paper_code"  — как в коде авторов: префикс только у e5-*, Qwen3-Embedding, e5-mistral;
    #                   BGE/GTE/остальные идут без префикса (так получены опубликованные числа);
    #   "model_card"  — как написано в §4 статьи словами: префиксы из карточек моделей.
    prefix_policy: str = "paper_code"
    # Chat template для LLM: в коде авторов его нет, промпт подаётся сырым текстом.
    use_chat_template: bool = False
    # Фоновая загрузка весов следующей модели, пока текущая считается на GPU.
    prefetch_next_model: bool = True
    llm_batch_size: int = 1
    load_in_4bit: bool = True
    extra: dict = field(default_factory=dict)

    def resolved_n(self) -> int | None:
        if self.n_samples is not None:
            return self.n_samples
        if self.mode == "smoke":
            return SMOKE_N
        if self.mode in {"block_a", "block_b"}:
            return BLOCK_N
        if self.mode in {"evaluate", "all", "reference"}:
            return EVALUATE_N
        if self.mode in {"replicate", "st_check", "baseline", "ours_full"}:
            return None     # полный тест, как у авторов
        if self.mode == "preflight":
            return 2
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dataset_keys"] = list(self.dataset_keys)
        d["model_ids"] = list(self.model_ids)
        d["encoder_batch_sizes"] = list(self.encoder_batch_sizes)
        d["resolved_n"] = self.resolved_n()
        return d

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def load_contracts(path: Path | str = CONTRACTS_PATH) -> dict:
    """Читает contracts.json (артефакт шага D3). Без сети."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"contracts.json не найден: {p}. Сначала выполнить шаг D3.")
    return json.loads(p.read_text(encoding="utf-8"))
