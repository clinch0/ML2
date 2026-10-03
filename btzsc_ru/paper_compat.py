"""Сравнимо ли наше число с опубликованным в статье.

Корневая проблема, которую закрывает модуль: число, посчитанное на подвыборке нашим кодом,
и число из Таблицы 2 статьи — разные величины, и класть их в одну таблицу нельзя.
В приложении A.4 статьи метрики считаются на ПОЛНОМ тестовом сплите, поэтому «сравнимо»
означает одновременно:

* датасет — из BTZSC (русские наборы сравнивать не с чем, в статье их нет);
* оценка прошла по полному сплиту, а не по подвыборке;
* код считал так же, как код авторов: бэкенд `st`, политика префиксов `paper_code`,
  chat template выключен.

Решение принимается **по каждому датасету отдельно**: при потолке в 3000 примеров
Rotten Tomatoes (1066) и Financial PhraseBank (690) остаются полными и сравнимыми,
а AG News (7600) и IMDb (10000) — нет.
"""

from __future__ import annotations

from dataclasses import dataclass

# Сколько multiclass-примеров в полном test-сплите каждого датасета статьи.
# Пары → примеры: строк в сплите делим на число классов (проверено через datasets-server).
FULL_SPLIT_EXAMPLES: dict[str, int] = {
    "btzsc_agnews": 7_600,                 # 30 400 пар / 4 класса
    "btzsc_imdb": 10_000,                  # 20 000 / 2
    "btzsc_rottentomatoes": 1_066,         # 2 132 / 2
    "btzsc_financialphrasebank": 690,      # 2 070 / 3
    "btzsc_emotiondair": 2_000,            # 12 000 / 6
    "btzsc_banking77": 2_880,              # 221 760 / 77
    "btzsc_massive": 2_974,                # 175 466 / 59
}

# Условия, при которых наш конвейер повторяет код авторов.
PAPER_CODE_SETTINGS = {
    "encoder_backend": "st",
    "prefix_policy": "paper_code",
    "use_chat_template": False,
}


@dataclass(frozen=True)
class Comparability:
    """Вердикт сравнимости и причина, по которой он такой."""

    comparable: bool
    reason: str

    def __bool__(self) -> bool:   # удобно писать `if comparability(...)`
        return self.comparable


def full_split_size(dataset_key: str) -> int | None:
    return FULL_SPLIT_EXAMPLES.get(dataset_key)


def comparable_to_paper(
    dataset_key: str,
    n_evaluated: int | None,
    *,
    encoder_backend: str = "st",
    prefix_policy: str = "paper_code",
    use_chat_template: bool = False,
) -> Comparability:
    """Можно ли класть это число рядом с опубликованным.

    `n_evaluated` — сколько примеров реально посчитано (None = весь сплит).
    """
    full = full_split_size(dataset_key)
    if full is None:
        return Comparability(False, "датасета нет в статье — сравнивать не с чем")

    if n_evaluated is not None and n_evaluated < full:
        return Comparability(
            False, f"подвыборка {n_evaluated} из {full}: в статье (A.4) метрика считается на полном сплите"
        )

    actual = {
        "encoder_backend": encoder_backend,
        "prefix_policy": prefix_policy,
        "use_chat_template": use_chat_template,
    }
    diffs = [f"{k}={actual[k]!r} вместо {v!r}" for k, v in PAPER_CODE_SETTINGS.items() if actual[k] != v]
    if diffs:
        return Comparability(False, "настройки расходятся с кодом авторов: " + "; ".join(diffs))

    return Comparability(True, f"полный сплит {full} примеров, настройки как в коде авторов")


def deviations() -> list[dict[str, str]]:
    """Где статья расходится сама с собой или мы расходимся со статьёй.

    Это не оправдание, а перечень, который должен быть виден проверяющему: каждая строка —
    место в статье, что там написано, что фактически делает её код и что делаем мы.
    """
    return [
        {
            "тема": "префиксы энкодеров",
            "статья": "§4: «следуем официальным инструкциям и префиксам из карточек моделей»",
            "код авторов": "models/embedding.py: префикс навешивается только на e5-*, "
                           "Qwen3-Embedding и e5-mistral; BGE и GTE идут без него",
            "у нас": "prefix_policy='paper_code' по умолчанию (как в коде); "
                     "'model_card' доступна для замера разницы",
            "влияние": "на BGE/GTE и на наши русские модели с префиксами в карточке",
        },
        {
            "тема": "объём оценки",
            "статья": "A.4: метрики на полном тестовом сплите",
            "код авторов": "max_samples=null в опубликованных артефактах",
            "у нас": "основная схема — 300 примеров; сравнимость помечается "
                     "признаком comparable_to_paper по каждому датасету",
            "влияние": "число на подвыборке нельзя класть рядом с Таблицей 2",
        },
        {
            "тема": "chat template у LLM",
            "статья": "§4 и C.3: промпт multiple-choice, шаблон чата не упоминается",
            "код авторов": "models/llm.py: шаблон не применяется, промпт подаётся сырым текстом",
            "у нас": "use_chat_template=False по умолчанию",
            "влияние": "шаблон сдвигает распределение первого токена",
        },
        {
            "тема": "точность вычислений",
            "статья": "Reproducibility: bfloat16 на A100 80GB",
            "код авторов": "torch_dtype передаётся вызывающим",
            "у нас": "LLM в 4-bit на T4 15GB",
            "влияние": "качество LLM может быть занижено; на энкодеры не влияет (fp32)",
        },
        {
            "тема": "усреднение macro-F1",
            "статья": "§3.1: macro-F1 по классам задачи",
            "код авторов": "metrics.py: sklearn f1_score(average='macro') по встреченным классам",
            "у нас": "labels=range(n_classes): непредсказанные классы тоже штрафуются",
            "влияние": "наши числа ниже на 59- и 77-классовых наборах",
        },
        {
            "тема": "отбор примеров",
            "статья": "полный сплит, отбор не требуется",
            "код авторов": "sampling с SAMPLING_SEED=0 только при max_samples",
            "у нас": "стратификация по gold-классу, seed 42, единый манифест на все модели",
            "влияние": "редкие классы представлены лучше, чем в естественном распределении",
        },
    ]
