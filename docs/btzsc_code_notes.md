# btzsc_code_notes.md — разбор официального репозитория (шаг D2)

Источник кода: <https://github.com/IliasAarab/btzsc>.
Лицензия: MIT. Пакет: `src/btzsc`, публикуется на PyPI как `btzsc`.

Счётчики grep по `src/btzsc`: `load_dataset` — 2, `hypothesis` — 2, `f1` — 14,
`SentenceTransformer` — 3, `logits` — 25, `generate` — **0** (генерации текста в репозитории нет:
и LLM, и генеративный реранкер скорятся по логитам одного следующего токена).

## Файлы и функции

| Путь | Функция / класс | Суть в одну строку |
|---|---|---|
| `src/btzsc/data.py` | `REPO_ID = "btzsc/btzsc"`, `load_btzsc_dataset(name, max_samples, cache_dir)` | грузит `load_dataset("btzsc/btzsc", name=<config>, split="test")` и превращает пары в multiclass |
| `src/btzsc/data.py` | `_get_n_classes(labels_col)` | число классов = индекс первого повтора значения `labels[0]` в бинарной колонке |
| `src/btzsc/data.py` | `BTZSCDataset` | контейнер: `texts`, `labels` (вербализации), `references` (индексы gold), `task`, `domain` |
| `src/btzsc/data.py` | `BTZSC_DATASETS`, `TASK_GROUPS` | 22 датасета и разбиение на sentiment/emotion/intent/topic |
| `src/btzsc/_meta/datasets.yml` | — | task/domain для каждого датасета |
| `src/btzsc/models/base.py` | `BaseModel.predict / predict_scores` | контракт адаптера: `(texts, labels, batch_size) → (n_texts, n_labels)`, gold в сигнатуре нет |
| `src/btzsc/models/embedding.py` | `EmbeddingModel.predict_scores` | `SentenceTransformer.encode(..., normalize_embeddings=True)` для текстов и меток, скор = скалярное произведение (= cosine) |
| `src/btzsc/models/embedding.py` | `_format_query / _format_label` | префиксы по имени модели: `query:`/`passage:` для E5, `Instruct: … Query:` для Qwen3-Embedding и e5-mistral |
| `src/btzsc/models/llm.py` | `LLMModel.predict_scores` | multiple-choice: промпт с буквенными опциями, softmax логитов последней позиции по токенам-буквам, нормировка |
| `src/btzsc/models/llm.py` | `_build_prompt`, `_SYMBOLS`, `_DEFAULT_PROMPT` | опции «A) …», инструкция «Answer with EXACTLY ONE LETTER», хвост «Answer: The correct option is letter » |
| `src/btzsc/models/llm.py` | `_get_letter_ids` | требует, чтобы каждая буква опции была одним токеном, иначе `ValueError` |
| `src/btzsc/models/nli.py` | `NLIModel._find_entailment_idx`, `predict_scores` | логит класса entailment как скор пары (текст, гипотеза) |
| `src/btzsc/models/reranker.py` | `RerankerModel._score_from_logits`, `_qwen_prompt` | реранкер: текст = запрос, метка = документ; для Qwen3-Reranker — промпт и вероятность «yes» |
| `src/btzsc/metrics.py` | `compute_metrics` | sklearn: `macro_f1`, `accuracy`, `macro_precision`, `macro_recall` (average="macro") |
| `src/btzsc/metrics.py` | `compute_task_summary` | невзвешенное среднее метрик внутри группы задач + `overall` |
| `src/btzsc/benchmark.py` | `BTZSCBenchmark.evaluate` | цикл по датасетам: `predict` → `compute_metrics` → `compute_task_summary` |
| `src/btzsc/benchmark.py` | `_resolve_model_revision / _resolve_precision / _resolve_device` | метаданные прогона в артефакт лидерборда (в опубликованных JSON стоит `"unknown"`) |
| `src/btzsc/benchmark.py` | `BTZSCResults.to_json / to_csv` | формат, совпадающий с файлами `paper_scores/*.json` |
| `src/btzsc/_baselines/*.csv` | — | опубликованные baseline-таблицы (acc/f1/precision/recall/roc) |
| `hf/results_repo/results/**` | — | 30 JSON лидерборда; 13 из них лежат в нашем `paper_scores/` без изменений |

## Реконструкция пар → multiclass (что именно копируем)

Формат строки HF-датасета: `text`, `hypothesis`, `labels ∈ {0,1}` (ClassLabel
`not_entailment/entailment`), `dataset_id`, `label_text`.
Подряд идущие `n_classes` строк относятся к одному тексту; gold — позиция строки с `labels == 1`.
`load_btzsc_dataset` берёт `texts[offset]`, список гипотез группы и `sample_binary.index(1)`.

Два места, где мы отклоняемся от официального кода (реализовано в `btzsc_ru/reconstruction.py`):

1. **Определение числа классов.** `_get_n_classes` ищет первый повтор `labels[0]`. Если gold первых
   двух примеров относятся к разным классам, эвристика ошибается: для паттерна
   `[1,0,0][0,1,0]` она вернёт 4 вместо 3 (воспроизведено в `tests/test_reconstruction.py::
   test_binary_pattern_heuristic_is_fragile_hypotheses_are_not`). Мы определяем период по колонке
   `hypothesis` и дополнительно проверяем, что порядок гипотез одинаков во всех группах.
2. **Группы без gold.** Официальный код делает `true_idx = ... if 1 in sample_binary else 0`,
   то есть молча приписывает класс 0. Мы такие группы отбрасываем и считаем в `dropped`
   (правило: ровно один gold). На agnews и imdb таких групп быть не должно;
   на banking77 в MVP их было 200.

Также: `load_btzsc_dataset(max_samples=...)` семплирует с `SAMPLING_SEED = 0` и всегда включает
примеры 0 и 1. Мы этот способ не используем — у нас стратифицированная выборка с seed 42 и
внешний `sample_manifest.csv`, потому что нужен один и тот же срез для всех моделей.

## Протокол LLM (основание для нашего адаптера)

`LLMModel` не вызывает `generate`: берётся `logits[:, last_pos, :]`, softmax, затем срез по
токенам-буквам опций и повторная нормировка. Padding side — `left` для mistral/qwen3.
Метки перед построением промпта сортируются (`sorted(set(labels))`), затем столбцы результата
возвращаются в исходный порядок. Chat template официальный адаптер не применяет.

Наш адаптер (`btzsc_ru/adapters.LLMAdapter`) повторяет схему «логиты первого токена по буквам»,
но: не пересортировывает метки (порядок задаёт датасет), оборачивает промпт в chat template из
карточки модели, если он есть, и грузит модель в 4-bit с batch = 1 (ограничения Colab).
Инструкция промпта переведена на русский, потому что оцениваются в том числе RU-датасеты.
Все отклонения перечислены в docstring класса; результаты прогона выводятся в `README.md`.

## Протокол реранкера и NLI (основание для наших адаптеров)

`RerankerModel`: `AutoModelForSequenceClassification`, пара «текст (запрос) × вербализация
класса (документ)» одним проходом; колонка логита выбирается в `_score_from_logits` по
`label2id` конфига (ключи `relevant`/`entailment`/`true`/`yes`, иначе последняя колонка;
при `num_labels=1` — единственная). Для Qwen3-Reranker — отдельная ветка: ChatML-промпт и
softmax(«no», «yes»)[1] по токенам 2152/9693. Длина не ограничивается снаружи
(`truncation=True` → предел модели). Дефолтный batch — 8 пар.

`NLIModel`: та же архитектура; скор пары — логит класса entailment, индекс которого ищет
`_find_entailment_idx` по ключам `entailment`/`label_2`/`true`/`yes`, иначе
`max(label2id.values())`, иначе 0. Дефолтный batch — 16 пар.

Наши `btzsc_ru/adapters.RerankerAdapter` и `NLIAdapter` — перенос буква в букву (логика
выбора колонки зафиксирована тестами `tests/test_cross_encoders.py` на пустышках).
Отличия только в обвязке: lazy-импорты, `revision` из `ModelSpec`, fp16 на CUDA для
реранкера 568M (авторы считали в bfloat16 на A100; на T4 эффективного bf16 нет).
Наши чекпоинты: `BAAI/bge-reranker-v2-m3` (num_labels=1 → `logits[:, 0]`),
`cointegrated/rubert-base-cased-nli-threeway` (`entailment=0`).
