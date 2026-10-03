# BTZSC-RU — задание 2: анализ статьи и собственный эксперимент

Разбор статьи **BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders,
Embedding Models, Rerankers and LLMs** (Ilias Aarab, arXiv:2603.11991, ICLR 2026) и перенос её
протокола на русский язык и компактные модели, помещающиеся в бесплатный Google Colab.

## Порядок чтения

1. **[`ТЗ_эксперимента.md`](ТЗ_эксперимента.md)** — что делает код: протокол, два блока, инварианты.
2. **[`Задание2_BTZSC.md`](Задание2_BTZSC.md)** — разбор статьи.
3. **[`Эксперимент_BTZSC.md`](Эксперимент_BTZSC.md)** — наш эксперимент и числа.
4. **[`Задание2_BTZSC.pptx`](Задание2_BTZSC.pptx)** — презентация.

Рабочие материалы (заметки по статье и коду, аудит моделей и датасетов, речь
к презентации) убраны в [`docs/`](docs). Инструкция по Colab — [`docs/colab.md`](docs/colab.md).

Основной прогон — `results/run5`: 162 строки, все `OK`. Он разделён на три теста колонкой `test`
(`baseline` — сверка кода на полных сплитах, `block_a`/`block_b` — обе группы моделей на 7
датасетах статьи, `ru_extension` — 7 русских задач). Средние **внутри одного теста**; графики
и слайды это соблюдают.

### Что лежит в git, а что нет

В репозитории хранятся сводные артефакты прогона: `results.csv`, `summary.csv`,
`run_config.json`, `environment.json`, `manifest_info.json`, `errors.log` — их хватает для всех
таблиц отчёта. Тяжёлые `predictions.jsonl` и `sample_manifest.csv` (десятки мегабайт, полностью
меняются после каждого прогона) и архивы `incoming/*.zip` в git не версионируются: они живут в
артефактах прогона Colab и в сборке CI.

`contracts.json` — проверенные контракты моделей и датасетов (revision, pooling, префиксы,
колонки, splits); из него берут значения `btzsc_ru/config.py` и тесты соответствия.

## Что читать в первую очередь

| Файл | О чём |
|---|---|
| [`ТЗ_эксперимента.md`](ТЗ_эксперимента.md) | что именно делает код: соответствие протоколу статьи, два блока, инварианты |
| [`Задание2_BTZSC.md`](Задание2_BTZSC.md) | анализ статьи по пунктам задания: постановка, цель, актуальность, новизна, методы, датасеты и бенчмарки, результаты авторов |
| [`Эксперимент_BTZSC.md`](Эксперимент_BTZSC.md) | наш эксперимент: модели, датасеты, протокол, результаты, сопоставление со статьёй, сверка кода |
| [`Задание2_BTZSC.pptx`](Задание2_BTZSC.pptx) | презентация на 13 слайдов (собирается скриптом) |
| [`figures/`](figures) | графики результатов, PNG с прозрачным фоном |

## Как устроен репозиторий

```
experiment.py              тонкий CLI: режимы all | preflight | smoke | evaluate | replicate | finetune | export
btzsc_ru/                  пакет эксперимента
  config.py                модели, датасеты, режимы, RunConfig; блок PAPER_MODELS для сверки
  data.py                  загрузка датасетов и вербализация меток
  reconstruction.py        восстановление multiclass-примеров из пар BTZSC
  sampling.py              единый sample_manifest.csv (seed 42)
  adapters.py              энкодеры (pooling/prefix → cosine) и LLM (multiple-choice по первому токену)
  metrics.py               macro-F1, accuracy, macro-P/R, покрытие классов, delta EN−RU
  io_utils.py              results.csv, predictions.jsonl, run_key, resume, бюджет времени, VRAM
  runner.py                оркестрация этапов, сверка на моделях статьи, экспорт архива
  finetune.py              дообучение rubert-tiny2 на MonoHime (пункт «8+»)
scripts/
  build_notebook.py        генерирует BTZSC_Colab.ipynb
  package_for_colab.py     собирает отдельный dist/colab_bundle.zip
  import_colab_outputs.py  импорт архива прогона: пересчёт метрик из predictions.jsonl
  make_figures.py          графики по артефактам прогона (PNG); руками графики не рисуются
  slide_figures.py         схемы-объяснялки для слайдов, тоже из кода и из чисел прогона
  check_replication.py     сверка наших чисел с опубликованными числами статьи
  build_reports.py         всё сразу: импорт → графики → сверка → презентация
BTZSC_Colab.ipynb          ноутбук для Colab; код проекта клонируется из репозитория
docs/colab.md            инструкция по запуску прогона
tests/                     132 теста на синтетике: реконструкция, выборка, метрики, resume, импорт, ноутбук, сверка
docs/                      заметки по статье и коду, аудит моделей и датасетов, речь к презентации
paper_scores/              опубликованные результаты авторов (13 JSON лидерборда)
data/                      локальная копия test-сплитов BTZSC (parquet)
results/<run_id>/          импортированные артефакты прогонов
incoming/                  сюда кладутся архивы colab_outputs_*.zip из Colab
```

## Рабочий цикл

1. **Прогон.** Открыть `BTZSC_Colab.ipynb` в Colab (T4), Runtime → Run all. Режим `all` сам проходит
   preflight → smoke → evaluate → replicate → finetune → export. Подробности — [`docs/colab.md`](docs/colab.md).
2. **Доставка результатов.** Скачанный `colab_outputs_<run_id>.zip` положить в `incoming/` и запушить.
3. **Сборка отчётов.** Автоматически: workflow [`import-results`](.github/workflows/import-results.yml)
   срабатывает на появление архива, распаковывает, пересчитывает метрики, обновляет отчёт, графики,
   сверку и презентацию.
   Вручную то же самое: `python scripts/build_reports.py`.
   Удаление обработанного архива пока отключено (флаг `--delete-zip`, в workflow закомментирован).

## Проверки

```bash
python -m pytest -q tests                      # 132 теста, без сети и без моделей
python scripts/package_for_colab.py --check-embedded   # в ноутбуке нет встроенного архива
python scripts/build_btzsc_pptx.py                     # пересобрать презентацию
```

CI ([`check.yml`](.github/workflows/check.yml)) прогоняет то же самое плюс компиляцию ячеек ноутбука
и проверку на отсутствие токенов. Модели и датасеты в CI не скачиваются: успешный CI означает,
что код проверен, а не что эксперимент выполнен.

## Сверка кода на моделях статьи

Чтобы не полагаться на слово «протокол совпадает», в конфиге есть блок `PAPER_MODELS`: пять
чекпоинтов из самой статьи (all-MiniLM-L6-v2, e5-base-v2, e5-large-v2, bge-base-en-v1.5,
bge-large-en-v1.5). Режим `replicate` прогоняет их нашим кодом на `agnews` и `imdb` на полном тесте,
а `scripts/check_replication.py` сравнивает результат с опубликованными macro-F1
(`btzsc_ru.config.PAPER_REFERENCE`, источник — `results.by_dataset` лидерборда официального
репозитория). Итог попадает в раздел 12 отчёта.

Результат сверки в прогоне `run5` (15 пар: 5 моделей × 3 датасета на полных тестовых сплитах):
**11 пар совпали** в пределах ±0.05, из них 6 — до третьего знака (обе модели BGE воспроизвелись
точно); 1 пара на границе (e5-base-v2 / Rotten Tomatoes, Δ = −0.059); 3 пары разошлись — все на
all-MiniLM-L6-v2 и все в одну сторону (+0.19…+0.32), что указывает на сбойные строки
в опубликованном рейтинге, а не на ошибку нашего кода. Для разбора есть переключатель кода кодирования:
`--encoder-backend st` считает энкодеры через `SentenceTransformer`, как в статье,
`--encoder-backend hf` (по умолчанию) — нашей реализацией.
