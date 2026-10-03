# model_audit.md — аудит моделей и датасетов (шаг D3)

Проверка без скачивания весов и без полной загрузки датасетов: HF API `/api/models/<id>`
(`expand[]=sha,safetensors,config,library_name,tags`), файлы карточек (`1_Pooling/config.json`,
`config_sentence_transformers.json`, `tokenizer_config.chat_template`), datasets-server
`/splits`, `/rows?length≤3`. Полные контракты — в `contracts.json`.

## Модели

| Модель | Роль | revision (SHA) | Параметры | ≤4B | Лицензия | Pooling | Префикс запроса / метки | Источник pooling/prefix | Есть в статье? | Статус |
|---|---|---|---:|:--:|---|---|---|---|:--:|---|
| cointegrated/rubert-tiny2 | энкодер | `e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae` | 29 378 550 | да | MIT | CLS | нет / нет | `1_Pooling/config.json` | нет | OK |
| sergeyzh/rubert-tiny-turbo | энкодер | `93769a3baad2b037e5c2e4312fccf6bcfe082bf1` | 29 378 550¹ | да | MIT | CLS | нет / нет | `1_Pooling/config.json` | нет | OK |
| ai-forever/ru-en-RoSBERTa | энкодер | `89fb1651989adbb1cfcfdedafd7d102951ad0555` | 403 707 904 | да | MIT | CLS | `classification: ` / `classification: ` | `config_sentence_transformers.json` → prompts | нет | OK |
| deepvk/USER-bge-m3 | энкодер | `0cc6cfe48e260fb0474c753087a69369e88709ae` | 359 026 688 | да | Apache-2.0 | CLS | нет / нет | `1_Pooling/config.json`, prompts = {} | нет | OK |
| intfloat/multilingual-e5-base | энкодер | `d128750597153bb5987e10b1c3493a34e5a4502a` | 278 044 162 | да | MIT | mean | `query: ` / `passage: ` | `1_Pooling/config.json` + карточка E5 | нет² | OK |
| Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct | LLM | `c9f763e5097587066e602ebabed4d93ea29bb60c` | 1 543 298 048 | да | Apache-2.0 | — | chat template ChatML | `tokenizer_config.chat_template` | нет | OK |
| Qwen/Qwen2.5-1.5B-Instruct | LLM | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 1 543 714 304 | да | Apache-2.0 | — | chat template ChatML | карточка Qwen2.5 | нет³ | OK |

¹ HF API не отдаёт блок `safetensors` для этого репозитория; число параметров указано по базовой
модели `cointegrated/rubert-tiny2`, заявленной в карточке (`base_model`). Точное число будет
зафиксировано в Colab на шаге preflight (`sum(p.numel() for p in model.parameters())`).
² В статье есть `e5-base-v2` и `e5-large-v2` — это англоязычные чекпоинты другого семейства весов;
точного ID `intfloat/multilingual-e5-base` в статье нет.
³ В статье есть `Qwen3-4B` и `Qwen3-8B`; `Qwen2.5-1.5B-Instruct` в статье отсутствует.

Замен моделей из фиксированного списка протокол эксперимента **не производилось**: все семь существуют,
доступны публично, укладываются в лимит 4B. Резервный список не задействован.

Происхождение моделей: rubert-tiny2 (cointegrated), rubert-tiny-turbo (sergeyzh),
ru-en-RoSBERTa (SberDevices / ai-forever), USER-bge-m3 (deepvk, VK) — российские разработчики;
multilingual-e5-base (Microsoft) и Qwen2.5-1.5B-Instruct (Alibaba) — **иностранные мультиязычные
baseline**, они не называются российскими; Vikhr-Qwen-2.5-1.5B-Instruct — российское дообучение
(Vikhrmodels) поверх иностранной базы Qwen2.5, что и указывается в отчёте.

## Датасеты

| Ключ | HF id | config | split | Строк | Колонки | Метки | Статус |
|---|---|---|---|---:|---|---|---|
| btzsc_agnews | btzsc/btzsc | `agnews` | `test` | 30 400 пар (7 600 примеров) | text, hypothesis, labels, dataset_id, label_text | ClassLabel `not_entailment/entailment`, 4 гипотезы на текст | OK |
| btzsc_imdb | btzsc/btzsc | `imdb` | `test` | 20 000 пар (10 000 примеров) | те же | 2 гипотезы на текст | OK |
| ru_sentiment | MonoHime/ru_sentiment_dataset | `default` | **`validation`** | 21 098 | Unnamed: 0, text, sentiment | 0=NEUTRAL, 1=POSITIVE, 2=NEGATIVE (README датасета) | OK |
| massive_en | mteb/amazon_massive_intent | `en` | `test` | 2 974 | id, label, label_text, text, lang | 59 слагов интентов (`alarm_set`, …) | OK |
| massive_ru | mteb/amazon_massive_intent | `ru` | `test` | 2 974 | id, label, label_text, text, lang | те же 59 слагов | OK |

Число классов MASSIVE (59) подтверждено полным прогоном `run1`; проба preflight по первым 50 строкам
показывает лишь 22 — по пробе число классов определять нельзя.

Реальные splits `MonoHime/ru_sentiment_dataset`: только `train` (190k) и `validation` (21 098).
Split `test` отсутствует — в отчёте и в коде используется имя `validation`.

### Замена датасета (обязательное обоснование)

протокол эксперимента предписывает `AmazonScience/massive` с конфигами `en-US` / `ru-RU`. Проверка:
datasets-server `/splits?dataset=AmazonScience/massive` возвращает
`RuntimeError: Dataset scripts are no longer supported, but found massive.py`.
То есть датасет существует, но опубликован в виде loading-скрипта и не загружается современным
`datasets` без `trust_remote_code` (в свежих версиях — вовсе). Это не «датасета нет», а
несовместимость формата.

Замена: **`mteb/amazon_massive_intent`**, конфиги `en` и `ru`, split `test` — тот же корпус
Amazon MASSIVE (intent detection, 60 интентов), опубликованный в parquet.
Роль в эксперименте сохраняется полностью: параллельная EN/RU задача с идентичным набором меток,
на которой считается delta EN−RU. Оба конфига содержат по 2 974 примера в `test`,
набор `label_text` совпадает — условие расчёта delta из протокол эксперимента выполнимо; проверка совпадения
наборов выполняется в коде (`btzsc_ru.data.label_sets_match`) перед расчётом.

## Вербализации (фиксируются до оценки)

* BTZSC (agnews, imdb) — гипотезы берутся из самого датасета (колонка `hypothesis`), т. е. те же
  формулировки, что у авторов.
* ru_sentiment — «Тональность этого русского текста {нейтральная|положительная|отрицательная}».
* massive_en — «The user request to the voice assistant is about {intent}»;
  massive_ru — «Запрос пользователя к голосовому ассистенту относится к теме: {intent}».
  Слаг интента остаётся английским в обоих языках (так он хранится в датасете) — это ограничение
  зафиксировано в отчёте: русифицирована только рамка шаблона.
