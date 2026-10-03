"""Адаптеры моделей.

Контракт адаптера (совпадает с официальным `btzsc.models.base.BaseModel`,
см. docs/btzsc_code_notes.md): в адаптер передаются ТОЛЬКО тексты и описания классов.
Gold-меток в сигнатурах нет — это проверяется тестом и вручную по сигнатурам.

    predict_scores(texts: list[str], labels: list[str], batch_size: int) -> np.ndarray  # (n_texts, n_labels)
    predict(texts, labels, batch_size) -> np.ndarray  # argmax по оси меток

torch/transformers импортируются лениво внутри методов.
"""

from __future__ import annotations

from typing import Protocol

from .config import ModelSpec


class Adapter(Protocol):
    """Протокол адаптера: gold в сигнатуры не входит."""

    def predict_scores(self, texts: list[str], labels: list[str], batch_size: int = 8): ...

    def predict(self, texts: list[str], labels: list[str], batch_size: int = 8): ...


def paper_code_prefixes(model_id: str) -> tuple[str, str]:
    """Префиксы ровно по коду авторов (btzsc/models/embedding.py::_format_query/_format_label).

    В §4 статьи написано «префиксы из карточек моделей», но фактический код навешивает их
    только на e5-*, Qwen3-Embedding и e5-mistral; BGE, GTE и остальные идут без префикса.
    Опубликованные числа получены именно этим кодом, поэтому для воспроизведения нужен он,
    а «карточная» политика доступна через prefix_policy="model_card".
    """
    name = model_id.lower()
    if "qwen3-embedding" in name or "e5-mistral" in name:
        inst = "Given a piece of text, retrieve relevant label descriptions that best match the text"
        return (f"Instruct: {inst}\nQuery: ", "")
    if "e5-" in name:
        return ("query: ", "passage: ")
    return ("", "")


# Списки ключей выбора колонки логита — дословно из официального кода статьи.
# btzsc/models/reranker.py::_score_from_logits — колонка релевантности:
RELEVANCE_KEYS = ["relevant", "entailment", "true", "yes"]
# btzsc/models/nli.py::_find_entailment_idx — индекс entailment:
ENTAILMENT_KEYS = ["entailment", "label_2", "true", "yes"]


def _silence_hf() -> None:
    """Понижает болтливость transformers/datasets после их импорта (переменных окружения мало)."""
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:  # noqa: BLE001
        pass
    try:
        import datasets

        datasets.logging.set_verbosity_error()
        datasets.disable_progress_bars()
    except Exception:  # noqa: BLE001
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:  # noqa: BLE001
        pass


class EncoderAdapter:
    """Энкодер: pooling/prefix из карточки → L2-норма → cosine → argmax.

    `backend="st"` (по умолчанию) — код-путь статьи: SentenceTransformer.encode с
    normalize_embeddings=True и собственным max_seq_length модели, как у авторов.
    `backend="hf"` — наша реализация на transformers с явным max_length; оставлена как
    диагностика, чтобы можно было измерить вклад различий в обработке длинных текстов.
    """

    role = "encoder"

    def __init__(self, spec: ModelSpec, *, device: str | None = None, max_length: int | None = None,
                 backend: str = "st", prefix_policy: str = "paper_code") -> None:
        import torch

        _silence_hf()

        if spec.pooling not in {"cls", "mean"}:
            raise ValueError(f"{spec.model_id}: неизвестный pooling {spec.pooling!r}")
        if prefix_policy not in {"paper_code", "model_card"}:
            raise ValueError(f"неизвестная политика префиксов: {prefix_policy!r}")
        self.prefix_policy = prefix_policy
        if prefix_policy == "paper_code":
            self.query_prefix, self.label_prefix = paper_code_prefixes(spec.model_id)
        else:
            self.query_prefix, self.label_prefix = spec.query_prefix, spec.label_prefix
        if backend not in {"hf", "st"}:
            raise ValueError(f"неизвестный backend энкодера: {backend!r}")
        self.spec = spec
        # None = длина самой модели (как в коде авторов, где SentenceTransformer берёт
        # свой max_seq_length). Жёсткие 256 обрезали длинные тексты IMDb и AG News.
        self.max_length = max_length
        self.backend = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if backend == "st":
            from sentence_transformers import SentenceTransformer

            self.st_model = SentenceTransformer(spec.model_id, revision=spec.revision, device=self.device)
            self.tokenizer = None
            self.model = None
            return

        from transformers import AutoModel, AutoTokenizer

        self.st_model = None
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
        self.model = AutoModel.from_pretrained(spec.model_id, revision=spec.revision)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(hidden, mask):
        mask_f = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)

    def _encode(self, texts: list[str], prefix: str, batch_size: int):
        import torch

        if self.backend == "st":
            import numpy as np

            vectors = self.st_model.encode(
                [prefix + t for t in texts],
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return torch.from_numpy(np.asarray(vectors, dtype="float32"))
        import torch.nn.functional as F

        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[start : start + batch_size]]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
            vec = hidden[:, 0] if self.spec.pooling == "cls" else self._mean_pool(hidden, enc["attention_mask"])
            chunks.append(F.normalize(vec.float(), p=2, dim=1).cpu())
        return torch.cat(chunks, dim=0)

    def predict_scores(self, texts: list[str], labels: list[str], batch_size: int = 8):
        q = self._encode(texts, self.query_prefix, batch_size)
        y = self._encode(labels, self.label_prefix, batch_size)
        return (q @ y.T).numpy()

    def predict(self, texts: list[str], labels: list[str], batch_size: int = 8):
        return self.predict_scores(texts, labels, batch_size=batch_size).argmax(axis=1)


class RerankerAdapter:
    """Реранкер (cross-encoder): текст = запрос, вербализация класса = документ.

    Перенос БУКВА В БУКВУ официального `btzsc/models/reranker.py::RerankerModel`:
    `AutoModelForSequenceClassification`, пара (текст, метка) одним проходом,
    выбор колонки логита в `_score_from_logits` по `label2id` из конфига модели
    (ключи relevant/entailment/true/yes, иначе последняя колонка; при num_labels=1 —
    единственная). Ветка Qwen3-Reranker (промпт + вероятность «yes») сохранена,
    хотя наш чекпоинт bge-reranker-v2-m3 в неё не попадает.

    Отличия от оригинала — только обвязка: lazy-импорты, revision из ModelSpec,
    fp16 на CUDA (у авторов bfloat16 на A100; на T4 эффективного bf16 нет,
    а fp32 для 568M вдвое медленнее). Длину не ограничиваем, как и авторы:
    truncation=True берёт предел самой модели.
    """

    role = "reranker"

    def __init__(self, spec: ModelSpec, *, device: str | None = None, torch_dtype=None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _silence_hf()
        self.spec = spec
        self.truncated_prompts = 0
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if torch_dtype is None and self.device == "cuda":
            torch_dtype = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            spec.model_id, revision=spec.revision, torch_dtype=torch_dtype
        )
        self.model.to(self.device)
        self.model.eval()
        self.is_qwen_reranker = "Qwen3-Reranker" in spec.model_id
        self.token_true_id, self.token_false_id = 9693, 2152

    def _score_from_logits(self, logits):
        """Колонка релевантности из логитов — дословно официальный `_score_from_logits`."""
        if logits.ndim == 1:
            return logits
        if logits.shape[-1] == 1:
            return logits[:, 0]
        mapping = getattr(self.model.config, "label2id", {}) or {}
        lower = {str(k).lower(): int(v) for k, v in mapping.items()}
        for key in RELEVANCE_KEYS:
            if key in lower:
                return logits[:, lower[key]]
        return logits[:, -1]

    @staticmethod
    def _qwen_prompt(text: str, label: str) -> str:
        """Промпт Qwen3-Reranker — дословно официальный `_qwen_prompt`."""
        inst = "Given a piece of text, retrieve relevant label descriptions that best match the text"
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
            'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return f"{prefix}<Instruct>: {inst}\n<Query>: {text}\n<Document>: {label}{suffix}"

    def predict_scores(self, texts: list[str], labels: list[str], batch_size: int = 8):
        import numpy as np
        import torch

        output: list[np.ndarray] = []
        with torch.no_grad():
            for text in texts:
                row: list[float] = []
                for i in range(0, len(labels), batch_size):
                    chunk = labels[i : i + batch_size]
                    if self.is_qwen_reranker:
                        prompts = [self._qwen_prompt(text, lbl) for lbl in chunk]
                        enc = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(
                            self.device
                        )
                        logits = self.model(**enc).logits[:, -1, :]
                        true_vec = logits[:, self.token_true_id]
                        false_vec = logits[:, self.token_false_id]
                        scores = torch.stack([false_vec, true_vec], dim=1).softmax(dim=1)[:, 1]
                    else:
                        enc = self.tokenizer(
                            [text] * len(chunk), chunk, padding=True, truncation=True, return_tensors="pt"
                        ).to(self.device)
                        logits = self.model(**enc).logits
                        scores = self._score_from_logits(logits)
                    row.extend(scores.detach().float().cpu().tolist())
                output.append(np.array(row, dtype=np.float32))
        return np.stack(output, axis=0)

    def predict(self, texts: list[str], labels: list[str], batch_size: int = 8):
        return self.predict_scores(texts, labels, batch_size=batch_size).argmax(axis=1)


class NLIAdapter:
    """NLI: логит класса entailment как скор пары (текст, гипотеза-вербализация).

    Перенос БУКВА В БУКВУ официального `btzsc/models/nli.py::NLIModel`: индекс entailment
    ищется в `label2id` по ключам entailment/label_2/true/yes (`_find_entailment_idx`),
    иначе max(label2id.values()); при num_labels=1 берётся единственная колонка.
    Отличия — только обвязка (lazy-импорты, revision из ModelSpec); dtype по умолчанию
    fp32, как у `from_pretrained` без аргумента (авторы передают dtype снаружи).
    """

    role = "nli"

    def __init__(self, spec: ModelSpec, *, device: str | None = None, torch_dtype=None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _silence_hf()
        self.spec = spec
        self.truncated_prompts = 0
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            spec.model_id, revision=spec.revision, torch_dtype=torch_dtype
        )
        self.model.to(self.device)
        self.model.eval()
        self.entailment_idx = self._find_entailment_idx()

    def _find_entailment_idx(self) -> int:
        """Индекс entailment из label2id — дословно официальный `_find_entailment_idx`."""
        mapping = getattr(self.model.config, "label2id", {}) or {}
        lower = {str(k).lower(): int(v) for k, v in mapping.items()}
        for key in ENTAILMENT_KEYS:
            if key in lower:
                return lower[key]
        if mapping:
            return int(max(mapping.values()))
        return 0

    def predict_scores(self, texts: list[str], labels: list[str], batch_size: int = 16):
        import numpy as np
        import torch

        all_scores: list[np.ndarray] = []
        with torch.no_grad():
            for text in texts:
                pairs = [(text, label) for label in labels]
                row_scores: list[float] = []
                for i in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[i : i + batch_size]
                    a = [x[0] for x in batch_pairs]
                    b = [x[1] for x in batch_pairs]
                    enc = self.tokenizer(a, b, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    logits = self.model(**enc).logits
                    if logits.ndim == 1:
                        vals = logits
                    elif logits.shape[-1] == 1:
                        vals = logits[:, 0]
                    else:
                        vals = logits[:, self.entailment_idx]
                    row_scores.extend(vals.detach().float().cpu().tolist())
                all_scores.append(np.array(row_scores, dtype=np.float32))
        return np.stack(all_scores, axis=0)

    def predict(self, texts: list[str], labels: list[str], batch_size: int = 16):
        return self.predict_scores(texts, labels, batch_size=batch_size).argmax(axis=1)


class LLMAdapter:
    """LLM: multiple-choice строго по btzsc/models/llm.py.

    Совпадает с официальным адаптером: тот же промпт, тот же алфавит из 100 символов,
    сортировка меток `sorted(set(labels))` с возвратом столбцов в исходный порядок,
    оценка по вероятности первого токена, без генерации, padding слева для mistral/qwen3.

    Единственное отличие — загрузка в 4-bit вместо bfloat16: это ограничение бесплатного
    Colab (15 ГБ против 80 ГБ A100 у авторов), оно задокументировано в отчёте.
    Chat template по умолчанию НЕ применяется (в коде статьи его нет); включается
    только явным `ModelSpec.chat_template=True` для диагностики.
    """

    role = "llm"

    # Алфавит вариантов — ровно как в btzsc/models/llm.py: латиница в двух регистрах
    # плюс греческий в двух регистрах = 100 символов (хватает на Banking77 с 77 классами).
    _SYMBOLS = (
        [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        + [chr(c) for c in range(ord("a"), ord("z") + 1)]
        + list("αβγδεζηθικλμνξοπρστυφχψω")
        + list("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")
    )
    # Промпт дословно из btzsc/models/llm.py (_DEFAULT_PROMPT).
    _INSTRUCTION = (
        "You are a text classifier.\n"
        "You will be given a text and several mutually exclusive options.\n"
        "Each option is prefixed by a single letter (e.g. A, b, γ, ...).\n"
        "Your task is to choose the single best option.\n\n"
        "IMPORTANT:\n"
        "- Answer with EXACTLY ONE LETTER used to prefix the options.\n"
        "- Do NOT output any words, punctuation, or explanation.\n\n"
        "TEXT:\n"
        "{text}\n\n"
        "OPTIONS:\n"
        "{options}\n\n"
        "Answer: The correct option is letter "
    )

    def __init__(
        self,
        spec: ModelSpec,
        *,
        device: str | None = None,
        max_length: int | None = None,
        load_in_4bit: bool | None = None,
        use_chat_template: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        # Лимит считается от задачи: промпт = текст + ВСЕ вербализации классов.
        # На Banking77 это 77 вариантов, в 256 токенов они не помещались и молча обрезались —
        # измерялась длина буфера, а не модель. None = взять лимит модели (но не больше 4096).
        self.max_length = max_length
        # В коде авторов chat template не применяется; включается только явно, для диагностики.
        self.use_chat_template = use_chat_template or bool(spec.chat_template and False)
        self.truncated_prompts = 0
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device != "cuda":
            raise RuntimeError("SKIPPED_NO_GPU: LLM запускается только на GPU (протокол эксперимента)")

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision, padding_side="left")
        kwargs: dict = {"revision": spec.revision}
        use_4bit = spec.load_in_4bit if load_in_4bit is None else load_in_4bit
        if use_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_prompt(self, text: str, labels: list[str]) -> str:
        options = "\n".join(f"{self._SYMBOLS[i]}) {label}" for i, label in enumerate(labels))
        body = self._INSTRUCTION.format(text=text, options=options)
        if self.spec.chat_template and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True
            )
        return body

    def _letter_ids(self, n_labels: int):
        import torch

        ids = []
        for sym in self._SYMBOLS[:n_labels]:
            tok = self.tokenizer(sym, add_special_tokens=False).input_ids
            if len(tok) != 1:
                raise ValueError(f"символ опции {sym!r} не однотокенный для {self.spec.model_id}")
            ids.append(tok[0])
        return torch.tensor(ids, device=self.model.device, dtype=torch.long)

    def predict_scores(self, texts: list[str], labels: list[str], batch_size: int = 1):
        import numpy as np
        import torch

        # Канонический порядок опций — как в официальном коде; столбцы возвращаются
        # в исходный порядок меток ниже.
        sorted_labels = sorted(set(labels))
        if len(sorted_labels) > len(self._SYMBOLS):
            raise ValueError(f"слишком много классов ({len(sorted_labels)}), максимум {len(self._SYMBOLS)}")
        letter_ids = self._letter_ids(len(sorted_labels))
        rows = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                prompts = [self.build_prompt(t, sorted_labels) for t in texts[start : start + batch_size]]
                limit = self.max_length or min(
                    getattr(self.tokenizer, "model_max_length", 4096) or 4096, 4096
                )
                enc = self.tokenizer(
                    prompts, padding=True, truncation=True, max_length=limit, return_tensors="pt"
                ).to(self.model.device)
                for prompt in prompts:   # считаем, сколько промптов не поместилось целиком
                    if len(self.tokenizer(prompt, add_special_tokens=False).input_ids) > limit:
                        self.truncated_prompts += 1
                logits = self.model(**enc).logits[:, -1, :]
                probs = logits.softmax(dim=-1)[:, letter_ids]
                probs = probs / probs.sum(dim=-1, keepdim=True)
                rows.append(probs.detach().float().cpu().numpy())
        scores_sorted = np.concatenate(rows, axis=0)
        # Возврат столбцов в исходный порядок меток (btzsc/models/llm.py делает то же самое).
        index = {label: i for i, label in enumerate(sorted_labels)}
        order = [index[label] for label in labels]
        return scores_sorted[:, order]

    def predict(self, texts: list[str], labels: list[str], batch_size: int = 1):
        return self.predict_scores(texts, labels, batch_size=batch_size).argmax(axis=1)


class DummyAdapter:
    """Детерминированный «псевдо-энкодер» для самопроверки конвейера без скачивания весов.

    Включается переменной окружения BTZSC_FAKE_MODELS=1. Считает примитивную лексическую
    близость текста и описания класса: числа бессмысленны, но проходят весь путь
    предсказания → метрики → results.csv → экспорт → импорт → отчёт.
    В настоящем прогоне не используется.
    """

    role = "encoder"

    def __init__(self, spec: ModelSpec, **_: object) -> None:
        self.spec = spec
        self.truncated_prompts = 0

    @staticmethod
    def _score(text: str, label: str) -> float:
        words = set(str(text).lower().split())
        return len(words & set(str(label).lower().split())) / (1 + len(str(label).split()))

    def predict_scores(self, texts, labels, batch_size: int = 8):
        import numpy as np

        shift = sum(ord(c) for c in self.spec.model_id) % 7
        return np.array([[self._score(t, lab) + ((shift + i) % 5) / 100 for i, lab in enumerate(labels)]
                         for t in texts])

    def predict(self, texts, labels, batch_size: int = 8):
        return self.predict_scores(texts, labels, batch_size=batch_size).argmax(axis=1)


def build_adapter(spec: ModelSpec, **kwargs) -> Adapter:
    """Фабрика адаптера по роли модели. Лишние для роли аргументы отбрасываются."""
    import os

    if os.environ.get("BTZSC_FAKE_MODELS") == "1":
        return DummyAdapter(spec)      # самопроверка конвейера, см. docstring DummyAdapter
    if spec.role == "encoder":
        kwargs.pop("use_chat_template", None)
        return EncoderAdapter(spec, **kwargs)
    kwargs.pop("backend", None)
    kwargs.pop("prefix_policy", None)
    if spec.role in ("reranker", "nli"):
        kwargs.pop("use_chat_template", None)
        kwargs.pop("max_length", None)       # кросс-энкодеры не ограничивают длину, как и авторы
        cls = RerankerAdapter if spec.role == "reranker" else NLIAdapter
        return cls(spec, **kwargs)
    if spec.role == "llm":
        return LLMAdapter(spec, **kwargs)
    raise ValueError(f"неизвестная роль: {spec.role}")
