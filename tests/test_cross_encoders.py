"""Контракт кросс-энкодеров (reranker, NLI) и расширенного якоря.

Тяжёлые модели не загружаются (нет сети/GPU в CI): логика выбора колонки логита
проверяется на объектах-пустышках через object.__new__, остальное — по config.py
и contracts.json. Эталон поведения — официальные btzsc/models/reranker.py и nli.py.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from btzsc_ru import config
from btzsc_ru.adapters import ENTAILMENT_KEYS, RELEVANCE_KEYS, NLIAdapter, RerankerAdapter

ROOT = Path(__file__).resolve().parent.parent


class _FakeConfig:
    def __init__(self, label2id):
        self.label2id = label2id


class _FakeModel:
    def __init__(self, label2id):
        self.config = _FakeConfig(label2id)


def _reranker_with(label2id) -> RerankerAdapter:
    obj = object.__new__(RerankerAdapter)
    obj.model = _FakeModel(label2id)
    return obj


def _nli_with(label2id) -> NLIAdapter:
    obj = object.__new__(NLIAdapter)
    obj.model = _FakeModel(label2id)
    return obj


# ── сигнатуры: gold-меток в predict нет ──────────────────────────────────────

def test_predict_signature_has_no_gold():
    for cls in (RerankerAdapter, NLIAdapter):
        params = list(inspect.signature(cls.predict).parameters)
        assert params[:3] == ["self", "texts", "labels"], (cls, params)
        assert not any(p in params for p in ("gold", "references", "y_true", "labels_true"))


# ── выбор колонки логита: дословно официальный код ───────────────────────────

def test_relevance_keys_match_official_code():
    """Списки ключей — ровно как в btzsc/models/{reranker,nli}.py, порядок важен."""
    assert RELEVANCE_KEYS == ["relevant", "entailment", "true", "yes"]
    assert ENTAILMENT_KEYS == ["entailment", "label_2", "true", "yes"]


def test_reranker_single_logit_column():
    """bge-reranker-v2-m3: num_labels=1 → берётся единственная колонка."""
    r = _reranker_with({"LABEL_0": 0})
    logits = np.array([[0.7], [-0.2]])
    assert np.allclose(r._score_from_logits(logits), [0.7, -0.2])


def test_reranker_column_by_label2id_keys():
    r = _reranker_with({"irrelevant": 0, "relevant": 1})
    logits = np.array([[0.1, 0.9], [0.8, 0.2]])
    assert np.allclose(r._score_from_logits(logits), [0.9, 0.2])
    # регистр не важен, как в официальном коде (str(k).lower())
    r = _reranker_with({"No": 0, "Yes": 1})
    assert np.allclose(r._score_from_logits(logits), [0.9, 0.2])


def test_reranker_falls_back_to_last_column():
    r = _reranker_with({"foo": 0, "bar": 1, "baz": 2})
    logits = np.array([[0.1, 0.2, 0.3]])
    assert np.allclose(r._score_from_logits(logits), [0.3])


def test_nli_entailment_idx_for_our_checkpoint():
    """rubert-base-cased-nli-threeway: entailment=0 (contracts.json)."""
    contracts = json.loads((ROOT / "contracts.json").read_text(encoding="utf-8"))
    assert "entailment': 0" in contracts["models"]["cointegrated/rubert-base-cased-nli-threeway"]["prefix_source"]
    n = _nli_with({"entailment": 0, "contradiction": 1, "neutral": 2})
    assert n._find_entailment_idx() == 0


def test_nli_entailment_idx_fallbacks():
    assert _nli_with({"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2})._find_entailment_idx() == 2  # ключ label_2
    assert _nli_with({"a": 0, "b": 1})._find_entailment_idx() == 1       # max(values) как у авторов
    assert _nli_with({})._find_entailment_idx() == 0


# ── конфигурация: спеки, контракты, блок Б, якорь ────────────────────────────

def test_cross_encoder_specs_match_contracts():
    contracts = json.loads((ROOT / "contracts.json").read_text(encoding="utf-8"))
    for spec in config.CROSS_ENCODERS:
        c = contracts["models"][spec.model_id]
        assert spec.revision == c["revision"], spec.model_id
        assert spec.role == c["role"], spec.model_id
        assert spec.params == c["params_total"], spec.model_id
        assert spec.pooling is None, "кросс-энкодеру pooling не нужен"


def test_cross_encoders_are_in_block_b_only():
    """Новые строки идут в block_b с тем же манифестом и тем же n; в ru_extension их нет."""
    ids = {m.model_id for m in config.CROSS_ENCODERS}
    assert ids <= set(config.BLOCK_B_MODEL_IDS)
    # ru_extension считается моделями по умолчанию (ENCODERS + LLMS) — кросс-энкодеры не входят
    default_ids = set(config.RunConfig().model_ids)
    assert not (ids & default_ids), "кросс-энкодеры не должны попадать в ru_extension по умолчанию"
    assert not (ids & set(config.EXTRA_MODEL_IDS))


def test_cross_encoder_batch_sizes_match_official_defaults():
    by_role = {m.role: m for m in config.CROSS_ENCODERS}
    assert by_role["reranker"].batch_size == 8    # дефолт btzsc/models/reranker.py
    assert by_role["nli"].batch_size == 16        # дефолт btzsc/models/nli.py


def test_build_adapter_accepts_everything_runner_sends():
    """Фабрика отбрасывает энкодерные kwargs для ролей reranker/nli (рассинхрон = FAILED-пары)."""
    from btzsc_ru import adapters

    for cls in (RerankerAdapter, NLIAdapter):
        accepted = set(inspect.signature(cls.__init__).parameters)
        assert "device" in accepted and "torch_dtype" in accepted
        assert "spec" in accepted
    src = inspect.getsource(adapters.build_adapter)
    for role in ("reranker", "nli"):
        assert role in src


def test_anchor_covers_long_datasets():
    """Якорь обязан включать IMDb и AG News: только там проверяется гипотеза об обрезке длины."""
    import os

    if os.environ.get("BTZSC_ANCHOR_DATASETS"):   # офлайн-самопроверки переопределяют состав
        return
    assert "btzsc_imdb" in config.ANCHOR_DATASET_KEYS
    assert "btzsc_agnews" in config.ANCHOR_DATASET_KEYS
    # короткая диагностика никуда не делась
    assert "btzsc_rottentomatoes" in config.ANCHOR_DATASET_KEYS
    assert config.ANCHOR_N is None, "якорь считается на полном сплите (A.4 статьи)"


def test_anchor_rows_remain_comparable_to_paper():
    """Полный сплит + st + paper_code → comparable_to_paper=True и для длинных датасетов."""
    from btzsc_ru.paper_compat import comparable_to_paper

    for key in ("btzsc_imdb", "btzsc_agnews"):
        verdict = comparable_to_paper(key, None, encoder_backend="st",
                                      prefix_policy="paper_code", use_chat_template=False)
        assert verdict.comparable, (key, verdict.reason)


def test_runner_gives_cross_encoders_their_own_batch_size():
    from btzsc_ru import runner

    src = inspect.getsource(runner.evaluate)
    assert '"reranker"' in src and '"nli"' in src, "batch_size кросс-энкодеров должен браться из ModelSpec"
