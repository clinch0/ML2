"""C3: контракт адаптеров (пакет btzsc_ru) — gold-метки не входят в сигнатуру predict.

Тяжёлые модели не загружаются (нет сети/GPU в CI); проверяем сигнатуры и
конфигурацию pooling/prefix из config.py против contracts.json.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from btzsc_ru import config
from btzsc_ru.adapters import EncoderAdapter, LLMAdapter

ROOT = Path(__file__).resolve().parent.parent


def test_predict_signature_has_no_gold():
    for cls in (EncoderAdapter, LLMAdapter):
        params = list(inspect.signature(cls.predict).parameters)
        assert params[:3] == ["self", "texts", "labels"], (cls, params)
        # никакого gold/references/y_true в сигнатуре
        assert not any(p in params for p in ("gold", "references", "y_true", "labels_true"))


def test_encoder_pooling_matches_contracts():
    contracts = json.loads((ROOT / "contracts.json").read_text(encoding="utf-8"))
    cmodels = contracts["models"]
    for spec in config.ENCODERS:
        c = cmodels[spec.model_id]
        assert spec.pooling == c["pooling"], spec.model_id
        assert spec.revision == c["revision"], spec.model_id
        assert spec.query_prefix == c["query_prefix"], spec.model_id
        assert spec.label_prefix == c["label_prefix"], spec.model_id


def test_llm_specs_match_contracts():
    contracts = json.loads((ROOT / "contracts.json").read_text(encoding="utf-8"))
    for spec in config.LLMS:
        assert spec.revision == contracts["models"][spec.model_id]["revision"], spec.model_id


def test_all_specs_have_revision():
    for spec in config.ENCODERS + config.LLMS:
        assert spec.revision and spec.revision != "unknown"


def test_build_adapter_accepts_everything_runner_sends():
    """Регрессия: раннер слал prefix_policy, а конструктор его не принимал — 126 пар FAILED.

    Тест ловит рассинхрон сигнатур без загрузки весов: проверяется, что каждый аргумент,
    который runner.evaluate передаёт в build_adapter, присутствует в конструкторе адаптера.
    """
    import inspect

    from btzsc_ru import adapters, runner
    from btzsc_ru.config import ENCODERS, LLMS, RunConfig

    source = inspect.getsource(runner.evaluate)
    start = source.index("build_adapter(")
    call = source[start:source.index(")", start)]
    sent = {line.split("=")[0].strip() for line in call.split("\n")[1:] if "=" in line}
    assert {"backend", "prefix_policy", "use_chat_template"} <= sent, f"раннер шлёт: {sent}"

    cfg = RunConfig()
    for spec, cls in ((ENCODERS[0], adapters.EncoderAdapter), (LLMS[0], adapters.LLMAdapter)):
        accepted = set(inspect.signature(cls.__init__).parameters)
        kwargs = {"backend": cfg.encoder_backend, "prefix_policy": cfg.prefix_policy,
                  "use_chat_template": cfg.use_chat_template}
        if spec.role == "encoder":
            kwargs.pop("use_chat_template")      # отбрасывается фабрикой
        else:
            kwargs.pop("backend"), kwargs.pop("prefix_policy")
        missing = set(kwargs) - accepted
        assert not missing, f"{cls.__name__} не принимает {missing}"


def test_encoder_uses_policy_prefixes_not_spec_prefixes():
    """Префиксы берутся из выбранной политики, а не напрямую из ModelSpec."""
    import inspect

    from btzsc_ru.adapters import EncoderAdapter

    src = inspect.getsource(EncoderAdapter.predict_scores)
    assert "self.query_prefix" in src and "self.label_prefix" in src
    assert "self.spec.query_prefix" not in src, "политика префиксов игнорировалась"
